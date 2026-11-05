# -*- coding: utf-8 -*-
"""
2FAST - Fully Automated Synthesis Technology

Core experiment control module for automated nanoparticle synthesis exploration
using Bayesian Optimization with Gaussian Process Regression.

This module provides the Experiment class which orchestrates:
- Hardware control (pumps, spectrometer, fraction collector)
- Bayesian Optimization loop
- Data acquisition and storage
- Real-time spectral analysis

Author: Sylvain David
Created: December 14, 2023
"""
import json
import os
import sys
import time
import math
import csv
import datetime
import pickle
import shutil
import logging
import threading
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

from toofast.hardware.pump import PumpSystem
from toofast.hardware.spectrometer import Spectrometer, Spectrum
from toofast.hardware.fraction_collector import FractionCollector
from toofast.acquisition import (
    ExpectedImprovement, ExplorationWrapper, LowerConfidenceBound, ThompsonSampling,
    FeasibilityWeightedAcquisition, fit_feasibility_classifier,
)
from toofast.hardware.mass_flow_controller import FlowController
from toofast.hardware.c3000 import C3000
from toofast.data.manager import DataManager
from toofast.core.cost import (
    SinglePointCost, L2SpectrumCost, CosineSpectrumCost, LambdaMaxCost,
    PCASpectrumCost, create_cost_function, huber_cost_nm,
)
from toofast.core import cost_warp
from toofast.core.diagnostics import GPDiagnostics
from toofast.core.growth import assess_growth, growth_time_from_rows, should_terminate
from toofast.config import TerminationConfig
from toofast.gui.callbacks import ExperimentCallbacks, HeadlessCallbacks

np.set_printoptions(precision=4, suppress=True)

logger = logging.getLogger(__name__)


def _freeze_kernel(kernel):
    """Return a copy of ``kernel`` with all hyperparameters frozen.

    Sets every ``*_bounds`` param to 'fixed' so a GaussianProcessRegressor keeps
    the (offline-tuned) hyperparameters instead of re-optimising them.
    """
    k = clone(kernel)
    k.set_params(**{n: "fixed" for n in k.get_params() if n.endswith("_bounds")})
    return k


def _prior_kernel_matches(kernel, n_dims):
    """True if the kernel's ARD lengthscales are compatible with ``n_dims``.

    A per-dimension (ARD) lengthscale vector must have length n_dims; scalar
    (isotropic) lengthscales are compatible with any dimensionality.
    """
    for name, v in kernel.get_params().items():
        if name.endswith("length_scale") and not name.endswith("_bounds"):
            v = np.atleast_1d(v)
            if v.size > 1 and v.size != n_dims:
                return False
    return True

class Experiment:
    """
    Main experiment controller for automated nanoparticle synthesis.

    Orchestrates hardware control, Bayesian Optimization, and data acquisition
    for microfluidic flow chemistry experiments.

    Attributes
    ----------
    experiment_name : str
        Name identifier for the experiment.
    experiment_path : str
        Directory path for storing experiment data.
    spectro : Spectrometer
        UV-Vis spectrometer interface.
    pump_syst : PumpSystem
        Syringe pump system controller.
    collector : FractionCollector
        Automated fraction collector interface.
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_path: str,
        logger: logging.Logger,
        number_of_pumps: int,
        inner_diameter_mm: float,
        max_piston_stroke_mm: float,
        config_file: str,
        chip_id: str,
        total_flow: float = 8.0,
        is_simulation: bool = False,
        collector_port: str = "",
        c3000_port: str = "",
        mfc_port: str = "",
    ) -> None:
        """
        Initialize an experiment instance.

        Parameters
        ----------
        experiment_name : str
            Unique identifier for this experiment.
        experiment_path : str
            Directory path where experiment data will be stored.
        logger : logging.Logger
            Logger instance for recording experiment events.
        number_of_pumps : int
            Number of syringe pumps in the system.
        inner_diameter_mm : float
            Inner diameter of syringes in millimeters.
        max_piston_stroke_mm : float
            Maximum piston stroke length in millimeters.
        config_file : str
            Path to pump configuration file.
        chip_id : str
            Identifier for the microfluidic chip being used.
        total_flow : float, optional
            Total flow rate in mL/min (default: 8.0).
        is_simulation : bool, optional
            If True, run in simulation mode without hardware (default: False).
        collector_port : str, optional
            Serial port for the fraction collector (e.g. "COM8").
        c3000_port : str, optional
            Serial port for the C3000 syringe pumps (e.g. "COM16").
        mfc_port : str, optional
            Serial port for the mass flow controller (e.g. "COM9").
        """
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.app = None
        self.logger = logger
        self.termination = TerminationConfig()

        # Callbacks for GUI decoupling (set by App.py via TkCallbacks)
        self.callbacks: ExperimentCallbacks = HeadlessCallbacks()
        self._gp_diagnostics = GPDiagnostics()

        self.is_simulation = is_simulation
        self.collector_port = collector_port
        self.c3000_port = c3000_port
        self.mfc_port = mfc_port

        self.spectro = Spectrometer(self.logger, is_simulation=self.is_simulation)
        self.spectra = {}
        self.tg_spectrum = None
        self.success = False
        self.neg_var = False

        self.total_flow = total_flow
        self.number_of_pumps = number_of_pumps
        self.pump_syst = PumpSystem(self.logger, number_of_pumps=number_of_pumps, inner_diameter_mm=inner_diameter_mm,
                                    max_piston_stroke_mm=max_piston_stroke_mm, config_file=config_file,
                                    is_simulation=is_simulation, max_flow=self.total_flow)

        self.collector = FractionCollector(self.collector_port, logger=self.logger, is_simulation=self.is_simulation)
        self.collector.connect()
        self.prime_status = False
        self.collector_status = False
        self.collection_status = False
        self.chip_id = chip_id

        # Previous experiment data for warm-starting BO
        self.previous_flows = None
        self.previous_costs = None
        self._n_warmstart = 0  # # of leading historical (soft) warm-start points

        # Multi-objective (ParEGO): when True, the GP is fit on a scalarization
        # of [pca_cost, growth_time] (both minimized) with a fresh random
        # simplex weight each iteration, yielding a quality/speed Pareto search.
        # Default off → the single-objective path is unchanged. See _scalarize_mo.
        self.multi_objective = False
        self.mo_rho = 0.05            # augmented-Tchebycheff rho
        self.growth_times = []        # growth time (s) aligned with costs
        self._last_growth_time = None  # set by _record_growth_time
        self.mo_scalarizer = None     # lazily built ParEGOScalarizer

        # Counts standard washes since last intensive wash
        self._wash_sample_counter = 0

        # Cooperative abort signal for long-running operations (flush, wait_for_idle, etc.)
        self._abort_event = threading.Event()

        self.update_optim_params()

    def connect_mfc(self, port: str = ''):
        """Connect the mass flow controller."""
        port = port or self.mfc_port
        self.mfc = FlowController(port)
        self.mfc.set_flow(0.0)

    def connect_c3000(self, port: str = ''):
        """Connect C3000 syringe pumps."""
        port = port or self.c3000_port
        self.c3000_1 = C3000(port,address=1, syringe_volume=2500, simulation=self.is_simulation, logger=self.logger)
        self.c3000_1.connect()
        self.c3000_2 = C3000(port, address=2, syringe_volume=2500, simulation=self.is_simulation, logger=self.logger)
        self.c3000_2.connect()



    def update_optim_params(self, maxiter=14, residence_time=None, reagents=None,
                            nb_init_samples=None, init_option="latinhypercube", init_flows=None,
                            target_style='single_point', target_init=None, tg_wl=None, tg_abs=None,
                            tolerance=None, rel_tol=False, tol_stop=False, threshold=None,
                            acq_func='EI', bounds=None, fixed_pumps=None, fixed_flows=None,
                            target_spectrum_file=None, cost_method='L2',
                            pca_model_file=None, pca_variance_weighted=False,
                            pca_lambda_weight=0.0, pca_lambda_scale_nm=30.0,
                            pca_amp_weight=0.0,
                            pca_cost_metric='sqdist', pca_n_components=0,
                            cost_transform='none'):

        self.maxiter = maxiter
        self.residence_time = residence_time or (0.5 if self.is_simulation else 40)
        self.reagents = reagents if reagents is not None else np.array([''] * self.number_of_pumps)

        # Bounds of the pumps
        self.fixed_pumps = fixed_pumps if fixed_pumps is not None else np.array(
            [False for _ in range(self.number_of_pumps)])
        self.fixed_flows = fixed_flows if fixed_flows is not None else np.array(
            [None for _ in range(self.number_of_pumps)])
        self.total_var_flow = self.total_flow - sum(self.fixed_flows[self.fixed_pumps])
        self.varying_pumps = ~self.fixed_pumps
        self.nb_of_var_pumps = self.varying_pumps.sum()

        self.default_bounds = np.array([(0, self.total_flow)] * self.number_of_pumps)
        self.bounds = bounds if bounds is not None else self.default_bounds
        for i in range(self.number_of_pumps):
            if self.fixed_pumps[i]:
                self.bounds[i] = (self.fixed_flows[i], self.fixed_flows[i])

        # Initialisation parameters
        self.nb_init_samples = nb_init_samples or self.number_of_pumps
        self.init_option = init_option
        self.initial_flows = init_flows

        # Target type: full spectrum target or single point target or lambda max
        self.target_style = target_style
        self.tg_wl = tg_wl or round(np.random.uniform(self.spectro.wavelengths[0],
                                                      self.spectro.wavelengths[-1]), 1)
        self.tg_abs = tg_abs or round(np.random.uniform(0.2, 2), 4)
        self.tg_init_flows = target_init
        self.target_spectrum_file = target_spectrum_file

        # Cost calculation method for full_spectrum mode
        self.cost_method = cost_method  # 'L2' or 'cosine'

        # Load target spectrum from file if provided
        if self.target_style in ('full_spectrum', 'pca') and target_spectrum_file is not None:
            self.tg_spectrum = self.load_spectrum_from_csv(target_spectrum_file)

        # PCA target: load the frozen basis fitted offline and build the cost.
        self.pca_model_file = pca_model_file
        self.pca_variance_weighted = pca_variance_weighted
        # Optional hybrid term: add a lambda_max penalty to a (non-normalized)
        # PCA cost so amplitude is kept while peak position stays sharp. 0 = off.
        self.pca_lambda_weight = pca_lambda_weight
        self.pca_lambda_scale_nm = pca_lambda_scale_nm
        # Optional amplitude term: on a max-normalized basis the PCA score
        # distance is amplitude-blind; this discriminates good-shape compositions
        # by yield/concentration. Keep small so shape stays primary. 0 = off.
        self.pca_amp_weight = pca_amp_weight
        # Score-space distance metric ('sqdist'/'dist'/'cosine') + how many leading
        # PCs to use (0 = all). 'cosine' (k~2-4) gives a bounded, smooth cost with
        # low GP uncertainty. See PCASpectrumCost.
        self.pca_cost_metric = pca_cost_metric or 'sqdist'
        self.pca_n_components = int(pca_n_components or 0)
        self.pca_model = None
        self.pca_cost_fn = None
        if self.target_style == 'pca':
            if pca_model_file is None or self.tg_spectrum is None:
                raise ValueError(
                    "pca target requires both pca_model_file and target_spectrum_file"
                )
            from toofast.core.pca_model import SpectralPCA
            self.pca_model = SpectralPCA.load(pca_model_file)
            self.pca_cost_fn = PCASpectrumCost(
                self.pca_model, self.tg_spectrum, self.pca_variance_weighted,
                lambda_weight=self.pca_lambda_weight,
                lambda_scale_nm=self.pca_lambda_scale_nm,
                amp_weight=self.pca_amp_weight,
                metric=self.pca_cost_metric,
                n_components=self.pca_n_components,
            )

        # Convergence tolerance
        self.default_tol = {'single_point': 1e-7,  # 0.0003
                            'full_spectrum': 3e-2,  # 0.003 average over 4094 pixels
                            'lambda_max': 25,  # 5 nm
                            'pca': 1e-2}
        self.tol = tolerance or self.default_tol[self.target_style]
        self.rel_tol = rel_tol
        if self.rel_tol:
            self.tol = (tolerance / 100) ** 2 or 0.05 ** 2
        # Stop when tolerance is reached ?
        self.tol_stop = tol_stop

        # No AgNP threshold
        self.threshold = threshold or 0.01

        # Regression parameters (may be overridden by advanced GUI settings)
        self.kernel_type_override = None
        self.rbf_length_scale_override = None
        self.white_noise_override = None
        # Hyperparameter prior: a kernel whose hyperparameters were tuned offline
        # on historical data (lengthscales + noise). Used as the live kernel so
        # the GP's hyperparameter optimisation starts WARM from the historical
        # regime, then keeps re-optimising as live points arrive (freeze=False).
        # Set via load_prior_kernel(); None disables it. See _rebuild_kernel.
        self.prior_kernel = None
        self.prior_kernel_freeze = False
        # Lower bound for the WhiteKernel noise_level the GP optimiser may reach.
        # With normalize_y=True this is a normalized variance, so 1e-3 ~ 3% noise
        # std. A non-trivial floor stops the marginal-likelihood optimiser from
        # driving noise to ~0 (exact interpolation), which collapses the length
        # scale (boundary_hit) and yields a misleading near-zero uncertainty.
        self.white_noise_floor = 1e-3
        # Upper bound for the WhiteKernel noise_level (normalized variance). The
        # legacy 1e5 let the marginal-likelihood optimiser attribute up to ~316x
        # the signal std to noise, which on a wide-dynamic-range cost (e.g. PCA
        # 0.1..500) drowns the low-cost valley: everything below the noise std
        # becomes indistinguishable, so the BO never refines toward the optimum.
        # Capping it keeps noise <= signal scale, so the valley stays resolvable.
        self.white_noise_ceiling = 1e1
        # Lower bound on the GP length scale (flow units). The marginal-likelihood
        # optimiser will happily shrink the length scale to interpolate scattered
        # / irreproducible points, producing a "bed of nails" surface (and huge
        # between-point uncertainty). This floor forces it to treat that local
        # scatter as noise instead, keeping the surface smooth. The log cost warp
        # steepens the valley and makes this collapse more likely, so the floor
        # matters more there. Raise it (1.5-2) if the surface still looks spiky.
        self.length_scale_min = 1.0
        # Optional monotone warp of the cost the GP is fit on (single-objective
        # only). 'log' fits log(cost): variance-stabilising, so the low-cost
        # valley gets as much GP resolution as the bad region is compressed into
        # a steep plateau — lets the BO push toward small costs (~0.1) instead of
        # treating 1..10 as "all fine". Display is inverse-warped (see
        # _gp_to_cost). 'none' (default) is a no-op: byte-identical behaviour.
        self.cost_transform = cost_transform or 'none'
        self.gp_n_restarts_override = None
        # BO preset: "default" (RBF + quadratic) or "robust" (Matern 2.5 ARD + Huber).
        self.bo_preset_override = "default"
        self.huber_delta_nm_override = 30.0
        # Flat-spectrum handling at GP-fit time (default: nearest_high).
        self.flat_strategy_override = "nearest_high"
        self.flat_penalty_factor_override = 2.0
        self.flat_nearest_margin_override = 1.25
        self.flat_nearest_cap_override = 3.0
        # Optional constrained BO (off by default).
        self.feasibility_aware_override = False
        # Physics-based ghost points: inject synthetic "no Ag" / "no NaBH4"
        # points at GP-fit time so the model knows these regions yield no
        # particles before any real measurement falls there. The ghost cost
        # is the flat-spectrum sentinel, so flat_strategy (nearest_high)
        # remaps it to a bounded penalty just like real flat measurements.
        self.use_physics_ghosts = True
        self.ghost_reagent_keywords = ("AgNO3", "NaBH4")
        # Per-point GP noise (alpha) for ghost points. Ghosts are virtual
        # observations, not real measurements: giving them a larger diagonal
        # noise than the real points means the GP treats them as soft
        # constraints instead of exact interpolation targets. This prevents
        # the length scale from collapsing to the lower bound (boundary_hit)
        # just to fit the sharp ghost->interior cost gradient at the simplex
        # edges. The value is in normalized-target units (the GP fits with
        # normalize_y=True, signal variance ~1), so it is scale-invariant.
        # 'auto' -> 0.05 ; a float sets a fixed normalized noise.
        self.ghost_noise_level = "auto"
        # Extra multiplier applied (on top of the nearest_high remap) to ghost
        # costs only, so the infeasible faces sit modestly above their neighbours
        # and the GP grows a soft slope toward them. 1.0 = off (plain
        # nearest_high). Kept bounded by the nearest_high cap, so no length-scale
        # collapsing discontinuity. ~1.5 gives a visible but gentle rise.
        self.ghost_penalty_lift = 1.5
        self._fit_ghost_mask = None
        self._fit_ghost_alpha = 0.0
        self._rebuild_kernel()
        self.acq_func = acq_func

        # Differential evolution parameters:
        self.de_tol = 1e-6  # Stricter tolerance for 4D optimization
        self.de_maxiter = 400  # Increased for 4D space (was 200)

    def check_if_ready(self):
        min_init = self.nb_of_var_pumps + 1
        if self.nb_init_samples < min_init:
            self.logger.warning(
                f"nb_init_samples={self.nb_init_samples} is too low for "
                f"{self.nb_of_var_pumps} varying pumps — raising to {min_init}"
            )
            self.nb_init_samples = min_init
        ready = np.array([self.spectro.reference is not None,
                          self.target_style not in ['single_point', 'lambda_max'] or self.spectro.wavelengths[
                              0] < self.tg_wl < self.spectro.wavelengths[-1],
                          self.target_style != 'single_point' or self.tg_abs > 0,
                          self.target_style != 'full_spectrum' or self.tg_spectrum is not None,
                          self.target_style != 'full_spectrum' or sum(self.tg_init_flows) == self.total_flow and all(
                              [0 <= flow <= self.total_flow for flow in
                               self.tg_init_flows]),
                          self.init_option != 'fixed' or (
                                      all([round(sum(flows), 2) == 8 for flows in self.initial_flows]
                                          ) and all([0 <= flow <= self.total_flow for flow in
                                                     [item for innerlist in self.initial_flows for
                                                      item in
                                                      innerlist]]))])
        reasons = ['\n- Reference not measured',
                   '\n- Invalid target wavelength',
                   '\n- Invalid absorbance (must be greater than 0)',
                   '\n- Target spectrum not measured or loaded',
                   f'\n- Invalid target flows (sum must be {self.total_flow})',
                   f'\n- Invalid initial flows (sums must be {self.total_flow})', ]

        fault_reasons = np.array(reasons)[~ready]
        return True if all(ready) else False, ''.join(fault_reasons)

    def load_spectrum_from_csv(self, csv_file_path):
        """
        Load a target spectrum from a CSV file.

        The CSV format is expected to have:
        - First row: 'timestamp' followed by wavelengths
        - Subsequent rows: timestamp followed by absorbance values
        - Uses the last row as the target spectrum

        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing the spectrum data.

        Returns
        -------
        Spectrum
            Spectrum object containing the loaded target spectrum.
        """
        try:
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                rows = list(reader)

                if len(rows) < 2:
                    raise ValueError("CSV file must contain at least a header and one data row")

                # First row contains 'timestamp' + wavelengths
                header = rows[0]
                wavelengths = np.array([float(wl) for wl in header[1:]])

                # Last row contains timestamp + absorbance values
                last_row = rows[-1]
                absorbance = np.array([float(abs_val) for abs_val in last_row[1:]])
                timestamp = last_row[0]

                # Create a Spectrum object
                # For target spectra, we don't have transmittance, so we'll use absorbance for both
                spectrum = Spectrum(wavelengths, absorbance, 'target', absorbance=absorbance)
                spectrum.timestamp = timestamp

                self.logger.info(f"Loaded target spectrum from {csv_file_path}")
                self.logger.info(f"Spectrum timestamp: {timestamp}")
                self.logger.info(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

                return spectrum

        except Exception as e:
            self.logger.error(f"Error loading spectrum from CSV: {e}")
            raise

    def save_target_spectrum(self, path=None):
        """Persist the target spectrum to ``target_spectrum.csv`` in the run folder.

        Same ``;``-delimited format as load_spectrum_from_csv (header = 'timestamp'
        + wavelengths, one data row = timestamp + absorbance), so it round-trips.
        No-op when there is no target spectrum (single_point / lambda_max styles).

        Returns the written path, or None if there was nothing to save.
        """
        if getattr(self, "tg_spectrum", None) is None:
            return None
        directory = path or self.experiment_path
        if not directory:
            return None
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "target_spectrum.csv")
        wl = np.asarray(self.tg_spectrum.wavelengths, dtype=float)
        ab = np.asarray(self.tg_spectrum.absorbance, dtype=float)
        timestamp = getattr(self.tg_spectrum, "timestamp", "")
        try:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(["timestamp"] + list(wl))
                writer.writerow([timestamp] + list(ab))
            self.logger.info("Saved target spectrum -> %s", file_path)
            return file_path
        except Exception as e:
            self.logger.error("Could not save target spectrum: %s", e)
            return None

    def cost_function(self, spectrum: Spectrum) -> Tuple[float, float]:
        """
        Calculate the cost of a measured spectrum compared to the target.

        Delegates to toofast.core.cost module for standard cases.
        Handles rel_tol and lambda_max edge cases (flat spectrum) locally
        to preserve exact backward compatibility.

        Parameters
        ----------
        spectrum : Spectrum
            Measured spectrum to compare against the target.

        Returns
        -------
        cost : float
            Cost value (lower is better, 0 = perfect match).
        lamb_max : float
            Wavelength of maximum absorbance (0 if no particles detected).
        """
        lamb_max = 0

        if self.target_style == 'full_spectrum':
            if self.cost_method == 'cosine':
                cost_fn = CosineSpectrumCost(self.tg_spectrum)
                cost, _ = cost_fn(spectrum)
            elif self.cost_method == 'L2':
                if self.rel_tol:
                    # Relative L2: divide by target absorbance and number of wavelengths
                    cost = ((spectrum.absorbance - self.tg_spectrum.absorbance)
                            / self.tg_spectrum.absorbance
                            ) ** 2 / len(self.spectro.wavelengths)
                    cost = float(cost.sum())
                else:
                    cost_fn = L2SpectrumCost(self.tg_spectrum)
                    cost, _ = cost_fn(spectrum)
            else:
                raise ValueError(f"Unknown cost method: {self.cost_method}. Use 'L2' or 'cosine'.")

        elif self.target_style == 'pca':
            # Flat spectrum (no particles) → PCA sentinel, remapped by the
            # flat_strategy at GP-fit time exactly like the lambda_max path.
            if np.std(spectrum.absorbance) < self.threshold:
                return float(self._flat_penalty_value()), 0
            cost, _ = self.pca_cost_fn(spectrum)

        elif self.target_style == 'single_point':
            if self.rel_tol:
                abs_at_tg_wl = spectrum.get_abs_at_wl(self.tg_wl)
                cost = float(((abs_at_tg_wl - self.tg_abs) / self.tg_abs) ** 2)
            else:
                cost_fn = SinglePointCost(self.tg_wl, self.tg_abs)
                cost, _ = cost_fn(spectrum)

        elif self.target_style == 'lambda_max':
            preset = getattr(self, "bo_preset_override", "default") or "default"
            use_huber = (preset == "robust")
            delta_nm = float(getattr(self, "huber_delta_nm_override", 30.0) or 30.0)
            divisor = self.tg_wl if self.rel_tol else 1

            # Flat spectrum check: no particles detected → maximum cost.
            # NB: this raw penalty is what gets stored in costs.csv. The
            # flat_strategy (nearest_high by default) re-maps it at GP-fit
            # time via _apply_flat_strategy(); flat points are detected by
            # matching this exact sentinel value.
            if np.std(spectrum.absorbance) < self.threshold:
                return float(self._flat_penalty_value()), 0
            else:
                # Apply savgol filter in-place (preserves original behavior)
                spectrum.absorbance = savgol_filter(spectrum.absorbance, 50, 3)
                lamb_max_index = spectrum.absorbance.argmax()
                lamb_max = spectrum.wavelengths[lamb_max_index]
                if use_huber:
                    cost = float(huber_cost_nm(lamb_max, self.tg_wl, delta_nm) / (divisor ** 2))
                else:
                    cost = float(((lamb_max - self.tg_wl) / divisor) ** 2)

        return float(cost), lamb_max

    def _flat_penalty_value(self) -> float:
        """Raw flat-spectrum penalty (max cost over the measurement domain).

        Single source of truth: used both by cost_function (what is stored)
        and as the sentinel to detect flat points in _apply_flat_strategy.
        Style-aware: the PCA target uses a PCA-score sentinel (see
        _pca_flat_sentinel) so the flat machinery and ghosts stay in the right
        units.
        """
        if self.target_style == 'pca':
            return self._pca_flat_sentinel()
        preset = getattr(self, "bo_preset_override", "default") or "default"
        use_huber = (preset == "robust")
        delta_nm = float(getattr(self, "huber_delta_nm_override", 30.0) or 30.0)
        divisor = self.tg_wl if self.rel_tol else 1
        min_wavelength, max_wavelength = 315, 750
        if use_huber:
            return max(
                huber_cost_nm(min_wavelength, self.tg_wl, delta_nm),
                huber_cost_nm(max_wavelength, self.tg_wl, delta_nm),
            ) / (divisor ** 2)
        return max(
            ((min_wavelength - self.tg_wl) / divisor) ** 2,
            ((max_wavelength - self.tg_wl) / divisor) ** 2,
        )

    def _pca_flat_sentinel(self) -> float:
        """Deterministic flat-spectrum cost for the PCA target.

        A no-particle spectrum is ~zero absorbance; its PCA cost vs the target
        is a reproducible, physically meaningful "no particle" penalty. Cached
        so the exact same value is both stored (cost_function) and matched as
        the sentinel in _apply_flat_strategy / carried by the ghost points.
        """
        cached = getattr(self, "_pca_flat_sentinel_value", None)
        if cached is not None:
            return cached
        wl = self.pca_model.wavelengths
        scores_flat = self.pca_model.transform(wl, np.zeros_like(wl, dtype=float))
        # Use the SAME metric as the live cost (sqdist/dist/cosine) so the flat
        # sentinel sits on the real cost scale and flat detection stays valid.
        val = float(self.pca_cost_fn._base_cost(scores_flat))
        # Stay consistent with PCASpectrumCost.__call__: a zero-amplitude
        # spectrum incurs the full amplitude term ((0 - A_t)/A_t)**2 = 1.
        if getattr(self.pca_cost_fn, "amp_weight", 0.0) > 0.0:
            val += float(self.pca_cost_fn.amp_weight)
        self._pca_flat_sentinel_value = val
        return val

    def _build_ghost_points_full(self):
        """Generate physics-based ghost flows in full pump space.

        For each varying pump carrying a reagent in ``ghost_reagent_keywords``
        (AgNO3, NaBH4), we sample ``n_per_face`` points on the hyperface
        ``flow_of_that_reagent = 0`` (sum of the remaining varying flows =
        total_var_flow, via Dirichlet). The ghost cost is the flat-spectrum
        sentinel so it gets remapped by the flat_strategy.

        Returns
        -------
        (ghost_full, ghost_costs) : (np.ndarray, np.ndarray) or (None, None)
            Ghost flows in full pump space and their sentinel costs.
        """
        if self.reagents is None or self.varying_pumps is None:
            return None, None

        keywords = tuple(k.lower() for k in self.ghost_reagent_keywords)
        var_global_idx = np.where(self.varying_pumps)[0]
        n_var = int(len(var_global_idx))
        if n_var < 2:
            # Need at least one other varying pump to absorb the flow.
            return None, None

        # Map: list of (zero_local_idx, zero_global_idx) for each ghost face.
        target_faces = []
        for local_i, global_i in enumerate(var_global_idx):
            r = str(self.reagents[global_i]).strip().lower()
            if any(k in r for k in keywords):
                target_faces.append((local_i, int(global_i)))

        if not target_faces:
            return None, None

        # Number of ghost points per face scales with the GP input dimension.
        # GP dim = n_var - 1 (sum constraint removes one DOF).
        # 2 var pumps (1D GP) → 3, 3 (2D) → 4, 4 (3D) → 6, 5 (4D) → 8.
        dim = max(1, n_var - 1)
        n_per_face = max(3, 2 * dim)

        # Resolve fixed-pump contributions to the full flow template.
        full_template = np.zeros(self.number_of_pumps, dtype=float)
        if self.fixed_pumps is not None and self.fixed_flows is not None:
            for k in range(self.number_of_pumps):
                if self.fixed_pumps[k] and self.fixed_flows[k] is not None:
                    full_template[k] = float(self.fixed_flows[k])

        rng = np.random.default_rng(seed=20251201)  # deterministic ghosts
        penalty = float(self._flat_penalty_value())
        ghost_full = []

        for zi_local, zi_global in target_faces:
            remaining_locals = [j for j in range(n_var) if j != zi_local]
            if not remaining_locals:
                continue

            if len(remaining_locals) == 1:
                # Only one absorber: the corner point (Ag=0, other=total_var_flow).
                full = full_template.copy()
                full[var_global_idx[remaining_locals[0]]] = self.total_var_flow
                ghost_full.append(full)
                continue

            for _ in range(n_per_face):
                alpha = np.ones(len(remaining_locals))
                sample = rng.dirichlet(alpha) * self.total_var_flow
                full = full_template.copy()
                for j, loc in enumerate(remaining_locals):
                    full[var_global_idx[loc]] = sample[j]
                # zi_global stays at 0
                ghost_full.append(full)

        if not ghost_full:
            return None, None

        ghost_full = np.asarray(ghost_full, dtype=float)
        ghost_costs = np.full(len(ghost_full), penalty, dtype=float)
        return ghost_full, ghost_costs

    def _augment_with_ghosts(self, X, costs):
        """Append ghost points (reduced to GP input space) to (X, costs).

        Ghosts are appended at the END so the last ``n_ghosts`` rows of the
        returned arrays are the ghost points (used to build the per-point
        alpha vector in _apply_flat_strategy).

        Returns
        -------
        (X, costs, n_ghosts) : the augmented arrays and the number of ghost
        rows appended (0 if ghosts are disabled, no relevant reagent varies,
        or the GP space cannot accept them).
        """
        if not getattr(self, "use_physics_ghosts", True):
            return X, costs, 0
        if getattr(self, "multi_objective", False):
            # Ghosts carry only a (flat) PCA cost, not a growth time, so they
            # don't fit the 2-objective scalarization cleanly. Disabled for the
            # ParEGO path in v1 (flats are still handled via the sentinel remap).
            return np.asarray(X, dtype=float), np.asarray(costs, dtype=float), 0
        ghost_full, ghost_costs = self._build_ghost_points_full()
        if ghost_full is None or len(ghost_full) == 0:
            return X, costs, 0
        # Apply the same reduction as the real flows: keep varying pumps,
        # drop the last column (its value is set by the sum constraint).
        ghost_reduced = ghost_full[:, self.varying_pumps][:, :-1]
        if ghost_reduced.shape[1] != X.shape[1]:
            # Shape mismatch (e.g. n_var=1 corner case) — skip rather than crash.
            self.logger.debug(
                "Ghost dim %d != X dim %d, skipping ghost injection",
                ghost_reduced.shape[1], X.shape[1])
            return X, costs, 0
        self.logger.debug(
            "Injecting %d physics ghost point(s) into GP training",
            len(ghost_reduced))
        return (np.concatenate([X, ghost_reduced], axis=0),
                np.concatenate([costs, ghost_costs], axis=0),
                len(ghost_reduced))

    def _apply_flat_strategy(self, X, costs):
        """Re-map flat-spectrum costs at GP-fit time per flat_strategy_override.

        Parameters
        ----------
        X : (m, d) array
            Reduced flows actually fed to the GP (flows_minus_one).
        costs : (m,) array
            Raw costs as stored (flat points hold the _flat_penalty_value).

        Returns
        -------
        (X_fit, costs_fit) : possibly row-filtered (for 'exclude'). Also
        stashes self._feas_X / self._feas_label (all points, reduced space)
        for the optional feasibility classifier.
        """
        X = np.asarray(X, dtype=float)
        costs = np.asarray(costs, dtype=float)

        # Inject physics-based ghost points before fitting. They carry the
        # flat-spectrum sentinel cost so they get remapped by 'nearest_high'
        # like any real flat measurement — no extreme outlier in the GP.
        X, costs, n_ghosts = self._augment_with_ghosts(X, costs)

        # Ghost rows are the last n_ghosts entries (see _augment_with_ghosts).
        ghost_mask = np.zeros(len(X), dtype=bool)
        if n_ghosts > 0:
            ghost_mask[-n_ghosts:] = True

        strategy = getattr(self, "flat_strategy_override", "nearest_high") or "nearest_high"
        if getattr(self, "multi_objective", False) and strategy == "exclude":
            # 'exclude' drops rows, breaking the row-alignment the ParEGO
            # scalarization relies on. Fall back to the row-preserving default.
            self.logger.info("multi_objective: 'exclude' flat_strategy -> nearest_high")
            strategy = "nearest_high"

        sentinel = float(self._flat_penalty_value())
        flat_mask = np.isclose(costs, sentinel, rtol=1e-9, atol=1e-9)

        # Per-point GP noise: ghosts get a larger diagonal alpha so the GP
        # treats them as soft constraints (see __init__ note on boundary_hit).
        # NB: the GP fits with normalize_y=True, so y is scaled to unit
        # variance before alpha is added to the kernel diagonal. The ghost
        # alpha is therefore expressed in NORMALIZED-target units (a fraction
        # of the signal variance) and is scale-invariant — a raw-cost value
        # like (0.2*std)^2 would be negligible for small costs. 'auto' uses
        # 0.05 (~ghosts carry ~22% relative uncertainty): low enough that the GP
        # mean actually RISES toward the ghost value (so the infeasible face
        # shows a soft slope, not just a variance dip), while still keeping the
        # length scale off its lower bound. Paired with ghost_penalty_lift.
        gnl = getattr(self, "ghost_noise_level", "auto")
        if gnl == "auto" or gnl is None:
            self._fit_ghost_alpha = 0.05
        else:
            self._fit_ghost_alpha = float(gnl)
        # Default alignment (full X order); overridden in the 'exclude' branch.
        self._fit_ghost_mask = ghost_mask

        # Soft warm-start: the leading _n_warmstart rows are historical points
        # (e.g. old .h5 data). They get a larger per-point alpha so the GP
        # treats them as a soft prior — robust to drift in stale experiments —
        # rather than trusting them as exact, fresh observations. Ghosts are
        # appended after, so warm-start rows keep their leading positions.
        n_ws = int(getattr(self, "_n_warmstart", 0))
        ws_mask = np.zeros(len(X), dtype=bool)
        if n_ws > 0:
            ws_mask[: min(n_ws, len(X))] = True
        wnl = getattr(self, "warmstart_noise_level", "auto")
        self._fit_warmstart_alpha = 0.15 if (wnl == "auto" or wnl is None) else float(wnl)
        self._fit_warmstart_mask = ws_mask

        # Stash labels for the (optional) feasibility classifier: 1 = feasible
        self._feas_X = X.copy()
        self._feas_label = (~flat_mask).astype(int)

        n_flat = int(flat_mask.sum())
        if n_flat == 0 or strategy == "penalty_max":
            self._gp_display_costs = costs.copy()
            return X, costs

        real_mask = ~flat_mask
        if real_mask.sum() == 0:
            self._gp_display_costs = costs.copy()
            return X, costs  # nothing to anchor on; leave as-is

        out = costs.copy()
        real_costs = costs[real_mask]

        if strategy == "exclude":
            # Flat points are not modelled; for display keep their true
            # measured cost (they will visibly sit off the GP surface,
            # which honestly reflects "excluded").
            self._gp_display_costs = costs.copy()
            # Ghosts carry the sentinel → they are in flat_mask → dropped here,
            # so realign the ghost/warm-start masks to the filtered rows.
            self._fit_ghost_mask = ghost_mask[real_mask]
            self._fit_warmstart_mask = ws_mask[real_mask]
            self.logger.info("flat_strategy=exclude -> %d flat point(s) dropped from GP fit", n_flat)
            return X[real_mask], costs[real_mask]

        if strategy == "penalty_moderate":
            factor = float(getattr(self, "flat_penalty_factor_override", 2.0) or 2.0)
            out[flat_mask] = factor * float(np.mean(real_costs))
            self._gp_display_costs = out.copy()
            self.logger.info(
                "flat_strategy=penalty_moderate -> %d flat cost(s) set to %.4g",
                n_flat, factor * float(np.mean(real_costs)))
            return X, out

        # nearest_high (default)
        margin = float(getattr(self, "flat_nearest_margin_override", 1.25) or 1.25)
        cap_mult = float(getattr(self, "flat_nearest_cap_override", 3.0) or 3.0)
        cap = cap_mult * float(np.max(real_costs))
        ri = np.where(real_mask)[0]
        fi = np.where(flat_mask)[0]
        Xr = X[ri]
        if len(X) >= 2:
            dm = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dm, np.inf)
            radius = 1.5 * float(np.median(dm.min(axis=1)))
        else:
            radius = np.inf
        for f in fi:
            dd = np.sqrt(((Xr - X[f]) ** 2).sum(-1))
            loc = costs[ri][dd <= radius]
            val = (margin * float(np.max(loc)) if loc.size
                   else float(np.percentile(real_costs, 90)))
            out[f] = min(val, cap)
        # Gentle ghost lift: ghosts (the infeasible Ag=0 / NaBH4=0 faces) sit a
        # bit ABOVE their nearest_high neighbour value so the GP mean rises into a
        # soft slope toward the face. Still bounded by the same cap, so no
        # discontinuity that would collapse the length scale. Applied to ghost
        # rows only (real flat measurements keep the plain nearest_high value).
        lift = float(getattr(self, "ghost_penalty_lift", 1.0) or 1.0)
        if n_ghosts > 0 and lift != 1.0:
            g = ghost_mask & flat_mask
            out[g] = np.minimum(out[g] * lift, cap)
        self._gp_display_costs = out.copy()
        self.logger.info(
            "flat_strategy=nearest_high -> %d flat cost(s) remapped "
            "(radius=%.3g, cap=%.4g)", n_flat, radius, cap)
        return X, out

    def _alpha_for_fit(self, base_alpha):
        """Per-point GP noise (alpha) aligned with the last _apply_flat_strategy
        output. Real points keep ``base_alpha``; ghost points get the (larger)
        ghost noise. Returns the scalar ``base_alpha`` when no ghosts are in
        the training set, so sklearn behaviour is unchanged in that case.

        Must be called AFTER _apply_flat_strategy and BEFORE the matching
        gaussian_process.fit(), passing the base alpha used at that fit site.
        """
        gmask = getattr(self, "_fit_ghost_mask", None)
        wmask = getattr(self, "_fit_warmstart_mask", None)
        has_g = gmask is not None and np.any(gmask)
        has_w = wmask is not None and np.any(wmask)
        if not has_g and not has_w:
            return base_alpha
        n = len(gmask) if gmask is not None else len(wmask)
        alpha = np.full(n, float(base_alpha), dtype=float)
        if has_g:
            gn = max(float(getattr(self, "_fit_ghost_alpha", 0.0)), float(base_alpha))
            alpha[gmask] = gn
        if has_w:
            wn = max(float(getattr(self, "_fit_warmstart_alpha", 0.0)), float(base_alpha))
            alpha[wmask] = np.maximum(alpha[wmask], wn)
        self.logger.debug(
            "Per-point alpha: %d ghost(s), %d warm-start, %d real at %.4g",
            int(np.sum(gmask)) if has_g else 0,
            int(np.sum(wmask)) if has_w else 0,
            int(n - (np.sum(gmask) if has_g else 0)), float(base_alpha))
        return alpha

    def _note_growth_time(self):
        """Record the growth time of the just-measured sample, aligned with the
        costs array. Called once per BO iteration right after a cost is
        appended. Missing/unparseable times are stored as NaN and neutral-filled
        at scalarization time, so multi-objective never crashes on a gap."""
        gt = getattr(self, "_last_growth_time", None)
        self.growth_times.append(float(gt) if gt is not None else float("nan"))
        self._last_growth_time = None

    def _scalarize_mo(self, pca_costs):
        """ParEGO-scalarize the objective matrix [pca_cost, growth_time].

        Both objectives are minimized. A fresh random simplex weight is drawn on
        every call (the ParEGO idea) so the search sweeps the quality/speed
        Pareto front across iterations. Growth times are defensively aligned to
        ``pca_costs``: gaps (NaN / not-yet-tracked rows such as init or
        warm-start points) are filled with the median so a single scalar GP can
        always be fit.
        """
        from toofast.core.scalarization import (
            ParEGOScalarizer, align_trailing_column)
        pca = np.asarray(pca_costs, dtype=float)
        gts = align_trailing_column(getattr(self, "growth_times", []), len(pca))
        Y = np.column_stack([pca, gts])
        if getattr(self, "mo_scalarizer", None) is None:
            self.mo_scalarizer = ParEGOScalarizer(
                n_obj=2, rho=float(getattr(self, "mo_rho", 0.05)),
                seed=getattr(self, "seed", None))
        return self.mo_scalarizer.scalarize(Y)

    def _cost_to_gp(self, y):
        """Map raw cost -> GP target space (see self.cost_transform / cost_warp)."""
        return cost_warp.cost_to_gp(
            y, getattr(self, "cost_transform", "none"),
            getattr(self, "multi_objective", False))

    def _gp_to_cost(self, v):
        """Inverse of _cost_to_gp: GP-space value -> raw cost units (for display)."""
        return cost_warp.gp_to_cost(
            v, getattr(self, "cost_transform", "none"),
            getattr(self, "multi_objective", False))

    def _gp_disp(self, v):
        """GP prediction in PLOT display space (identity).

        Plots stay in the GP's native fit space: log-cost when
        cost_transform=='log' (so the +/-sigma band is symmetric and bounded —
        no exp() explosion), raw cost otherwise. Measured points are mapped into
        the same space with _cost_to_gp; see _cost_axis_suffix for the label.
        """
        return np.asarray(v, dtype=float)

    def _cost_axis_suffix(self):
        """Label suffix telling the reader the cost axis is in log units."""
        return " (log)" if getattr(self, "cost_transform", "none") == "log" else ""

    def _fit_gp(self, gaussian_process, flows_minus_one, costs, base_alpha):
        """Fit the GP for one BO step and return the y array the acquisition
        function should treat as 'costs' (best_y / signal scale).

        Single-objective: the GP is fit on the (flat-remapped, ghost-augmented)
        PCA costs and the raw ``costs`` drive acquisition — identical to the
        previous inline code. Multi-objective: the same fit targets are ParEGO-
        scalarized with [growth_time] and that scalar drives both fit and
        acquisition (one shared weight per call).
        """
        _Xfit, _yfit = self._apply_flat_strategy(flows_minus_one, costs)
        if getattr(self, "multi_objective", False):
            _yfit = self._scalarize_mo(_yfit)
            self._gp_display_costs = _yfit.copy()
            y_acq = _yfit
        else:
            y_acq = costs
        gaussian_process.alpha = self._alpha_for_fit(base_alpha)
        # Fit in GP-target space (identity unless cost_transform == 'log').
        gaussian_process.fit(_Xfit, self._cost_to_gp(_yfit))
        return y_acq

    def _display_costs(self, costs):
        """Costs to SHOW on the GP figure / cost-evolution panel: the values
        the GP was actually trained on (flat points remapped per
        flat_strategy), so measured points sit on the GP surface instead
        of floating at the raw penalty. costs.csv keeps the raw values.
        """
        disp = getattr(self, "_gp_display_costs", None)
        costs = np.asarray(costs, dtype=float)
        if disp is not None and len(disp) == len(costs):
            return np.asarray(disp, dtype=float)
        return costs

    def _validate_previous_data(self) -> bool:
        """Check that previous_flows/previous_costs are usable with the current
        experiment configuration. If invalid, clear them and warn.

        Returns True if previous data is set AND valid, False otherwise.
        """
        if self.previous_flows is None or self.previous_costs is None:
            return False
        n_prev = len(self.previous_costs)
        n_pumps_prev = self.previous_flows.shape[1] if self.previous_flows.ndim > 1 else 1
        reasons = []
        if n_pumps_prev != self.number_of_pumps:
            reasons.append(
                f"pump count mismatch (previous={n_pumps_prev}, current={self.number_of_pumps})")
        if n_prev < self.nb_init_samples:
            reasons.append(
                f"too few previous samples ({n_prev}) for current "
                f"nb_init_samples={self.nb_init_samples}")
        if reasons:
            self.logger.warning(
                "Discarding previous experiment data — " + "; ".join(reasons))
            self.previous_flows = None
            self.previous_costs = None
            self._n_warmstart = 0
            return False
        return True

    def warm_start_from_h5(self, h5_dir, drop_flat=False, flat_threshold=0.05,
                           min_wl=315.0, max_wl=750.0, warmstart_noise_level="auto"):
        """Warm-start the BO from a directory of historical .h5 samples.

        For the *current* PCA target, project every historical spectrum onto the
        frozen basis and compute its PCA cost; the (flows, cost) pairs seed the
        GP via previous_flows / previous_costs, so the BO starts from a model
        already informed by all those experiments instead of a cold GP.

        Because the PCA cost is target-dependent, this recomputes the costs for
        the target currently configured (call after update_optim_params). No new
        experiment is run — costs come from the historical spectra.

        Parameters
        ----------
        h5_dir : str
            Directory of per-sample .h5 files (flows_set + statistics/abs_final).
        drop_flat : bool
            Drop no-particle / flat spectra (handled by the flat machinery at
            run time, not by the PCA cost) before seeding.
        flat_threshold, min_wl, max_wl : float
            Flat-detection parameters (peak-to-peak amplitude over the window).

        Returns
        -------
        (flows, costs) : (np.ndarray, np.ndarray)
            The warm-start data that was loaded (also stored on the experiment).
        """
        if self.target_style != 'pca' or getattr(self, 'pca_cost_fn', None) is None:
            raise ValueError(
                "warm_start_from_h5 requires a configured PCA target "
                "(call update_optim_params with target_style='pca' first)."
            )

        from toofast.data.h5_dataset import load_h5_dataset

        ds = load_h5_dataset(h5_dir)
        if drop_flat:
            ds, flat = ds.drop_flat(threshold=flat_threshold, min_wl=min_wl, max_wl=max_wl)
            self.logger.info(f"Warm-start: dropped {int(flat.sum())} flat spectra")

        # Flat (no-particle) spectra are KEPT by default and assigned the flat
        # sentinel — the same value cost_function stores for a live flat — so the
        # flat machinery (nearest_high remap + soft noise) handles them exactly
        # like a measured flat, instead of feeding their (meaningless) raw PCA
        # projection to the GP. Detection mirrors the live criterion in
        # cost_function (std of absorbance below self.threshold).
        sentinel = float(self._flat_penalty_value())
        flat_thr = float(getattr(self, "threshold", flat_threshold))
        costs = []
        n_flat = 0
        for i in range(ds.n_samples):
            absb = ds.spectra[i]
            spec = Spectrum(ds.wavelengths, 10 ** (-absb), 'measurement', absorbance=absb)
            if float(np.std(spec.absorbance)) < flat_thr:
                costs.append(sentinel)
                n_flat += 1
            else:
                cost, _ = self.pca_cost_fn(spec)
                costs.append(float(cost))
        if not drop_flat and n_flat:
            self.logger.info(
                f"Warm-start: {n_flat} flat spectra kept with the flat sentinel "
                f"(handled by flat_strategy, not dropped)")

        self.previous_flows = np.asarray(ds.flows, dtype=float)
        self.previous_costs = np.asarray(costs, dtype=float)
        # Soft warm-start: tag these historical points so the GP fit gives them
        # an inflated per-point noise (prior, not gospel) — robust to drift in
        # old experiments. Reset to 0 wherever previous_* are cleared.
        self._n_warmstart = int(len(self.previous_costs))
        self.warmstart_noise_level = warmstart_noise_level
        self.logger.info(
            f"Warm-start: seeded BO with {ds.n_samples} historical points "
            f"(best PCA cost {self.previous_costs.min():.5f}); "
            f"soft prior noise={warmstart_noise_level}"
        )
        return self.previous_flows, self.previous_costs

    def load_previous_experiment(self, previous_exp_path):
        """
        Load costs, pump flows, and other data from a previous experiment.
        Also copies all the previous experiment files to the current experiment directory.
        Validates that the target parameters match the current experiment.

        Parameters
        ----------
        previous_exp_path : str
            Path to the directory containing the previous experiment data.

        Returns
        -------
        flows : np.ndarray or None
            Array of shape (n_samples, n_pumps) containing the pump flows.
        costs : np.ndarray or None
            Array of shape (n_samples,) containing the costs.
        """
        costs_file = os.path.join(previous_exp_path, "costs.csv")
        flows_file = os.path.join(previous_exp_path, "pump_flows.csv")

        # Check if both files exist
        if not os.path.exists(costs_file):
            self.logger.error(f"Costs file not found: {costs_file}")
            return None, None
        if not os.path.exists(flows_file):
            self.logger.error(f"Pump flows file not found: {flows_file}")
            return None, None

        # Load and validate target parameters from previous experiment
        params_file = None
        for filename in os.listdir(previous_exp_path):
            if filename.endswith("_parameters.txt"):
                params_file = os.path.join(previous_exp_path, filename)
                break

        if params_file and os.path.exists(params_file):
            try:
                prev_target_style = None
                prev_tg_wl = None
                prev_tg_abs = None

                with open(params_file, 'r') as f:
                    for line in f:
                        if 'Optimization strategy:' in line:
                            prev_target_style = line.split(':')[1].strip().rstrip('.')
                        elif 'Lambda_max target at' in line:
                            prev_tg_wl = float(line.split('at')[1].strip().split()[0])
                        elif 'Target absorbance of' in line:
                            parts = line.split()
                            prev_tg_abs = float(parts[3])
                            prev_tg_wl = float(parts[5])

                # Check if target parameters match
                target_mismatch = False
                mismatch_msg = []

                if prev_target_style and prev_target_style != self.target_style:
                    target_mismatch = True
                    mismatch_msg.append(f"Target style: previous={prev_target_style}, current={self.target_style}")

                if prev_tg_wl is not None and abs(prev_tg_wl - self.tg_wl) > 0.1:
                    target_mismatch = True
                    mismatch_msg.append(f"Target wavelength: previous={prev_tg_wl} nm, current={self.tg_wl} nm")

                if prev_tg_abs is not None and abs(prev_tg_abs - self.tg_abs) > 0.01:
                    target_mismatch = True
                    mismatch_msg.append(f"Target absorbance: previous={prev_tg_abs}, current={self.tg_abs}")

                if target_mismatch:
                    self.logger.warning("Target parameters mismatch detected!")
                    for msg in mismatch_msg:
                        self.logger.warning(f"  - {msg}")
                    self.logger.warning("The loaded costs were calculated for a different target.")
                    self.logger.warning("You should use the same target parameters as the previous experiment.")

                    # Store the previous target info for the user
                    self.prev_target_info = {
                        'style': prev_target_style,
                        'wavelength': prev_tg_wl,
                        'absorbance': prev_tg_abs
                    }
                    return None, None
                else:
                    self.logger.info("Target parameters match - safe to load previous data")

            except Exception as e:
                self.logger.warning(f"Could not validate target parameters: {e}")
                self.logger.warning("Proceeding with caution - target may not match!")

        # Restore experiment parameters from machine-readable config
        cfg = self._load_config_json(previous_exp_path)
        if cfg is not None:
            self._apply_loaded_config(cfg)
            self.loaded_config = cfg
        else:
            self.loaded_config = None
            self.logger.warning("No _config.json found — parameters not restored")

        try:
            # Load costs
            costs = []
            with open(costs_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row:  # Skip empty rows
                        costs.append(float(row[0]))
            costs = np.array(costs)

            # Load pump flows
            flows = []
            with open(flows_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row:  # Skip empty rows
                        flows.append([float(x) for x in row])
            flows = np.array(flows)

            # Validate data consistency
            if len(costs) != len(flows):
                self.logger.error(f"Data mismatch: {len(costs)} costs but {len(flows)} flow sets")
                return None, None

            if len(flows) > 0 and flows.shape[1] != self.number_of_pumps:
                self.logger.error(f"Flow data has {flows.shape[1]} pumps but experiment expects {self.number_of_pumps}")
                return None, None

            self.logger.info(f"Successfully loaded {len(costs)} data points from previous experiment")
            self.logger.info(f"Previous costs range: [{costs.min():.4f}, {costs.max():.4f}]")

            # Store source path — files will be copied when the BO sub-folder
            # is created (in _copy_previous_files, called from optimise/stop_flow)
            self._previous_exp_source = previous_exp_path

            return flows, costs

        except Exception as e:
            self.logger.error(f"Error loading previous experiment data: {e}")
            return None, None

    def _copy_previous_files(self):
        """Copy previous experiment files into the current BO sub-folder.

        Called from optimise/stop_flow once experiment_path points to the
        sub-folder (SDXXX/SDXXX_XX).
        """
        src = getattr(self, '_previous_exp_source', None)
        if src is None:
            return

        try:
            copied_files = []

            costs_file = os.path.join(src, "costs.csv")
            if os.path.exists(costs_file):
                shutil.copy2(costs_file, os.path.join(self.experiment_path, "previous_costs.csv"))
                copied_files.append("costs.csv")

            flows_file = os.path.join(src, "pump_flows.csv")
            if os.path.exists(flows_file):
                shutil.copy2(flows_file, os.path.join(self.experiment_path, "previous_pump_flows.csv"))
                copied_files.append("pump_flows.csv")

            lambda_max_file = os.path.join(src, "lambda_max.csv")
            if os.path.exists(lambda_max_file):
                shutil.copy2(lambda_max_file, os.path.join(self.experiment_path, "previous_lambda_max.csv"))
                copied_files.append("lambda_max.csv")

            for filename in os.listdir(src):
                if filename.startswith("absorbance_at_") and filename.endswith("_nm.csv"):
                    shutil.copy2(
                        os.path.join(src, filename),
                        os.path.join(self.experiment_path, f"previous_{filename}"),
                    )
                    copied_files.append(filename)

            for filename in os.listdir(src):
                if filename.endswith("_parameters.txt"):
                    shutil.copy2(
                        os.path.join(src, filename),
                        os.path.join(self.experiment_path, "previous_parameters.txt"),
                    )
                    copied_files.append(filename)
                    break

            samples_dir = os.path.join(src, "samples")
            if os.path.exists(samples_dir):
                dest_samples = os.path.join(self.experiment_path, "previous_samples")
                shutil.copytree(samples_dir, dest_samples, dirs_exist_ok=True)
                copied_files.append("samples/")

            metadata_file = os.path.join(self.experiment_path, "previous_experiment_info.txt")
            with open(metadata_file, 'w') as f:
                f.write(f"Source path: {src}\n")
                f.write(f"Loaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data points: {len(self.previous_costs)}\n")
                f.write(f"Best cost: {self.previous_costs.min():.6f}\n")
                f.write(f"Copied files: {', '.join(copied_files)}\n")

            self.logger.info(f"Copied {len(copied_files)} previous experiment files to {self.experiment_path}")

        except Exception as e:
            self.logger.warning(f"Could not copy previous experiment files: {e}")

    def _sample_candidates_ts(self, n: int = 2000) -> np.ndarray:
        """Return ``n`` feasible candidates in reduced GP space (n_var-1 dims).

        Each candidate satisfies the sum constraint
        ``sum(var_flows) == total_var_flow`` and stays within ``bounds``.
        Uses a Sobol sequence for uniform coverage, then projects onto the
        simplex face.
        """
        bounds_var = self.bounds[self.varying_pumps]   # (n_var, 2)
        lo = bounds_var[:-1, 0]
        hi = bounds_var[:-1, 1]
        dim = len(lo)                                  # n_var - 1

        # Sobol for low-discrepancy coverage, fall back to uniform if unavailable.
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=dim, scramble=True, seed=42)
            # next_power_of_2 >= n
            m = int(np.ceil(np.log2(max(n, 1))))
            raw = sampler.random_base2(m=m)[:n]        # (n, dim) in [0,1]
        except Exception:
            rng = np.random.default_rng(42)
            raw = rng.random(size=(n, dim))

        # Scale to individual bounds.
        cands = lo + raw * (hi - lo)                   # (n, dim)

        # Enforce sum constraint: last varying flow = total_var_flow - sum(rest).
        # Drop candidates where the last flow would be out of bounds.
        last = self.total_var_flow - cands.sum(axis=1)
        lo_last, hi_last = bounds_var[-1, 0], bounds_var[-1, 1]
        valid = (last >= lo_last - 1e-9) & (last <= hi_last + 1e-9)
        cands = cands[valid]

        if len(cands) == 0:
            # Degenerate: return the centroid as the only candidate.
            centre = np.full(dim, self.total_var_flow / (dim + 1))
            return centre.reshape(1, -1)

        return cands

    def _predict_next_ts(self, gaussian_process, rand_int: int) -> np.ndarray:
        """Thompson Sampling next point via a coherent GP posterior draw.

        Draws *one* function from the GP posterior using ``sample_y`` over a
        dense candidate grid (preserving inter-point correlations), then
        returns the full predicted flows for the candidate minimising that
        sample.  This is statistically correct and avoids passing a stochastic
        callable to ``differential_evolution``.

        Parameters
        ----------
        gaussian_process : GaussianProcessRegressor
            Fitted GP (already trained on the current data).
        rand_int : int
            Random seed — logged and stored in ``self._last_ts_seed`` for
            post-hoc traceability.

        Returns
        -------
        np.ndarray of shape (n_pumps,)
            Full predicted flows (varying + fixed).
        """
        candidates = self._sample_candidates_ts(n=2000)    # (m, dim)
        # Draw one coherent function from the GP posterior.
        sample = gaussian_process.sample_y(
            candidates, n_samples=1, random_state=rand_int
        ).ravel()                                           # (m,)
        best_idx = int(np.argmin(sample))
        x_reduced = candidates[best_idx]                   # (dim,)

        self.logger.info(
            f"TS: {len(candidates)} candidates | "
            f"sample min={sample.min():.4g} | "
            f"selected idx={best_idx} | seed={rand_int}"
        )

        # Reconstruct full flows (append last var flow + fixed flows).
        predicted_var_flows = np.append(x_reduced,
                                        self.total_var_flow - x_reduced.sum())
        predicted_flows = np.zeros(self.number_of_pumps)
        j = 0
        for i in range(self.number_of_pumps):
            if self.fixed_pumps[i]:
                predicted_flows[i] = self.fixed_flows[i]
            else:
                predicted_flows[i] = predicted_var_flows[j]
                j += 1
        return predicted_flows

    def predict_next(self, costs, gaussian_process, iteration=None):
        """
        Find the next flow set to evaluate by optimizing the acquisition function
        over the GP posterior, using differential evolution.

        If the GP fit is unreliable (too few points or length scale at bounds),
        returns a random exploration point instead.

        Parameters
        ----------
        costs : array-like of shape (n_samples)
            List of the known costs.
        gaussian_process : sk-learn GaussianProcessRegressor
            Regressor against which the expected improvement is calculated.
        iteration : int, optional
            Current BO iteration (post-init). Used for adaptive exploration.

        Returns
        -------
        array-like of shape (n_features-1)
            Flows optimizing the acquisition function.
        """
        if not self._is_gp_fit_reliable(gaussian_process):
            self.logger.info("GP fit unreliable — returning random exploration point")
            return self.pump_syst.generate_random_flows(
                self.bounds, self.total_var_flow, self.fixed_pumps, self.fixed_flows
            )

        # Signal scale (raw cost units) used to make xi dimensionless. Use the
        # costs the GP was actually trained on (_gp_display_costs: flats
        # remapped + ghosts), which matches gp.predict's denormalised output
        # scale and avoids the raw flat-sentinel inflating the std. Falls back
        # to the raw costs, then to 1.0 (legacy behaviour) if unusable.
        # gp.predict returns values in GP-target space (warped if cost_transform
        # == 'log'); best_y and y_scale must live in that same space.
        disp = getattr(self, "_gp_display_costs", None)
        scale_src = disp if (disp is not None and len(disp) > 1) else costs
        y_scale = float(np.std(self._cost_to_gp(scale_src)))
        if not np.isfinite(y_scale) or y_scale < 1e-12:
            y_scale = 1.0

        if self.acq_func == 'EI':
            best_y_gp = float(self._cost_to_gp(np.asarray(costs)).min())
            acq_func = ExpectedImprovement(gp_model=gaussian_process,
                                           best_y=best_y_gp, y_scale=y_scale)
        elif self.acq_func == 'LCB':
            acq_func = LowerConfidenceBound(gp_model=gaussian_process)
        elif self.acq_func == 'TS':
            # Thompson Sampling requires a coherent draw from the GP posterior.
            # Passing a stochastic callable to differential_evolution is
            # incorrect (non-deterministic objective → DE never converges).
            # Instead, use gp.sample_y over a dense candidate grid — see
            # _predict_next_ts / _sample_candidates_ts.
            rand_int = np.random.randint(0, 10_000)
            self._last_ts_seed = rand_int          # stored for traceability
            if iteration is not None:
                self.logger.info(
                    f"TS: exploration ratio N/A (coherent sample_y draw, "
                    f"seed={rand_int}, iter={iteration})"
                )
            return self._predict_next_ts(gaussian_process, rand_int)
        else:
            raise ValueError('Acquisition function not recognized.')

        if iteration is not None:
            acq_func = ExplorationWrapper(
                base_acq=acq_func,
                iteration=iteration,
            )
            self.logger.info(
                f"Exploration ratio: {acq_func.explore_ratio:.2f} "
                f"(iter {iteration})"
            )

        # Optional constrained-BO: multiply EI by P(feasible | x).
        if getattr(self, "feasibility_aware_override", False):
            if self.acq_func != 'EI':
                self.logger.info(
                    "Feasibility-aware requested but acquisition is %s; "
                    "the feasibility weight is only applied to EI — skipped.",
                    self.acq_func)
            else:
                feas_X = getattr(self, "_feas_X", None)
                feas_y = getattr(self, "_feas_label", None)
                if feas_X is not None and feas_y is not None:
                    clf = fit_feasibility_classifier(feas_X, feas_y, min_per_class=4)
                    if clf is not None:
                        acq_func = FeasibilityWeightedAcquisition(acq_func, clf)
                        self.logger.info("Feasibility-weighted EI active.")

        bounds = self.bounds[self.varying_pumps]
        constraints = LinearConstraint(np.ones(self.nb_of_var_pumps - 1), self.total_var_flow - bounds[-1][1],
                                       self.total_var_flow - bounds[-1][0])

        # Time the optimization
        start_time = time.time()
        res = differential_evolution(acq_func,
                                     bounds=bounds[:-1],
                                     constraints=constraints,
                                     tol=self.de_tol,
                                     maxiter=self.de_maxiter,
                                     popsize=25,  # 25×(n_dims) individuals for better 4D exploration
                                     init='sobol',
                                     polish=True,  # Critical: local refinement with L-BFGS-B
                                     atol=0,  # Disable absolute tolerance to avoid premature stopping
                                     updating='deferred')
        elapsed_time = time.time() - start_time

        # Detailed convergence diagnostics
        self.logger.info(f"Acquisition optimization (dim={self.nb_of_var_pumps-1}):")
        self.logger.info(f"  Time: {elapsed_time:.3f}s")
        self.logger.info(f"  Success: {res.success}")
        self.logger.info(f"  Iterations: {res.nit}/{self.de_maxiter}")
        self.logger.info(f"  Function evals: {res.nfev}")
        self.logger.info(f"  Acquisition value: {res.fun:.8e}")
        if res.nit >= self.de_maxiter * 0.9:
            self.logger.warning(f"  ⚠️ Nearly hit maxiter ({res.nit}/{self.de_maxiter})! Consider increasing de_maxiter")
        if not res.success:
            self.logger.warning(f"  ⚠️ Optimization did not converge: {res.message}")

        # Explore-vs-exploit diagnostic for the chosen point
        try:
            from scipy.stats import norm as _norm
            x_next_reduced = np.atleast_2d(res.x)
            mu_next, sigma_next = gaussian_process.predict(x_next_reduced, return_std=True)
            mu_next, sigma_next = float(mu_next[0]), float(sigma_next[0])

            # Reference sigma over the search domain (random sample of feasible points)
            n_ref = 200
            X_train = gaussian_process.X_train_
            lo = np.minimum(bounds[:-1, 0], X_train.min(axis=0))
            hi = np.maximum(bounds[:-1, 1], X_train.max(axis=0))
            X_ref = np.random.uniform(lo, hi, size=(n_ref, X_train.shape[1]))
            sigma_ref = gaussian_process.predict(X_ref, return_std=True)[1]
            sigma_pool_mean = float(np.mean(sigma_ref)) or 1e-12
            sigma_norm = sigma_next / sigma_pool_mean

            # EI decomposition (uses the same xi as the acquisition wrapper, if EI)
            base = getattr(acq_func, 'base_acq', acq_func)
            xi_used = float(getattr(base, 'xi', 0.0))
            # xi is dimensionless; scale it by y_scale (matching ExpectedImprovement).
            xi_scale = float(getattr(base, 'y_scale', 1.0))
            # mu_next is in GP-target space, so best_y must be too.
            best_y = float(self._cost_to_gp(np.asarray(costs)).min())
            improvement = best_y - mu_next - xi_used * xi_scale
            if sigma_next > 1e-12:
                z = improvement / sigma_next
                ei_mean = max(0.0, improvement) * _norm.cdf(z)
                ei_sigma = sigma_next * _norm.pdf(z)
            else:
                ei_mean = max(0.0, improvement)
                ei_sigma = 0.0
            ei_ratio = ei_sigma / (ei_mean + 1e-12)

            # Distance from x_next to nearest training point (reduced space)
            d_min = float(np.min(np.linalg.norm(X_train - x_next_reduced, axis=1)))

            tag = "EXPLORE" if (sigma_norm > 1.5 or ei_ratio > 1.0) else (
                  "EXPLOIT" if (sigma_norm < 0.5 and ei_ratio < 0.5) else "MIXED")
            self.logger.info(
                f"  Explore/exploit [{tag}]: "
                f"sigma_next={sigma_next:.3e} "
                f"sigma_norm={sigma_norm:.2f} "
                f"d_min={d_min:.3f} "
                f"EI_mean={ei_mean:.3e} EI_sigma={ei_sigma:.3e} ratio={ei_ratio:.2f} "
                f"xi={xi_used:.3f}"
            )
        except Exception as e:
            self.logger.debug(f"Explore/exploit diagnostic failed: {e}")

        # Add the last varying flow and fixed flows
        predicted_var_flows = np.append(res.x, self.total_var_flow - res.x.sum())
        j = 0
        predicted_flows = np.zeros(self.number_of_pumps)
        for i in range(self.number_of_pumps):
            if self.fixed_pumps[i]:
                predicted_flows[i] = self.fixed_flows[i]
            else:
                predicted_flows[i] = predicted_var_flows[j]
                j += 1
        return predicted_flows

    def init_points(self):
        """
        Initialize the Gaussian Process Regressor model.
        Measure initial samples and calculate costs.
        """

        # --- Generate initial points ---
        match self.init_option:
            case "random":
                self.initial_flows = self.pump_syst.generate_random_flows_sets(
                    self.nb_init_samples, self.bounds, self.total_var_flow,
                    self.fixed_pumps, self.fixed_flows
                )
            case "latinhypercube":
                self.initial_flows = self.pump_syst.generate_lhs_flows_sets(
                    self.nb_init_samples, self.bounds, self.total_var_flow,
                    self.fixed_pumps, self.fixed_flows
                )
            case "fixed":
                pass
            case _:
                raise ValueError("Invalid initialization_option. Please use 'random', 'latinhypercube' or 'fixed'.")

        self.logger.info(f"Initial flows: \n{self.initial_flows}")

        # --- Initialize results ---
        initial_costs = np.zeros(self.nb_init_samples)
        spectrum = None  # Initialize to None to avoid false empty spectrum

        # --- Loop over initial samples ---
        for i, flows in enumerate(self.initial_flows):
            self.logger.info(f"Initial measurement {i + 1}/{self.nb_init_samples}")

            sample_id = f"init_{i}"
            spectrum, self.initial_flows[i] = self.stop_flow_and_measure(flows, sample_id=sample_id)

            # --- Handle measurement failures ---
            if spectrum is None:
                self.logger.error("Measurement failed, stopping initialization.")
                return None, None, None

            # --- Store spectrum ---
            self.spectra[f'init {i + 1}'] = spectrum

            # --- Calculate cost ---
            cost, lamb_max = self.cost_function(spectrum)

            if self.target_style == 'single_point':
                self.logger.info(f"Absorbance at {self.tg_wl} nm: {spectrum.get_abs_at_wl(self.tg_wl)}")
            elif self.target_style == 'lambda_max':
                self.logger.info(f'Maximum absorbance at {spectrum.wavelengths[spectrum.absorbance.argmax()]} nm.')

            self.logger.info(f"cost = {cost}")

            # --- Save ---
            self.write_measurements(cost, spectrum, lamb_max)

            # --- Add cost ---
            initial_costs[i] = cost

        return self.initial_flows, initial_costs, spectrum

    # ------------------------------------------------------------------
    # Thread-safe GUI helpers (for use from worker threads)
    # ------------------------------------------------------------------
    def _gui(self, func, *args):
        """Schedule a fire-and-forget GUI call on the Tk main thread."""
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            self.app.after(0, func, *args)

    def _gui_call(self, func, *args):
        """Call *func* on the Tk main thread and wait for the return value."""
        if threading.current_thread() is threading.main_thread():
            return func(*args)
        import queue as _q
        result_q = _q.Queue()

        def _wrapper():
            try:
                result_q.put(('ok', func(*args)))
            except Exception as exc:
                result_q.put(('err', exc))

        self.app.after(0, _wrapper)
        tag, value = result_q.get()
        if tag == 'err':
            raise value
        return value

    def _gui_get(self, widget_or_var):
        """Thread-safe .get() on a tkinter Entry / StringVar / IntVar / BooleanVar."""
        return self._gui_call(widget_or_var.get)

    def start_exp_fixed(self):

        """Start experiment with fixed flows or stop-flow mode."""
        self._current_exp_type = 'Fixed'
        self.logger.info("Experiment started -- Fixed flows")
        number_of_samples = int(self._gui_get(self.app.number_of_samples_entry))
        self.residence_time = float(self._gui_get(self.app.residence_time_entry))
        # create metadata file

        # Initialize valve to trash position
        self._gui(self.app.switch_valve_to_trash)

        for i in range(number_of_samples):
            if self.app.abort_exp_flag:
                self._gui(self.app.stop_exp, 'Aborted via abort button')
                return

            sample_id = f"sample_{i + 1}"
            self.logger.info(f"Processing {sample_id}")
            if not self._gui_get(self.app.stop_flow_exp):  # Lost flow mode
                self._gui(self.app.move_to_next_well)
                self._gui(self.app.start_flow)

                if i == 0:
                    self.wait_for_steady_state()

                self.logger.info(f"Flows: {self.pump_syst.get_flows()}")
                self.collect_sample_with_spectro(sample_id)
            else:  # Stop flow mode
                #get flow in the app
                flows = [self._gui_call(entry.get) for entry in self.app.pump_flow_entry]
                flows = [float(f) for f in flows]
                self.stop_flow_and_measure(flows, sample_id=sample_id)

        self._gui(self.app.stop_exp, 'Normal ending')

    def start_exp_random(self):
        """Start experiment with random or LHS flows."""
        self._current_exp_type = self._gui_get(self.app.exp_type_var)  # 'Random' or 'LHS Flows'
        number_of_samples = int(self._gui_get(self.app.number_of_samples_entry))
        self.residence_time = int(self._gui_get(self.app.residence_time_entry))

        # Initialize valve to trash position
        self._gui(self.app.switch_valve_to_trash)

        # Generate flow sets based on experiment type
        exp_type = self._gui_get(self.app.exp_type_var)
        if exp_type == 'Random':
            flow_sets = self.pump_syst.generate_random_flows_sets(number_of_samples)
            self.logger.info("Experiment started -- Random flows")
        elif exp_type == 'LHS Flows':
            flow_sets = self.pump_syst.generate_lhs_flows_sets(number_of_samples)
            self.logger.info("Experiment started -- LHS flows")
        else:
            self.logger.error("No flow generation method selected")
            return

        # Iterate through the samples
        for i in range(number_of_samples):
            if self.app.abort_exp_flag:
                self._gui(self.app.stop_exp, 'Aborted via abort button')
                return

            sample_id = f"sample_{i + 1}"
            self.logger.info(f"Processing {sample_id}")

            if not self._gui_get(self.app.stop_flow_exp):  # Lost flow mode
                self._gui(self.app.move_to_next_well)
                self.pump_syst.set_flows(flow_sets[i])

                if i == 0:
                    self.wait_for_steady_state()

                self.logger.info(f"Flows: {self.pump_syst.get_flows()}")
                self.collect_sample_with_spectro(sample_id)
            else:  # Stop flow mode
                self.stop_flow_and_measure(flow_sets[i], sample_id=sample_id)

        self._gui(self.app.stop_exp, 'Normal ending')

    def start_exp_from_csv(self, csv_path):
        """Run experiment from a CSV file containing flow rates.

        Supports flow_plan.py format with a ``status`` column (todo/done).
        Only rows with status ``todo`` are executed; each is marked ``done``
        after completion.  Plain CSV files (no status column) are fully executed.
        """
        self._current_exp_type = 'CSV Flows'
        self.residence_time = float(self._gui_get(self.app.residence_time_entry))

        import pandas as pd
        try:
            df = pd.read_csv(csv_path, sep=";")
        except Exception:
            df = pd.read_csv(csv_path)

        has_status = "status" in df.columns
        pump_cols = [c for c in df.columns if c != "status"]

        if len(pump_cols) != self.number_of_pumps:
            self.logger.error(
                f"CSV has {len(pump_cols)} pump columns but expected {self.number_of_pumps}"
            )
            return

        if has_status:
            todo_df = df[df["status"] == "todo"]
            self.logger.info(f"{len(todo_df)} todo rows out of {len(df)} total")
        else:
            todo_df = df

        flow_sets = todo_df[pump_cols].values.astype(float)
        todo_indices = todo_df.index.tolist()
        csv_count = len(flow_sets)

        if csv_count == 0:
            self.logger.info("No remaining todo samples in CSV")
            self.callbacks.on_experiment_complete("All CSV samples already done")
            return

        # Cross-check with the GUI "number of samples" field
        try:
            requested = int(self._gui_get(self.app.number_of_samples_entry))
        except (ValueError, TypeError):
            requested = csv_count

        if requested < csv_count:
            # User asked for fewer samples than the CSV provides → just truncate.
            self.logger.info(
                f"CSV has {csv_count} todo rows; running only the first {requested} "
                f"as requested by the GUI sample count")
            flow_sets = flow_sets[:requested]
            todo_indices = todo_indices[:requested]
            number_of_samples = requested
        elif requested > csv_count:
            # User asked for more than the CSV has → confirm running just the CSV
            import tkinter.messagebox as _mb
            run_csv = self._gui_call(
                _mb.askyesno,
                "CSV shorter than requested",
                f"CSV file contains only {csv_count} todo flow sets but the "
                f"experiment is configured for {requested} samples.\n\n"
                f"Run the {csv_count} CSV rows anyway?\n"
                f"  Yes  →  run the {csv_count} available rows\n"
                f"  No   →  cancel experiment",
            )
            if not run_csv:
                self.logger.info(
                    f"CSV experiment cancelled by user (CSV={csv_count}, requested={requested})")
                self.callbacks.on_experiment_complete("Cancelled (CSV too short)")
                return
            self.logger.info(
                f"User confirmed running {csv_count} CSV rows (requested was {requested})")
            number_of_samples = csv_count
        else:
            number_of_samples = csv_count

        self.logger.info(f"Experiment started -- CSV Flows ({number_of_samples} flow sets from {csv_path})")
        self._gui(self.app.switch_valve_to_trash)

        for i in range(number_of_samples):
            if self.app.abort_exp_flag:
                self._gui(self.app.stop_exp, 'Aborted via abort button')
                return

            sample_id = f"sample_{i + 1}"
            self.logger.info(f"Processing {sample_id}")

            if not self._gui_get(self.app.stop_flow_exp):
                self._gui(self.app.move_to_next_well)
                self.pump_syst.set_flows(flow_sets[i])
                if i == 0:
                    self.wait_for_steady_state()
                self.logger.info(f"Flows: {self.pump_syst.get_flows()}")
                self.collect_sample_with_spectro(sample_id)
            else:
                self.stop_flow_and_measure(flow_sets[i], sample_id=sample_id)

            if has_status:
                df.loc[todo_indices[i], "status"] = "done"
                df.to_csv(csv_path, index=False, sep=";")

        self._gui(self.app.stop_exp, 'Normal ending')

    def collect_sample_with_spectro(self, sample_id):
        """Collect sample in lost-flow mode and perform inline spectroscopy."""
        self._gui(self.app.switch_valve_to_collection)
        #reset sample level

        self.pump_syst.set_flows(self.pump_syst.get_flows())
        self.collection_status = True
        self._gui(self.app.check_sample_level)
        timer = time.time()
        while self.collection_status:
            if self.app.abort_exp_flag:
                self._gui(self.app.stop_exp, 'Aborted via abort button')
                return

            # NOTE: self.app.update() was removed here — calling Tk update()
            # from a worker thread causes "main loop is not in main thread".
            # The main loop already processes events via mainloop(); periodic
            # after() callbacks handle GUI refreshes.

            if self._gui_get(self.app.inline_spectro_var):
                sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
                if round(time.time() - timer, 1) % sampling_rate == 0:
                    self._gui(self.app.measure_spectrum, sample_id)

            self._gui(self.app.check_sample_level)
            time.sleep(0.1)

        self._gui(self.app.switch_valve_to_trash)
        self.collector.move_z_to_position(999)
        self._gui(self.app.stop_flow)
    def stop_flow_and_measure(self, flows, path=None, meas_type='measurement', sample_id="", measure_ref=False):
        """Perform stop-flow measurement."""
        # alert = self.check_if_stopped()
        # if alert in ['stop', 'pause']:
        #     return self.handle_alert(alert, flows, path, meas_type, sample_id)

        ### STEP 1 : Prepare the system for stop-flow measurements
        self.callbacks.set_phase("Pumping")
        self.collector.switch_8_port_valve(0)
        self.collector.switch_10_port_valve(0)
        self.logger.info(f"Updating flows to {flows}...")
        self.prepare_flows(flows)
        # Capture the requested flows now, before any manual pump intervention
        # during stabilization can zero out get_flows() (e.g. when the user
        # stops idle pumps during ramp-up to save reagent).
        performed_flows = np.asarray(flows, dtype=float).copy()
        # Fill rinse syringes (water + HNO3)
        self.c3000_1.refill(flow=150)
        time.sleep(0.1)
        self.c3000_2.refill(flow=150)
        self.logger.info("Waiting for stabilization...")
        self.callbacks.set_phase("Stabilizing")
        self.wait_for_steady_state()
        # Purge sample collection loop
        if self._gui_get(self.app.sample_collection_var):
            # Valve collector to trash
            self.collector.switch_collection_valve(0)
            self.collector.move_z_to_position(999)
            # start carrier gas flow
            self.mfc.set_flow(8)

        ### STEP 2 : Monitor growth and perform spectroscopy in stop-flow mode while washing the chip
        self.callbacks.set_phase("Measuring")
        self.pump_syst.stop_all_pumps()

        # FOR PENTANE FLOW, check if there is en pentane bubble in the uv cell, if yes, push a little further
        self.collector.switch_8_port_valve(1)
        self.collector.switch_10_port_valve(1)

        intensive = self._should_do_intensive_wash(flows)
        if intensive:
            self._wash_sample_counter = 0
        else:
            self._wash_sample_counter += 1
        self._gui(self.app.update_wash_counter_display)
        wash_fn = self.flush_chip_intensive if intensive else self.flush_chip
        flush_chip_thread = threading.Thread(target=wash_fn)
        flush_chip_thread.start()
        if self._gui_get(self.app.sample_collection_var):
            self._gui(self.app.move_to_next_well)
            self._gui(self.app.switch_valve_to_collection)
            self.mfc.set_flow(20)

        # # While acid wash is running, start measurement
        # spectrum = self.monitor_growth_and_measure(sample_id)

        # Prepare metadata for recording
        metadata = {
            'chip_id': self.chip_id,
            'measurement_type': meas_type,
        }

        # Read spectrometer parameters from the live instrument
        spectro_params = {
            'spectro_integration_time_ms': getattr(self.spectro, 'integration_time', None),
            'spectro_number_averages': getattr(self.spectro, 'number_average', None),
            'spectro_smoothing_px': getattr(self.spectro, 'smoothing', None),
            'spectro_sampling_rate_s': float(self._gui_get(self.app.spectro_sampling_rate_entry)),
        }

        # Read wash / flush parameters from the GUI
        wash_params = {
            'wash_acid_volume_uL': float(self._gui_get(self.app.nitric_acid_volume_entry)),
            'wash_water_volume_uL': float(self._gui_get(self.app.water_volume_entry)),
            'wash_backpressure_flow': float(self._gui_get(self.app.back_pressure_entry)),
        }

        tc = self.termination
        termination_params = {
            'termination_correlation_threshold': tc.correlation_threshold,
            'termination_amplitude_threshold': tc.amplitude_threshold,
            'termination_min_spectra': tc.min_spectra,
            'termination_timeout_s': tc.monitoring_timeout_s,
        }

        exp_params = {
            'experiment_type': getattr(self, '_current_exp_type', 'Stop flow'),
            'reagents': self.reagents,
            'residence_time': self.residence_time,
            'total_flow': self.total_flow,
            'flows_set': flows.tolist() if isinstance(flows, np.ndarray) else flows,
            'target_style': self.target_style,
            'target_wavelength': self.tg_wl,
            'target_absorbance': self.tg_abs,
            'tolerance': self.tol,
            'cost_method': getattr(self, 'cost_method', None),
            'acquisition_function': getattr(self, 'acq_func', None),
            'fixed_pumps': getattr(self, 'fixed_pumps', None),
            'varying_pumps': getattr(self, 'varying_pumps', None),
            # PCA decomposition file used for the score-space cost (if any).
            'pca_model_file': getattr(self, 'pca_model_file', None),
            **spectro_params,
            **wash_params,
            **termination_params,
        }
        # Target spectrum (full_spectrum / pca): store the actual curve in the h5.
        if getattr(self, 'tg_spectrum', None) is not None:
            exp_params['target_spectrum_wavelengths'] = np.asarray(
                self.tg_spectrum.wavelengths, dtype=float)
            exp_params['target_spectrum_absorbance'] = np.asarray(
                self.tg_spectrum.absorbance, dtype=float)
        # Create file for time series
        self.app.data_manager.create_sample_file(sample_id, metadata, **exp_params)

        # Collect time series of spectra
        spectrum_series, spectrum = self.monitor_growth_and_measure_series(sample_id)

        # Save time evolution figure with color gradient
        if spectrum_series and len(spectrum_series) >= 2:
            from toofast.visualization.figure_plotting import save_time_evolution_figure
            spectra_data = [s.absorbance for s in spectrum_series if s.absorbance is not None]
            if spectra_data:
                save_time_evolution_figure(
                    self.experiment_path, sample_id,
                    spectrum_series[0].wavelengths, spectra_data,
                    flows=flows)

        # Finalize with actual flow rates
        self.app.data_manager.finalize_sample(sample_id, performed_flows)

        # Return last spectrum for compatibility
        final_spectrum = spectrum_series[-1] if spectrum_series else None

        # Ensure acid wash is complete before continuing.
        # On abort, the flush thread exits quickly (cooperative checks) so a short
        # join timeout is sufficient; we then run the emergency hardware stop.
        self.callbacks.set_phase("Cleaning")
        join_timeout = 5 if self._abort_event.is_set() else 120
        flush_chip_thread.join(timeout=join_timeout)

        if self._abort_event.is_set():
            self._emergency_hardware_stop()
            self._abort_event.clear()
            return None, None

        # Check C3000 pump status after wash
        # Trace each step so a freeze in serial I/O is visible in the log.
        for pump, name in [(self.c3000_1, "Water"), (self.c3000_2, "HNO3")]:
            self.logger.debug(f"Post-wash: checking C3000 {name} error status")
            error = pump.check_error()
            self.logger.debug(f"Post-wash: C3000 {name} error status returned ({error!r})")
            if error:
                self.logger.error(f"C3000 {name} pump error after wash: {error}")
                self.callbacks.set_phase("Paused")
                import tkinter.messagebox as _mb
                self._gui_call(
                    _mb.showwarning,
                    "C3000 Pump Error",
                    f"{error}\n\nResolve the issue, then press OK to continue.")
                self.logger.info(f"C3000 {name} pump error resolved, continuing.")

        self.logger.debug("Post-wash: reading sample_collection_var (GUI round-trip)")
        if self._gui_get(self.app.sample_collection_var):
            self.logger.debug("Post-wash: collector.move_z_to_position(999)")
            self.collector.move_z_to_position(999)
            self._gui(self.app.switch_valve_to_trash)
            self.mfc.set_flow(0)  # Stop carrier gas flow

        # self.save_measurement_fig(sample_id, flows)
        self.save_or_plot_spectrum(spectrum, path, meas_type)
        self.logger.debug("Post-wash: switch_10_port_valve(0)")
        self.collector.switch_10_port_valve(0)
        self.logger.debug("Post-wash: switch_10_port_valve done; entering UV cell wash")

        if self.app.abort_exp_flag:
            self.logger.info("Abort detected — skipping post-measurement wash sequence")
            return final_spectrum, performed_flows

        if measure_ref: #remove for testing purpose.
            self.perform_acid_wash_and_reference_measurement()
        else:
            time.sleep(1)
            self.dispense_acid_and_flush()
            time.sleep(1) # why ?
            if self._gui_get(self.app.manual_spectrum_validation):
                self._gui_call(self.app.ask_for_spectrum_validation)
        return final_spectrum, performed_flows

    def monitor_growth_and_measure_series(self, sample_id: str, duration: float = None):
        """
        Monitor particle growth by collecting a time series of spectra.

        Args:
            sample_id: Sample identifier
            duration: Total monitoring duration in seconds
            interval: Interval between measurements in seconds

        Returns:
            List[Spectrum]: List of collected spectra
        """
        spectra_collected = []
        spectrum = None
        if duration is None:
            duration = self.termination.monitoring_timeout_s
        start_time = time.time()
        sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
        next_sampling_time = time.perf_counter() + sampling_rate

        self.logger.info(f"Starting time series collection for sample {sample_id}")
        self.logger.info(f"Duration: {duration}s, Interval: {sampling_rate}s")

        try:
            while (time.time() - start_time) < duration:
                if self.app.abort_exp_flag:
                    self.logger.info("Time series collection aborted by user")
                    break
                # self.app.update()
                current_time = time.perf_counter()
                # Mesurer un spectre

                if current_time >= next_sampling_time:
                    spectrum = self.measure_spectrum(sample_id=sample_id, meas_type='measurement')
                    self.app.data_manager.add_spectrum_to_sample(sample_id, spectrum)
                    # Keep in memory for compatibility
                    spectra_collected.append(spectrum)
                    next_sampling_time = current_time + sampling_rate  # Schedule next measurement

                    if self.np_growth_done(sample_id):
                        break

                time.sleep(0.1)

            # Move these lines outside the loop to avoid premature return
            if spectrum is not None:
                try:
                    self._gui(self.app.plot_spectrum, spectrum)
                    self.logger.info("Spectrum measurement done.")
                except Exception as e:
                    self.logger.warning(f"Unable to display spectrum: {e}")

        except Exception as e:
            self.logger.error(f"Error during time series collection: {str(e)}")

        finally:
            total_collected = len(spectra_collected)
            self.logger.info(f"Collection complete. {total_collected} spectra collected for sample {sample_id}")

        # Use last collected spectrum if it exists, otherwise None
        final_spectrum = spectra_collected[-1] if spectra_collected else None
        return spectra_collected, final_spectrum

    def _check_abort(self) -> bool:
        """Return True if abort has been requested. Used to short-circuit long sequences."""
        return self._abort_event.is_set()

    def flush_chip(self):
        """Flush the chip with nitric acid and water from c3000 pumps
        Reagent pumps are activated at low flow to maintain pressure in the syringes
        """
        ev = self._abort_event
        # Flush acid
        if self._check_abort(): return
        self.c3000_1.switch_valve_to_position("left")
        time.sleep(0.2)
        if self._check_abort(): return
        self.c3000_2.switch_valve_to_position("right")
        time.sleep(0.1)
        self.c3000_1.switch_valve_to_position("left")  # Remove when proper valve position check is implemented
        back_pressure_flow = float(self._gui_get(self.app.back_pressure_entry))
        flow_set = [back_pressure_flow]*self.number_of_pumps
        self.pump_syst.set_flows(flow_set)
        self.app.logger.info(f"set flow to {flow_set}")
        volume_acid = float(self._gui_get(self.app.nitric_acid_volume_entry))
        if self._check_abort(): return
        self.app.logger.info(f"dispense {volume_acid} µL of acid")
        self.c3000_2.dispense(volume=volume_acid, flow=10)
        self.c3000_2.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        time.sleep(0.5)
        # Maintain pressure in reagent syringes
        self.c3000_1.switch_valve_to_position("right")
        time.sleep(0.2)
        self.c3000_2.switch_valve_to_position("left")
        volume_eau = float(self._gui_get(self.app.water_volume_entry))
        if self._check_abort(): return
        self.app.logger.info(f"dispense {volume_eau} µL of water")
        self.c3000_1.dispense(volume=volume_eau, flow=11)
        self.c3000_1.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        self.pump_syst.stop_all_pumps()
        self.c3000_1.switch_valve_to_position("left")
        self.app.logger.info("Chip flush done.")

    def flush_chip_intensive(self):
        """Intensive chip flush: more acid, split water phase with mid-rinse back-pressure stop.

        Volumes and split point are read from the dedicated GUI entries (temporary,
        to be hard-coded in final version).
        """
        ev = self._abort_event
        volume_acid = float(self._gui_get(self.app.intensive_acid_volume_entry))
        volume_water_1 = float(self._gui_get(self.app.intensive_water_bp_stop_entry))
        volume_water_total = float(self._gui_get(self.app.intensive_water_volume_entry))
        volume_water_2 = max(0.0, volume_water_total - volume_water_1)

        # Acid phase
        if self._check_abort(): return
        self.c3000_1.switch_valve_to_position("left")
        time.sleep(0.2)
        if self._check_abort(): return
        self.c3000_2.switch_valve_to_position("right")
        time.sleep(0.1)
        self.c3000_1.switch_valve_to_position("left")
        back_pressure_flow = float(self._gui_get(self.app.back_pressure_entry))
        flow_set = [back_pressure_flow] * self.number_of_pumps
        self.pump_syst.set_flows(flow_set)
        if self._check_abort(): return
        self.app.logger.info(f"[intensive] dispense {volume_acid} µL acid")
        self.c3000_2.dispense(volume=volume_acid, flow=10)
        self.c3000_2.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        time.sleep(0.5)

        # Water phase 1: with back-pressure
        self.c3000_1.switch_valve_to_position("right")
        time.sleep(0.2)
        self.c3000_2.switch_valve_to_position("left")
        if self._check_abort(): return
        self.app.logger.info(f"[intensive] dispense {volume_water_1} µL water (back-pressure on)")
        self.c3000_1.dispense(volume=volume_water_1, flow=11)
        self.c3000_1.wait_for_idle(stop_event=ev)
        if self._check_abort(): return

        # Stop back-pressure from reagent syringes
        self.pump_syst.stop_all_pumps()

        # Water phase 2: without back-pressure
        if volume_water_2 > 0:
            self.app.logger.info(f"[intensive] dispense {volume_water_2} µL water (back-pressure off)")
            self.c3000_1.dispense(volume=volume_water_2, flow=11)
            self.c3000_1.wait_for_idle(stop_event=ev)
            if self._check_abort(): return

        self.c3000_1.switch_valve_to_position("left")
        self.app.logger.info("Intensive chip flush done.")

    def _compute_ag_pvp_ratio(self, flows) -> Optional[float]:
        """Return molar Ag/PVP flow ratio for the given flow set, or None if not determinable."""
        ag_molar_flow = None
        pvp_molar_flow = None
        for i in range(self.number_of_pumps):
            reagent = self._gui_get(self.app.selected_reagents[i])
            try:
                conc = float(self._gui_get(self.app.reagent_concentrations[i]))
            except (ValueError, TypeError):
                continue
            flow = flows[i] if i < len(flows) else 0.0
            if reagent == "AgNO3":
                ag_molar_flow = flow * conc
            elif reagent == "PVP":
                pvp_molar_flow = flow * conc
        if ag_molar_flow is None or pvp_molar_flow is None or pvp_molar_flow == 0:
            return None
        return ag_molar_flow / pvp_molar_flow

    def _should_do_intensive_wash(self, flows) -> bool:
        """Return True if conditions require an intensive wash."""
        self.app.logger.debug(
            f"Wash counter before decision: {self._wash_sample_counter} "
            f"(samples since last intensive wash)")
        if self._gui_get(self.app.regular_wash_var):
            if self._wash_sample_counter >= 9:
                self.app.logger.info(
                    f"Regular intensive wash triggered (counter={self._wash_sample_counter})")
                return True
        if self._gui_get(self.app.pvp_threshold_wash_var):
            try:
                threshold = float(self._gui_get(self.app.pvp_threshold_entry))
            except (ValueError, TypeError):
                threshold = 3.0
            ratio = self._compute_ag_pvp_ratio(flows)
            if ratio is not None and ratio > threshold:
                self.app.logger.info(
                    f"PVP-threshold intensive wash triggered (Ag/PVP={ratio:.2f} > {threshold})")
                return True
        return False

    def trigger_manual_wash(self):
        """Run an intensive chip flush immediately (called from the manual wash button)."""
        self.app.logger.info("Manual intensive wash triggered.")
        self._wash_sample_counter = 0
        self._gui(self.app.update_wash_counter_display)
        self.flush_chip_intensive()

    def _emergency_hardware_stop(self):
        """Bring all hardware to a safe state.

        Called by abort/stop_all. Each step is wrapped in try/except so that one
        failure (e.g. unconnected MFC) does not prevent the others from running.
        Safe state:
          - Nemesys pumps: stopped
          - C3000 pumps: stopped, valves to "left" (refill)
          - MFC: flow set to 0
          - Collector: valve to trash, Z bar to 999
        Resets the wash counter as a new session boundary.
        """
        self.logger.warning("Emergency hardware stop initiated.")
        try:
            self.pump_syst.stop_all_pumps()
        except Exception as e:
            self.logger.error(f"Emergency stop: Nemesys stop failed: {e}")
        for pump, name in [(getattr(self, 'c3000_1', None), "C3000_1"),
                           (getattr(self, 'c3000_2', None), "C3000_2")]:
            if pump is None:
                continue
            try:
                pump.stop_pump()
            except Exception as e:
                self.logger.error(f"Emergency stop: {name} stop failed: {e}")
            try:
                pump.switch_valve_to_position("left")
            except Exception as e:
                self.logger.error(f"Emergency stop: {name} valve→left failed: {e}")
        try:
            if getattr(self, 'mfc', None) is not None:
                self.mfc.set_flow(0)
        except Exception as e:
            self.logger.error(f"Emergency stop: MFC set_flow(0) failed: {e}")
        try:
            if getattr(self, 'collector', None) is not None and self.collector_status:
                self.collector.switch_collection_valve(0)
                self.collector.move_z_to_position(999)
        except Exception as e:
            self.logger.error(f"Emergency stop: collector failed: {e}")

        self._wash_sample_counter = 0
        self._gui(self.app.update_wash_counter_display)
        self.logger.warning("Emergency hardware stop done.")

    def trigger_emergency_stop(self):
        """Public entry point for the Stop All button (always available).

        Sets the abort event so any cooperative loops exit, then performs the
        hardware stop on the calling thread (typically a daemon thread spawned
        by the GUI).
        """
        self._abort_event.set()
        # Brief pause to let in-flight serial transactions complete before we
        # touch the shared C3000 serial port from this thread.
        time.sleep(1.0)
        self._emergency_hardware_stop()
        self._abort_event.clear()

    def prepare_flows(self, flows):
        """Prepare the system for stop-flow measurements."""
        standard_flows = [self.total_flow / self.number_of_pumps] * self.number_of_pumps  # Not sure about keeping this step
        self.pump_syst.set_flows(standard_flows)
        time.sleep(5)
        self.pump_syst.set_flows(flows)
        self.pump_syst.write_flows(self.experiment_path)

    def monitor_growth_and_measure(self, sample_id):
        """Monitor growth and perform spectroscopy in stop-flow mode."""
        sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
        next_sampling_time = time.perf_counter() + sampling_rate
        try:
            while True:
                # NOTE: self.app.update() removed — causes threading crash.
                # The Tk mainloop already runs event processing.
                current_time = time.perf_counter()

                if current_time >= next_sampling_time:
                    spectrum = self.measure_spectrum(sample_id=sample_id)
                    next_sampling_time = current_time + sampling_rate  # Planifier la prochaine mesure

                    if self.np_growth_done(sample_id):
                        break

                time.sleep(0.1)

            self._gui(self.app.plot_spectrum, spectrum)
            self.logger.info("Spectrum measurement done.")
            return spectrum

        except Exception as e:
            self.logger.error(f"Error in monitor_growth_and_measure: {e}")
            return None

    def save_or_plot_spectrum(self, spectrum, path, meas_type):
        """Save or plot the spectrum based on measurement type."""
        if meas_type == "measurement":
            # self.pump_syst.write_flows(self.experiment_path)
            self.logger.debug("Measurement spectrum acquired")
        else:
            self._gui(self.app.plot_spectrum, spectrum, path or self.experiment_path, f'{meas_type} spectrum')
            if meas_type == 'target':
                self.tg_spectrum = spectrum

    def perform_acid_wash(self):
        """Perform an acid wash for cleaning."""
        acid_pump_index = 0
        for i in range(self.number_of_pumps):
            if self._gui_get(self.app.selected_reagents[i]) == "Acetic Acid":
                acid_pump_index = i
                break

        self.pump_syst.acid_wash(acid_pump_index)
        self.logger.info("Acid wash done.")

    def perform_acid_wash_and_reference_measurement(self):
        """Perform an acid wash and measure the reference spectrum."""
        self.dispense_acid_and_flush()
        time.sleep(5)
        while True:
            self.collector.open_lamp_shutter()
            self.measure_reference()
            self.collector.close_lamp_shutter()
            if self._gui_call(self.app.ask_for_spectrum_validation):
                break

    def handle_alert(self, alert, flows, path, meas_type, sample_id):
        """Handle 'stop' and 'pause' alerts during an experiment."""
        if alert == 'stop':
            self.logger.info("Experiment stopped due to 'stop' alert.")
            self._gui(self.app.stop_exp, 'Experiment stopped by user.')
            return None, None
        elif alert == 'pause':
            self.logger.info("Experiment paused due to 'pause' alert.")
            self.pump_syst.stop_all_pumps()
            self.pause()  # Pause method to implement if it doesn't exist
            spectrum, performed_flows = self.stop_flow_and_measure(flows, path, meas_type, sample_id)
            return spectrum, performed_flows

    def dispense_acid_and_flush(self):
        """Dispense acid and flush the system."""
        # c3000_2 is used for nitric acid, c3000_1 for water
        self.c3000_1.switch_valve_to_position("left")
        self.c3000_2.switch_valve_to_position("center")
        time.sleep(0.5)
        self.c3000_2.generate_flow(20)
        time.sleep(5)
        self.c3000_2.stop_pump()
        time.sleep(0.5)
        self.c3000_2.switch_valve_to_position("left")
        self.c3000_1.switch_valve_to_position("center")
        time.sleep(0.5)
        self.c3000_1.generate_flow(100)
        time.sleep(20)
        self.c3000_1.stop_pump()
        self.c3000_1.switch_valve_to_position("left")
        time.sleep(0.5)

    def flush_nitric_acid(self):
        """Flush the system with nitric acid."""
        self.c3000_2.dispense(volume=25, flow=10)
        time.sleep(0.5)

    def optimise(
        self,
        plot_figs: bool = True,
        save_figs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, GaussianProcessRegressor]:
        """
        Run Bayesian Optimization loop to find optimal flow parameters.

        This is the core optimization method that iteratively:
        1. Fits a Gaussian Process to observed data
        2. Selects next point using acquisition function
        3. Performs experiment at selected flows
        4. Updates the model with new observation

        Parameters
        ----------
        plot_figs : bool, optional
            If True, plots intermediate figures during optimization.
            The final figure is always plotted (default: True).
        save_figs : bool, optional
            If True, saves intermediate figures to disk.
            The final figure is always saved (default: True).

        Returns
        -------
        flows : np.ndarray of shape (n_samples, n_pumps)
            Array of all flow rate combinations tested.
        costs : np.ndarray of shape (n_samples,)
            Array of corresponding cost values.
        gaussian_process : GaussianProcessRegressor
            The fitted Gaussian Process model.
        """
        self.logger.debug("Starting optimisation")
        self._rebuild_kernel()
        base_alpha = 1e-8
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=base_alpha,
            normalize_y=True,
            n_restarts_optimizer=self.gp_n_restarts_override or 100,
        )

        self.write_params(gaussian_process)
        self._copy_previous_files()
        self.save_target_spectrum()
        best_min = []
        times = []  # contains the computation times
        predicted_flows = None
        self.growth_times = []        # reset MO objective-2 history for this run
        self.mo_scalarizer = None     # fresh, seed-deterministic scalarizer

        # Load previous experiment data if available (validated against current config)
        if self._validate_previous_data():
            flows = self.previous_flows.copy()
            costs = self.previous_costs.copy()
            self.logger.info(f"Starting from {len(costs)} previous data points")
            self.logger.info(f"Previous best cost: {costs.min():.4f}")

            # Calculate correct starting iteration to match plot_fig expectations
            # If we have N costs with nb_init_samples init points, we've done (N - nb_init_samples) BO iterations
            # So we should continue from iteration (N - nb_init_samples + 1)
            starting_iter = len(costs) - self.nb_init_samples + 1
            if getattr(self, "_n_warmstart", 0) > 0:
                # Seeded from a historical DB (warm_start_from_h5): the loaded
                # points ARE the initial design, so no init phase is run and
                # maxiter counts NEW BO iterations (i=1..maxiter) instead of an
                # absolute index — otherwise 100 warm-start points would push
                # starting_iter past maxiter and the BO would do nothing.
                starting_iter = 1
                self.logger.info(
                    f"Warm-start: {len(costs)} historical points used as the "
                    f"initial design (no init phase); running {self.maxiter} "
                    f"new BO iteration(s).")
            self.logger.info(f"Continuing from iteration {starting_iter}")

            # Initialize best_min with the previous data progression
            for j in range(len(costs)):
                best_min.append(costs[:j+1].min())

            # Pre-train the GP with previous data and compute next point
            self.logger.info("Pre-training GP with previous data...")
            variant_flows = flows[:, self.varying_pumps]  # Dimensionality reduction
            flows_minus_one = variant_flows[:, :-1]
            y_acq = self._fit_gp(gaussian_process, flows_minus_one, costs, base_alpha)

            # Save the pre-trained GP model
            if hasattr(self, 'experiment_path') and self.experiment_path:
                try:
                    gp_model_file = os.path.join(self.experiment_path, "previous_gp_model.pkl")
                    with open(gp_model_file, 'wb') as f:
                        pickle.dump(gaussian_process, f)
                    self.logger.info(f"Saved pre-trained GP model to: {gp_model_file}")
                except Exception as e:
                    self.logger.warning(f"Could not save GP model: {e}")

            # Compute the first predicted flow to continue optimization
            predicted_flows = self.predict_next(y_acq, gaussian_process, iteration=starting_iter)
            self.logger.info(f"Next predicted flows: {predicted_flows}")
        else:
            flows = None
            costs = None
            starting_iter = 0  # Start from iteration 0 for initialization

        for i in range(starting_iter, self.maxiter + 1):
            if self.check_if_stopped() == 'stop':
                i -= 1
                break

            self.logger.info("-------------")
            self.logger.info(f'   iter {i} ')
            self.logger.info("-------------")

            if i == 0:
                flows, costs, spectrum = self.init_points()
                if flows is None:
                    return
                self.write_params(gaussian_process)  # To include the initial flows
            else:
                spectrum, performed_flows = self.make_flow_and_measure(predicted_flows)
                if spectrum is None:
                    i -= 1
                    break
                self.spectra[f'iter {i}'] = spectrum
                cost_at_predict, lamb_max = self.cost_function(spectrum)
                self.logger.info(f'measured cost: {cost_at_predict}')
                self.write_measurements(cost_at_predict, spectrum, lamb_max)
                if self.target_style == 'single_point':
                    self.logger.info(f"Absorbance at {self.tg_wl} nm: {spectrum.get_abs_at_wl(self.tg_wl)}")
                elif self.target_style == 'lambda_max':
                    self.logger.info(f'Maximum absorbance at {spectrum.wavelengths[spectrum.absorbance.argmax()]} nm.')

                # Update database
                flows = np.concatenate((flows, [performed_flows]))
                costs = np.concatenate((costs, [cost_at_predict]))
                self._note_growth_time()  # align growth_times with costs (MO)

            # Only append to best_min if not starting from previous data (already initialized)
            if i >= starting_iter or starting_iter == 0:
                best_min.append(costs.min())

            tmp = time.time()

            # Update the GPR
            variant_flows = flows[:, self.varying_pumps]  # Dimensionality reduction
            flows_minus_one = variant_flows[:, :-1]

            y_acq = self._fit_gp(gaussian_process, flows_minus_one, costs, base_alpha)

            # Diagnose GP convergence (only log detailed diagnostics every 3 iterations to avoid spam)
            if i % 3 == 0 or i == self.maxiter:
                self.diagnose_gp_convergence(gaussian_process, flows_minus_one, costs)

            # Prediction of the next flow to evaluate
            predicted_flows = self.predict_next(y_acq, gaussian_process, iteration=i)

            times.append(time.time() - tmp)

            # Plot the figure via callbacks
            self.callbacks.on_iteration_complete(
                i, flows, self._display_costs(costs), gaussian_process, best_min, times,
                spectrum, predicted_flows, False, save_figs,
            )

            # If tolerance is reached, stops loop
            if best_min[-1] < self.tol:
                self.success = True
                if self.tol_stop:
                    break

        self.logger.info("------------")
        self.logger.info(f'final figure ({i})')
        self.logger.info("------------")

        # Optimal results
        self.min_cost = costs.min()
        self.opt_flows = flows[costs.argmin()]
        if costs.argmin() >= self.nb_init_samples:
            self.best_iter = costs.argmin() - self.nb_init_samples + 1
            best_spectrum = self.spectra[f'iter {self.best_iter}']
        else:
            self.best_iter = 0
            best_spectrum = self.spectra[f'init {costs.argmin() + 1}']

        self.callbacks.on_iteration_complete(
            i, flows, self._display_costs(costs), gaussian_process, best_min, times,
            best_spectrum, None, True, True,
        )

        self.logger.info(f"Optimal flows = {self.opt_flows}")
        if self.target_style == 'single_point':
            self.logger.info(f"Absorbance at {self.tg_wl} nm: {best_spectrum.get_abs_at_wl(self.tg_wl)}")
        elif self.target_style == 'lambda_max':
            self.logger.info(f'Maximum absorbance at {best_spectrum.absorbance.argmax()} nm.')
        self.logger.info(f"Minimum measured cost: {self.min_cost}")

        self.write_params(gaussian_process, end=True)

        # Load-previous is one-shot: clear so the next experiment starts fresh
        # unless the user explicitly clicks "Load Previous" again.
        self.previous_flows = None
        self.previous_costs = None
        self._n_warmstart = 0

        self.callbacks.on_experiment_complete('Success' if self.success else 'Maxiter reached')
        return flows, costs, gaussian_process

    def make_flow_and_measure(self, flows, path=None, meas_type='measurement'):
        alert = self.check_if_stopped()
        if alert == 'stop':
            return None, None
        elif alert == 'pause':
            self.pause()
        self.logger.info(f"updating flows to {flows}... ")
        self.pump_syst.set_flows(flows)
        # Capture the requested flows now, before any manual pump intervention
        # during stabilization can zero out get_flows().
        performed_flows = np.asarray(flows, dtype=float).copy()
        alert = self.wait_for_steady_state()
        if alert == 'stop':
            return None, None
        elif alert == 'pause':
            self.pump_syst.stop_all_pumps()
            self.pause()
            spectrum, performed_flows = self.make_flow_and_measure(flows, path, meas_type)
            return spectrum, performed_flows
        self.logger.info('Done.')
        self.logger.info("Measuring spectrum...")
        while True:
            self.collector.open_lamp_shutter()
            spectrum = self.spectro.measure_spectrum()
            self.collector.close_lamp_shutter()
            self._gui(self.app.plot_spectrum, spectrum)
            # ask user if the spectrum is good
            if self._gui_call(self.app.ask_for_spectrum_validation):
                break
            else:
                time.sleep(1)
                alert = self.wait_for_steady_state()
                if alert == 'stop':
                    return None, None
                elif alert == 'pause':
                    self.pump_syst.stop_all_pumps()
                    self.pause()
                    spectrum, performed_flows = self.make_flow_and_measure(flows, path, meas_type)
                    return spectrum, performed_flows
                self.logger.info("Remaking spectrum...")
        self.logger.info('Done.')
        if meas_type == "measurement":
            self.pump_syst.write_flows(self.experiment_path)
        else:
            self._gui(self.app.plot_spectrum, spectrum,
                      self.experiment_path if path is None else path, f'{meas_type} spectrum')
            if meas_type == 'target':
                self.tg_spectrum = spectrum
                self._gui(self.app.plot_spectrum, spectrum)
        self.pump_syst.stop_all_pumps()
        spectrum.save_data(self.experiment_path if path is None else path)
        return spectrum, performed_flows


    def stop_flow(self, plot_figs=True, save_figs=True):
        self._current_exp_type = 'Bayesian Optimization'
        self._rebuild_kernel()
        base_alpha = 1e-3
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=base_alpha,
            normalize_y=True,
            n_restarts_optimizer=self.gp_n_restarts_override or 100,
        )
        # self.write_parameters_file(gaussian_process)

        self.write_params(gaussian_process)
        self._copy_previous_files()
        self.save_target_spectrum()

        best_min = []
        times = []  # contains the computation times
        predicted_flows = None
        self.growth_times = []        # reset MO objective-2 history for this run
        self.mo_scalarizer = None     # fresh, seed-deterministic scalarizer

        # Load previous experiment data if available (validated against current config)
        if self._validate_previous_data():
            flows = self.previous_flows.copy()
            costs = self.previous_costs.copy()
            self.logger.info(f"Starting from {len(costs)} previous data points")
            self.logger.info(f"Previous best cost: {costs.min():.4f}")

            # Calculate correct starting iteration to match plot_fig expectations
            # If we have N costs with nb_init_samples init points, we've done (N - nb_init_samples) BO iterations
            # So we should continue from iteration (N - nb_init_samples + 1)
            starting_iter = len(costs) - self.nb_init_samples + 1
            if getattr(self, "_n_warmstart", 0) > 0:
                # Seeded from a historical DB (warm_start_from_h5): the loaded
                # points ARE the initial design, so no init phase is run and
                # maxiter counts NEW BO iterations (i=1..maxiter) instead of an
                # absolute index — otherwise 100 warm-start points would push
                # starting_iter past maxiter and the BO would do nothing.
                starting_iter = 1
                self.logger.info(
                    f"Warm-start: {len(costs)} historical points used as the "
                    f"initial design (no init phase); running {self.maxiter} "
                    f"new BO iteration(s).")
            self.logger.info(f"Continuing from iteration {starting_iter}")

            # Initialize best_min with the previous data progression
            for j in range(len(costs)):
                best_min.append(costs[:j+1].min())

            # Pre-train the GP with previous data and compute next point
            self.logger.info("Pre-training GP with previous data...")
            variant_flows = flows[:, self.varying_pumps]  # Dimensionality reduction
            flows_minus_one = variant_flows[:, :-1]
            y_acq = self._fit_gp(gaussian_process, flows_minus_one, costs, base_alpha)

            # Save the pre-trained GP model
            if hasattr(self, 'experiment_path') and self.experiment_path:
                try:
                    gp_model_file = os.path.join(self.experiment_path, "previous_gp_model.pkl")
                    with open(gp_model_file, 'wb') as f:
                        pickle.dump(gaussian_process, f)
                    self.logger.info(f"Saved pre-trained GP model to: {gp_model_file}")
                except Exception as e:
                    self.logger.warning(f"Could not save GP model: {e}")

            # Compute the first predicted flow to continue optimization
            predicted_flows = self.predict_next(y_acq, gaussian_process, iteration=starting_iter)
            self.logger.info(f"Next predicted flows: {predicted_flows}")
        else:
            flows = None
            costs = None
            starting_iter = 0  # Start from iteration 0 for initialization

        for i in range(starting_iter, self.maxiter + 1):
            if self.check_if_stopped() == 'stop':
                i -= 1
                break

            self.logger.info("-------------")
            self.logger.info(f'   iter {i} ')
            self.logger.info("-------------")

            if i == 0:
                flows, costs, spectrum = self.init_points()
                if flows is None:
                    return
                self.write_params(gaussian_process)  # To include the initial flows
            else:
                sample_id = f"iter_{i}"
                # Loop allowing to redo measurement if spectrum is not good
                while True:
                    spectrum, performed_flows = self.stop_flow_and_measure(predicted_flows, sample_id=sample_id)
                    if spectrum is None:
                        i -= 1
                        break
                    self.spectra[f'iter {i}'] = spectrum
                    cost_at_predict, lamb_max = self.cost_function(spectrum)
                    if self.callbacks.validate_cost(cost_at_predict):
                        break
                    else:
                        self.logger.info(f"iter_{i} is measured again.")

                self.logger.info(f'measured cost: {cost_at_predict}')
                self.write_measurements(cost_at_predict, spectrum, lamb_max)

                if self.target_style == 'single_point':
                    self.logger.info(f"Absorbance at {self.tg_wl} nm: {spectrum.get_abs_at_wl(self.tg_wl)}")
                elif self.target_style == 'lambda_max':
                    self.logger.info(f'Maximum absorbance at {spectrum.wavelengths[spectrum.absorbance.argmax()]} nm.')

                # Update database
                flows = np.concatenate((flows, [performed_flows]))
                costs = np.concatenate((costs, [cost_at_predict]))
                self._note_growth_time()  # align growth_times with costs (MO)

            # Only append to best_min if not starting from previous data (already initialized)
            if i >= starting_iter or starting_iter == 0:
                best_min.append(costs.min())

            tmp = time.time()

            # Update the GPR
            variant_flows = flows[:, self.varying_pumps]  # Dimensionality reduction
            flows_minus_one = variant_flows[:, :-1]

            # Normalization
            y_acq = self._fit_gp(gaussian_process, flows_minus_one, costs, base_alpha)

            # Diagnose GP convergence (only log detailed diagnostics every 3 iterations to avoid spam)
            if i % 3 == 0 or i == self.maxiter:
                self.diagnose_gp_convergence(gaussian_process, flows_minus_one, costs)

            # Prediction of the next flow to evaluate
            predicted_flows = self.predict_next(y_acq, gaussian_process, iteration=i)

            times.append(time.time() - tmp)

            # Plot the figure via callbacks
            self.callbacks.on_iteration_complete(
                i, flows, self._display_costs(costs), gaussian_process, best_min, times,
                spectrum, predicted_flows, False, save_figs,
            )

            # If tolerance is reached, stops loop
            if best_min[-1] < self.tol:
                self.success = True
                if self.tol_stop:
                    break

        self.logger.info("------------")
        self.logger.info(f'final figure ({i})')
        self.logger.info("------------")

        # Optimal results
        self.min_cost = costs.min()
        self.opt_flows = flows[costs.argmin()]
        if costs.argmin() >= self.nb_init_samples:
            self.best_iter = costs.argmin() - self.nb_init_samples + 1
            best_spectrum = self.spectra[f'iter {self.best_iter}']
        else:
            self.best_iter = 0
            best_spectrum = self.spectra[f'init {costs.argmin() + 1}']

        self.callbacks.on_iteration_complete(
            i, flows, self._display_costs(costs), gaussian_process, best_min, times,
            best_spectrum, None, True, True,
        )

        self.logger.info(f"Optimal flows = {self.opt_flows}")
        if self.target_style == 'single_point':
            self.logger.info(f"Absorbance at {self.tg_wl} nm: {best_spectrum.get_abs_at_wl(self.tg_wl)}")
        elif self.target_style == 'lambda_max':
            self.logger.info(f'Maximum absorbance at {best_spectrum.absorbance.argmax()} nm.')
        self.logger.info(f"Minimum measured cost: {self.min_cost}")

        self.write_params(gaussian_process, end=True)

        # Load-previous is one-shot: clear so the next experiment starts fresh
        # unless the user explicitly clicks "Load Previous" again.
        self.previous_flows = None
        self.previous_costs = None
        self._n_warmstart = 0

        self.callbacks.on_experiment_complete('Success' if self.success else 'Maxiter reached')
        return flows, costs, gaussian_process

    def load_prior_kernel(self, path, freeze=False):
        """Load an offline-tuned kernel to use as a hyperparameter prior.

        The kernel (e.g. produced by notebooks/kernel_benchmark_pca.ipynb by
        fitting a GP on historical data) carries learned lengthscales + noise.
        Used by _rebuild_kernel as the live kernel: the historical data informs
        the function's SHAPE (smoothness/noise), not where the optimum is.

        Parameters
        ----------
        path : str
            Pickle file containing a fitted sklearn kernel.
        freeze : bool
            If False (default), the offline hyperparameters are only a WARM
            START: the live GP keeps re-optimising them as points arrive (i.e.
            just a hyperparameter optimisation seeded from history). If True,
            freeze them (bounds -> 'fixed') so the prior regime is kept fixed.
        """
        with open(path, "rb") as f:
            self.prior_kernel = pickle.load(f)
        self.prior_kernel_freeze = bool(freeze)
        self.logger.info("Loaded prior kernel from %s (freeze=%s): %s",
                         path, freeze, self.prior_kernel)
        return self.prior_kernel

    def _rebuild_kernel(self):
        """Rebuild the GP kernel from current override settings."""
        preset = getattr(self, "bo_preset_override", "default") or "default"
        n_dims = max(1, int(np.sum(self.varying_pumps)) - 1) if hasattr(self, 'varying_pumps') else 1

        # Hyperparameter prior takes precedence: use the offline-tuned kernel.
        if getattr(self, "prior_kernel", None) is not None:
            if _prior_kernel_matches(self.prior_kernel, n_dims):
                self.kernel = (_freeze_kernel(self.prior_kernel)
                               if getattr(self, "prior_kernel_freeze", False)
                               else clone(self.prior_kernel))
                self.logger.info(
                    "Using PRIOR kernel (hyperparameter prior, freeze=%s): %s",
                    getattr(self, "prior_kernel_freeze", False), self.kernel)
                return
            self.logger.warning(
                "Prior kernel dim mismatch (live n_dims=%d) — ignoring it and "
                "rebuilding the default kernel.", n_dims)

        if preset == "robust":
            # Robust preset is atomic: ignore UI length_scale / noise overrides.
            # Matern 2.5 ISOTROPIC: ARD's per-dimension length scales came out
            # ~equal on well-fit data (mechanically redundant) and fitted noise
            # on poor data — isotropic is the defensible choice.
            self.kernel = (
                Matern(length_scale=max(1.0, self.length_scale_min),
                       length_scale_bounds=(self.length_scale_min, 10.0),
                       nu=2.5)
                + WhiteKernel(noise_level=1e-3,
                              noise_level_bounds=(self.white_noise_floor,
                                                   self.white_noise_ceiling))
            )
            self.logger.info(
                "Robust BO preset: Matern 2.5 isotropic kernel "
                "(length_scale_bounds=(%.3g, 10.0), white noise initial 1e-3). "
                "UI Length scale / Noise level entries are ignored.",
                self.length_scale_min,
            )
            return

        # Default preset: honour UI overrides
        ls = self.rbf_length_scale_override or 1.0
        noise = self.white_noise_override or 1e-3
        kernel_type = self.kernel_type_override or "RBF"

        if kernel_type == "ARD":
            ls = np.ones(n_dims) * ls
            self.logger.warning(
                "ARD kernel: per-dimension length scales (n_dims=%d). "
                "Requires sufficient data to avoid overfitting.", n_dims,
            )
        # Keep the initial length scale within the (length_scale_min, 1e5) bounds.
        ls = np.maximum(ls, self.length_scale_min)

        self.kernel = (
            RBF(length_scale=ls, length_scale_bounds=(self.length_scale_min, 1e5))
            + WhiteKernel(noise_level=noise,
                          noise_level_bounds=(self.white_noise_floor,
                                              self.white_noise_ceiling))
        )

    def np_growth_done(self, sample_id=""):
        """Check if nanoparticle growth has stabilized using correlation and amplitude criteria.

        The historical ``min_spectra`` gate can be bypassed: when
        ``termination.allow_early_stop`` is set, growth may be declared done
        before ``min_spectra`` measurements provided the convergence criteria
        have held for ``early_stop_confirmations`` consecutive checks (and we
        have at least ``minimum_line`` rows to compute the correlation). This
        avoids wasting time/reagent on syntheses that finish fast.
        """
        if self.app.next_sample_pressed:
            self.app.next_sample_pressed = False
            self._reset_growth_streak(sample_id)
            return True

        tc = self.termination
        try:
            with open(os.path.join(self.experiment_path, f"measurement_spectra_{sample_id}.csv"), mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                rows = list(reader)

            sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
            assessment = assess_growth(
                rows, sampling_rate,
                correlation_threshold=tc.correlation_threshold,
                amplitude_threshold=tc.amplitude_threshold,
                min_spectra=tc.min_spectra,
                savgol_window=tc.savgol_window,
                savgol_polyorder=tc.savgol_polyorder,
            )

            if not assessment.computable:
                return False

            # Track consecutive criteria-met checks for the early-stop bypass.
            streak = self._update_growth_streak(sample_id, assessment.criteria_met)
            allow_early = getattr(tc, "allow_early_stop", False)
            confirmations = getattr(tc, "early_stop_confirmations", 2)

            corr_ok = assessment.correlation > tc.correlation_threshold
            amp_ok = assessment.amplitude_change < tc.amplitude_threshold
            early_note = ""
            if allow_early and not assessment.enough_spectra and assessment.criteria_met:
                early_note = f" | early-stop streak {streak}/{confirmations}"
            self.logger.info(
                f"Termination check — Pearson r: {assessment.correlation:.6f} "
                f"(threshold: {tc.correlation_threshold}) {'OK' if corr_ok else 'not met'} | "
                f"Ext max diff: {assessment.amplitude_change:.4%} "
                f"(threshold: {tc.amplitude_threshold:.2%}) {'OK' if amp_ok else 'not met'} | "
                f"Spectra: {assessment.line_count}/{tc.min_spectra}{early_note}")
            print(
                f"  >> Pearson r = {assessment.correlation:.6f} "
                f"({'OK' if corr_ok else 'waiting'})  |  "
                f"Ext max diff = {assessment.amplitude_change:.4%} "
                f"({'OK' if amp_ok else 'waiting'})  |  "
                f"Spectra: {assessment.line_count}/{tc.min_spectra}{early_note}")

            done = should_terminate(assessment, allow_early, streak, confirmations)
            if done:
                self._record_growth_time(sample_id, rows)
            return done

        except Exception as e:
            self.logger.error(f"Error in np_growth_done: {e}")
            return False

    def _update_growth_streak(self, sample_id: str, criteria_met: bool) -> int:
        """Increment (or reset) the consecutive criteria-met streak for a sample."""
        if getattr(self, "_growth_streak_sample", None) != sample_id:
            self._growth_streak_sample = sample_id
            self._growth_streak = 0
        self._growth_streak = self._growth_streak + 1 if criteria_met else 0
        return self._growth_streak

    def _reset_growth_streak(self, sample_id: str) -> None:
        self._growth_streak_sample = sample_id
        self._growth_streak = 0

    def _record_growth_time(self, sample_id: str, rows) -> None:
        """Append the measured growth time (s) for this sample to growth_times.csv.

        Makes growth time available as a second BO objective (see ParEGO /
        toofast.core.multi_objective). Best-effort: never raises.
        """
        try:
            seconds = growth_time_from_rows(rows)
            if seconds is None:
                return
            # Stash for the multi-objective loop (read by _note_growth_time).
            self._last_growth_time = float(seconds)
            path = os.path.join(self.experiment_path, "growth_times.csv")
            with open(path, mode='a', newline='') as f:
                csv.writer(f, delimiter=';').writerow([sample_id, f"{seconds:.3f}"])
            self.logger.info(f"Sample {sample_id}: growth time = {seconds:.1f}s")
        except Exception as e:
            self.logger.warning(f"Could not record growth time: {e}")

    def wait_for_steady_state(self):
        """
        Wait for 2.5 times the residence time to ensure steady state is reached. Returns True if the experiment is
        stopped during the waiting time.
        """
        start_time = time.time()
        while time.time() - start_time < 2.5 * self.residence_time:
            if self._abort_event.is_set():
                return 'stop'
            alert = self.check_if_stopped()
            if alert:
                return alert
            time.sleep(0.5)
        return False

    def check_if_stopped(self):
        if self.callbacks.should_stop():
            self.callbacks.on_experiment_complete('Experiment externally stopped.')
            return 'stop'
        elif self.callbacks.should_pause():
            return 'pause'
        else:
            return False

    def pause(self):
        self.logger.info('Entered pause mode.')
        self.callbacks.set_phase("Paused")
        while self.callbacks.should_pause():
            time.sleep(0.5)
        if not self.callbacks.should_stop():
            self.logger.info('Exit pause mode : restarting current measurement.')

    def write_measurements(self, cost, spectrum, lamb_max):
        with open(os.path.join(self.experiment_path, "costs.csv"), mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([cost])
        if self.target_style == 'single_point':
            file_name = f'absorbance_at_{self.tg_wl}_nm'
            value = spectrum.get_abs_at_wl(self.tg_wl)
        elif self.target_style == 'lambda_max':
            file_name = 'lambda_max'
            value = lamb_max
        else:
            return

        with open(os.path.join(self.experiment_path, file_name + '.csv'), mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([value])

    def measure_reference(self, overwrite=False):
        """
        Generate a water flow with maximum flow
        Stops all other pumps
        Measure a reference spectrum and write it in a reference file
        Stops all the pump after measurement

        Returns the reference spectrum
        ------- 
        """
        self.logger.info("Measuring new reference spectrum")

        reference = self.spectro.measure_reference()
        self._gui(self.app.plot_spectrum, reference)
        reference.save_data(self.app.experiment_path)
        self.logger.info('Reference measured.')
        # self.pump_syst.stop_all_pumps()

    def measure_spectrum(
        self,
        sample_id: str = "",
        meas_type: str = 'measurement'
    ) -> Spectrum:
        """
        Acquire a UV-Vis spectrum and save it to disk.

        Parameters
        ----------
        sample_id : str, optional
            Identifier for the sample being measured.
        meas_type : str, optional
            Type of measurement ('measurement', 'reference', 'target').

        Returns
        -------
        Spectrum
            The acquired spectrum object containing wavelengths and absorbance.
        """
        self.collector.open_lamp_shutter()
        spectrum = self.spectro.measure_spectrum(meas_type=meas_type)
        self._gui(self.app.plot_spectrum, spectrum)
        spectrum.save_data(self.experiment_path, sample_id)
        self.logger.info("Spectrum saved")
        self.collector.close_lamp_shutter()
        return spectrum

    def smart_plot_spectrum(self, sample_id: str = "") -> Spectrum:
        """
        Acquire spectrum with smart averaging for noise reduction.

        Parameters
        ----------
        sample_id : str, optional
            Identifier for the sample being measured.

        Returns
        -------
        Spectrum
            The acquired spectrum with smart averaging applied.
        """
        smart_spectrum = self.spectro.smart_measure()
        self._gui(self.app.plot_spectrum, smart_spectrum)
        return smart_spectrum

    def exploratory_synthesis(self, number_of_samples=4, sample_volume=400, previous_samples=None):
        self.pump_syst.prime_all_pumps()
        for i in range(number_of_samples):
            rd_flows = self.pump_syst.generate_random_flows()
            self.pump_syst.set_flows(rd_flows)
            self.logger.info(self.pump_syst.get_flows())
            self.pump_syst.write_flows(self.experiment_path)
            self.logger.info("wait for steady state")
            self.wait_for_steady_state()
            self.logger.info("ok to collect")
            self.collection_status = True
            start_time = time.time()
            while self.collection_status:
                current_time = time.time()
                elapsed_time = current_time - start_time
                self.logger.info(elapsed_time)
                if elapsed_time >= 50:
                    self.collection_status = False
                elif elapsed_time % 10 == 0:  # Measure a spectrum every 2 seconds
                    self.measure_spectrum()
            self.logger.info(f"sample {i} complete")
        self.logger.info("experiment done")
        # if volume ok, break, sinon continue
        # pump acide qq seconde
        # purge
        # collection status = false

    def _is_gp_fit_reliable(self, gaussian_process):
        """Quick check: is the GP fit reliable enough for acquisition optimization?

        Returns False when there are too few samples or when the RBF length
        scale has hit its upper bound (sign of underfitting with sparse data).
        """
        try:
            n_samples = len(gaussian_process.y_train_)
            n_features = gaussian_process.X_train_.shape[1]
            if n_samples < n_features + 2:
                self.logger.info(
                    f"Only {n_samples} samples for {n_features}D — "
                    "insufficient for reliable GP"
                )
                return False

            kernel = gaussian_process.kernel_
            params = kernel.get_params()
            for key, value in params.items():
                if "length_scale" in key and "bounds" not in key:
                    ls = np.atleast_1d(value)
                    bounds_key = key + "_bounds"
                    if bounds_key in params and params[bounds_key] != "fixed":
                        lo, hi = params[bounds_key]
                        if np.any(ls >= hi * 0.9):
                            self.logger.info(
                                f"Length scale {ls} near upper bound {hi:.0f} "
                                "— GP likely underfitting"
                            )
                            return False
            return True
        except Exception:
            return True

    def diagnose_gp_convergence(self, gaussian_process, flows, costs):
        """
        Diagnose GP convergence and parameter quality after fitting.

        Delegates to toofast.core.diagnostics.GPDiagnostics module.

        Parameters
        ----------
        gaussian_process : GaussianProcessRegressor
            Fitted GP model
        flows : array-like
            Training flows
        costs : array-like
            Training costs

        Returns
        -------
        DiagnosticReport : Diagnostic information with converged, warnings, suggestions
        """
        # The GP is fit in GP-target space (warped if cost_transform == 'log'),
        # so residuals must be computed against the warped targets; the 4th arg
        # (raw cost history) stays in raw units for range/scale reporting.
        report = self._gp_diagnostics.diagnose(
            gaussian_process, flows, self._cost_to_gp(np.asarray(costs)), costs
        )

        # Log report using the experiment logger
        lml = gaussian_process.log_marginal_likelihood_value_
        self.logger.info("=" * 60)
        self.logger.info("GP CONVERGENCE DIAGNOSTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Log Marginal Likelihood: {lml:.2f}")
        self.logger.info(f"Converged: {report.is_converged}")

        if report.rbf_length_scale is not None:
            self.logger.info(f"Length scale: {report.rbf_length_scale}")
        if report.noise_level is not None:
            self.logger.info(f"Noise level: {report.noise_level:.4f}")

        if report.warnings:
            for warn in report.warnings:
                self.logger.warning(f"  - {warn}")
        if report.suggestions:
            self.logger.info("Suggestions:")
            for sug in report.suggestions:
                self.logger.info(f"  - {sug}")

        if report.is_converged and not report.warnings:
            self.logger.info("GP converged successfully with no issues detected")
        self.logger.info("=" * 60)

        # Return legacy-compatible dict
        kernel_params = {}
        if report.rbf_length_scale is not None:
            kernel_params['length_scale'] = report.rbf_length_scale
        if report.noise_level is not None:
            kernel_params['noise_level'] = report.noise_level

        return {
            'converged': report.is_converged,
            'warnings': report.warnings,
            'suggestions': report.suggestions,
            'kernel_params': kernel_params,
        }

    # ------------------------------------------------------------------
    # Machine-readable config save / load for load_previous
    # ------------------------------------------------------------------
    def _save_config_json(self):
        """Save all experiment parameters as JSON for reliable reload."""
        config = {
            # BO parameters
            'maxiter': self.maxiter,
            'acq_func': self.acq_func,
            'nb_init_samples': self.nb_init_samples,
            'init_option': self.init_option,
            'de_tol': self.de_tol,
            'de_maxiter': self.de_maxiter,
            # Target
            'target_style': self.target_style,
            'tg_wl': self.tg_wl,
            'tg_abs': self.tg_abs,
            'tol': self.tol,
            'rel_tol': self.rel_tol,
            'tol_stop': self.tol_stop,
            'cost_method': getattr(self, 'cost_method', 'L2'),
            'threshold': self.threshold,
            # Timing
            'residence_time': self.residence_time,
            # Pump constraints
            'bounds': self.bounds.tolist() if self.bounds is not None else None,
            'fixed_pumps': self.fixed_pumps.tolist() if self.fixed_pumps is not None else None,
            'fixed_flows': [float(f) if f is not None else None for f in self.fixed_flows],
            'reagents': list(self.reagents) if self.reagents is not None else None,
            'total_flow': self.total_flow,
            'number_of_pumps': self.number_of_pumps,
            # GP kernel overrides
            'kernel_type_override': self.kernel_type_override,
            'rbf_length_scale_override': self.rbf_length_scale_override,
            'white_noise_override': self.white_noise_override,
            'white_noise_floor': self.white_noise_floor,
            'white_noise_ceiling': self.white_noise_ceiling,
            'length_scale_min': self.length_scale_min,
            'cost_transform': self.cost_transform,
            'pca_cost_metric': self.pca_cost_metric,
            'pca_n_components': self.pca_n_components,
            'gp_n_restarts_override': self.gp_n_restarts_override,
            'bo_preset_override': getattr(self, 'bo_preset_override', 'default'),
            'huber_delta_nm_override': getattr(self, 'huber_delta_nm_override', 30.0),
            'flat_strategy_override': getattr(self, 'flat_strategy_override', 'nearest_high'),
            'flat_penalty_factor_override': getattr(self, 'flat_penalty_factor_override', 2.0),
            'flat_nearest_margin_override': getattr(self, 'flat_nearest_margin_override', 1.25),
            'flat_nearest_cap_override': getattr(self, 'flat_nearest_cap_override', 3.0),
            'feasibility_aware_override': getattr(self, 'feasibility_aware_override', False),
            'use_physics_ghosts': getattr(self, 'use_physics_ghosts', True),
            'ghost_noise_level': getattr(self, 'ghost_noise_level', 'auto'),
            'ghost_penalty_lift': getattr(self, 'ghost_penalty_lift', 1.5),
            # Spectrometer
            'spectro_integration_time': getattr(self.spectro, 'integration_time', None),
            'spectro_number_average': getattr(self.spectro, 'number_average', None),
            'spectro_smoothing': getattr(self.spectro, 'smoothing', None),
            # Wash / flush (read from GUI if available)
            'wash_acid_volume': self._gui_get_safe('nitric_acid_volume_entry'),
            'wash_water_volume': self._gui_get_safe('water_volume_entry'),
            'wash_backpressure_flow': self._gui_get_safe('back_pressure_entry'),
            'spectro_sampling_rate': self._gui_get_safe('spectro_sampling_rate_entry'),
            # Termination
            'termination_correlation': self.termination.correlation_threshold,
            'termination_amplitude': self.termination.amplitude_threshold,
            'termination_min_spectra': self.termination.min_spectra,
            'termination_timeout': self.termination.monitoring_timeout_s,
        }
        path = os.path.join(self.experiment_path, '_config.json')
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"Config JSON saved: {path}")
        except Exception as e:
            self.logger.warning(f"Could not save config JSON: {e}")

    def _gui_get_safe(self, attr_name):
        """Read a GUI entry value without crashing if the GUI is absent."""
        try:
            widget = getattr(self.app, attr_name, None)
            if widget is not None:
                return float(self._gui_get(widget))
        except Exception:
            pass
        return None

    @staticmethod
    def _load_config_json(previous_exp_path):
        """Load config JSON from a previous experiment directory."""
        path = os.path.join(previous_exp_path, '_config.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _apply_loaded_config(self, cfg):
        """Apply parameters from a loaded config JSON to this experiment."""
        self.maxiter = cfg.get('maxiter', self.maxiter)
        self.acq_func = cfg.get('acq_func', self.acq_func)
        self.nb_init_samples = cfg.get('nb_init_samples', self.nb_init_samples)
        self.init_option = cfg.get('init_option', self.init_option)
        self.de_tol = cfg.get('de_tol', self.de_tol)
        self.de_maxiter = cfg.get('de_maxiter', self.de_maxiter)

        self.target_style = cfg.get('target_style', self.target_style)
        self.tg_wl = cfg.get('tg_wl', self.tg_wl)
        self.tg_abs = cfg.get('tg_abs', self.tg_abs)
        self.tol = cfg.get('tol', self.tol)
        self.rel_tol = cfg.get('rel_tol', self.rel_tol)
        self.tol_stop = cfg.get('tol_stop', self.tol_stop)
        self.cost_method = cfg.get('cost_method', getattr(self, 'cost_method', 'L2'))
        self.threshold = cfg.get('threshold', self.threshold)

        self.residence_time = cfg.get('residence_time', self.residence_time)

        if cfg.get('total_flow') is not None:
            self.total_flow = cfg['total_flow']
        if cfg.get('bounds') is not None:
            self.bounds = np.array(cfg['bounds'])
        if cfg.get('fixed_pumps') is not None:
            self.fixed_pumps = np.array(cfg['fixed_pumps'])
            self.varying_pumps = ~self.fixed_pumps
            self.nb_of_var_pumps = self.varying_pumps.sum()
        if cfg.get('fixed_flows') is not None:
            self.fixed_flows = np.array([
                f if f is not None else 0.0 for f in cfg['fixed_flows']
            ])
        self.total_var_flow = self.total_flow - sum(
            self.fixed_flows[self.fixed_pumps]
        )
        if cfg.get('reagents') is not None:
            self.reagents = np.array(cfg['reagents'])

        self.kernel_type_override = cfg.get('kernel_type_override', self.kernel_type_override)
        self.rbf_length_scale_override = cfg.get('rbf_length_scale_override', self.rbf_length_scale_override)
        self.white_noise_override = cfg.get('white_noise_override', self.white_noise_override)
        self.white_noise_floor = cfg.get('white_noise_floor', self.white_noise_floor)
        self.white_noise_ceiling = cfg.get('white_noise_ceiling', self.white_noise_ceiling)
        self.length_scale_min = cfg.get('length_scale_min', self.length_scale_min)
        self.cost_transform = cfg.get('cost_transform', self.cost_transform)
        self.pca_cost_metric = cfg.get('pca_cost_metric', self.pca_cost_metric)
        self.pca_n_components = cfg.get('pca_n_components', self.pca_n_components)
        self.gp_n_restarts_override = cfg.get('gp_n_restarts_override', self.gp_n_restarts_override)
        self.bo_preset_override = cfg.get('bo_preset_override', getattr(self, 'bo_preset_override', 'default'))
        self.huber_delta_nm_override = cfg.get('huber_delta_nm_override', getattr(self, 'huber_delta_nm_override', 30.0))
        self.flat_strategy_override = cfg.get('flat_strategy_override', getattr(self, 'flat_strategy_override', 'nearest_high'))
        self.flat_penalty_factor_override = cfg.get('flat_penalty_factor_override', getattr(self, 'flat_penalty_factor_override', 2.0))
        self.flat_nearest_margin_override = cfg.get('flat_nearest_margin_override', getattr(self, 'flat_nearest_margin_override', 1.25))
        self.flat_nearest_cap_override = cfg.get('flat_nearest_cap_override', getattr(self, 'flat_nearest_cap_override', 3.0))
        self.feasibility_aware_override = cfg.get('feasibility_aware_override', getattr(self, 'feasibility_aware_override', False))
        self.use_physics_ghosts = bool(cfg.get('use_physics_ghosts', getattr(self, 'use_physics_ghosts', True)))
        self.ghost_noise_level = cfg.get('ghost_noise_level', getattr(self, 'ghost_noise_level', 'auto'))
        self.ghost_penalty_lift = cfg.get('ghost_penalty_lift', getattr(self, 'ghost_penalty_lift', 1.5))
        self._rebuild_kernel()

        tc = self.termination
        tc.correlation_threshold = cfg.get('termination_correlation', tc.correlation_threshold)
        tc.amplitude_threshold = cfg.get('termination_amplitude', tc.amplitude_threshold)
        tc.min_spectra = cfg.get('termination_min_spectra', tc.min_spectra)
        tc.monitoring_timeout_s = cfg.get('termination_timeout', tc.monitoring_timeout_s)

        self.logger.info("Experiment parameters restored from previous config")

    def write_params(self, gaussian_process, end=False):
        """
        Writes a .txt file containing the parameters of the experiment.

        Parameters
        ----------
        gaussian_process : sk-learn GaussianProcessRegressor
            The GPR.
        end : bool, optional
            If set to True, writes the optimal measured points' characteristics.
            The default is False.

        """
        parameters_file_path = os.path.join(self.experiment_path,
                                            f"{self.experiment_name}_parameters.txt")
        with open(parameters_file_path, 'w') as file:
            file.write(f'{self.experiment_name}:   {datetime.datetime.now().isoformat()}\n\n')

            # Add information about loaded previous experiment if applicable
            if self.previous_flows is not None and self.previous_costs is not None:
                file.write('=' * 60 + '\n')
                file.write('WARM START FROM PREVIOUS EXPERIMENT\n')
                file.write('=' * 60 + '\n')
                file.write(f'Loaded {len(self.previous_costs)} data points from previous experiment\n')
                file.write(f'Previous best cost: {self.previous_costs.min():.6f}\n')

                # Check if metadata file exists
                metadata_file = os.path.join(self.experiment_path, "previous_experiment_info.txt")
                if os.path.exists(metadata_file):
                    file.write(f'See "previous_experiment_info.txt" for complete details\n')
                file.write('\n')

            file.write(f'Features: {[pump.dye for pump in self.pump_syst.pumps]}\n')
            file.write(f"Total Flow: {self.total_flow}\n")
            file.write(f'Bounds to the pumps: {self.bounds}\n\n')

            file.write(f"Number of maximum iterations: {self.maxiter} iterations.\n")
            file.write(f'Optimization strategy: {self.target_style}.\n')
            if self.target_style == 'lambda_max':
                file.write(f'Lambda_max target at {self.tg_wl} nm.\n')
            elif self.target_style == 'single_point':
                file.write(f'Target absorbance of {self.tg_abs} at {self.tg_wl} nm.\n')
            elif self.target_style == 'full_spectrum':
                file.write(f'Full spectrum target measured at {self.tg_init_flows}.\n')
            elif self.target_style == 'pca':
                file.write('PCA target (score-space distance).\n')
                if getattr(self, 'pca_model_file', None):
                    file.write(f'PCA decomposition file: {self.pca_model_file}\n')
                file.write(f'PCA cost metric: {self.pca_cost_metric} | '
                           f'n_components: {self.pca_n_components or "all"}\n')
                file.write(f'PCA variance-weighted: {self.pca_variance_weighted}\n')
                if getattr(self, 'pca_lambda_weight', 0.0):
                    file.write(f'Hybrid lambda_max term: weight={self.pca_lambda_weight}, '
                               f'scale={self.pca_lambda_scale_nm} nm\n')
                if getattr(self, 'pca_amp_weight', 0.0):
                    file.write(f'Hybrid amplitude term: weight={self.pca_amp_weight} '
                               f'(relative to target peak absorbance)\n')
            # Target spectrum (full_spectrum / pca): saved alongside as a CSV.
            if getattr(self, 'tg_spectrum', None) is not None:
                wl = np.asarray(self.tg_spectrum.wavelengths, dtype=float)
                ab = np.asarray(self.tg_spectrum.absorbance, dtype=float)
                src = (os.path.basename(self.target_spectrum_file)
                       if getattr(self, 'target_spectrum_file', None) else 'in-memory')
                file.write(
                    f'Target spectrum: {len(wl)} points, {wl[0]:.1f}-{wl[-1]:.1f} nm, '
                    f'peak abs {np.max(ab):.4f} at {wl[int(np.argmax(ab))]:.1f} nm '
                    f'(source: {src}; saved as target_spectrum.csv).\n')
            file.write(f'Tolerence: {self.tol}\n\n')

            file.write(f'Initiation option: {self.init_option} ({self.nb_init_samples} init samples).\n')
            file.write('Initialisation points:\n')
            file.write(f'{self.initial_flows}')

            file.write('\n\nGPR parameters:\n')
            file.write(f'Acquisition function: {self.acq_func}.\n')
            file.write(str(gaussian_process.get_params()) + '\n\n')
            file.write('Differential evolution parameters:\nMaxiter: '
                       f'{self.de_maxiter}\nTolerance: {self.de_tol}\n')
            file.write('\nSpectrometer parameters:\n')
            file.write(f'Integration time = {self.spectro.integration_time} ms\n')
            file.write(f'Average = {self.spectro.number_average} measurements\n')

            if end:
                file.write('\n----------------\n')
                file.write('Results: ' + ('Success' if self.success else 'Failure') + '\n\n')
                file.write(f"Optimal flows: {self.opt_flows}\n")
                file.write(f"Minimal cost: {self.min_cost}\n")
                file.write(f'Obtained after {self.best_iter} iterations.\n')
                file.write(f'\nLast kernel parameters : {gaussian_process.kernel_}\n')
                file.write("\nErrors during the GP:\n")
                file.write(f"Predicted negative variance: {self.neg_var}.")

        self._save_config_json()
