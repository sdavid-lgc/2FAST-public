# -*- coding: utf-8 -*-
"""
2FAST - Fully Automated Synthesis Technology

Core experiment control module for automated nanoparticle synthesis exploration
using Bayesian Optimization with Gaussian Process Regression.

Groups of methods live in mixins of the same package: flat_strategy (flat spectra, ghost points), warm_start,
next_point (acquisition + differential evolution), wash, run_config (_config.json, parameters file), gp_model (kernel,
prior, diagnostics).

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

from toofast import clock
from toofast.hardware.factory import create_c3000, create_hardware, create_mfc
from toofast.hardware.spectrometer import Spectrum
from toofast.acquisition import (
    ExpectedImprovement, ExplorationWrapper, LowerConfidenceBound, ThompsonSampling,
    FeasibilityWeightedAcquisition, fit_feasibility_classifier,
)
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
from toofast.core.flat_strategy import FlatStrategyMixin
from toofast.core.warm_start import WarmStartMixin
from toofast.core.next_point import NextPointMixin
from toofast.core.wash import WashMixin
from toofast.core.run_config import RunConfigMixin
from toofast.core.gp_model import GPModelMixin
from toofast.core.gp_model import _freeze_kernel, _prior_kernel_matches  # noqa: F401

np.set_printoptions(precision=4, suppress=True)

logger = logging.getLogger(__name__)


class Experiment(FlatStrategyMixin, WarmStartMixin, NextPointMixin, WashMixin, RunConfigMixin, GPModelMixin):
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
        pump_driver: Optional[str] = None,
        spectrometer_driver: Optional[str] = None,
        rig=None,
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
        pump_driver, spectrometer_driver : str, optional
            Laboratory drivers of the vendor instruments ("package.module:Class", see toofast.hardware.drivers);
            default: the TOOFAST_PUMP_DRIVER / TOOFAST_SPECTROMETER_DRIVER environment variables. Ignored in
            simulation.
        rig : toofast.hardware.simulated.SimulatedRig, optional
            Simulation only: the simulated instruments and their virtual reactor (a default one is created).
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

        self.total_flow = total_flow
        self.number_of_pumps = number_of_pumps
        hardware = create_hardware(self.logger, number_of_pumps, inner_diameter_mm, max_piston_stroke_mm,
                                   total_flow, is_simulation=is_simulation, collector_port=self.collector_port,
                                   config_file=config_file, pump_driver=pump_driver,
                                   spectrometer_driver=spectrometer_driver, rig=rig)
        self.rig = hardware.rig
        self.spectro = hardware.spectro
        self.pump_syst = hardware.pump_syst
        self.collector = hardware.collector
        self.spectra = {}
        self.tg_spectrum = None
        self.success = False
        self.neg_var = False

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
        self.mfc = create_mfc(port, self.logger, is_simulation=self.is_simulation, rig=self.rig)
        self.mfc.set_flow(0.0)

    def connect_c3000(self, port: str = ''):
        """Connect C3000 syringe pumps."""
        port = port or self.c3000_port
        self.c3000_1 = create_c3000(port, 1, self.logger, is_simulation=self.is_simulation, rig=self.rig)
        self.c3000_2 = create_c3000(port, 2, self.logger, is_simulation=self.is_simulation, rig=self.rig)


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
            case "latinhypercube":          # Latin hypercube on the simplex of compositions
                self.initial_flows = self.pump_syst.generate_lhs_flows_sets(
                    self.nb_init_samples, self.bounds, self.total_var_flow,
                    self.fixed_pumps, self.fixed_flows
                )
            case "projected_lhs":           # toofast 1.0 "latinhypercube" (the paper's runs)
                self.initial_flows = self.pump_syst.generate_projected_lhs_flows_sets(
                    self.nb_init_samples, self.bounds, self.total_var_flow,
                    self.fixed_pumps, self.fixed_flows
                )
            case "fixed":
                pass
            case _:
                raise ValueError("Invalid initialization_option. Please use 'random', 'latinhypercube', "
                                 "'projected_lhs' or 'fixed'.")

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
    def _to_main_thread(self, func, *args):
        """Hand `func(*args)` to the GUI's main thread: its thread-safe `call_soon` queue (Tk must not be called
        from worker threads); `after(0, ...)` only for an app without one."""
        call_soon = getattr(self.app, "call_soon", None)
        if call_soon is not None:
            call_soon(func, *args)
        else:
            self.app.after(0, func, *args)

    def _gui(self, func, *args):
        """Schedule a fire-and-forget GUI call on the Tk main thread."""
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            self._to_main_thread(func, *args)

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

        self._to_main_thread(_wrapper)
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
        self._current_exp_type = self._gui_get(self.app.exp_type_var)  # 'Random', 'LHS Flows', 'Projected LHS Flows'
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
            self.logger.info("Experiment started -- LHS flows (Latin hypercube on the simplex)")
        elif exp_type == 'Projected LHS Flows':
            flow_sets = self.pump_syst.generate_projected_lhs_flows_sets(number_of_samples)
            self.logger.info("Experiment started -- projected LHS flows")
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
        timer = clock.time()
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
                if round(clock.time() - timer, 1) % sampling_rate == 0:
                    self._gui(self.app.measure_spectrum, sample_id)

            self._gui(self.app.check_sample_level)
            clock.sleep(0.1)

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
        clock.sleep(0.1)
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
            'measurement_type': meas_type,
        }

        # Read spectrometer parameters from the live instrument
        identity = getattr(self.spectro, 'identity', None) or {}
        spectro_params = {
            'spectrometer_model': identity.get('model'),
            'spectrometer_serial': identity.get('serial'),
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
            'termination_allow_early_stop': bool(getattr(tc, 'allow_early_stop', False)),
            'termination_early_stop_confirmations': int(getattr(tc, 'early_stop_confirmations', 0)),
            'termination_savgol_window': tc.savgol_window,
            'termination_savgol_polyorder': tc.savgol_polyorder,
        }
        # provenance: which software produced the file, and whether the instruments were simulated
        from toofast import __version__ as _toofast_version
        provenance = {'software': f"toofast {_toofast_version}", 'simulated': bool(self.is_simulation)}

        # Stock concentrations (mM) entered on the pump cards of the GUI
        stock_mM = []
        for i in range(self.number_of_pumps):
            try:
                stock_mM.append(float(self._gui_get(self.app.reagent_concentrations[i])))
            except (ValueError, TypeError, IndexError, AttributeError):
                stock_mM.append(float('nan'))

        exp_params = {
            'experiment_type': getattr(self, '_current_exp_type', 'Stop flow'),
            'reagents': self.reagents,
            'stock_concentration': stock_mM,
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
            **provenance,
        }
        # Target spectrum (full_spectrum / pca): store the actual curve in the h5.
        if getattr(self, 'tg_spectrum', None) is not None:
            exp_params['target_spectrum_wavelengths'] = np.asarray(
                self.tg_spectrum.wavelengths, dtype=float)
            exp_params['target_spectrum_absorbance'] = np.asarray(
                self.tg_spectrum.absorbance, dtype=float)
        # Create file for time series
        self.app.data_manager.create_sample_file(sample_id, metadata, **exp_params)
        self._sample_growth_time = None    # set by _record_growth_time when the termination rule is met

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
        self.app.data_manager.finalize_sample(sample_id, performed_flows,
                                              growth_time_s=getattr(self, '_sample_growth_time', None))

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
            clock.sleep(1)
            self.dispense_acid_and_flush()
            clock.sleep(1) # why ?
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
        start_time = clock.time()
        sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
        next_sampling_time = clock.perf_counter() + sampling_rate

        self.logger.info(f"Starting time series collection for sample {sample_id}")
        self.logger.info(f"Duration: {duration}s, Interval: {sampling_rate}s")

        try:
            while (clock.time() - start_time) < duration:
                if self.app.abort_exp_flag:
                    self.logger.info("Time series collection aborted by user")
                    break
                # self.app.update()
                current_time = clock.perf_counter()
                # Mesurer un spectre

                if current_time >= next_sampling_time:
                    spectrum = self.measure_spectrum(sample_id=sample_id, meas_type='measurement')
                    self.app.data_manager.add_spectrum_to_sample(sample_id, spectrum)
                    # Keep in memory for compatibility
                    spectra_collected.append(spectrum)
                    next_sampling_time = current_time + sampling_rate  # Schedule next measurement

                    if self.np_growth_done(sample_id):
                        break

                clock.sleep(0.1)

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


    def prepare_flows(self, flows):
        """Prepare the system for stop-flow measurements."""
        standard_flows = [self.total_flow / self.number_of_pumps] * self.number_of_pumps  # Not sure about keeping this step
        self.pump_syst.set_flows(standard_flows)
        clock.sleep(5)
        self.pump_syst.set_flows(flows)
        self.pump_syst.write_flows(self.experiment_path)

    def monitor_growth_and_measure(self, sample_id):
        """Monitor growth and perform spectroscopy in stop-flow mode."""
        sampling_rate = float(self._gui_get(self.app.spectro_sampling_rate_entry))
        next_sampling_time = clock.perf_counter() + sampling_rate
        try:
            while True:
                # NOTE: self.app.update() removed — causes threading crash.
                # The Tk mainloop already runs event processing.
                current_time = clock.perf_counter()

                if current_time >= next_sampling_time:
                    spectrum = self.measure_spectrum(sample_id=sample_id)
                    next_sampling_time = current_time + sampling_rate  # schedule the next measurement

                    if self.np_growth_done(sample_id):
                        break

                clock.sleep(0.1)

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
        return self._run_bo(self._measure_iteration_continuous, base_alpha=1e-8, plot_figs=plot_figs,
                            save_figs=save_figs)

    def stop_flow(self, plot_figs=True, save_figs=True):
        """Bayesian optimisation in stop-flow mode (the platform's mode for the paper's campaigns).

        Same loop as `optimise` (see there), with the stop-flow measurement and the operator's validation of each
        cost (a rejected measurement is repeated), and a GP jitter alpha of 1e-3 instead of 1e-8.
        """
        self._current_exp_type = 'Bayesian Optimization'
        return self._run_bo(self._measure_iteration_stop_flow, base_alpha=1e-3, plot_figs=plot_figs,
                            save_figs=save_figs)

    def _measure_iteration_continuous(self, i, predicted_flows):
        """BO iteration `i`, continuous flow: (spectrum, performed flows, cost, lambda_max), None if aborted."""
        spectrum, performed_flows = self.make_flow_and_measure(predicted_flows)
        if spectrum is None:
            return None
        self.spectra[f'iter {i}'] = spectrum
        cost_at_predict, lamb_max = self.cost_function(spectrum)
        return spectrum, performed_flows, cost_at_predict, lamb_max

    def _measure_iteration_stop_flow(self, i, predicted_flows):
        """BO iteration `i`, stop flow, until the operator accepts the cost; None if aborted."""
        sample_id = f"iter_{i}"
        # Loop allowing to redo measurement if spectrum is not good
        while True:
            spectrum, performed_flows = self.stop_flow_and_measure(predicted_flows, sample_id=sample_id)
            if spectrum is None:
                return None
            self.spectra[f'iter {i}'] = spectrum
            cost_at_predict, lamb_max = self.cost_function(spectrum)
            if self.callbacks.validate_cost(cost_at_predict):
                return spectrum, performed_flows, cost_at_predict, lamb_max
            self.logger.info(f"iter_{i} is measured again.")

    def _run_bo(self, measure_iteration, base_alpha, plot_figs=True, save_figs=True):
        """The BO loop shared by `optimise` and `stop_flow`: GP (+ warm start from previous data), initial design
        (`init_points`, stop-flow), then `maxiter` iterations of acquisition -> `measure_iteration` -> refit."""
        self._rebuild_kernel()
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
                measured = measure_iteration(i, predicted_flows)
                if measured is None:          # measurement aborted
                    i -= 1
                    break
                spectrum, performed_flows, cost_at_predict, lamb_max = measured
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

            tmp = clock.time()

            # Update the GPR
            variant_flows = flows[:, self.varying_pumps]  # Dimensionality reduction
            flows_minus_one = variant_flows[:, :-1]

            y_acq = self._fit_gp(gaussian_process, flows_minus_one, costs, base_alpha)

            # Diagnose GP convergence (only log detailed diagnostics every 3 iterations to avoid spam)
            if i % 3 == 0 or i == self.maxiter:
                self.diagnose_gp_convergence(gaussian_process, flows_minus_one, costs)

            # Prediction of the next flow to evaluate
            predicted_flows = self.predict_next(y_acq, gaussian_process, iteration=i)

            times.append(clock.time() - tmp)

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
                clock.sleep(1)
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
            # NB: `rows` includes the CSV header line, so assessment.line_count = spectra + 1 and the
            # "line_count > min_spectra" gate means ">= min_spectra spectra" (the rule stated in the paper).
            # Kept as is: this is how the paper's data were acquired (replay of the 947 RG samples of raw/).
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
            if rows and rows[0] and rows[0][0] == "timestamp":
                rows = rows[1:]                    # header line of the measurement CSV (no timestamp in it)
            seconds = growth_time_from_rows(rows)
            if seconds is None:
                return
            # Stash for the multi-objective loop (read by _note_growth_time).
            self._last_growth_time = float(seconds)
            self._sample_growth_time = float(seconds)      # written in the sample file (result@growth_time_s)
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
        start_time = clock.time()
        while clock.time() - start_time < 2.5 * self.residence_time:
            if self._abort_event.is_set():
                return 'stop'
            alert = self.check_if_stopped()
            if alert:
                return alert
            clock.sleep(0.5)
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
            clock.sleep(0.5)
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
            start_time = clock.time()
            while self.collection_status:
                current_time = clock.time()
                elapsed_time = current_time - start_time
                self.logger.info(elapsed_time)
                if elapsed_time >= 50:
                    self.collection_status = False
                elif elapsed_time % 10 == 0:  # Measure a spectrum every 2 seconds
                    self.measure_spectrum()
            self.logger.info(f"sample {i} complete")
        self.logger.info("experiment done")
        # if volume ok, break, sinon continue
        # pump acid for a few seconds
        # purge
        # collection status = false


