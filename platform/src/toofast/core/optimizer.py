"""
Unified Bayesian Optimization loop.

Eliminates the ~170-line duplication between tooFAST.optimise() and
tooFAST.stop_flow() by parameterizing the measurement strategy.
Both methods follow the exact same structure:
  1. Optionally load previous data + pre-train GP
  2. For each iteration: measure -> compute cost -> fit GP -> predict next
  3. Report optimal results

The only difference was the measurement function called. Now that
difference is captured by the `measure_fn` callable parameter.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from toofast.acquisition import ExpectedImprovement, LowerConfidenceBound, ThompsonSampling
from toofast.config import OptimizationConfig
from toofast.core.cost import CostFunction
from toofast.core.diagnostics import GPDiagnostics
from toofast.data.models import ExperimentResult, Spectrum
from toofast.gui.callbacks import ExperimentCallbacks, HeadlessCallbacks

logger = logging.getLogger(__name__)


# Type alias for measurement functions
MeasureFn = Callable[[np.ndarray], Tuple[Optional[Spectrum], np.ndarray]]

# Type alias for init function
InitFn = Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Spectrum]]]


class BayesianOptimizer:
    """
    Unified Bayesian Optimization loop.

    Parameters
    ----------
    config : OptimizationConfig
        Optimization parameters.
    cost_fn : CostFunction
        Callable computing cost from a Spectrum.
    varying_pumps : np.ndarray
        Boolean mask for pumps participating in optimization.
    total_flow : float
        Total flow constraint.
    callbacks : ExperimentCallbacks
        UI/controller callbacks for reporting progress.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        cost_fn: CostFunction,
        varying_pumps: np.ndarray,
        total_flow: float,
        target_style: str = "single_point",
        target_wavelength: Optional[float] = None,
        callbacks: Optional[ExperimentCallbacks] = None,
    ):
        self.config = config
        self.cost_fn = cost_fn
        self.varying_pumps = varying_pumps
        self.total_flow = total_flow
        self.target_style = target_style
        self.target_wavelength = target_wavelength
        self.callbacks = callbacks or HeadlessCallbacks()
        self.diagnostics = GPDiagnostics()

        # Build GP kernel
        n_dims = max(1, int(np.sum(varying_pumps)) - 1)
        if getattr(config, "bo_preset", "default") == "robust":
            # Robust preset: Matern 2.5 ISOTROPIC + tighter LS bounds
            # (0.5, 10) µL/min. ARD was redundant (per-dim length scales
            # ~equal on good data, noise-fitting on poor data).
            # Atomic preset → ignore config.rbf_length_scale / white_noise_level.
            self.kernel = Matern(
                length_scale=1.0,
                length_scale_bounds=(0.5, 10.0),
                nu=2.5,
            ) + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=(1e-5, 1e5),
            )
            logger.info(
                "Robust BO preset enabled: Matern 2.5 isotropic kernel "
                "(length_scale_bounds=(0.5, 10.0)). "
                "config.rbf_length_scale and white_noise_level are ignored.",
            )
        else:
            if config.kernel_type == "ARD":
                length_scale = np.ones(n_dims) * config.rbf_length_scale
                logger.warning(
                    "ARD kernel enabled: using per-dimension length scales "
                    "(n_dims=%d). This increases fitting time and requires "
                    "sufficient data to avoid overfitting.", n_dims,
                )
            else:
                length_scale = config.rbf_length_scale
            self.kernel = RBF(
                length_scale=length_scale,
                length_scale_bounds=config.rbf_length_scale_bounds,
            ) + WhiteKernel(
                noise_level=config.white_noise_level,
                noise_level_bounds=config.white_noise_bounds,
            )

        # State
        self.spectra: Dict[str, Spectrum] = {}
        self.success = False

    def run(
        self,
        measure_fn: MeasureFn,
        init_fn: InitFn,
        experiment_path: str = "",
        previous_flows: Optional[np.ndarray] = None,
        previous_costs: Optional[np.ndarray] = None,
        nb_init_samples: int = 3,
        save_figs: bool = True,
    ) -> Optional[ExperimentResult]:
        """
        Execute the BO loop.

        Parameters
        ----------
        measure_fn : callable
            (predicted_flows) -> (Spectrum, actual_flows). The measurement strategy.
        init_fn : callable
            () -> (flows, costs, last_spectrum). Generates initial experimental data.
        experiment_path : str
            Path for saving intermediate results.
        previous_flows, previous_costs : np.ndarray, optional
            Previous experiment data for warm-starting.
        nb_init_samples : int
            Number of initial samples before BO starts.
        save_figs : bool
            Whether to save intermediate figures.

        Returns
        -------
        ExperimentResult or None if aborted.
        """
        gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.config.gp_alpha,
            normalize_y=self.config.gp_normalize_y,
            n_restarts_optimizer=self.config.gp_n_restarts,
        )

        best_min: List[float] = []
        times: List[float] = []
        predicted_flows: Optional[np.ndarray] = None
        self.spectra = {}
        self.success = False
        spectrum: Optional[Spectrum] = None

        # --- Load previous data (warm start) ---
        flows, costs, starting_iter, predicted_flows = self._warm_start(
            gp, previous_flows, previous_costs, nb_init_samples,
            best_min, experiment_path,
        )

        # --- Main BO loop ---
        final_iter = starting_iter
        for i in range(starting_iter, self.config.max_iterations + 1):
            final_iter = i

            if self.callbacks.should_stop():
                final_iter = i - 1
                break

            logger.info(f"--- Iteration {i} ---")

            if i == 0:
                # Initialization phase
                flows, costs, spectrum = init_fn()
                if flows is None:
                    return None
            else:
                # Measurement phase (the only part that differs between strategies)
                spectrum, performed_flows = measure_fn(predicted_flows)
                if spectrum is None:
                    final_iter = i - 1
                    break

                self.spectra[f"iter {i}"] = spectrum
                cost_at_predict, measured_value = self.cost_fn(spectrum)
                logger.info(f"Measured cost: {cost_at_predict:.6f}")
                self._log_measurement(spectrum, measured_value)

                flows = np.concatenate((flows, [performed_flows]))
                costs = np.concatenate((costs, [cost_at_predict]))

            # Track best cost
            if i >= starting_iter or starting_iter == 0:
                best_min.append(float(costs.min()))

            # Fit GP
            t0 = time.time()
            flows_reduced = self._reduce_dimensions(flows)
            gp.fit(flows_reduced, costs)

            # Diagnose periodically
            if i % 3 == 0 or i == self.config.max_iterations:
                self.diagnostics.diagnose(gp, flows_reduced, costs, costs)

            # Predict next point
            predicted_flows = self._predict_next(costs, gp)
            times.append(time.time() - t0)

            # Report progress
            self.callbacks.on_iteration_complete(
                i, flows, costs, gp, best_min, times,
                spectrum, predicted_flows, False, save_figs,
            )

            # Save intermediate data
            if experiment_path:
                self._save_intermediate(flows, costs, experiment_path)

            # Check tolerance
            if best_min[-1] < self._get_tolerance():
                self.success = True
                if self.config.nb_init_samples is not None:
                    logger.info("Tolerance reached!")
                    break

        # --- Final reporting ---
        return self._finalize(
            flows, costs, gp, best_min, times, nb_init_samples,
            final_iter, save_figs,
        )

    def _warm_start(
        self,
        gp: GaussianProcessRegressor,
        previous_flows: Optional[np.ndarray],
        previous_costs: Optional[np.ndarray],
        nb_init_samples: int,
        best_min: List[float],
        experiment_path: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Optional[np.ndarray]]:
        """Load previous data and pre-train GP if available."""
        if previous_flows is not None and previous_costs is not None:
            flows = previous_flows.copy()
            costs = previous_costs.copy()
            logger.info(f"Warm-starting from {len(costs)} previous data points")
            logger.info(f"Previous best cost: {costs.min():.4f}")

            starting_iter = len(costs) - nb_init_samples + 1

            for j in range(len(costs)):
                best_min.append(float(costs[: j + 1].min()))

            # Pre-train GP
            flows_reduced = self._reduce_dimensions(flows)
            gp.fit(flows_reduced, costs)

            # Save pre-trained model
            if experiment_path:
                try:
                    path = os.path.join(experiment_path, "previous_gp_model.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(gp, f)
                except Exception as e:
                    logger.warning(f"Could not save pre-trained GP: {e}")

            predicted = self._predict_next(costs, gp)
            logger.info(f"Next predicted flows: {predicted}")
            return flows, costs, starting_iter, predicted

        return None, None, 0, None

    def _reduce_dimensions(self, flows: np.ndarray) -> np.ndarray:
        """Reduce flow dimensions: keep varying pumps, drop last (constrained)."""
        variant = flows[:, self.varying_pumps]
        return variant[:, :-1]

    def _predict_next(
        self, costs: np.ndarray, gp: GaussianProcessRegressor
    ) -> np.ndarray:
        """Use acquisition function + differential evolution to find next point."""
        n_var = int(self.varying_pumps.sum())

        # Create acquisition function
        acq_fn = self._create_acq_fn(gp, costs)

        # Bounds for DE: each varying pump (except last) in [0, total_flow]
        n_dim = n_var - 1
        var_bounds = self.config.bounds
        if var_bounds is not None:
            varying_indices = np.where(self.varying_pumps)[0]
            de_bounds = [
                (float(var_bounds[idx][0]), float(var_bounds[idx][1]))
                for idx in varying_indices[:-1]
            ]
        else:
            de_bounds = [(0, self.total_flow)] * n_dim

        # Sum constraint: sum of varying flows <= total_flow
        if n_dim > 1:
            constraint = LinearConstraint(
                np.ones(n_dim), lb=0, ub=self.total_flow
            )
            result = differential_evolution(
                acq_fn,
                bounds=de_bounds,
                constraints=[constraint],
                tol=self.config.de_tolerance,
                maxiter=self.config.de_max_iterations,
                popsize=self.config.de_popsize,
            )
        elif n_dim == 1:
            result = differential_evolution(
                acq_fn,
                bounds=de_bounds,
                tol=self.config.de_tolerance,
                maxiter=self.config.de_max_iterations,
            )
        else:
            # 0D case (single pump)
            return np.array([self.total_flow])

        # Reconstruct full flow array
        predicted_var = np.zeros(n_var)
        predicted_var[:-1] = result.x
        predicted_var[-1] = self.total_flow - np.sum(result.x)

        # Map back to full pump array
        full_flows = np.zeros(len(self.varying_pumps))
        full_flows[self.varying_pumps] = predicted_var

        # Add fixed flows
        if self.config.fixed_flows is not None:
            fixed_mask = ~self.varying_pumps
            full_flows[fixed_mask] = self.config.fixed_flows[fixed_mask]

        return np.round(full_flows, 4)

    def _create_acq_fn(self, gp: GaussianProcessRegressor, costs: np.ndarray):
        """Instantiate the configured acquisition function."""
        acq_type = self.config.acquisition_function
        if acq_type == "EI":
            # xi is a dimensionless fraction of the signal scale; gp.predict
            # returns raw units (normalize_y denormalises), so pass the cost
            # std as y_scale to make xi meaningful. Guard degenerate spreads.
            y_scale = float(np.std(costs))
            if not np.isfinite(y_scale) or y_scale < 1e-12:
                y_scale = 1.0
            return ExpectedImprovement(gp, best_y=float(costs.min()), y_scale=y_scale)
        elif acq_type == "LCB":
            return LowerConfidenceBound(gp)
        elif acq_type == "TS":
            return ThompsonSampling(gp)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_type}")

    def _get_tolerance(self) -> float:
        """Get the tolerance threshold for early stopping."""
        # This will be set from the ExperimentConfig
        return getattr(self, '_tolerance', float('inf'))

    def set_tolerance(self, tol: float, stop_on_tol: bool = True) -> None:
        """Set the tolerance threshold."""
        self._tolerance = tol
        self._stop_on_tol = stop_on_tol

    def _log_measurement(self, spectrum: Spectrum, measured_value: float) -> None:
        """Log measurement details based on target style."""
        if self.target_style == "single_point" and self.target_wavelength:
            abs_val = spectrum.get_abs_at_wl(self.target_wavelength)
            logger.info(f"Absorbance at {self.target_wavelength}nm: {abs_val:.4f}")
        elif self.target_style == "lambda_max":
            idx = int(np.argmax(spectrum.absorbance))
            logger.info(f"Lambda max at {spectrum.wavelengths[idx]:.1f}nm")

    def _save_intermediate(
        self, flows: np.ndarray, costs: np.ndarray, path: str
    ) -> None:
        """Save flows and costs to CSV after each iteration."""
        try:
            np.savetxt(os.path.join(path, "flows.csv"), flows, delimiter=";", fmt="%.4f")
            np.savetxt(os.path.join(path, "cost.csv"), costs, delimiter=";", fmt="%.8f")
        except Exception as e:
            logger.warning(f"Failed to save intermediate data: {e}")

    def _finalize(
        self,
        flows: np.ndarray,
        costs: np.ndarray,
        gp: GaussianProcessRegressor,
        best_min: List[float],
        times: List[float],
        nb_init_samples: int,
        final_iter: int,
        save_figs: bool,
    ) -> ExperimentResult:
        """Compute and report final results."""
        logger.info(f"=== Final results (iteration {final_iter}) ===")

        min_cost = float(costs.min())
        best_idx = int(costs.argmin())
        opt_flows = flows[best_idx]

        if best_idx >= nb_init_samples:
            best_iter = best_idx - nb_init_samples + 1
            best_key = f"iter {best_iter}"
        else:
            best_iter = 0
            best_key = f"init {best_idx + 1}"

        best_spectrum = self.spectra.get(best_key)

        # Report final figure
        if best_spectrum is not None:
            self.callbacks.on_iteration_complete(
                final_iter, flows, costs, gp, best_min, times,
                best_spectrum, None, True, True,
            )

        logger.info(f"Optimal flows: {opt_flows}")
        logger.info(f"Minimum cost: {min_cost:.6f}")

        status = "Success" if self.success else "Max iterations reached"
        self.callbacks.on_experiment_complete(status)

        return ExperimentResult(
            flows=flows,
            costs=costs,
            spectra=self.spectra,
            optimal_flows=opt_flows,
            minimum_cost=min_cost,
            best_iteration=best_iter,
            success=self.success,
            gp_model=gp,
        )
