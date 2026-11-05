"""
Configuration dataclasses for 2FAST experiments.

Replaces the 17-parameter update_optim_params() method and hardcoded
constants with structured, serializable configuration objects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware connection and simulation settings."""

    nemesys_config_file: str = ""
    collector_port: str = "COM8"
    c3000_port: str = "COM19"
    mfc_port: str = "COM9"
    is_simulation: bool = False

    # Syringe parameters
    number_of_pumps: int = 5
    inner_diameter_mm: float = 10.301
    max_piston_stroke_mm: float = 60.0
    total_flow: float = 8.0


@dataclass
class TargetConfig:
    """Target spectrum configuration for the cost function."""

    style: Literal["single_point", "full_spectrum", "lambda_max", "pca"] = "single_point"
    wavelength: Optional[float] = None
    absorbance: Optional[float] = None
    spectrum_file: Optional[str] = None
    cost_method: Literal["L2", "cosine"] = "L2"
    tolerance: Optional[float] = None
    relative_tolerance: bool = False
    stop_on_tolerance: bool = False
    no_particle_threshold: float = 0.01

    # PCA target ('pca' style): frozen basis fitted offline via
    # toofast.core.pca_model. The cost is the squared distance to the target
    # spectrum in PCA score space (see PCASpectrumCost).
    pca_model_file: Optional[str] = None
    pca_variance_weighted: bool = False

    @property
    def default_tolerance(self) -> float:
        defaults = {
            "single_point": 1e-7,
            "full_spectrum": 3e-2,
            "lambda_max": 25.0,
            "pca": 1e-2,
        }
        return defaults[self.style]

    def get_tolerance(self) -> float:
        if self.tolerance is not None:
            if self.relative_tolerance:
                return (self.tolerance / 100) ** 2
            return self.tolerance
        return self.default_tolerance


@dataclass
class TerminationConfig:
    """Criteria for determining when a stop-flow measurement is complete."""

    correlation_threshold: float = 0.998
    amplitude_threshold: float = 0.02
    min_spectra: int = 10
    monitoring_timeout_s: float = 300.0
    savgol_window: int = 11
    savgol_polyorder: int = 3

    # Early-stop bypass of the min_spectra gate. When a synthesis genuinely
    # stabilizes before min_spectra measurements, allow termination provided
    # the convergence criteria have held for early_stop_confirmations
    # consecutive checks (guards against a single transient early plateau).
    allow_early_stop: bool = True
    early_stop_confirmations: int = 2


@dataclass
class OptimizationConfig:
    """Bayesian Optimization parameters."""

    max_iterations: int = 14
    acquisition_function: Literal["EI", "LCB", "TS"] = "EI"

    # BO preset — selects an atomic combination of kernel + cost.
    #   "default": RBF + quadratic cost on lambda_max (historical behaviour).
    #   "robust" : Matern 2.5 ARD + Huber loss (delta=huber_delta_nm) on
    #              lambda_max + tighter length-scale bounds. Validated against
    #              multiple BO runs as a more stable choice in low-data regimes.
    # When set to "robust" the kernel_type / rbf_length_scale_bounds fields are
    # ignored for kernel construction and the lambda_max cost path uses Huber.
    bo_preset: Literal["default", "robust"] = "default"
    huber_delta_nm: float = 30.0

    # Flat-spectrum (no-particle) handling at GP-fit time. The raw penalty is
    # still stored in costs.csv; this only transforms the GP training targets.
    #   "nearest_high"     : flat cost = margin * max(real costs of neighbours
    #                        within an adaptive radius), capped (default — keeps
    #                        the region "bad" without an extreme outlier that
    #                        collapses the length scale).
    #   "penalty_max"      : keep the cost_function max penalty (historical).
    #   "penalty_moderate" : flat cost = factor * mean(real costs).
    #   "exclude"          : drop flat points from the GP fit.
    flat_strategy: Literal[
        "nearest_high", "penalty_max", "penalty_moderate", "exclude"
    ] = "nearest_high"
    flat_penalty_factor: float = 2.0     # 'penalty_moderate'
    flat_nearest_margin: float = 1.25    # 'nearest_high'
    flat_nearest_cap: float = 3.0        # 'nearest_high' (× worst real cost)

    # Constrained BO (optional, off by default): fit a GP classifier on
    # P(particles | flows) and multiply the EI acquisition by it. Needs
    # enough points of BOTH classes — falls back to plain EI otherwise.
    feasibility_aware: bool = False

    # Physics-based ghost points injected at GP-fit time: synthetic samples
    # on the hyperfaces "AgNO3 flow = 0" and "NaBH4 flow = 0" (no particles
    # form there) carrying the flat-spectrum sentinel cost. The flat_strategy
    # (nearest_high) remaps them to a bounded penalty.
    use_physics_ghosts: bool = True

    # Per-point GP noise for ghost points. Ghosts are virtual observations, so
    # they get a larger diagonal alpha than real measurements (which keeps the
    # GP from collapsing its length scale to interpolate them exactly). The GP
    # fits with normalize_y=True, so this is in normalized-target units (a
    # fraction of the unit signal variance) and is scale-invariant. 'auto' ->
    # 0.1 ; a float sets a fixed normalized noise.
    ghost_noise_level: float | str = "auto"

    # GP kernel parameters
    kernel_type: Literal["RBF", "ARD"] = "RBF"
    rbf_length_scale: float = 1.0
    rbf_length_scale_bounds: Tuple[float, float] = (0.1, 1e5)
    white_noise_level: float = 1e-3
    white_noise_bounds: Tuple[float, float] = (1e-3, 1e5)
    gp_alpha: float = 1e-8
    gp_normalize_y: bool = True
    gp_n_restarts: int = 100

    # Differential evolution parameters
    de_tolerance: float = 1e-6
    de_max_iterations: int = 400
    de_popsize: int = 25

    # Initialization
    nb_init_samples: Optional[int] = None
    init_option: Literal["random", "latinhypercube", "projected_lhs", "fixed"] = "latinhypercube"
    init_flows: Optional[np.ndarray] = None

    # Pump bounds and constraints
    bounds: Optional[np.ndarray] = None
    fixed_pumps: Optional[np.ndarray] = None
    fixed_flows: Optional[np.ndarray] = None

    # Multi-objective BO (ParEGO). When enabled, each iteration scalarizes the
    # objective vector (e.g. ["lambda_max", "growth_time"]) with a fresh random
    # simplex weight and an augmented Tchebycheff scalarization, tracing the
    # Pareto front with the existing scalar GP/EI loop. See
    # toofast.core.multi_objective.run_parego and core.scalarization.
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["lambda_max"])
    parego_rho: float = 0.05


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    experiment_name: str = "experiment"
    experiment_path: str = ""
    chip_id: str = ""
    reagents: Optional[List[str]] = None
    residence_time: float = 40.0
    sampling_rate: float = 5.0
    back_pressure_flow: float = 0.75
    nitric_acid_volume: float = 80.0
    water_volume: float = 400.0

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    def save(self, path: str) -> None:
        """Save configuration to JSON."""
        data = self._to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Configuration loaded from {path}")
        return cls._from_dict(data)

    def _to_dict(self) -> Dict:
        """Convert to serializable dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif hasattr(value, "__dict__"):
                result[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.__dict__.items()
                }
            else:
                result[key] = value
        return result

    @classmethod
    def _from_dict(cls, data: Dict) -> "ExperimentConfig":
        """Construct from dict."""
        hw_data = data.pop("hardware", {})
        tg_data = data.pop("target", {})
        opt_data = data.pop("optimization", {})
        term_data = data.pop("termination", {})

        # Reconstruct numpy arrays
        for key in ("bounds", "fixed_pumps", "fixed_flows", "init_flows"):
            if key in opt_data and opt_data[key] is not None:
                opt_data[key] = np.array(opt_data[key])

        return cls(
            hardware=HardwareConfig(**hw_data),
            target=TargetConfig(**tg_data),
            optimization=OptimizationConfig(**opt_data),
            termination=TerminationConfig(**term_data),
            **data,
        )
