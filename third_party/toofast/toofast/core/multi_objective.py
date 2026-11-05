"""
ParEGO multi-objective Bayesian optimization driver.

A self-contained ParEGO loop that reuses the same surrogate (sklearn GP) and
acquisition optimizer (differential evolution + Expected Improvement) as the
single-objective ``BayesianOptimizer``, but optimizes a *vector* of objectives.

It is deliberately decoupled from the hardware loop so it can be unit-tested
and driven from a notebook: you pass an ``evaluate`` callable mapping a point
(flow vector in reduced space) to an objective vector, e.g.

    def evaluate(x):
        spectrum, growth_time = run_synthesis(x)
        return np.array([lambda_max_cost(spectrum), growth_time])

    result = run_parego(evaluate, bounds, n_obj=2, n_init=6, n_iter=20)

Each iteration draws a fresh random simplex weight (the ParEGO idea), collapses
the observed objective matrix to a scalar cost via the augmented Tchebycheff
scalarization, fits one GP on it and maximizes EI to pick the next point.
Sweeping the weights over iterations traces the Pareto front.

To use it with the real 2FAST setup, wrap a synthesis+measurement call as
``evaluate`` and return [lambda_max cost, growth time]. Growth time is recorded
per sample by ``Experiment`` (see growth_time_from_rows / growth_times.csv).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from toofast.acquisition import ExpectedImprovement
from toofast.core.scalarization import ParEGOScalarizer, pareto_mask

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ParEGOResult:
    """Outcome of a ParEGO run."""

    X: np.ndarray              # evaluated points (n_points, n_dim)
    objectives: np.ndarray     # raw objective matrix (n_points, n_obj)
    pareto_X: np.ndarray       # non-dominated points
    pareto_objectives: np.ndarray  # the Pareto front
    weights: List[np.ndarray] = field(default_factory=list)  # weight per BO iter

    @property
    def n_evaluations(self) -> int:
        return int(self.X.shape[0])


def _build_gp() -> GaussianProcessRegressor:
    kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e3), nu=2.5) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-6, 1e2)
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, alpha=1e-8
    )


def _propose(
    gp: GaussianProcessRegressor,
    scalar_costs: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    sum_constraint: Optional[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Maximize EI (minimize the scalar cost) via differential evolution."""
    acq = ExpectedImprovement(
        gp, best_y=float(scalar_costs.min()),
        y_scale=max(float(np.std(scalar_costs)), 1e-9),
    )
    constraints = ()
    if sum_constraint is not None and len(bounds) > 1:
        constraints = [LinearConstraint(np.ones(len(bounds)), lb=0, ub=sum_constraint)]
    result = differential_evolution(
        acq, bounds=list(bounds), constraints=constraints,
        tol=1e-6, maxiter=200, popsize=20, seed=int(rng.integers(0, 2**31 - 1)),
    )
    return np.asarray(result.x, dtype=float)


def run_parego(
    evaluate: EvaluateFn,
    bounds: Sequence[Tuple[float, float]],
    n_obj: int,
    n_init: int = 6,
    n_iter: int = 20,
    rho: float = 0.05,
    sum_constraint: Optional[float] = None,
    seed: Optional[int] = None,
) -> ParEGOResult:
    """Run ParEGO multi-objective Bayesian optimization.

    Parameters
    ----------
    evaluate : callable
        Maps a point (n_dim,) to an objective vector (n_obj,), all minimized.
    bounds : sequence of (low, high)
        Box bounds per dimension.
    n_obj : int
        Number of objectives.
    n_init : int
        Random initial evaluations before BO starts.
    n_iter : int
        Number of BO iterations (each = one fresh weight + one evaluation).
    rho : float
        Augmented-Tchebycheff augmentation coefficient.
    sum_constraint : float, optional
        If set, enforce sum(x) <= sum_constraint (e.g. total flow budget).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    ParEGOResult
    """
    rng = np.random.default_rng(seed)
    bounds = list(bounds)
    n_dim = len(bounds)
    scalarizer = ParEGOScalarizer(n_obj=n_obj, rho=rho, seed=seed)

    # --- Initial design (random within bounds, respecting the budget) ---
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    while len(X) < n_init:
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        if sum_constraint is not None and x.sum() > sum_constraint:
            continue
        X.append(x)
        Y.append(np.asarray(evaluate(x), dtype=float))
    X_arr = np.vstack(X)
    Y_arr = np.vstack(Y)

    weights_log: List[np.ndarray] = []

    # --- BO loop ---
    for it in range(n_iter):
        scalar_costs = scalarizer.scalarize(Y_arr)  # fresh random weight each iter
        weights_log.append(scalarizer.last_weight)

        gp = _build_gp()
        gp.fit(X_arr, scalar_costs)

        x_next = _propose(gp, scalar_costs, bounds, sum_constraint, rng)
        y_next = np.asarray(evaluate(x_next), dtype=float)

        X_arr = np.vstack([X_arr, x_next])
        Y_arr = np.vstack([Y_arr, y_next])
        logger.info(
            "ParEGO iter %d/%d: weight=%s objective=%s",
            it + 1, n_iter, np.round(scalarizer.last_weight, 3).tolist(),
            np.round(y_next, 4).tolist(),
        )

    mask = pareto_mask(Y_arr)
    return ParEGOResult(
        X=X_arr,
        objectives=Y_arr,
        pareto_X=X_arr[mask],
        pareto_objectives=Y_arr[mask],
        weights=weights_log,
    )
