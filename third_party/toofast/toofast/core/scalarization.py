"""
Scalarization utilities for multi-objective Bayesian optimization (ParEGO).

ParEGO (Knowles, 2006) turns a multi-objective problem into a sequence of
single-objective ones: at every BO iteration a *fresh random weight vector*
is drawn on the simplex and the objective vectors are collapsed with an
augmented Tchebycheff scalarization. Fitting one GP on that scalar reuses the
existing single-objective machinery, yet sweeping the weights traces out the
whole Pareto front instead of a single trade-off point.

This is the lowest-effort path to multi-objective BO here (e.g. lambda_max
accuracy + growth time) because it plugs into the scalar GP/EI loop already in
the codebase — no per-objective GPs or hypervolume machinery required.

Key pieces
----------
- ``random_simplex_weight``: draw a weight vector summing to 1.
- ``normalize_objectives``: map each objective to [0, 1] from observed extents
  (mandatory — objectives have different units, e.g. nm^2 vs seconds).
- ``augmented_tchebycheff``: the scalarization itself.
- ``ParEGOScalarizer``: convenience wrapper holding rho + a RNG, exposing
  ``scalarize(objectives)`` which normalizes, draws a fresh weight and reduces
  a whole objective matrix to a scalar-cost vector for GP fitting.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def random_simplex_weight(n_obj: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw a weight vector uniformly on the (n_obj-1)-simplex (sums to 1)."""
    rng = rng or np.random.default_rng()
    # Uniform on the simplex via the normalized-exponential (Dirichlet(1,...,1)).
    w = rng.dirichlet(np.ones(int(n_obj)))
    return w


def normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """Min-max normalize each objective (column) to [0, 1].

    Objectives are assumed to be *minimized* and to have heterogeneous units,
    so normalization to a common scale is required before scalarizing. A
    column with zero spread maps to all-zeros (it carries no information yet).
    """
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    mins = objectives.min(axis=0, keepdims=True)
    maxs = objectives.max(axis=0, keepdims=True)
    spread = maxs - mins
    safe = np.where(spread > 1e-12, spread, 1.0)
    normalized = (objectives - mins) / safe
    # Columns with no spread -> 0 (avoid 0/1 artefacts from the safe divisor).
    normalized = np.where(spread > 1e-12, normalized, 0.0)
    return normalized


def augmented_tchebycheff(
    normalized: np.ndarray, weights: np.ndarray, rho: float = 0.05
) -> np.ndarray:
    """Augmented Tchebycheff scalarization (to be *minimized*).

        s(y) = max_k [ w_k * y_k ] + rho * sum_k [ w_k * y_k ]

    The max term drives points toward the Pareto front; the small weighted-sum
    term (rho ~ 0.05) breaks ties and avoids weakly-dominated solutions.

    Parameters
    ----------
    normalized : np.ndarray
        Objective matrix already normalized to [0, 1] (n_points, n_obj).
    weights : np.ndarray
        Weight vector on the simplex (n_obj,).
    rho : float
        Augmentation coefficient.
    """
    normalized = np.atleast_2d(np.asarray(normalized, dtype=float))
    weights = np.asarray(weights, dtype=float)
    weighted = normalized * weights  # (n_points, n_obj)
    return weighted.max(axis=1) + rho * weighted.sum(axis=1)


class ParEGOScalarizer:
    """Stateful ParEGO scalarizer: holds rho and an RNG for reproducibility.

    Each call to :meth:`scalarize` draws a *new* random weight (that is the
    whole point of ParEGO), normalizes the objective matrix and reduces it to a
    scalar cost vector suitable for a single GP fit.
    """

    def __init__(self, n_obj: int, rho: float = 0.05, seed: Optional[int] = None):
        self.n_obj = int(n_obj)
        self.rho = float(rho)
        self.rng = np.random.default_rng(seed)
        self.last_weight: Optional[np.ndarray] = None

    def scalarize(
        self, objectives: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Normalize + scalarize an objective matrix to a scalar-cost vector.

        Parameters
        ----------
        objectives : np.ndarray
            Raw objective matrix (n_points, n_obj), all minimized.
        weights : np.ndarray, optional
            Fixed weight vector. If None, a fresh random simplex weight is
            drawn (standard ParEGO behaviour).
        """
        objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
        if objectives.shape[1] != self.n_obj:
            raise ValueError(
                f"Expected {self.n_obj} objectives, got {objectives.shape[1]}"
            )
        if weights is None:
            weights = random_simplex_weight(self.n_obj, self.rng)
        self.last_weight = weights
        normalized = normalize_objectives(objectives)
        return augmented_tchebycheff(normalized, weights, self.rho)


def align_trailing_column(values, n: int) -> np.ndarray:
    """Align a trailing-tracked objective column to ``n`` rows.

    In the online BO loop, a secondary objective such as growth time is only
    recorded for the most recent (BO) measurements; the leading init / warm-start
    rows have no value. ``values`` therefore correspond to the LAST ``len(values)``
    of ``n`` rows. Missing leading rows (and any NaNs) are neutral-filled with the
    median of the known values (0.0 if none are known) so a single scalar GP can
    always be fit. Returns a float array of length ``n``.
    """
    vals = np.asarray(list(values or []), dtype=float)
    if len(vals) < n:
        vals = np.concatenate([np.full(n - len(vals), np.nan), vals])
    else:
        vals = vals[-n:]
    if not np.any(np.isfinite(vals)):
        return np.zeros(n)
    return np.where(np.isfinite(vals), vals, np.nanmedian(vals))


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (objectives are minimized)."""
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = objectives.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # A point j dominates i if j <= i on all objectives and < on at least one.
        dominates = np.all(objectives <= objectives[i], axis=1) & np.any(
            objectives < objectives[i], axis=1
        )
        if np.any(dominates):
            is_efficient[i] = False
    return is_efficient


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Return the non-dominated objective vectors (the Pareto front)."""
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    return objectives[pareto_mask(objectives)]
