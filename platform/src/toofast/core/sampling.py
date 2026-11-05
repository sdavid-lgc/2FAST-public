"""
Space-filling designs of compositions (flow sets with a fixed total flow).

Latin hypercube on the simplex (`simplex_lhs`): a Latin hypercube in d−1 dimensions is mapped onto the simplex
{q ≥ lower bounds, Σ q = total} by the inverse Rosenblatt transform of the uniform (Dirichlet(1,…,1)) distribution
(stick breaking: x_k = (1 − Σ_{j<k} x_j)·(1 − u_k^{1/(d−k)})). The design is uniformly distributed over the
compositions and stratified: each coordinate of the underlying hypercube has exactly one point per 1/n stratum.
The marginal of each flow is then the Beta(1, d−1) law of a uniform composition, not uniform (no design can have
uniform marginals on every flow axis and a fixed sum: their means would add up to Σ (lo+hi)/2 ≠ total).
Upper bounds are enforced by redrawing the whole design (stratification kept).

The previous "projected LHS" (hypercube LHS rescaled to the total flow, as used for the paper's runs) stays available
as `PumpSystem.generate_projected_lhs_flows_sets`.

Random numbers come from numpy's global generator, like the other samplers of the platform (seeded by the caller).
"""
from __future__ import annotations

import numpy as np


def latin_hypercube(n_samples: int, n_dims: int) -> np.ndarray:
    """(n_samples, n_dims) points in [0, 1): one point per 1/n_samples stratum on every axis."""
    u = np.empty((n_samples, n_dims))
    for k in range(n_dims):
        u[:, k] = (np.random.permutation(n_samples) + np.random.random(n_samples)) / n_samples
    return u


def cube_to_simplex(u: np.ndarray) -> np.ndarray:
    """Inverse Rosenblatt transform: uniform points of [0,1]^(d−1) -> uniform points of the (d−1)-simplex (sum 1)."""
    n, dm1 = u.shape
    d = dm1 + 1
    x = np.zeros((n, d))
    remaining = np.ones(n)
    for k in range(dm1):
        x[:, k] = remaining * (1.0 - u[:, k] ** (1.0 / (d - 1 - k)))
        remaining = remaining - x[:, k]
    x[:, -1] = remaining
    return x


def simplex_to_cube(x: np.ndarray) -> np.ndarray:
    """Inverse of `cube_to_simplex` (used to check the stratification)."""
    n, d = x.shape
    u = np.zeros((n, d - 1))
    remaining = np.ones(n)
    for k in range(d - 1):
        frac = np.divide(x[:, k], remaining, out=np.zeros(n), where=remaining > 0)
        u[:, k] = (1.0 - frac) ** (d - 1 - k)
        remaining = remaining - x[:, k]
    return u


def simplex_lhs(n_samples: int, bounds, total: float, max_tries: int = 10000) -> np.ndarray:
    """Latin hypercube on the simplex {lo_i <= q_i <= hi_i, Σ q_i = total}: (n_samples, d) flows.

    bounds: (d, 2) per-flow (lo, hi). Raises ValueError if the bounds leave no room (Σ lo > total or Σ hi < total)
    or if no design satisfies the upper bounds within `max_tries` draws.
    """
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(bounds)
    free = float(total) - lo.sum()
    if free < -1e-12 or hi.sum() < total - 1e-12:
        raise ValueError(f"no composition satisfies the bounds: sum of lower bounds {lo.sum():g}, "
                         f"sum of upper bounds {hi.sum():g}, total {total:g}")
    if d == 1:
        return np.full((n_samples, 1), float(total))
    for _ in range(max_tries):
        q = lo + cube_to_simplex(latin_hypercube(n_samples, d - 1)) * max(free, 0.0)
        if np.all(q <= hi + 1e-12):
            return np.minimum(q, hi)
    raise ValueError(f"no Latin hypercube design within the upper bounds after {max_tries} draws: "
                     "the bounds cut most of the simplex; use the projected LHS or random initialisation")
