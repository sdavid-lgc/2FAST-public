"""Monotone warps between raw cost and the GP-target space.

Pure, dependency-light helpers so the warp logic is unit-testable without the
hardware stack that ``experiment`` pulls in. ``Experiment._cost_to_gp`` /
``_gp_to_cost`` delegate here.

The 'log' warp fits the GP on ``log(cost)``: variance-stabilising, so the
low-cost valley keeps GP resolution while the high-cost region compresses into a
steep plateau. It is disabled for multi-objective (the ParEGO scalar can be
negative, so ``log`` is undefined).
"""

import numpy as np

# Floor applied before log() so a zero / near-zero cost cannot map to -inf.
LOG_FLOOR = 1e-9


def cost_to_gp(y, transform="none", multi_objective=False):
    """Map raw cost -> GP-target space."""
    y = np.asarray(y, dtype=float)
    if transform == "log" and not multi_objective:
        return np.log(np.maximum(y, LOG_FLOOR))
    return y


def gp_to_cost(v, transform="none", multi_objective=False):
    """Inverse of :func:`cost_to_gp`: GP-target value -> raw cost units."""
    v = np.asarray(v, dtype=float)
    if transform == "log" and not multi_objective:
        return np.exp(v)
    return v
