"""
Feasibility-weighted acquisition for constrained Bayesian Optimization.

Implements the "two stacked models" pattern (Gardner 2014, Gelbart 2014):
a cost GP fitted on feasible points only, and a GP classifier modelling
P(feasible | x) = P(particles form | flows). The base (improvement-based)
acquisition is multiplied by that probability so the optimiser stops
chasing barren regions where the cost GP is merely uncertain.

Only meaningful for improvement-based acquisitions (EI): with the sign
convention used here the base acquisition returns ``-EI`` (<= 0), so
``base * P`` (P in [0, 1]) keeps the minimisation well-posed and maximises
``EI * P``. Not applied to LCB/TS.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def fit_feasibility_classifier(X: np.ndarray, labels: np.ndarray,
                               min_per_class: int = 4):
    """Fit a GP classifier on P(feasible | x).

    Parameters
    ----------
    X : (n, d) array
        Training inputs (reduced flow space, same as the cost GP).
    labels : (n,) int array
        1 = feasible (particles formed), 0 = flat/infeasible.
    min_per_class : int
        Minimum count required in BOTH classes; below this the classifier
        is not informative and None is returned (caller falls back to
        plain acquisition).

    Returns
    -------
    Fitted GaussianProcessClassifier, or None if there is not enough data.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n0 < min_per_class or n1 < min_per_class:
        logger.info(
            "Feasibility classifier skipped: too few per class "
            "(feasible=%d, infeasible=%d, need >= %d each).",
            n1, n0, min_per_class)
        return None
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    n_dims = X.shape[1] if X.ndim == 2 else 1
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(max(n_dims, 1)),
        length_scale_bounds=(0.3, 1e3),
    )
    clf = GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=5, random_state=0)
    clf.fit(X, labels)
    logger.info(
        "Feasibility classifier fitted (feasible=%d, infeasible=%d).", n1, n0)
    return clf


class FeasibilityWeightedAcquisition:
    """Multiply a base acquisition by P(feasible | x).

    Parameters
    ----------
    base_acq : callable
        The base acquisition (EI, possibly already exploration-wrapped).
        Must return the minimisation-oriented value (``-EI``).
    classifier : fitted GaussianProcessClassifier
        Predicts P(feasible | x) via ``predict_proba``.
    """

    def __init__(self, base_acq, classifier):
        self.base_acq = base_acq
        self.classifier = classifier

    def __call__(self, x, gp_model: Optional[object] = None) -> float:
        base = float(self.base_acq(x))
        xx = np.asarray(x, dtype=float).reshape(1, -1)
        p_feas = float(self.classifier.predict_proba(xx)[0, 1])
        # base is <= 0 (-EI); scaling by p_feas in [0,1] keeps DE minimisation
        # consistent and maximises EI * P(feasible).
        return base * p_feas
