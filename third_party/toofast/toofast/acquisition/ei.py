"""Expected Improvement acquisition function."""

from typing import Union

import numpy as np
from scipy.stats import norm

from toofast.acquisition.base import AcquisitionFunction


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) for minimization.

    Quantifies the expected amount by which a candidate point will
    improve upon the current best observed value.

    Parameters
    ----------
    gp_model : GaussianProcessRegressor
        Fitted GP model.
    best_y : float
        Current best (minimum) observed value.
    xi : float
        Exploration-exploitation trade-off, expressed as a DIMENSIONLESS
        fraction of the signal scale. The effective raw-unit margin is
        ``xi * y_scale``. Larger = more exploration.
    eps : float
        Numerical stability threshold for standard deviation.
    y_scale : float
        Scale of the target signal in RAW cost units (typically the std of
        the costs the GP was trained on). ``gp.predict`` returns mu/sigma in
        raw units (sklearn normalize_y denormalises its output), so ``xi``
        must be multiplied by this scale to be commensurate with
        ``best_y - mu``. Default 1.0 reproduces the legacy behaviour where
        xi was a raw-unit constant.
    """

    def __init__(self, gp_model, best_y: float, xi: float = 0.01,
                 eps: float = 1e-12, y_scale: float = 1.0):
        self.gp = gp_model
        self.best_y = float(best_y)
        self.xi = float(xi)
        self.eps = float(eps)
        self.y_scale = float(y_scale)

    def __call__(self, x: Union[np.ndarray, list], gp_model=None) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        gp = gp_model if gp_model is not None else self.gp
        mean, std = gp.predict(x, return_std=True)
        mu = float(mean.ravel()[0])
        sigma = float(std.ravel()[0])

        # xi is a dimensionless fraction of the signal scale; convert it to
        # raw cost units so it is commensurate with (best_y - mu).
        improvement = self.best_y - mu - self.xi * self.y_scale

        if sigma <= self.eps:
            ei = max(0.0, improvement)
        else:
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        return float(-ei)
