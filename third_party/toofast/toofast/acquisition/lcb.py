"""Lower Confidence Bound acquisition function."""

from typing import Union

import numpy as np

from toofast.acquisition.base import AcquisitionFunction


class LowerConfidenceBound(AcquisitionFunction):
    """
    Lower Confidence Bound (LCB) for minimization.

    Selects points with low predicted mean and/or high uncertainty.

    Parameters
    ----------
    gp_model : GaussianProcessRegressor
        Fitted GP model.
    beta : float
        Exploration weight. Higher = more exploration. Typical range [1, 3].
    """

    def __init__(self, gp_model, beta: float = 2.0):
        self.gp = gp_model
        self.beta = float(beta)

    def __call__(self, x: Union[np.ndarray, list], gp_model=None) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        gp = gp_model if gp_model is not None else self.gp
        mean, std = gp.predict(x, return_std=True)
        mu = float(mean.ravel()[0])
        sigma = float(std.ravel()[0])
        return float(mu - self.beta * sigma)
