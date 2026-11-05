"""Thompson Sampling acquisition function.

Design note
-----------
The *correct* way to use Thompson Sampling with a GP is:

    candidates = <dense feasible grid>
    sample = gp.sample_y(candidates, n_samples=1, random_state=seed).ravel()
    next_x = candidates[sample.argmin()]

This draws a *single coherent function* from the GP posterior and takes its
minimum.  ``Experiment.predict_next`` uses exactly this path when
``acq_func == 'TS'`` — it never passes ``ThompsonSampling`` to
``differential_evolution``, which would be incorrect because DE requires a
deterministic objective.

``ThompsonSampling.__call__`` (independent marginal draws, one per x) is kept
for analysis/notebook use only (e.g. visualising the acquisition landscape).
Do NOT pass an instance of this class to ``differential_evolution``.
"""

from typing import Optional, Union

import numpy as np

from toofast.acquisition.base import AcquisitionFunction


class ThompsonSampling(AcquisitionFunction):
    """
    Thompson Sampling for Gaussian Processes — analysis / notebook use only.

    Each call draws an **independent** sample from N(mu(x), sigma(x)).  This
    is useful for visualisation but is statistically incorrect for
    optimisation (it ignores posterior covariance).  The correct path —
    ``gp.sample_y`` over a candidate grid — is implemented in
    ``Experiment._predict_next_ts``.

    Parameters
    ----------
    gp_model : GaussianProcessRegressor
        Fitted GP model.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, gp_model, random_state: Optional[int] = None):
        self.gp = gp_model
        self.rng = np.random.default_rng(random_state)

    def __call__(self, x: Union[np.ndarray, list], gp_model=None) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        gp = gp_model if gp_model is not None else self.gp
        mean, std = gp.predict(x, return_std=True)
        mu = float(mean.ravel()[0])
        sigma = float(std.ravel()[0])
        return float(self.rng.normal(mu, sigma))
