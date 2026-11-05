"""
Base class for Bayesian Optimization acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AcquisitionFunction(ABC):
    """Base class for BO acquisition functions."""

    @abstractmethod
    def __call__(self, x: np.ndarray, gp_model: object) -> float:
        """
        Evaluate the acquisition function at point x.

        Parameters
        ----------
        x : np.ndarray
            Point to evaluate (1D array).
        gp_model : GaussianProcessRegressor
            Fitted GP model.

        Returns
        -------
        float
            Acquisition value (negated for minimization with scipy.optimize).
        """
