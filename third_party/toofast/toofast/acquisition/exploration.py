"""
Exploration-aware acquisition wrapper.

Wraps any acquisition function with an adaptive exploration schedule:
small fixed boost in early iterations, decaying linearly to zero
after EXPLORATION_HORIZON BO samples.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from toofast.acquisition.base import AcquisitionFunction
from toofast.acquisition.ei import ExpectedImprovement
from toofast.acquisition.lcb import LowerConfidenceBound

EXPLORATION_HORIZON = 10


class ExplorationWrapper:
    """
    Wraps an acquisition function with adaptive exploration.

    A small, fixed exploration boost is applied in early BO iterations
    and decays linearly to zero after ``EXPLORATION_HORIZON`` samples.

    Parameters
    ----------
    base_acq : AcquisitionFunction
        The underlying acquisition function.
    iteration : int
        Current BO iteration (1-based, post-init).
    """

    def __init__(
        self,
        base_acq: AcquisitionFunction,
        iteration: int,
    ):
        self.base_acq = base_acq
        self.iteration = iteration

        horizon = max(EXPLORATION_HORIZON - 1, 1)
        self.explore_ratio = max(0.0, 1.0 - (iteration - 1) / horizon)

        self._apply_exploration_to_base()

    def _apply_exploration_to_base(self):
        """Boost exploration parameters on the base acquisition function."""
        if isinstance(self.base_acq, ExpectedImprovement):
            self.base_acq.xi += 0.03 * self.explore_ratio

        elif isinstance(self.base_acq, LowerConfidenceBound):
            self.base_acq.beta += 1.0 * self.explore_ratio

    def __call__(self, x: Union[np.ndarray, list], gp_model=None) -> float:
        return self.base_acq(x, gp_model)
