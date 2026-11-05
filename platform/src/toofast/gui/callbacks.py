"""
Callback protocol for decoupling experiment logic from the GUI.

The Experiment class communicates with the outside world through this
protocol. The Tkinter GUI implements it for interactive use; a default
headless implementation is provided for testing and CLI automation.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from toofast.data.models import Spectrum

logger = logging.getLogger(__name__)


@runtime_checkable
class ExperimentCallbacks(Protocol):
    """Protocol that any UI/controller must implement to drive an experiment."""

    def on_spectrum_acquired(self, spectrum: Spectrum) -> None:
        """Called when a new spectrum is measured."""
        ...

    def on_iteration_complete(
        self,
        iteration: int,
        flows: np.ndarray,
        costs: np.ndarray,
        gp_model: object,
        best_min: list,
        times: list,
        spectrum: Spectrum,
        predicted_flows: Optional[np.ndarray],
        final: bool,
        save_figs: bool,
    ) -> None:
        """Called after each BO iteration with updated data for plotting."""
        ...

    def on_experiment_complete(self, message: str) -> None:
        """Called when the experiment ends."""
        ...

    def should_stop(self) -> bool:
        """Return True to abort the experiment."""
        ...

    def should_pause(self) -> bool:
        """Return True to pause the experiment."""
        ...

    def validate_spectrum(self, spectrum: Spectrum) -> bool:
        """Ask user/auto-validate a spectrum. Return True to accept."""
        ...

    def validate_cost(self, cost: float) -> bool:
        """Ask user/auto-validate a cost. Return True to accept."""
        ...

    def move_to_next_well(self) -> None:
        """Advance the fraction collector to the next well."""
        ...

    def switch_valve_to_trash(self) -> None:
        """Switch collection valve to waste."""
        ...

    def switch_valve_to_collection(self) -> None:
        """Switch collection valve to collection."""
        ...

    def set_phase(self, phase: str) -> None:
        """Update the current experiment phase display."""
        ...


class HeadlessCallbacks:
    """Default callbacks for headless/test operation. Auto-accepts everything."""

    def __init__(self) -> None:
        self._stop = False

    def on_spectrum_acquired(self, spectrum: Spectrum) -> None:
        logger.debug(f"Spectrum acquired: {spectrum.meas_type}")

    def on_iteration_complete(self, iteration, flows, costs, gp_model,
                              best_min, times, spectrum, predicted_flows,
                              final, save_figs) -> None:
        logger.info(f"Iteration {iteration} complete, best cost: {best_min[-1]:.6f}")

    def on_experiment_complete(self, message: str) -> None:
        logger.info(f"Experiment complete: {message}")

    def should_stop(self) -> bool:
        return self._stop

    def should_pause(self) -> bool:
        return False

    def validate_spectrum(self, spectrum: Spectrum) -> bool:
        return True

    def validate_cost(self, cost: float) -> bool:
        return True

    def move_to_next_well(self) -> None:
        pass

    def switch_valve_to_trash(self) -> None:
        pass

    def switch_valve_to_collection(self) -> None:
        pass

    def set_phase(self, phase: str) -> None:
        logger.debug(f"Phase: {phase}")

    def request_stop(self) -> None:
        """External signal to stop the experiment."""
        self._stop = True
