"""
Abstract base classes for hardware devices.

All hardware drivers must implement these interfaces. This allows
simulation, testing, and alternative implementations without changing
the experiment logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class PumpInterface(ABC):
    """Interface for a single syringe pump."""

    @abstractmethod
    def generate_flow(self, flow: float) -> None:
        """Start pumping at the given flow rate (uL/s)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop this pump."""

    @abstractmethod
    def get_flow(self) -> float:
        """Return current flow rate."""

    @abstractmethod
    def get_fill_level(self) -> float:
        """Return current syringe fill level in uL."""

    @abstractmethod
    def refill(self, flow: float = 500, fill_level: Optional[float] = None) -> bool:
        """Refill syringe to fill_level at the given flow rate."""

    @abstractmethod
    def is_pumping(self) -> bool:
        """Return True if pump is currently dispensing."""


class PumpSystemInterface(ABC):
    """Interface for a multi-pump system."""

    @abstractmethod
    def set_flows(self, flows: np.ndarray) -> None:
        """Set flow rates for all pumps."""

    @abstractmethod
    def get_flows(self) -> np.ndarray:
        """Get current flow rates from all pumps."""

    @abstractmethod
    def stop_all_pumps(self) -> None:
        """Stop all pumps."""

    @abstractmethod
    def get_fill_levels(self) -> np.ndarray:
        """Get syringe fill levels for all pumps."""

    @abstractmethod
    def write_flows(self, experiment_directory: str) -> None:
        """Write current flows to CSV."""

    @abstractmethod
    def generate_random_flows(
        self,
        bounds: Optional[np.ndarray] = None,
        total_var_flow: Optional[float] = None,
        fixed_pumps: Optional[np.ndarray] = None,
        fixed_flows: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a random flow set summing to max flow."""

    @abstractmethod
    def generate_random_flows_sets(
        self,
        n_samples: int,
        bounds: Optional[np.ndarray] = None,
        total_var_flow: Optional[float] = None,
        fixed_pumps: Optional[np.ndarray] = None,
        fixed_flows: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate multiple random flow sets."""

    @abstractmethod
    def generate_lhs_flows_sets(self, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube sampled flow sets."""


class SpectrometerInterface(ABC):
    """Interface for a UV-Vis spectrometer."""

    wavelengths: np.ndarray
    reference: object  # Spectrum or None

    @abstractmethod
    def measure_reference(self, scans: int = 1) -> object:
        """Measure and store a reference spectrum."""

    @abstractmethod
    def measure_spectrum(self, scans: int = 1, meas_type: str = "measurement") -> object:
        """Measure and return a spectrum (requires reference)."""

    @abstractmethod
    def update_parameters(self, **kwargs) -> None:
        """Update spectrometer acquisition parameters."""

    @abstractmethod
    def find_index_for_wavelength(self, target_wavelength: float) -> int:
        """Find pixel index for a given wavelength."""


class FractionCollectorInterface(ABC):
    """Interface for a fraction collector."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""

    @abstractmethod
    def homing(self) -> bool:
        """Perform homing sequence."""

    @abstractmethod
    def move_to_position(self, position) -> None:
        """Move to a well position."""

    @abstractmethod
    def switch_collection_valve(self, position: int) -> None:
        """Switch valve: 0=waste, 1=collection."""

    @abstractmethod
    def switch_8_port_valve(self, position: int) -> None:
        """Switch 8-port valve."""

    @abstractmethod
    def switch_10_port_valve(self, position: int) -> None:
        """Switch 10-port valve."""

    @abstractmethod
    def move_z_axis(self, level: int) -> None:
        """Move Z axis to a height level."""

    @abstractmethod
    def open_lamp_shutter(self) -> None:
        """Open lamp shutter."""

    @abstractmethod
    def close_lamp_shutter(self) -> None:
        """Close lamp shutter."""


class MassFlowControllerInterface(ABC):
    """Interface for a mass flow controller."""

    @abstractmethod
    def set_flow(self, flow: float) -> None:
        """Set flow rate setpoint."""

    @abstractmethod
    def measure_flow(self) -> float:
        """Read current measured flow rate."""

    @abstractmethod
    def close(self) -> None:
        """Close connection."""


class C3000Interface(ABC):
    """Interface for a Tricontinent C3000 syringe pump."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""

    @abstractmethod
    def refill(self, flow: float = 150) -> None:
        """Refill syringe."""

    @abstractmethod
    def dispense(self, volume: float, flow: float) -> None:
        """Dispense a specific volume at a given flow rate."""

    @abstractmethod
    def generate_flow(self, flow: float) -> None:
        """Start continuous flow."""

    @abstractmethod
    def stop_pump(self) -> None:
        """Stop the pump."""

    @abstractmethod
    def switch_valve_to_position(self, position: str) -> None:
        """Switch valve to named position (left/center/right)."""

    @abstractmethod
    def wait_for_idle(self) -> None:
        """Block until pump operation completes."""
