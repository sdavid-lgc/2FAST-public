"""
Vendor-neutral hardware contracts.

The platform drives two instruments through vendor SDKs that cannot be redistributed: the syringe pumps
(Cetoni Nemesys, QmixSDK) and the UV-Vis spectrometer (Avantes AvaSpec, avaspec DLL). The public package only
defines what it needs from them — the classes below — and ships simulated implementations
(``toofast.hardware.simulated``). A laboratory plugs its own drivers in by naming them (``"package.module:Class"``,
see ``toofast.hardware.drivers``); they live outside this repository, next to the vendor SDK they wrap.

The serial instruments (C3000 rinse pumps, fraction collector, Bronkhorst MFC) use documented ASCII / propar
protocols: their drivers are part of the package and run in simulation against an emulated serial port.

Method names follow the calls made by the experiment logic and the GUI (they were written against the Nemesys
API), so that a driver is a thin adapter.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class SyringePump(ABC):
    """One syringe pump (flows in µL/s, volumes in µL)."""

    #: True for simulated pumps (the GUI shows a fixed calibration status / sample level for them)
    is_simulation: bool = False

    def __init__(self, logger, inner_diameter_mm: float, max_piston_stroke_mm: float,
                 max_flow: float = 8, min_flow: float = 0):
        self.logger = logger
        self.inner_diameter_mm = inner_diameter_mm
        self.max_piston_stroke_mm = max_piston_stroke_mm
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.dye = ""
        self.color = None
        self.volume = round(math.pi * (inner_diameter_mm / 2) ** 2 * max_piston_stroke_mm / 1000, 1)   # mL

    @property
    def max_volume(self) -> float:
        """Syringe volume in µL."""
        return self.volume * 1000

    # -- flow ------------------------------------------------------------------------------------------------
    @abstractmethod
    def generate_flow(self, flow: float) -> None:
        """Dispense continuously at `flow` µL/s (negative = aspirate)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop this pump."""

    @abstractmethod
    def get_flow_is(self) -> float:
        """Actual flow (µL/s)."""

    @abstractmethod
    def is_pumping(self) -> bool:
        """True while the pump is moving."""

    # -- volume ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get_fill_level(self) -> float:
        """Current syringe content (µL)."""

    @abstractmethod
    def set_fill_level(self, fill_level: float, flow: float) -> None:
        """Move the piston until the syringe holds `fill_level` µL, at `flow` µL/s (non-blocking)."""

    @abstractmethod
    def get_dosed_volume(self) -> float:
        """Volume dispensed since the last reset (µL)."""

    def refill(self, flow: float = 500, fill_level: Optional[float] = None) -> bool:
        """Refill to `fill_level` (default: full). Drivers may block until done; returns True on success."""
        self.set_fill_level(self.max_volume if fill_level is None else fill_level, flow)
        return True

    # -- set-up and faults -----------------------------------------------------------------------------------
    def set_parameters(self) -> None:
        """Send units and syringe geometry to the device."""
        self.logger.info(f"Pump set with parameters: inner_diameter_mm={self.inner_diameter_mm}, "
                         f"max_piston_stroke_mm={self.max_piston_stroke_mm}, Syringe volume = {self.volume} mL")

    def calibration(self) -> bool:
        """Calibrate the position sensing if needed; returns the calibration status."""
        return self.is_position_sensing_initialized()

    def is_position_sensing_initialized(self) -> bool:
        return True

    def is_in_fault_state(self) -> bool:
        return False

    def clear_fault(self) -> None:
        pass


class PumpBackend(ABC):
    """Vendor layer of the pump system: opens the device bus and returns one `SyringePump` per syringe."""

    @abstractmethod
    def open(self, logger, number_of_pumps: int, inner_diameter_mm: float, max_piston_stroke_mm: float,
             max_flow: float, **options) -> List[SyringePump]:
        """Connect, enable and return the pumps (index order = pump order of the experiment).

        `options` carries driver-specific settings (e.g. ``config_file``, the device configuration folder of
        the vendor software), passed through untouched from the experiment set-up.
        """

    def stop_all(self, pumps: List[SyringePump]) -> None:
        """Stop every pump (drivers may use a single bus-level command)."""
        for pump in pumps:
            pump.stop()

    def close(self) -> None:
        pass


class SpectrometerBackend(ABC):
    """Vendor layer of the spectrometer: raw detector counts, no processing."""

    @abstractmethod
    def open(self) -> np.ndarray:
        """Connect and return the wavelength (nm) of every detector pixel."""

    @abstractmethod
    def configure(self, start_pixel: int, stop_pixel: int, integration_time_ms: float, number_average: int,
                  smoothing: int) -> None:
        """Prepare acquisitions over pixels start_pixel..stop_pixel (inclusive)."""

    @abstractmethod
    def read(self, scans: int = 1) -> np.ndarray:
        """Acquire and return the counts of the configured pixels (blocking)."""

    def identity(self) -> dict:
        """Instrument identity written in the sample files: {"model": ..., "serial": ...} (after open)."""
        return {}

    def close(self) -> None:
        pass
