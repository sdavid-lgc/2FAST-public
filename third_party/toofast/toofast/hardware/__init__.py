"""Hardware control drivers for 2FAST instruments.

Hardware modules are imported on-demand since they require
platform-specific dependencies (PyQt5, qmixsdk, propar, pyserial).

Usage:
    from toofast.hardware.pump import PumpSystem
    from toofast.hardware.spectrometer import Spectrometer
"""

__all__ = [
    "avaspec",
    "spectrometer",
    "pump",
    "fraction_collector",
    "mass_flow_controller",
    "c3000",
]
