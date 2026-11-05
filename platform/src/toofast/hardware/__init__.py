"""Instrument layer of the platform.

- ``base``        vendor-neutral contracts (`SyringePump`, `PumpBackend`, `SpectrometerBackend`)
- ``pump``        `PumpSystem` (multi-syringe flows, flow-set generation), any pump backend
- ``spectrometer`` `Spectrometer` (reference / extinction spectra), any spectrometer backend
- ``c3000``, ``fraction_collector``, ``mass_flow_controller``  serial instruments (pyserial / propar, optional)
- ``simulated``   simulated pumps, spectrometer and serial firmwares sharing a virtual reactor
- ``drivers``     loading of the laboratory drivers of the vendor instruments (not part of this package)
- ``factory``     assembly of an experiment's instruments, real or simulated
"""

__all__ = ["base", "pump", "spectrometer", "c3000", "fraction_collector", "mass_flow_controller", "simulated",
           "drivers", "factory"]
