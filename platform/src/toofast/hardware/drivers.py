"""
Laboratory drivers for the vendor instruments.

The public package has no driver for the Cetoni Nemesys pumps or the Avantes spectrometer: their SDKs are
proprietary. A laboratory provides them as classes implementing ``toofast.hardware.base.PumpBackend`` and
``SpectrometerBackend``, in its own package (installed next to the vendor SDK), and names them:

- in the environment: ``TOOFAST_PUMP_DRIVER=mylab_drivers.nemesys:NemesysBackend``,
  ``TOOFAST_SPECTROMETER_DRIVER=mylab_drivers.avantes:AvantesBackend``;
- or explicitly: ``Experiment(..., pump_driver="...", spectrometer_driver="...")``.

The class is instantiated without arguments. See ``docs/drivers.md`` for the contract and a skeleton.
"""
from __future__ import annotations

import importlib
import os
from typing import Optional

from toofast.hardware.base import PumpBackend, SpectrometerBackend

PUMP_DRIVER_ENV = "TOOFAST_PUMP_DRIVER"
SPECTROMETER_DRIVER_ENV = "TOOFAST_SPECTROMETER_DRIVER"


class DriverNotConfigured(RuntimeError):
    pass


def load_class(spec: str):
    """Import ``"package.module:Class"``."""
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"driver must be given as 'package.module:Class', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def _backend(spec: Optional[str], env: str, base: type, what: str):
    spec = spec or os.environ.get(env, "")
    if not spec:
        raise DriverNotConfigured(
            f"No {what} driver configured. The vendor SDK drivers are not part of this package: set {env} to "
            f"'package.module:Class' (a {base.__name__} implementation installed with the vendor SDK), or run in "
            f"simulation mode.")
    backend = load_class(spec)()
    if not isinstance(backend, base):
        raise TypeError(f"{spec} must implement toofast.hardware.base.{base.__name__}")
    return backend


def pump_backend(spec: Optional[str] = None) -> PumpBackend:
    return _backend(spec, PUMP_DRIVER_ENV, PumpBackend, "pump")


def spectrometer_backend(spec: Optional[str] = None) -> SpectrometerBackend:
    return _backend(spec, SPECTROMETER_DRIVER_ENV, SpectrometerBackend, "spectrometer")
