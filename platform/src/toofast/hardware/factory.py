"""
Builds the instruments of an experiment: simulated (one shared virtual reactor) or real (laboratory drivers for
the vendor instruments, package drivers for the serial ones).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from toofast.hardware import drivers
from toofast.hardware.c3000 import C3000
from toofast.hardware.fraction_collector import FractionCollector
from toofast.hardware.mass_flow_controller import FlowController
from toofast.hardware.pump import PumpSystem
from toofast.hardware.simulated import SimulatedRig
from toofast.hardware.spectrometer import Spectrometer


@dataclass
class Hardware:
    spectro: Spectrometer
    pump_syst: PumpSystem
    collector: FractionCollector
    rig: Optional[SimulatedRig] = None          # simulation only


def create_hardware(logger, number_of_pumps, inner_diameter_mm, max_piston_stroke_mm, total_flow,
                    is_simulation=False, collector_port="", config_file=None, pump_driver=None,
                    spectrometer_driver=None, rig=None) -> Hardware:
    """Spectrometer, reagent pumps and fraction collector (connected)."""
    if is_simulation:
        rig = rig or SimulatedRig()
        pump_backend, spectro_backend = rig.pump_backend, rig.spectrometer_backend
    else:
        rig = None
        pump_backend = drivers.pump_backend(pump_driver)
        spectro_backend = drivers.spectrometer_backend(spectrometer_driver)

    spectro = Spectrometer(logger, spectro_backend, is_simulation=is_simulation)
    pumps = pump_backend.open(logger, number_of_pumps, inner_diameter_mm=inner_diameter_mm,
                              max_piston_stroke_mm=max_piston_stroke_mm, max_flow=total_flow,
                              config_file=config_file)
    pump_syst = PumpSystem(logger, pumps, max_flow=total_flow, backend=pump_backend)
    collector = FractionCollector(collector_port, logger=logger, is_simulation=is_simulation,
                                  sim_serial=rig.collector_serial() if rig else None)
    collector.connect()
    return Hardware(spectro, pump_syst, collector, rig)


def create_c3000(port, address, logger, is_simulation=False, rig=None, syringe_volume=2500) -> C3000:
    """One C3000 rinse pump (connected). Simulated pumps of the same rig share one emulated port."""
    sim_serial = None
    if is_simulation and rig is not None:
        if getattr(rig, "_c3000_serial", None) is None:
            rig._c3000_serial = rig.c3000_serial()
        sim_serial = rig._c3000_serial
    pump = C3000(port, address=address, syringe_volume=syringe_volume, simulation=is_simulation, logger=logger,
                 sim_serial=sim_serial)
    pump.connect()
    return pump


def create_mfc(port, logger, is_simulation=False, rig=None) -> FlowController:
    return FlowController(port, logger=logger, is_simulation=is_simulation,
                          sim_instrument=rig.mfc_instrument() if (is_simulation and rig) else None)
