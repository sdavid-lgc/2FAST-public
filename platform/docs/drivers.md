# Laboratory drivers for the vendor instruments

`toofast` does not ship drivers for the syringe pumps or the spectrometer: they wrap the vendors' SDKs (Cetoni QmixSDK
for the Nemesys pumps, Avantes AvaSpec library for the spectrometer), which cannot be redistributed. A laboratory
writes two small adapter classes in its own package, installed next to the SDK, and names them with
`"package.module:Class"`:

- `TOOFAST_PUMP_DRIVER` / `Experiment(..., pump_driver=...)` → a `toofast.hardware.base.PumpBackend`
- `TOOFAST_SPECTROMETER_DRIVER` / `Experiment(..., spectrometer_driver=...)` → a `toofast.hardware.base.SpectrometerBackend`

Both classes are instantiated without arguments. Without them the platform only runs in simulation; asking for the
real instrument raises `DriverNotConfigured` with this explanation.

## Pumps
```python
from toofast.hardware.base import PumpBackend, SyringePump

class MyPump(SyringePump):
    """One syringe. Flows in µL/s, volumes in µL."""
    def __init__(self, logger, device, **geometry):
        super().__init__(logger, **geometry)          # inner_diameter_mm, max_piston_stroke_mm, max_flow
        self.device = device                          # SDK object of this pump

    def generate_flow(self, flow): ...                # dispense continuously (negative = aspirate)
    def stop(self): ...
    def get_flow_is(self): ...                        # actual flow
    def is_pumping(self): ...
    def get_fill_level(self): ...
    def set_fill_level(self, fill_level, flow): ...   # move to a fill level, non-blocking
    def get_dosed_volume(self): ...
    # optional, with defaults in SyringePump:
    # set_parameters (units, syringe geometry), calibration, is_position_sensing_initialized,
    # is_in_fault_state, clear_fault, refill


class MyPumpBackend(PumpBackend):
    def open(self, logger, number_of_pumps, inner_diameter_mm, max_piston_stroke_mm, max_flow, **options):
        # options["config_file"]: device configuration folder of the vendor software for <n> pumps, given by the GUI
        # as $TOOFAST_DEVICE_CONFIG_DIR/<n>syringes (default <project>/Configurations/<n>syringes)
        bus = ...                                     # open the SDK bus, enable the pumps
        return [MyPump(logger, bus.pump(i), inner_diameter_mm=inner_diameter_mm,
                       max_piston_stroke_mm=max_piston_stroke_mm, max_flow=max_flow)
                for i in range(number_of_pumps)]

    def stop_all(self, pumps):                        # optional: one bus-level stop
        ...
```
The pump order is the reagent order of the experiment (pump *i* ↔ reagent *i* in the GUI).

## Spectrometer
```python
from toofast.hardware.base import SpectrometerBackend

class MySpectrometerBackend(SpectrometerBackend):
    def open(self):
        """Connect; return the wavelength (nm) of every detector pixel."""
    def configure(self, start_pixel, stop_pixel, integration_time_ms, number_average, smoothing):
        """Prepare acquisitions over start_pixel..stop_pixel (inclusive)."""
    def read(self, scans=1):
        """Acquire (blocking) and return the raw counts of the configured pixels, in pixel order."""
```
`toofast.hardware.spectrometer.Spectrometer` does everything else (pixel window from the wavelength range, reference,
extinction = −log10(T / T_ref), averaged measurements).

## Checking a driver without the instrument
The simulated backends (`toofast.hardware.simulated.SimulatedPumpBackend`, `SimulatedSpectrometerBackend`) implement
the same contracts and can be loaded by name, e.g.
`TOOFAST_PUMP_DRIVER=toofast.hardware.simulated:SimulatedPumpBackend`.
