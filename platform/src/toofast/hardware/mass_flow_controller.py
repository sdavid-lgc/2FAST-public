"""
Mass Flow Controller Interface.

This module provides a Python interface for controlling mass flow controllers (MFCs)
using the Bronkhorst propar protocol.

The controller supports:
- Serial communication via the propar protocol (Bronkhorst instruments)
- Flow rate setpoint control
- Real-time flow measurement
- Fluid type configuration
- Unit configuration
- Simulation mode for testing without hardware


Communication Protocol:
    Uses the Bronkhorst propar protocol via RS232/RS485 serial communication.
    Default settings: address=0x80, baudrate=38400

    Key Parameters:
    - Parameter 24: Fluid type selection
    - Parameter 25: Fluid name (read-only)
    - Parameter 45: Unit selection
    - Parameter 205: Measured flow value
    - Parameter 206: Flow setpoint

Example:
    >>> mfc = FlowController(port="COM9")
    >>> mfc.set_fluid(fluid=0)  # Set to nitrogen
    >>> print(mfc.read_fluid_name())
    >>> mfc.set_flow(5.0)  # Set flow to 5.0 µL/s
    >>> measured = mfc.measure_flow()
    >>> print(f"Current flow: {measured} µL/s")

References:
    Bronkhorst High-Tech B.V. propar Manual
    https://www.bronkhorst.com/
    Digital Communication Protocol for Flow Instruments
"""

import logging
from typing import Optional, Union
import time
from toofast import clock

try:                     # Bronkhorst propar (pip install bronkhorst-propar): only needed for the real MFC
    import propar
except ImportError:      # pragma: no cover
    propar = None

logger = logging.getLogger(__name__)


class FlowController:
    """
    Controller for Bronkhorst mass flow controllers.

    This class provides an interface to control mass flow controllers using
    the Bronkhorst propar protocol. It supports flow setpoint control,
    measurement, fluid configuration, and unit selection.

    Parameters
    ----------
    port : str
        Serial port identifier (e.g., "COM9" on Windows, "/dev/ttyUSB0" on Linux).
    address : int, optional (default=0x80)
        Instrument address for multi-drop configurations. Default 0x80 for
        single instrument.
    baudrate : int, optional (default=38400)
        Serial communication baud rate. Default 38400 for propar protocol.
    logger : logging.Logger, optional (default=None)
        Logger instance for recording operations. If None, operations are not logged.
    is_simulation : bool, optional (default=False)
        If True, runs in simulation mode without requiring hardware connection.
        Useful for testing and development.

    Attributes
    ----------
    instrument : propar.instrument or SimulatedProparInstrument
        Propar instrument object or mock for simulation.
    logger : logging.Logger or None
        Logger instance for operation recording.
    is_simulation : bool
        Simulation mode flag.
    max_flow : float
        Maximum allowed flow rate in µL/s (default: 20.0).

    Notes
    -----
    The mass flow controller measures and controls gas flow rates with high
    precision. The actual flow range depends on the instrument's full-scale
    capacity and calibration.

    For accurate control:
    - Select appropriate fluid type (affects density and calibration)
    - Verify correct units are configured
    - Allow settling time after setpoint changes
    - Monitor measured vs. setpoint for control stability

    Examples
    --------
    Basic usage for nitrogen flow control:

    >>> mfc = FlowController(port="COM9")
    >>> mfc.set_fluid(0)  # Nitrogen
    >>> print(f"Fluid: {mfc.read_fluid_name()}")
    >>> mfc.set_flow(10.0)  # 10 µL/s
    >>> clock.sleep(2)  # Allow settling
    >>> measured = mfc.measure_flow()
    >>> print(f"Setpoint: 10.0, Measured: {measured:.2f} µL/s")

    Simulation mode for testing:

    >>> mfc = FlowController(port="COM9", is_simulation=True)
    >>> mfc.set_flow(5.0)  # No hardware required
    >>> # Returns simulated measurement
    """

    def __init__(
        self,
        port: str,
        address: int = 0x80,
        baudrate: int = 38400,
        logger: Optional[object] = None,
        is_simulation: bool = False,
        sim_instrument=None,
    ):
        self.logger = logger
        self.is_simulation = is_simulation
        self.max_flow = 20.0  # Maximum flow in µL/s

        if is_simulation:
            if sim_instrument is None:
                from toofast.hardware.simulated import SimulatedProparInstrument
                sim_instrument = SimulatedProparInstrument()
            self.instrument = sim_instrument
            if logger:
                logger.info("Simulated Mass Flow Controller initialized")
        else:
            if propar is None:
                raise ImportError("the Bronkhorst MFC needs the 'propar' package (pip install bronkhorst-propar)")
            self.instrument = propar.instrument(port, address=address, baudrate=baudrate)
            if logger:
                logger.info(f"Mass Flow Controller connected on {port}")

        # Verify parameters: fluid, unit, flow

    def set_fluid(self, fluid: int):
        """
        Set the fluid type for accurate flow control.

        Parameters
        ----------
        fluid : int
            Fluid type identifier. Common values:
            - 0: Nitrogen (N₂)
            - 1: Argon (Ar)
            - 2: Helium (He)
            - 3: Oxygen (O₂)
            - 4: Carbon Dioxide (CO₂)
            - (Refer to instrument manual for complete list)

        Notes
        -----
        The fluid type affects density compensation and flow calculations.
        Setting the correct fluid type is essential for accurate flow control.

        The controller uses fluid-specific calibration factors stored in firmware.
        Changing the fluid type updates the internal conversion factors.

        Examples
        --------
        >>> mfc.set_fluid(0)  # Set to nitrogen
        >>> print(mfc.read_fluid_name())
        'Nitrogen'
        """
        self.instrument.writeParameter(24, fluid)
        if self.logger:
            self.logger.info(f"Fluid type set to {fluid}")

    def read_fluid(self) -> int:
        """
        Read the current fluid type identifier.

        Returns
        -------
        int
            Fluid type identifier currently configured.

        Examples
        --------
        >>> fluid_id = mfc.read_fluid()
        >>> print(f"Current fluid ID: {fluid_id}")
        """
        return self.instrument.readParameter(24)

    def read_fluid_name(self) -> str:
        """
        Read the human-readable fluid name.

        Returns
        -------
        str
            Name of the currently configured fluid (e.g., "Nitrogen", "Argon").

        Notes
        -----
        This parameter is read-only and automatically corresponds to the
        fluid type set via `set_fluid()`.

        Examples
        --------
        >>> name = mfc.read_fluid_name()
        >>> print(f"Current fluid: {name}")
        """
        return self.instrument.readParameter(25)

    def read_unit(self) -> int:
        """
        Read the current flow unit configuration.

        Returns
        -------
        int
            Unit identifier for flow measurements.

        Notes
        -----
        Common unit identifiers:
        - Flow units: mL/min, L/min, µL/s, etc.
        Refer to instrument manual for complete unit code list.
        """
        # Bug fix: changed from writeParameter to readParameter
        return self.instrument.readParameter(45)

    def set_unit(self, unit: int):
        """
        Set the flow unit for measurements and setpoints.

        Parameters
        ----------
        unit : int
            Unit identifier code. Refer to instrument manual for valid codes.

        Notes
        -----
        Changing the unit affects both measured values and setpoint interpretation.
        Ensure consistent units throughout your application.

        This method is currently a placeholder and needs implementation.
        """
        # Not yet implemented
        self.instrument.writeParameter(45, unit)
        if self.logger:
            self.logger.info(f"Flow unit set to {unit}")

    def set_flow(self, flow: float):
        """
        Set the flow rate setpoint.

        Parameters
        ----------
        flow : float
            Desired flow rate in microliters per second (µL/s).
            Must be between 0 and max_flow (default: 20.0 µL/s).

        Raises
        ------
        ValueError
            If flow is outside the valid range [0, max_flow].

        Notes
        -----
        The controller will adjust the valve position to achieve the requested
        flow rate. Allow 1-2 seconds for the flow to stabilize after setting
        a new setpoint.

        Actual achievable flow depends on:
        - Inlet pressure
        - Downstream restrictions
        - Fluid properties
        - Instrument full-scale range

        The setpoint is written to parameter 206 (flow setpoint register).
        """
        if flow < 0 or flow > self.max_flow:
            raise ValueError(f"Flow must be between 0 and {self.max_flow} µL/s")

        self.instrument.writeParameter(206, flow)      # simulation: SimulatedProparInstrument
        if self.logger:
            self.logger.info(f"Flow setpoint set to {flow:.2f} µL/s")

    def measure_flow(self) -> float:
        """
        Measure the current actual flow rate.

        Returns
        -------
        float
            Measured flow rate in µL/s (or configured units).

        Notes
        -----
        This reads the real-time flow measurement from parameter 205.
        The measured value may differ from the setpoint due to:
        - Transient response (flow settling)
        - Pressure variations
        - Control loop limitations
        - Physical constraints

        For accurate measurements:
        - Allow sufficient settling time after setpoint changes
        - Ensure stable inlet pressure
        - Verify no downstream blockages
        """
        return self.instrument.readParameter(205)      # simulation: SimulatedProparInstrument (follows the setpoint)

    def close(self):
        """
        Close the connection to the mass flow controller.

        Performs cleanup operations before disconnecting. In the current
        implementation, this is handled automatically by the propar library.

        Notes
        -----
        This method is called automatically when the object is deleted,
        but can be called explicitly to ensure proper cleanup.
        """
        if self.logger:
            self.logger.info("Mass Flow Controller closed")

    def __del__(self):
        """
        Destructor called when object is garbage collected.

        Ensures proper cleanup of resources.
        """
        self.close()


def main():
    """
    Example usage demonstrating mass flow controller operations.

    This function shows a typical workflow:
    1. Connect to the controller
    2. Configure fluid type
    3. Set flow rate
    4. Monitor measured flow
    5. Adjust flow dynamically
    """
    # Initialize controller
    mfc = FlowController(port='COM9')

    # Set to zero flow initially
    mfc.set_flow(0.0)
    clock.sleep(1)

    # Read fluid configuration
    logger.info(f"Fluid ID: {mfc.read_fluid()}")
    logger.info(f"Fluid Name: {mfc.read_fluid_name()}")

    # Set flow and monitor
    mfc.set_flow(10.0)
    clock.sleep(2)
    measured = mfc.measure_flow()
    logger.info(f"Setpoint: 10.0 µL/s, Measured: {measured:.2f} µL/s")

    # Ramp down
    mfc.set_flow(0.0)
    clock.sleep(1)

    logger.info("Mass flow controller test complete")


if __name__ == "__main__":
    main()
