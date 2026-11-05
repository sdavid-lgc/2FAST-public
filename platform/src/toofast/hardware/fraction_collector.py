"""
Fraction Collector Controller.

This module provides a Python interface for controlling a custom automated fraction collector.

Communication Protocol:
    Commands are sent as ASCII strings terminated with \\n
    Common commands:
    - H: Homing sequence
    - #<pos>: Move to well position
    - &<pos>: Switch collection valve
    - V<pos>: Switch refill/8-port valve
    - W<pos>: Switch 10-port valve
    - Z<level>: Move Z-axis
    - M: Query motor status
    - ?: Query current position
"""

from typing import Optional, Literal
import logging
import time
try:                     # pyserial: only needed for the real instrument
    import serial
    _SerialException = serial.SerialException
except ImportError:      # pragma: no cover
    serial = None
    _SerialException = OSError
from toofast import clock

logger = logging.getLogger(__name__)


class FractionCollector:
    """
    Controller for automated fraction collector

    This class provides an interface to control fraction collectors via serial
    communication. It supports well plate navigation, valve switching, Z-axis
    control, and automated sample collection.

    Parameters
    ----------
    port : str
        Serial port identifier (e.g., "COM8" on Windows, "/dev/ttyUSB0" on Linux).
    logger : logging.Logger
        Logger instance for recording fraction collector operations.
    is_simulation : bool, optional (default=False)
        If True, runs in simulation mode without requiring hardware connection.
        Useful for testing and development.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for operation recording.
    ser : serial.Serial or SimulatedSerial
        Serial connection object or mock for simulation.
    port : str
        Serial port identifier.
    con_status : bool
        Connection status flag.
    init_status : bool
        Initialization (homing) status flag.
    current_well : int or None
        Current well position number.
    filled_wells : list
        List of wells that have been filled.
    wells : list
        List of available wells.
    current_valve_position : str or None
        Current collection valve position ("Waste" or "Collection").
    current_refill_valve_position : str
        Current refill valve position ("Chip" or "Reagent").
    current_8_port_valve_position : int
        Current 8-port valve position (0 or 1).
    current_10_port_valve_position : int
        Current 10-port valve position (0 or 1).
    current_z_position : int
        Current Z-axis position (1-999, where 999 is top/safe position).
    is_simulation : bool
        Simulation mode flag.

    Notes
    -----
    The fraction collector uses a coordinate system for well plates:
    - XY positioning for well selection (horizontal movement)
    - Z-axis for vertical positioning (droplet height control)
    - Multiple valves for routing liquids to waste or collection

    Safety features:
    - Automatic Z-axis retraction before XY movement
    - Waste position default during movement
    - Position 999 on Z-axis is the safe "up" position
    """

    def __init__(
        self,
        port: str,
        logger,
        is_simulation: bool = False,
        sim_serial=None,
    ):
        self.logger = logger
        self.ser = None
        self.port = port
        self.con_status = False
        self.init_status = False
        self.current_well = None
        self.filled_wells = []
        self.wells = []
        self.current_valve_position = None
        self.current_refill_valve_position = "Chip"
        self.current_8_port_valve_position = 0
        self.current_10_port_valve_position = 0
        self.current_z_position = 999  # Start at safe position
        self.is_simulation = is_simulation
        self._sim_serial = sim_serial

    def connect(self) -> bool:
        """
        Establish serial connection to the fraction collector.

        For physical hardware, creates a serial connection.
        For simulation mode, creates a mock serial object.

        Returns
        -------
        bool
            True if connection successful, False otherwise.

        Notes
        -----
        Upon connection, the Z-axis is automatically moved to position 999
        (safe/up position) to prevent collisions during initialization.

        Examples
        --------
        >>> collector = FractionCollector(port="COM8", logger=logger)
        >>> if collector.connect():
        ...     print("Collector connected and ready")
        """
        try:
            if self.is_simulation:
                self.con_status = True
                if self._sim_serial is None:
                    from toofast.hardware.simulated import CollectorFirmware, SimulatedSerial
                    self._sim_serial = SimulatedSerial(CollectorFirmware(), port="SIM-collector")
                self.ser = self._sim_serial
                self.logger.info("Simulated Fraction Collector Connected")
                return self.con_status

            if serial is None:
                raise ImportError("the fraction collector driver needs pyserial (pip install pyserial)")
            self.ser = serial.Serial(port=self.port)
            self.con_status = True  # Verify connection is open and established
            self.logger.info("Fraction Collector Connected")
            self.move_z_axis(999)  # Move to safe position
        except _SerialException as e:
            self.logger.error(f"Serial connection error: {e}")
        return self.con_status

    def homing(self) -> bool:
        """
        Perform homing sequence to calibrate position.

        The homing sequence moves all axes to their reference positions,
        establishing a known coordinate system for subsequent movements.

        Returns
        -------
        bool
            True if homing completed successfully.

        Notes
        -----
        This method:
        1. Moves Z-axis to safe position (999)
        2. Sends homing command (H)
        3. Waits for "Homing Completed" response
        4. Sets init_status to True upon completion

        Homing must be performed after connection and before any positioning
        commands to ensure accurate well plate navigation.

        Examples
        --------
        >>> collector.connect()
        >>> collector.homing()  # Initialize coordinate system
        >>> # Now ready for positioning commands
        """
        # Force Z to safe position (999) before homing
        self.move_z_axis(999)

        homing_str = "H\n"
        homing_byt = homing_str.encode('utf-8')
        self.ser.write(homing_byt)

        while True:
            response = self.ser.readline().decode().strip()
            self.logger.info(f"Homing response: {response}")
            if response == "Homing Completed":
                self.init_status = True
                break

        return self.init_status

    def is_busy(self) -> bool:
        """
        Check if the motors are currently moving.

        Queries the hardware to determine if any motion is in progress.

        Returns
        -------
        bool
            True if motors are moving, False if stopped.

        Notes
        -----
        This method is useful for:
        - Waiting for movements to complete before next command
        - Monitoring operation status
        - Preventing command conflicts

        The hardware responds with either "Motors are moving" or "Motors are stopped".

        Examples
        --------
        >>> collector.move_to_position(5)
        >>> while collector.is_busy():
        ...     clock.sleep(0.1)  # Wait for movement to complete
        >>> # Now safe to proceed with next operation
        """
        busy_str = "M\n"
        busy_byt = busy_str.encode('utf-8')
        self.ser.write(busy_byt)

        response = self.ser.readline().decode('utf-8').strip()
        if response == "Motors are moving":
            return True
        elif response == "Motors are stopped":
            return False
        else:
            self.logger.error(f"Unexpected motor status response: {response}")
            return False

    def switch_collection_valve(self, position: int) -> str:
        """
        Switch the collection valve between waste and collection positions.

        Parameters
        ----------
        position : int
            Valve position:
            - 0: Waste position (discard liquid)
            - 1: Collection position (collect into well)

        Returns
        -------
        str
            The command string sent to the hardware.

        Notes
        -----
        The collection valve determines whether liquid flows to:
        - Waste: Used during setup, washing, or unwanted fractions
        - Collection: Used to collect desired fractions into wells

        This is a critical safety feature to prevent contamination of samples
        with unwanted material.
        """
        switch_str = "&" + str(position) + "\n"
        switch_byt = switch_str.encode('utf-8')
        self.ser.write(switch_byt)

        if position == 0:
            self.current_valve_position = "Waste"
            self.logger.info("Collection valve: Waste")
        elif position == 1:
            self.current_valve_position = "Collection"
            self.logger.info("Collection valve: Collection")

        return switch_str

    def switch_refill_valve(self, position: int):
        """
        Switch the refill valve between chip and reagent positions.

        Parameters
        ----------
        position : int
            Valve position:
            - 0: Chip position
            - 1: Reagent position

        Notes
        -----
        The refill valve controls the source of liquid for the system.
        This is typically used in microfluidic or flow chemistry setups
        where multiple input sources are available.
        """
        switch_str = "V" + str(position) + "\n"
        switch_byt = switch_str.encode('utf-8')
        self.ser.write(switch_byt)

        if position == 0:
            self.current_refill_valve_position = "Chip"
            self.logger.info("Refill valve: Chip")
        elif position == 1:
            self.current_refill_valve_position = "Reagent"
            self.logger.info("Refill valve: Reagent")

    def switch_8_port_valve(self, position: int):
        """
        Switch the 8-port valve position.

        Parameters
        ----------
        position : int
            Valve position:
            - 0: Position 0 (Chip)
            - 1: Position 1 (Reagent)

        """
        switch_str = "V" + str(position) + "\n"
        switch_byt = switch_str.encode('utf-8')
        self.ser.write(switch_byt)

        if position == 0:
            self.current_8_port_valve_position = 0
            self.current_valve_position = "Chip"
            self.logger.info("8-port valve: Chip")
        elif position == 1:
            self.current_8_port_valve_position = 1
            self.current_valve_position = "Reagent"
            self.logger.info("8-port valve: Reagent")

    def switch_10_port_valve(self, position: int):
        """
        Switch the 10-port valve position.

        Parameters
        ----------
        position : int
            Valve position:
            - 0: Position 0 (Chip)
            - 1: Position 1 (Reagent)
        """
        switch_str = "W" + str(position) + "\n"
        switch_byt = switch_str.encode('utf-8')
        self.ser.write(switch_byt)

        if position == 0:
            self.current_10_port_valve_position = 0
            self.current_valve_position = "Chip"
            self.logger.info("10-port valve: Chip")
        elif position == 1:
            self.current_10_port_valve_position = 1
            self.current_valve_position = "Reagent"
            self.logger.info("10-port valve: Reagent")

    def move_to_position(self, position: int) -> str:
        """
        Move to a specific well position on the plate.

        Parameters
        ----------
        position : int
            Well number to move to (plate-specific numbering).

        Returns
        -------
        str
            The command string sent to the hardware.
        """
        # Force Z to safe position (999) and valve to waste before moving
        if self.current_valve_position != "Waste":
            self.switch_collection_valve(0)

        if self.current_z_position != 999:
            self.move_z_axis(999)

        # Move to target well position
        move_str = "#" + str(position) + "\n"
        move_byt = move_str.encode('utf-8')
        self.ser.write(move_byt)

        # Wait for movement to complete
        while self.is_busy():
            clock.sleep(0.5)

        # Lower to collection height
        self.move_z_axis(850)
        self.current_well = position
        self.logger.info(f"Moved to well position {position}")

        return move_str

    def get_position(self):
        """
        Query and log the current position of the collector.

        Sends a position query command and logs the response.

        Notes
        -----
        The response format depends on the hardware implementation.
        Typically returns the current well number or XY coordinates.
        """
        position_str = "?\n"
        position_byt = position_str.encode('utf-8')
        self.ser.write(position_byt)
        response = self.ser.readline().decode('utf-8').strip()
        self.logger.info(f"Current position: {response}")
        return response

    def move_z_axis(self, level: int) -> str:
        """
        Move the Z-axis to a specific height level.

        Parameters
        ----------
        level : int
            Z-axis position (typically 1-999):
            - 999: Top/safe position (maximum height)
            - 850: Collection position (typical droplet height)
            - 1: Bottom position (lowest height)

        Returns
        -------
        str
            The command string sent to the hardware.

        Notes
        -----
        The Z-axis controls the vertical height of the collection nozzle
        above the well plate. Higher values raise the nozzle, lower values
        bring it closer to the wells.

        Position 999 is used as a safe position to prevent collisions during
        XY movements.

        """
        move_str = "Z" + str(level) + "\n"
        move_byt = move_str.encode('utf-8')
        self.ser.write(move_byt)
        self.logger.info(f"Moving Z-axis to {level}")
        self.current_z_position = level
        return move_str

    def move_z_to_position(self, position: Literal["Top", "Collection", "Bottom"]):
        """
        Move Z-axis to a named position.

        Parameters
        ----------
        position : {"Top", "Collection", "Bottom"}
            Named Z-axis position:
            - "Top": Safe position at maximum height (999)
            - "Collection": Standard collection height (850)
            - "Bottom": Minimum height (1)

        Notes
        -----
        This is a convenience method that maps named positions to numeric
        Z-axis levels. Use this for more readable code instead of magic numbers.

        """
        match position:
            case "Top":
                level = 999
            case "Collection":
                level = 850
            case "Bottom":
                level = 1
            case _:
                level = 999  # Default to safe position

        self.move_z_axis(level)
        self.current_z_position = level

    def open_lamp_shutter(self):
        """
        Open the lamp shutter (not implemented).

        Notes
        -----
        This method is a placeholder for systems with integrated optical
        detection or UV lamps. Implementation would send command "*1\\n".
        """
        pass
        # Placeholder for optical systems
        # open_str = "*1\n"
        # self.ser.write(open_str.encode())
        # clock.sleep(0.2)

    def close_lamp_shutter(self):
        """
        Close the lamp shutter (not implemented).

        Notes
        -----
        This method is a placeholder for systems with integrated optical
        detection or UV lamps. Implementation would send command "*0\\n".
        """
        pass
        # Placeholder for optical systems
        # close_str = "*0\n"
        # self.ser.write(close_str.encode())


def menu():
    """
    Interactive menu for manual fraction collector control.

    Provides a command-line interface for testing and manual operation
    of the fraction collector. Useful for setup, calibration, and
    troubleshooting.

    Menu Options:
        1. Homing - Perform homing sequence
        2. Switch Valve - Change collection valve position
        3. Move to Position - Navigate to specific well
        4. Get Position - Query current position
        5. is Busy - Check motor status
        6. Exit - Close the menu

    Notes
    -----
    This function is intended for testing and manual operation only.
    For automated operation, use the FractionCollector class methods
    directly in your scripts.
    """
    logging.basicConfig(level=logging.INFO)

    collector = FractionCollector("COM8", logger=logger)
    collector.connect()

    while True:
        logger.info("\n=== Fraction Collector Menu ===")
        logger.info("1. Homing")
        logger.info("2. Switch Valve")
        logger.info("3. Move to Position")
        logger.info("4. Get Position")
        logger.info("5. Check if Busy")
        logger.info("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            collector.homing()
        elif choice == "2":
            position = int(input("Enter valve position (0=Waste, 1=Collection): "))
            collector.switch_collection_valve(position)
        elif choice == "3":
            position = int(input("Enter well position number: "))
            collector.move_to_position(position)
        elif choice == "4":
            collector.get_position()
        elif choice == "5":
            busy = collector.is_busy()
            logger.info(f"Motors are {'moving' if busy else 'stopped'}")
        elif choice == "6":
            logger.info("Exiting...")
            break
        else:
            logger.warning("Invalid choice. Please try again.")


if __name__ == "__main__":
    menu()
