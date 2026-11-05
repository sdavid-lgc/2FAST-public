"""
C3000 Syringe Pump Controller.

This module provides a Python interface for controlling Tricontinent C3000 syringe pumps
via serial communication. The C3000 is a precision syringe pump commonly used in
microfluidics, analytical chemistry, and automated liquid handling applications.

Communication Protocol:
    Commands follow the format: /<address><command>R\\n
    where <address> is the pump's unique identifier (0-9)

Example:
    >>> pump = C3000(port="COM4", address=1, syringe_volume=2500)
    >>> pump.connect()
    >>> pump.initialize()
    >>> pump.refill(flow=100)
    >>> pump.dispense(volume=500, flow=50)
    >>> pump.stop_pump()

References:
    Tricontinent C3000 Modular Syringe Pump User Manual
    http://www.tricontinent.com/
"""

from typing import Optional, Literal
import logging
import serial
import time
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


class C3000:
    """
    Controller for Tricontinent C3000 syringe pumps.

    This class provides an interface to control C3000 syringe pumps via serial
    communication. It supports initialization, volume dispensing, aspiration,
    flow generation, valve control, and status monitoring.

    Parameters
    ----------
    port : str
        Serial port identifier (e.g., "COM4" on Windows, "/dev/ttyUSB0" on Linux).
    address : int or str
        Pump address for daisy-chained configurations (typically 0-9).
    syringe_volume : int, optional (default=2500)
        Maximum syringe volume in microliters (µL). Common values: 1000, 2500, 5000.
    logger : logging.Logger, optional (default=None)
        Logger instance for recording pump operations. If None, operations are
        printed to console.
    increment_mode : str, optional (default="N2")
        Resolution mode for step increments:
        - "N0": 3000 steps (lower resolution, faster)
        - "N1": 12000 steps (medium resolution)
        - "N2": 24000 steps (highest resolution, recommended)
    simulation : bool, optional (default=False)
        If True, runs in simulation mode without requiring hardware connection.
        Useful for testing and development.

    Attributes
    ----------
    ser : serial.Serial or MagicMock
        Serial connection object or mock for simulation.
    port : str
        Serial port identifier.
    con_status : bool
        Connection status flag.
    init_status : bool
        Initialization status flag.
    address : str
        Pump address as string.
    max_speed : int
        Maximum speed in steps per second (default: 48000).
    max_steps : int
        Maximum number of steps for the syringe (depends on increment_mode).
    syringe_volume : int
        Syringe capacity in microliters.
    is_simulation : bool
        Simulation mode flag.
    increment_mode : str
        Current increment mode setting.
    logger : logging.Logger or None
        Logger instance for operation recording.

    """

    def __init__(
        self,
        port: str,
        address: int,
        syringe_volume: int = 2500,
        logger: Optional[object] = None,
        increment_mode: Literal["N0", "N1", "N2"] = "N2",
        simulation: bool = False
    ):
        self.ser = None
        self.port = port
        self.con_status = False
        self.init_status = False
        self.address = str(address)
        self.max_speed = 48000
        self.max_steps = 24000
        self.syringe_volume = syringe_volume
        self.is_simulation = simulation
        self.increment_mode = increment_mode
        self.logger = logger

    def connect(self) -> bool:
        """
        Establish serial connection to the pump.

        For physical hardware, creates or reuses a shared serial connection.
        For simulation mode, creates a mock serial object.

        Returns
        -------
        bool
            True if connection successful, False otherwise.

        """
        try:
            if self.is_simulation:
                self.con_status = True
                self.ser = MagicMock()
                (self.logger or logger).info("Simulated C3000 Connected")
                return self.con_status

            if not hasattr(C3000, 'shared_serial'):
                C3000.shared_serial = serial.Serial(port=self.port, timeout=2)
            self.ser = C3000.shared_serial
            self.con_status = True
            (self.logger or logger).info(f"C3000 {self.address} Connected")
        except serial.SerialException as e:
            (self.logger or logger).error(f"Serial connection error: {e}")
        return self.con_status

    def is_initialized(self) -> bool:
        """
        Check if the pump has been initialized.

        Queries the pump's initialization status using the ?19 command.

        Returns
        -------
        bool
            True if pump is initialized, False otherwise.

        Notes
        -----
        The pump must be initialized after connection before performing operations.
        Initialization calibrates the syringe position and prepares the pump for use.
        """
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        try:
            is_init_str = "/" + self.address + "?19\n"
            self.ser.write(is_init_str.encode())
            response = self.decode_response(response=self.ser.readline())
            (self.logger or logger).debug(f"Initialization status response: {response}")
            if response[2] == "1":
                return True
            else:
                return False
        except Exception as e:
            (self.logger or logger).error(f"Error checking initialization status: {e}")
            return False

    def initialize(self):
        """
        Initialize the pump if not already initialized.

        Sets the increment mode and performs initialization sequence (Y command).
        The initialization process moves the syringe to a known reference position.

        Notes
        -----
        This method checks if the pump is already initialized to avoid unnecessary
        operations. Use `force_initialize()` to reinitialize regardless of status.

        The initialization process:
        1. Sets the increment mode (resolution)
        2. Checks current initialization status
        3. Executes initialization sequence if needed
        4. Waits for completion
        """
        self.set_increment_mode(self.increment_mode)
        time.sleep(1)
        if self.is_initialized():
            (self.logger or logger).info("Pump already initialized")
            return
        init_str = "/" + self.address + "YR\n"
        self.ser.write(init_str.encode())
        self.wait_for_idle()
        (self.logger or logger).info(f"C3000 {self.address} Initialized")

    def force_initialize(self):
        """
        Force pump initialization regardless of current status.

        Performs initialization even if the pump reports it is already initialized.
        Useful when recovering from errors or ensuring a known state.

        Notes
        -----
        Use this method with caution as it will interrupt any ongoing operations.
        Prefer `initialize()` for normal startup procedures.
        """
        self.set_increment_mode(self.increment_mode)
        time.sleep(1)
        init_str = "/" + self.address + "YR\n"
        self.ser.write(init_str.encode())
        self.wait_for_idle()
        (self.logger or logger).info(f"C3000 {self.address} Force Initialized")

    def wait_for_idle(self, stop_event=None):
        """
        Block until the pump becomes idle.

        Continuously polls the pump status using the Q command until it reports
        idle state (indicated by '`' character in response).

        Parameters
        ----------
        stop_event : threading.Event, optional
            If provided and set, sends a terminate (T) command and returns
            immediately. Used for cooperative abort during long operations.

        Notes
        -----
        This is a blocking method that should be called after commands that require
        time to complete (dispensing, aspirating, initializing, etc.).

        The polling interval is 0.2 seconds to balance responsiveness with
        communication overhead.

        In simulation mode, returns immediately.
        """
        # Clear any pending responses from previous commands
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        # Wait for any command confirmation to arrive and be discarded
        time.sleep(0.15)
        self.ser.reset_input_buffer()

        while True:
            if stop_event is not None and stop_event.is_set():
                stop_str = "/" + self.address + "TR\n"
                self.ser.write(stop_str.encode())
                (self.logger or logger).info(
                    f"C3000 {self.address} wait_for_idle aborted, T sent")
                return
            # Send status query command
            is_busy_str = "/" + self.address + "Q\n"
            self.ser.write(is_busy_str.encode())
            # Wait a bit for the response to arrive
            time.sleep(0.05)
            # Read and decode response
            response = self.decode_response(self.ser.read_until())
            if self.logger:
                self.logger.debug(f"C3000 {self.address} status: {response}")
            # Check for idle status
            if len(response) > 1:
                # Second character '`' indicates idle state
                if response[1] == "`" or self.is_simulation:
                    break
            # Wait before next check
            time.sleep(0.2)

    def check_error(self) -> Optional[str]:
        """
        Query the pump status and return an error description if in error state.

        Returns None if the pump is OK, or a string describing the error.
        The C3000 status byte meanings:
          ` (0x60) = idle/OK
          @ (0x40) = busy
          Other values indicate errors (overpressure, stall, etc.)
        """
        if self.is_simulation:
            return None
        try:
            query = "/" + self.address + "Q\n"
            self.ser.write(query.encode())
            time.sleep(0.05)
            response = self.decode_response(self.ser.read_until())
            if len(response) > 1:
                status_byte = response[1]
                if status_byte == '`':
                    return None
                elif status_byte == '@':
                    return None
                else:
                    error_msg = f"C3000 pump {self.address} error (status={status_byte!r}, response={response!r})"
                    (self.logger or logger).error(error_msg)
                    return error_msg
        except Exception as e:
            error_msg = f"C3000 pump {self.address} communication error: {e}"
            (self.logger or logger).error(error_msg)
            return error_msg
        return None

    def refill(self, flow: int = 100, fill_level: Optional[int] = None):
        """
        Refill the syringe from the reservoir.

        Switches valve to inlet position and aspirates to maximum volume.

        Parameters
        ----------
        flow : int, optional (default=100)
            Flow rate in microliters per second (µL/s).
        fill_level : int, optional (default=None)
            Target fill level in steps. If None, fills to maximum capacity.
            Currently not implemented.

        Notes
        -----
        The refill sequence:
        1. Switch valve to "left" (inlet) position
        2. Calculate flow rate in steps/second
        3. Execute aspiration command to maximum volume
        4. Use `wait_for_idle()` after this method to ensure completion

        Examples
        --------
        >>> pump.refill(flow=150)  # Refill at 150 µL/s
        >>> pump.wait_for_idle()   # Wait for completion
        """
        self.switch_valve_to_position("left")
        flow_steps = int(2 * flow * self.max_steps / self.syringe_volume)
        refill_str = "/" + self.address + "V" + str(flow_steps) + "A24000R\n"
        self.ser.write(refill_str.encode())
        (self.logger or logger).info(f"C3000 {self.address} Refilled at {flow} µL/s")

    def set_fill_level(self, fill_level: int, flow: int):
        """
        Set syringe fill level (not yet implemented).

        Parameters
        ----------
        fill_level : int
            Target fill level in steps.
        flow : int
            Flow rate in µL/s.
        """
        pass

    def step_to_volume(self, steps: int) -> float:
        """
        Convert step position to volume in microliters.

        Parameters
        ----------
        steps : int
            Number of steps from zero position.

        Returns
        -------
        float
            Volume in microliters (µL).

        Notes
        -----
        The conversion depends on the syringe volume and maximum steps:
        volume = steps × syringe_volume / max_steps

        Examples
        --------
        >>> pump.step_to_volume(12000)  # Half of 24000 max steps
        1250.0  # Half of 2500 µL syringe
        """
        volume = steps * self.syringe_volume / self.max_steps
        (self.logger or logger).debug(f"Volume: {volume} µL")
        return volume

    def volume_to_step(self, volume: float) -> int:
        """
        Convert volume in microliters to step position.

        Parameters
        ----------
        volume : float
            Volume in microliters (µL).

        Returns
        -------
        int
            Number of steps corresponding to the volume.

        Notes
        -----
        The conversion depends on the syringe volume and maximum steps:
        steps = volume × max_steps / syringe_volume

        Examples
        --------
        >>> pump.volume_to_step(1250)  # Half volume
        12000  # Half of 24000 max steps for 2500 µL syringe
        """
        steps = int(volume * self.max_steps / self.syringe_volume)
        (self.logger or logger).debug(f"Steps: {steps}")
        return steps

    def set_increment_mode(self, mode: Literal["N0", "N1", "N2"]):
        """
        Set the resolution mode for pump movements.

        """
        # Validate mode
        if mode not in ["N0", "N1", "N2"]:
            raise ValueError("Invalid mode. Choose from 'N0', 'N1', 'N2'.")

        # Set increment mode attribute
        self.increment_mode = mode

        # Adjust maximum steps based on mode
        if mode == "N0":
            self.max_steps = 3000
        elif mode == "N2":
            self.max_steps = 24000

        # Prepare and send serial command
        set_increment_mode = f"/{self.address}{self.increment_mode}R\n"
        self.ser.write(set_increment_mode.encode())

    def decode_response(self, response: bytes) -> str:
        """
        Decode and clean the serial response from the pump.

        Parameters
        ----------
        response : bytes
            Raw bytes received from serial port.

        Returns
        -------
        str
            Cleaned and decoded response string.

        Notes
        -----
        The method:
        1. Strips whitespace and ETX control character (\\x03)
        2. Decodes bytes to UTF-8 string
        3. Removes leading '/' character
        4. Resets serial buffers for next communication

        The response format from pump is typically: /<address><status><data>\\x03
        """
        # Reset buffers for next communication
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Guard against empty response (timeout / lost connection)
        if not response:
            return ""

        # Decode with 'replace' so a garbled serial byte (e.g. 0xfe from line
        # noise or a dropped connection) produces U+FFFD instead of raising
        # UnicodeDecodeError and crashing the calling thread.
        decoded_response = response.strip().rstrip(b'\x03').decode('ascii', errors='replace')
        decoded_response = decoded_response.lstrip("/")  # Remove leading '/'
        return decoded_response

    def get_fill_level(self) -> str:
        """
        Query and return the current syringe fill level.

        Returns
        -------
        str
            Current position in steps as a string.

        Notes
        -----
        The method:
        1. Sends position query command (?)
        2. Receives response with current step position
        3. Displays position in both steps and microliters
        4. Returns position in steps (excluding address/status prefix)

        Examples
        --------
        >>> fill_level = pump.get_fill_level()
        Fill level: 12000 steps
        Volume: 1250.0 µL
        Fill level: 1250.0 µL
        """
        # Clear input buffer
        self.ser.reset_input_buffer()

        # Prepare and send fill level query command
        fill_level_str = f"/{self.address}?\n"
        self.ser.write(fill_level_str.encode())

        # Read response
        response = self.ser.readline()

        # Decode response
        decoded_response = self.decode_response(response)
        (self.logger or logger).debug(f"Fill level: {decoded_response[2:]} steps")
        self.step_to_volume(int(decoded_response[2:]))
        (self.logger or logger).debug(f"Fill level: {self.step_to_volume(int(decoded_response[2:]))} µL")
        return decoded_response[2:]

    def get_valve_position(self) -> str:
        """
        Query the current valve position from the pump.

        Returns
        -------
        str
            Current valve position: "left", "right", or "center".
            Returns "unknown" if the response cannot be decoded.

        Notes
        -----
        Sends the ?6 command to query valve position.
        Response format after decode: ``<address><status><data>``.
        The data portion contains the valve position indicator.
        If the pump is busy (status ``@``), retries up to 5 times.
        """
        position_map = {
            'o': "left", '2': "left",
            'i': "right", '1': "right",
            'e': "center", '3': "center",
        }

        for attempt in range(5):
            self.ser.reset_input_buffer()
            query_str = f"/{self.address}?6\n"
            self.ser.write(query_str.encode())

            response = self.ser.readline()
            if not response:
                (self.logger or logger).debug("Valve position query: no response (timeout)")
                time.sleep(0.3)
                continue

            decoded = self.decode_response(response)
            (self.logger or logger).debug(f"Valve position response: {decoded}")

            if len(decoded) >= 2 and decoded[1] == '@':
                (self.logger or logger).debug("Pump busy, waiting before retry")
                time.sleep(0.3)
                continue

            data = decoded[2:].strip().lower() if len(decoded) > 2 else ""
            for key, pos in position_map.items():
                if key in data:
                    return pos

            (self.logger or logger).warning(
                f"Could not determine valve position from response: {decoded}"
            )
            return "unknown"

        (self.logger or logger).warning("Pump stayed busy — could not read valve position")
        return "unknown"

    def verify_valve_position(self, expected_position: Literal["left", "right", "center"]) -> bool:
        """
        Verify that the valve is in the expected position.

        Parameters
        ----------
        expected_position : {"left", "right", "center"}
            The expected valve position to verify.

        Returns
        -------
        bool
            True if valve is in expected position, False otherwise.

        Notes
        -----
        This method queries the actual valve position and compares it to the
        expected position. Useful for safety checks and debugging.

        Examples
        --------
        >>> pump.switch_valve_to_position("left")
        >>> if pump.verify_valve_position("left"):
        ...     print("Valve successfully switched to left")
        """
        actual_position = self.get_valve_position()
        is_correct = actual_position == expected_position

        if is_correct:
            (self.logger or logger).info(f"C3000 {self.address} valve position verified: {actual_position}")
        else:
            (self.logger or logger).warning(
                f"C3000 {self.address} valve position mismatch! "
                f"Expected: {expected_position}, Actual: {actual_position}"
            )

        return is_correct

    def switch_valve_to_position(
        self,
        position: Literal["left", "right", "center"],
        verify: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        on_failure: Literal["raise", "warn", "ignore"] = "warn"
    ):
        """
        Switch the pump valve to the specified position with automatic retry.

        Parameters
        ----------
        position : {"left", "right", "center"}
            Target valve position:
            - "left": Input/inlet position (typically for aspiration from reservoir)
            - "right": Output/outlet position (typically for dispensing)
            - "center": Bypass/neutral position (if available)
        verify : bool, optional (default=True)
            If True, verify the valve position after switching.
        max_retries : int, optional (default=3)
            Maximum number of retry attempts if verification fails.
            The valve switch command will be resent up to this many times.
        retry_delay : float, optional (default=0.5)
            Delay in seconds between retry attempts.
            Allows extra time for the valve to settle.
        on_failure : {"raise", "warn", "ignore"}, optional (default="warn")
            Action to take if all retries fail:
            - "raise": Raise RuntimeError (stops execution)
            - "warn": Log warning and continue (prints warning message)
            - "ignore": Silently continue without verification

        Raises
        ------
        ValueError
            If position is not one of the valid options.
        RuntimeError
            If verification fails after all retries and on_failure="raise".

        Notes
        -----
        Valve positions are application-specific. Common configurations:
        - Left: Reservoir/refill
        - Right: Output/dispense
        - Center: Closed/bypass

        The retry mechanism helps handle transient communication issues or
        mechanical delays in valve actuation. Each retry attempt:
        1. Resends the valve position command
        2. Waits for valve to settle (retry_delay)
        3. Queries and verifies the actual position

        Using on_failure="warn" (default) allows experiments to continue
        even if valve verification fails, while still logging the issue
        for later investigation.

        Examples
        --------
        >>> # Default behavior: 3 retries, warn on failure
        >>> pump.switch_valve_to_position("left")

        >>> # More retries with longer delays for problematic valves
        >>> pump.switch_valve_to_position("right", max_retries=5, retry_delay=1.0)

        >>> # Strict mode: raise error if verification fails
        >>> pump.switch_valve_to_position("center", on_failure="raise")

        >>> # Fast mode: no verification
        >>> pump.switch_valve_to_position("left", verify=False)
        """
        # Validate position and get command string
        match position:
            case "left":
                switch_valve_str = f"/{self.address}OR\n"
            case "right":
                switch_valve_str = f"/{self.address}IR\n"
            case "center":
                switch_valve_str = f"/{self.address}ER\n"
            case _:
                raise ValueError("Invalid valve position. Choose from 'left', 'right', 'center'.")

        # Attempt to switch valve with retries
        for attempt in range(max_retries + 1):
            # Send valve switch command
            self.ser.write(switch_valve_str.encode())
            if attempt == 0:
                (self.logger or logger).info(f"C3000 {self.address} valve: {position}")
            else:
                (self.logger or logger).info(f"C3000 {self.address} valve: {position} (retry {attempt}/{max_retries})")

            # Wait for valve to move
            if attempt == 0:
                time.sleep(0.5)  # Initial short delay
            else:
                time.sleep(retry_delay)  # Longer delay for retries

            # Verify position if requested and not in simulation
            if verify and not self.is_simulation:
                time.sleep(0.1)  # Additional settle time
                if self.verify_valve_position(position):
                    # Success!
                    if attempt > 0:
                        (self.logger or logger).info(
                            f"C3000 {self.address} valve position verified after {attempt} retry(ies)"
                        )
                    return  # Exit successfully

                # Verification failed
                if attempt < max_retries:
                    (self.logger or logger).warning(
                        f"C3000 {self.address} valve verification failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}), retrying..."
                    )
            else:
                # No verification needed
                return

        # All retries exhausted - handle failure
        error_msg = (
            f"Failed to switch C3000 {self.address} valve to {position} position "
            f"after {max_retries + 1} attempts"
        )

        if on_failure == "raise":
            (self.logger or logger).error(error_msg)
            raise RuntimeError(error_msg)
        elif on_failure == "warn":
            (self.logger or logger).warning(f"{error_msg} - continuing anyway, please check valve position manually!")
        # on_failure == "ignore": silently continue

    def switch_valve_with_manual_fallback(
        self,
        position: Literal["left", "right", "center"],
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        """
        Switch valve with automatic retry and manual intervention fallback.

        This method attempts automatic valve switching with retries. If all
        automatic attempts fail, it prompts for manual intervention rather
        than raising an error or continuing with uncertainty.

        Parameters
        ----------
        position : {"left", "right", "center"}
            Target valve position.
        max_retries : int, optional (default=3)
            Maximum number of automatic retry attempts.
        retry_delay : float, optional (default=0.5)
            Delay between retry attempts in seconds.

        Notes
        -----
        This method is useful for critical operations where valve position
        must be correct. If automatic switching fails:
        1. User is prompted to manually position the valve
        2. User presses Enter when valve is in correct position
        3. System verifies the position
        4. Process repeats until position is verified

        This prevents experiment crashes while ensuring valve is correctly
        positioned before continuing.

        Examples
        --------
        >>> # Use in critical operations like flush_chip
        >>> pump.switch_valve_with_manual_fallback("left")
        """
        try:
            # Try automatic switching with retries
            self.switch_valve_to_position(
                position,
                verify=True,
                max_retries=max_retries,
                retry_delay=retry_delay,
                on_failure="raise"
            )
        except RuntimeError:
            # Automatic switching failed - request manual intervention
            (self.logger or logger).warning(
                f"C3000 {self.address}: Manual intervention required for valve position {position}. "
                f"Please manually position the valve to: {position}"
            )

            # Wait for manual correction and verification
            while True:
                input(f"Press Enter when valve is in '{position}' position...")

                # Verify the position
                if self.verify_valve_position(position):
                    (self.logger or logger).info(
                        f"C3000 {self.address}: Valve position manually corrected and verified"
                    )
                    break
                else:
                    (self.logger or logger).warning(
                        f"C3000 {self.address}: Valve still not in correct position. Please adjust and try again."
                    )

    def aspirate(self, volume: float):
        """
        Aspirate a specified volume into the syringe.

        Parameters
        ----------
        volume : float
            Volume to aspirate in microliters (µL).

        Notes
        -----
        This method uses a fixed speed of 1000 steps/second for aspiration.
        The P command moves to an absolute position, pulling liquid into the syringe.

        Ensure the valve is in the correct position (typically "left") before
        aspirating. Use `wait_for_idle()` after this method to ensure completion.

        Examples
        --------
        >>> pump.switch_valve_to_position("left")
        >>> pump.aspirate(500)  # Aspirate 500 µL
        >>> pump.wait_for_idle()
        """
        steps = self.volume_to_step(volume)
        asp_str = "/" + self.address + "V1000" + "P" + str(steps) + "R\n"
        (self.logger or logger).debug(f"Aspirate command: {asp_str}")
        asp_byt = asp_str.encode()
        self.ser.write(asp_byt)

    def get_top_velocity(self):
        """
        Query the current top velocity setting.

        Sends the ?2 query command to retrieve the maximum velocity configuration.

        Notes
        -----
        This method is primarily for diagnostics and configuration verification.
        The response format depends on the pump's current settings.
        """
        get_top_velocity_str = "/" + self.address + "?2\n"
        self.ser.write(get_top_velocity_str.encode())
        response = self.decode_response(response=self.ser.readline())
        (self.logger or logger).debug(f"Top velocity: {response}")

    def generate_flow(self, flow: float):
        """
        Generate continuous flow at the specified rate.

        Parameters
        ----------
        flow : float
            Flow rate in microliters per second (µL/s).

        Notes
        -----
        This command sets the pump to dispense continuously at the specified flow
        rate. The pump will continue until:
        - The syringe is empty
        - A stop command is issued
        - An error occurs

        The 'A0' in the command sets the target position to 0 (empty syringe).
        Use `stop_pump()` to halt flow before the syringe empties.

        Examples
        --------
        >>> pump.generate_flow(25)  # Start 25 µL/s continuous flow
        >>> time.sleep(10)          # Flow for 10 seconds
        >>> pump.stop_pump()        # Stop the flow
        """
        # Calculate flow rate in steps/second
        flow_steps = int(2 * flow * self.max_steps / self.syringe_volume)
        (self.logger or logger).debug(f"Flow in steps/s: {flow_steps}")
        generate_flow_str = "/" + self.address + "V" + str(flow_steps) + "A0R\n"
        (self.logger or logger).debug(f"Generate flow command: {generate_flow_str}")
        self.ser.write(generate_flow_str.encode())
        (self.logger or logger).info(f"C3000 {self.address} flow = {flow} µL/s")

    def dispense(self, volume: float, flow: float):
        """
        Dispense a specified volume at a given flow rate.

        Parameters
        ----------
        volume : float
            Volume to dispense in microliters (µL).
        flow : float
            Flow rate in microliters per second (µL/s).

        Notes
        -----
        This method calculates the required steps for both volume and flow rate,
        then sends the D (dispense) command. The pump will automatically stop
        after dispensing the specified volume.

        Ensure:
        - Syringe contains sufficient volume
        - Valve is in correct position (typically "right")
        - Use `wait_for_idle()` after to ensure completion

        Examples
        --------
        >>> pump.switch_valve_to_position("right")
        >>> pump.dispense(volume=250, flow=50)  # Dispense 250 µL at 50 µL/s
        >>> pump.wait_for_idle()                # Wait for completion
        """
        steps = self.volume_to_step(volume)
        flow_steps = int(2 * flow * self.max_steps / self.syringe_volume)
        disp_str = "/" + self.address + "V" + str(flow_steps) + "D" + str(steps) + "R\n"
        (self.logger or logger).debug(f"Dispense command: {disp_str}")
        self.ser.write(disp_str.encode())
        # Read and discard the command confirmation to avoid interference with wait_for_idle()
        time.sleep(0.05)
        if self.ser.in_waiting > 0:
            self.ser.read_until()
        (self.logger or logger).info(f"C3000 {self.address} dispense {volume} µL at {flow} µL/s")

    def stop_pump(self):
        """
        Immediately stop all pump motion.

        Sends the T (terminate) command to halt the current operation.
        This is an emergency stop that immediately terminates movement.

        Notes
        -----
        This command can be used to:
        - Stop continuous flow operations
        - Abort ongoing dispensing or aspiration
        - Emergency shutdown

        After stopping, the pump can immediately accept new commands.

        Examples
        --------
        >>> pump.generate_flow(50)  # Start continuous flow
        >>> time.sleep(5)           # Flow for 5 seconds
        >>> pump.stop_pump()        # Emergency stop
        """
        stop_pump_str = "/" + self.address + "TR\n"
        (self.logger or logger).info(f"C3000 {self.address} stopped")
        self.ser.write(stop_pump_str.encode())


def main():
    """
    Example usage demonstrating basic pump operations.

    This function shows a typical workflow:
    1. Create pump instance
    2. Connect to hardware
    3. Initialize pump
    4. Perform aspiration
    5. Refill syringe
    """
    pump_1 = C3000("COM4", 1)
    pump_1.connect()
    pump_1.initialize()
    pump_1.aspirate(100)
    pump_1.refill()


if __name__ == "__main__":
    main()
