"""
Simulated instruments: the whole platform runs without hardware.

- `SimulatedPumpBackend` / `SimulatedPump`: syringe pumps with flow, piston position and dosed volume integrated
  over (virtual) time.
- `SimulatedSpectrometerBackend`: detector counts = halogen lamp × 10^(−extinction of the cell contents) + noise,
  on the 1219-pixel grid of the AvaSpec-2048 (310–1000 nm) used for the paper's 2026 sessions.
- `SimulatedSerial` + firmware emulators: the real serial drivers (C3000 rinse pumps, fraction collector) run
  unchanged against an emulated port that answers their ASCII protocol.
- `SimulatedProparInstrument`: parameter store standing in for a Bronkhorst MFC.

The pumps, valves and rinse syringes report to a shared `VirtualReactor` (``toofast.simulation``), which the
spectrometer reads: the measured spectra follow what the experiment does (mixing, stop-flow growth, washing).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, List, Optional

import numpy as np

from toofast import clock
from toofast.hardware.base import PumpBackend, SpectrometerBackend, SyringePump
from toofast.simulation.reactor import VirtualReactor

AVASPEC_2048_GRID = (310.2, 1000.1, 1219)     # nm, nm, pixels


# ── pumps ──────────────────────────────────────────────────────────────────────────────────────────────────────
class SimulatedPump(SyringePump):
    is_simulation = True

    def __init__(self, logger, index: int, reactor: Optional[VirtualReactor] = None, fill_fraction: float = 0.5,
                 **geometry):
        super().__init__(logger, **geometry)
        self.index = index
        self.reactor = reactor
        self._lock = threading.Lock()
        self._fill = fill_fraction * self.max_volume       # µL
        self._flow = 0.0                                     # µL/s, > 0 dispenses
        self._target: Optional[float] = None                 # fill level aimed at by set_fill_level
        self._dosed = 0.0
        self._t = clock.time()

    def _advance(self) -> None:
        now = clock.time()
        dt, self._t = max(0.0, now - self._t), now
        if self._flow == 0 or dt == 0:
            return
        new = self._fill - self._flow * dt
        stop_at = self._target if self._target is not None else (0.0 if self._flow > 0 else self.max_volume)
        if (self._flow > 0 and new <= stop_at) or (self._flow < 0 and new >= stop_at):
            new = stop_at
            self._set_flow(0.0)
            self._target = None
        self._dosed += max(0.0, self._fill - new)
        self._fill = new

    def _set_flow(self, flow: float) -> None:
        self._flow = float(flow)
        if self.reactor is not None:
            self.reactor.set_pump_flow(self.index, max(0.0, self._flow))

    def generate_flow(self, flow: float) -> None:
        with self._lock:
            self._advance()
            self._target = None
            self._set_flow(flow)

    def stop(self) -> None:
        with self._lock:
            self._advance()
            self._target = None
            self._set_flow(0.0)

    def get_flow_is(self) -> float:
        with self._lock:
            self._advance()
            return self._flow

    def is_pumping(self) -> bool:
        return self.get_flow_is() != 0.0

    def get_fill_level(self) -> float:
        with self._lock:
            self._advance()
            return self._fill

    def set_fill_level(self, fill_level: float, flow: float) -> None:
        with self._lock:
            self._advance()
            target = float(np.clip(fill_level, 0.0, self.max_volume))
            if target == self._fill:
                return
            self._target = target
            self._set_flow(abs(flow) if target < self._fill else -abs(flow))

    def get_dosed_volume(self) -> float:
        with self._lock:
            self._advance()
            return self._dosed


class SimulatedPumpBackend(PumpBackend):
    def __init__(self, reactor: Optional[VirtualReactor] = None):
        self.reactor = reactor

    def open(self, logger, number_of_pumps, inner_diameter_mm, max_piston_stroke_mm, max_flow, **options):
        logger.info("Simulated pumps initialized")
        return [SimulatedPump(logger, i, self.reactor, inner_diameter_mm=inner_diameter_mm,
                              max_piston_stroke_mm=max_piston_stroke_mm, max_flow=max_flow)
                for i in range(number_of_pumps)]


# ── spectrometer ───────────────────────────────────────────────────────────────────────────────────────────────
def halogen_counts(wavelengths: np.ndarray, temperature: float = 3000.0, peak_counts: float = 45000.0) -> np.ndarray:
    """Black-body lamp spectrum (counts), with a floor so that no pixel is dark."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    wl = np.asarray(wavelengths, dtype=float) * 1e-9
    radiance = 1.0 / (wl ** 5 * (np.exp(h * c / (wl * k * temperature)) - 1.0))
    return peak_counts * (0.03 + 0.97 * radiance / radiance.max())


class SimulatedSpectrometerBackend(SpectrometerBackend):
    def __init__(self, reactor: Optional[VirtualReactor] = None, grid=AVASPEC_2048_GRID, count_noise: float = 2e-4,
                 seed: int = 1):
        self.reactor = reactor
        self.grid = grid
        self.count_noise = count_noise
        self._rng = np.random.default_rng(seed)
        self._window = slice(None)
        self._acquisition_s = 0.0

    def open(self):
        self._wl = np.linspace(*self.grid)
        self._lamp = halogen_counts(self._wl)
        return self._wl

    def identity(self):
        return {"model": f"simulated ({len(self._wl)}-pixel grid)", "serial": "SIM"}

    def configure(self, start_pixel, stop_pixel, integration_time_ms, number_average, smoothing):
        self._window = slice(int(start_pixel), int(stop_pixel) + 1)
        self._acquisition_s = float(integration_time_ms) * int(number_average) / 1000.0

    def read(self, scans=1):
        clock.sleep(self._acquisition_s * max(1, int(scans)))
        wl, lamp = self._wl[self._window], self._lamp[self._window]
        extinction = self.reactor.absorbance(wl) if self.reactor is not None else np.zeros(len(wl))
        counts = lamp * 10.0 ** (-extinction)
        return counts * (1.0 + self._rng.normal(0.0, self.count_noise, size=len(wl)))


# ── serial instruments ─────────────────────────────────────────────────────────────────────────────────────────
class SimulatedSerial:
    """Minimal pyserial stand-in: each written line goes to `responder`, whose reply (if any) is queued for reading."""

    def __init__(self, responder: Callable[[str], Optional[str]], port: str = "SIM"):
        self.responder = responder
        self.port = port
        self.is_open = True
        self._rx: deque = deque()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        for line in data.decode("utf-8", errors="replace").splitlines():
            reply = self.responder(line.strip()) if line.strip() else None
            if reply is not None:
                with self._lock:
                    self._rx.append(reply.encode("utf-8") + b"\r\n")
        return len(data)

    def readline(self, *args, **kwargs) -> bytes:
        with self._lock:
            return self._rx.popleft() if self._rx else b""

    read_until = readline

    @property
    def in_waiting(self) -> int:
        return sum(len(r) for r in self._rx)

    def reset_input_buffer(self) -> None:
        with self._lock:
            self._rx.clear()

    def reset_output_buffer(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


class CollectorFirmware:
    """Answers the fraction-collector protocol (H homing, M motor status, ? position, W/V valves, #, Z, &)."""

    def __init__(self, reactor: Optional[VirtualReactor] = None):
        self.reactor = reactor
        self.position = 0

    def __call__(self, cmd: str) -> Optional[str]:
        if cmd == "H":
            return "Homing Completed"
        if cmd == "M":
            return "Motors are stopped"
        if cmd == "?":
            return f"Position: {self.position}"
        if cmd.startswith("#"):
            self.position = int(cmd[1:] or 0)
        elif cmd.startswith("W") and self.reactor is not None:      # 10-port valve: 0 = UV cell in line, 1 = isolated
            self.reactor.set_cell_in_line(cmd[1:] == "0")
        return None


class C3000Firmware:
    """Answers the Tricontinent C3000 protocol (/<addr><commands>R, queries ?, ?6, ?19, Q) for one pump address.

    Replies use the master address 0 and the idle status byte '`'. Plunger output with the valve on the chip side
    (right / centre) rinses the virtual UV cell.
    """

    VALVE_CODE = {"O": "o", "I": "i", "E": "e"}       # left, right, centre

    def __init__(self, reactor: Optional[VirtualReactor] = None, max_steps: int = 24000):
        self.reactor = reactor
        self.max_steps = max_steps
        self.position = 0
        self.valve = "O"

    def __call__(self, cmd: str) -> Optional[str]:
        if not cmd.startswith("/") or len(cmd) < 3:
            return None
        body = cmd[2:]                                  # drop "/<addr>"
        if body == "?19":
            return "/0`1"
        if body == "?6":
            return f"/0`{self.VALVE_CODE[self.valve]}"
        if body == "?":
            return f"/0`{self.position}"
        if body == "?2":
            return "/0`1000"
        if body == "Q":
            return "/0`"
        self._execute(body)
        return "/0`"

    def _execute(self, body: str) -> None:
        i, out = 0, False
        while i < len(body):
            ch = body[i]
            j = i + 1
            while j < len(body) and body[j].isdigit():
                j += 1
            arg = int(body[i + 1:j]) if j > i + 1 else None
            if ch in self.VALVE_CODE:
                self.valve = ch
            elif ch == "A" and arg is not None:
                out |= arg < self.position
                self.position = arg
            elif ch == "D" and arg is not None:
                out = True
                self.position = max(0, self.position - arg)
            elif ch == "P" and arg is not None:
                self.position = min(self.max_steps, self.position + arg)
            elif ch == "Y":
                self.position = 0
            i = j
        if out and self.valve in ("I", "E") and self.reactor is not None:
            self.reactor.rinse()


class SimulatedProparInstrument:
    """Bronkhorst propar parameter store: the measured flow (205) follows the setpoint (206)."""

    def __init__(self):
        self.params = {24: 0, 25: "N2", 45: 0, 205: 0.0, 206: 0.0}

    def writeParameter(self, number, value):
        self.params[number] = value
        if number == 206:
            self.params[205] = value
        return True

    def readParameter(self, number):
        return self.params.get(number)


# ── factory ────────────────────────────────────────────────────────────────────────────────────────────────────
class SimulatedRig:
    """One virtual reactor shared by every simulated instrument of an experiment."""

    def __init__(self, reactor: Optional[VirtualReactor] = None):
        self.reactor = reactor or VirtualReactor()
        self.pump_backend = SimulatedPumpBackend(self.reactor)
        self.spectrometer_backend = SimulatedSpectrometerBackend(self.reactor)
        self._c3000 = {}

    def collector_serial(self) -> SimulatedSerial:
        return SimulatedSerial(CollectorFirmware(self.reactor), port="SIM-collector")

    def c3000_serial(self) -> SimulatedSerial:
        """One shared port for the daisy-chained C3000 pumps; each address has its own plunger and valve."""
        def responder(cmd: str):
            addr = cmd[1:2] if cmd.startswith("/") else ""
            fw = self._c3000.setdefault(addr, C3000Firmware(self.reactor))
            return fw(cmd)
        return SimulatedSerial(responder, port="SIM-C3000")

    def mfc_instrument(self) -> SimulatedProparInstrument:
        return SimulatedProparInstrument()


__all__ = ["SimulatedPump", "SimulatedPumpBackend", "SimulatedSpectrometerBackend", "SimulatedSerial",
           "CollectorFirmware", "C3000Firmware", "SimulatedProparInstrument", "SimulatedRig", "halogen_counts",
           "AVASPEC_2048_GRID"]
