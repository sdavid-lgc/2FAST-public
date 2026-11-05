"""
Time source of the platform.

Every wait, timeout and timestamp of the experiment logic and of the hardware drivers goes through this module
instead of calling ``time`` / ``datetime`` directly. On the instrument the clock is the wall clock (``RealClock``,
the default): behaviour is unchanged. In simulation a ``VirtualClock`` lets a whole stop-flow BO campaign
(minutes of waits per sample) run in seconds, while keeping consistent timestamps — the growth-termination rule
and the recorded growth times only see virtual seconds.

Usage::

    from toofast import clock
    clock.sleep(0.5); t0 = clock.time(); stamp = clock.now().isoformat()

    with clock.use(clock.VirtualClock(speed=0.0)):
        ...   # simulated run, waits are instantaneous
"""
from __future__ import annotations

import datetime as _dt
import threading
import time as _time
from contextlib import contextmanager


class RealClock:
    """Wall clock (the instrument)."""

    def time(self) -> float:
        return _time.time()

    def perf_counter(self) -> float:
        return _time.perf_counter()

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)

    def now(self) -> _dt.datetime:
        return _dt.datetime.now()


class VirtualClock:
    """Simulated time: ``sleep(dt)`` advances the clock by ``dt`` and really waits ``speed * dt``.

    speed = 0 → waits are instantaneous (tests, batch simulations); speed = 1 → real time; 0.05 → 20× faster.
    The clock is shared by all threads (the experiment runs its chip wash in a thread while it measures); a
    small real yield keeps the threads interleaved when speed = 0.

    owner: if set (a thread), only that thread advances the clock; the waits of other threads only yield. The
    timeline then does not depend on thread scheduling, so a seeded simulation is reproducible. Leave it None when
    the experiment does not run in a single known thread (e.g. the GUI's worker threads).
    """

    def __init__(self, speed: float = 0.0, start: _dt.datetime | None = None, owner: threading.Thread | None = None):
        self.speed = float(speed)
        self.owner = owner
        self._t = 0.0
        self._origin = start or _dt.datetime(2026, 1, 1, 9, 0, 0)
        self._epoch = self._origin.timestamp()
        self._lock = threading.Lock()

    def time(self) -> float:
        return self._epoch + self._t

    def perf_counter(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if self.owner is not None and threading.current_thread() is not self.owner:
            _time.sleep(0.0005)
            return
        with self._lock:
            self._t += seconds
        _time.sleep(self.speed * seconds if self.speed > 0 else 0.0005)

    def now(self) -> _dt.datetime:
        return self._origin + _dt.timedelta(seconds=self._t)

    def elapsed(self) -> float:
        """Virtual seconds since the clock was created."""
        return self._t


_clock = RealClock()


def get_clock():
    return _clock


def set_clock(new_clock) -> None:
    global _clock
    _clock = new_clock


@contextmanager
def use(new_clock):
    """Temporarily replace the clock (restored on exit)."""
    old = get_clock()
    set_clock(new_clock)
    try:
        yield new_clock
    finally:
        set_clock(old)


def time() -> float:
    return _clock.time()


def perf_counter() -> float:
    return _clock.perf_counter()


def sleep(seconds: float) -> None:
    _clock.sleep(seconds)


def now() -> _dt.datetime:
    return _clock.now()
