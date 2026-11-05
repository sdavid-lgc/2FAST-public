"""
Thread-safe hand-off of GUI work to the Tk main thread.

Tkinter is not thread-safe: calling any widget method — including ``widget.after`` — from a worker thread can crash
the Tcl interpreter (native access violation), which the experiment's worker threads used to do for every spectrum.
Worker threads now only put callables in a queue; the main thread drains it every `period_ms`, in order.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading

logger = logging.getLogger(__name__)


class MainThreadDispatcher:
    def __init__(self, widget, period_ms: int = 10):
        self._widget = widget
        self._period = int(period_ms)
        self._queue: "queue.Queue" = queue.Queue()
        self._stopped = False
        self._schedule()

    def call_soon(self, func, *args) -> None:
        """Run `func(*args)` on the main thread: now if called from it, else at the next drain (FIFO)."""
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            self._queue.put((func, args))

    def stop(self) -> None:
        self._stopped = True

    def _schedule(self) -> None:
        if self._stopped:
            return
        try:
            self._widget.after(self._period, self._drain)
        except Exception:          # widget destroyed: nothing left to run on
            self._stopped = True

    def _drain(self) -> None:
        while True:
            try:
                func, args = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args)
            except Exception:      # same reporting as an exception in a Tk after() callback
                report = getattr(self._widget, "report_callback_exception", None)
                if report is not None:
                    report(*sys.exc_info())
                else:
                    logger.exception("GUI callback failed")
        self._schedule()
