import time
import os
import shutil
import threading
import queue
import ctypes
import gc
import tkinter as tk
import sys
from tkinter import ttk
import logging
import math
from tkinter import messagebox
from tkinter import filedialog


# ---------------------------------------------------------------------------
# Safety net: prevent process crashes when tkinter Variables / Images get
# garbage-collected from a non-main thread. Without this, Variable.__del__ /
# Image.__del__ try to call into Tcl from the wrong thread, which raises
# RuntimeError("main thread is not in main loop") and can ultimately trigger
# Tcl_AsyncDelete and a fatal process exit (-2147483645 on Windows).
#
# We swallow the RuntimeError silently — the Tcl resources will be cleaned up
# when the interpreter is torn down. The proper fix is still to keep tkinter
# objects on the main thread (see the _call_on_main / _gui_call helpers), but
# this guard turns any remaining slip-up into a harmless ignored exception.
# ---------------------------------------------------------------------------
def _install_tk_threadsafe_destructors():
    for cls in (tk.Variable, tk.Image):
        original_del = cls.__del__

        def _safe_del(self, _orig=original_del):
            try:
                _orig(self)
            except RuntimeError:
                pass
            except Exception:
                pass

        cls.__del__ = _safe_del


_install_tk_threadsafe_destructors()

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if __package__ in (None, ""):   # run as a script (python .../toofast/gui/app.py): make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toofast.core.experiment import Experiment
from toofast.visualization.figure_plotting import plot_fig
from toofast.data.manager import DataManager
from toofast.gui.callbacks import ExperimentCallbacks
from toofast.gui.dispatch import MainThreadDispatcher

logger = logging.getLogger(__name__)

# Icon path — resolve relative to the project root (3 levels up from this file)
_ICON_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, '2FAST.ico')
_ICON_PATH = os.path.normpath(_ICON_PATH)

# ---------------------------------------------------------------------------
# Modern theme constants
# ---------------------------------------------------------------------------
COLORS = {
    'bg': '#f0f2f5',
    'frame_bg': '#ffffff',
    'header_bg': '#1e293b',
    'header_fg': '#ffffff',
    'primary': '#3b82f6',
    'accent': '#3b82f6',
    'accent_hover': '#2563eb',
    'success': '#16a34a',
    'warning': '#ea580c',
    'danger': '#dc2626',
    'text': '#1e293b',
    'text_secondary': '#64748b',
    'text_light': '#94a3b8',
    'border': '#e2e8f0',
    'entry_bg': '#ffffff',
    'button_bg': '#f8fafc',
    'section_bg': '#ffffff',
    'card_bg': '#ffffff',
    'card_border': '#e2e8f0',
    'teal': '#0d9488',
    'teal_light': '#ccfbf1',
}

FONTS = {
    'title': ('Segoe UI', 12, 'bold'),
    'section': ('Segoe UI', 11, 'bold'),
    'subsection': ('Segoe UI', 10, 'bold'),
    'body': ('Segoe UI', 9),
    'small': ('Segoe UI', 8),
    'mono': ('Consolas', 10),
    'mono_sm': ('Consolas', 9),
    'header': ('Segoe UI', 10, 'bold'),
    'header_sm': ('Segoe UI', 9),
    'value': ('Consolas', 11),
}


def _configure_ttk_style():
    """Set up modern ttk styles."""
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    style.configure('TNotebook', background=COLORS['bg'], borderwidth=0)
    style.configure('TNotebook.Tab',
                    font=('Segoe UI', 10), padding=[14, 6],
                    background=COLORS['bg'], foreground=COLORS['text_secondary'])
    style.map('TNotebook.Tab',
              background=[('selected', COLORS['card_bg'])],
              foreground=[('selected', COLORS['primary'])],
              expand=[('selected', [0, 0, 0, 2])])

    style.configure('Teal.Horizontal.TProgressbar',
                    troughcolor=COLORS['border'], background=COLORS['teal'],
                    thickness=8)
    style.configure('Danger.Horizontal.TProgressbar',
                    troughcolor=COLORS['border'], background=COLORS['danger'],
                    thickness=8)

    style.configure('TFrame', background=COLORS['bg'])
    style.configure('Card.TFrame', background=COLORS['card_bg'],
                    relief='solid', borderwidth=1)
    style.configure('TLabel', background=COLORS['bg'],
                    foreground=COLORS['text'], font=FONTS['body'])
    style.configure('TCheckbutton', background=COLORS['bg'], font=FONTS['body'])
    style.configure('TRadiobutton', background=COLORS['bg'], font=FONTS['body'])

    return style


def _style_button(btn, style='default'):
    """Apply sober button styling — dark text on tinted backgrounds for readability."""
    base = {
        'font': ('Segoe UI', 9),
        'relief': 'solid',
        'borderwidth': 1,
        'padx': 12,
        'pady': 4,
        'cursor': 'hand2',
    }
    if style == 'primary':
        base.update({'bg': '#dbeafe', 'fg': '#1e40af',
                     'activebackground': '#bfdbfe', 'activeforeground': '#1e3a8a',
                     'font': ('Segoe UI', 9, 'bold')})
    elif style == 'success':
        base.update({'bg': '#dcfce7', 'fg': '#166534',
                     'activebackground': '#bbf7d0', 'activeforeground': '#14532d',
                     'font': ('Segoe UI', 9, 'bold')})
    elif style == 'danger':
        base.update({'bg': '#fee2e2', 'fg': '#991b1b',
                     'activebackground': '#fecaca', 'activeforeground': '#7f1d1d',
                     'font': ('Segoe UI', 9, 'bold')})
    elif style == 'warning':
        base.update({'bg': '#fef3c7', 'fg': '#92400e',
                     'activebackground': '#fde68a', 'activeforeground': '#78350f',
                     'font': ('Segoe UI', 9, 'bold')})
    elif style == 'teal':
        base.update({'bg': '#ccfbf1', 'fg': '#115e59',
                     'activebackground': '#99f6e4', 'activeforeground': '#134e4a',
                     'font': ('Segoe UI', 9, 'bold')})
    else:
        base.update({'bg': '#f1f5f9', 'fg': '#334155',
                     'activebackground': '#e2e8f0', 'activeforeground': '#1e293b'})
    btn.config(**base)


def _style_entry(entry):
    """Apply modern entry styling."""
    entry.config(
        font=FONTS['body'],
        relief='solid',
        borderwidth=1,
        bg=COLORS['entry_bg'],
        fg=COLORS['text'],
        insertbackground=COLORS['text'],
        highlightthickness=1,
        highlightcolor=COLORS['primary'],
        highlightbackground=COLORS['border'],
    )


def _style_label(label, style='body'):
    label.config(font=FONTS.get(style, FONTS['body']), fg=COLORS['text'])


def _make_card(parent, **grid_kw):
    """Create a card-style frame with subtle border."""
    card = tk.Frame(parent, bg=COLORS['card_bg'], relief='solid',
                    bd=1, highlightthickness=0)
    card.config(highlightbackground=COLORS['card_border'])
    if grid_kw:
        card.grid(**grid_kw)
    return card


class TkCallbacks:
    """
    Bridges the ExperimentCallbacks protocol to the Tkinter GUI.

    This class implements the ExperimentCallbacks interface by delegating
    to the ConfigurationWindow (Tkinter) methods. It decouples tooFAST.py's
    experiment logic from direct Tkinter widget access.

    All GUI calls are routed through the main thread to prevent
    "main loop is not in the main thread" crashes.
    """

    def __init__(self, window: 'ConfigurationWindow') -> None:
        self.window = window

    # ------------------------------------------------------------------
    # Thread-safe helpers
    # ------------------------------------------------------------------
    def _run_on_main(self, func, *args):
        """Schedule *func* on the Tk main thread (fire-and-forget)."""
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            self.window.call_soon(func, *args)

    def _call_on_main(self, func, *args):
        """
        Call *func* on the main thread and **wait** for the result.
        Use this for calls that need a return value (e.g. messageboxes).
        """
        if threading.current_thread() is threading.main_thread():
            return func(*args)
        result_queue = queue.Queue()

        def _wrapper():
            try:
                result_queue.put(('ok', func(*args)))
            except Exception as exc:
                result_queue.put(('err', exc))

        self.window.call_soon(_wrapper)
        tag, value = result_queue.get()  # blocks until main thread finishes
        if tag == 'err':
            raise value
        return value

    # ------------------------------------------------------------------
    # ExperimentCallbacks implementation
    # ------------------------------------------------------------------
    def on_spectrum_acquired(self, spectrum) -> None:
        self._run_on_main(self.window.plot_spectrum, spectrum)

    def on_iteration_complete(
        self, iteration, flows, costs, gp_model,
        best_min, times, spectrum, predicted_flows,
        final, save_figs,
    ) -> None:
        # Use _call_on_main (synchronous) instead of _run_on_main so the BO
        # worker thread waits until the figure has been rendered on the main
        # thread before proceeding. This avoids races where matplotlib /
        # tkinter PhotoImage objects get garbage collected from the wrong
        # thread when the user interacts with the BO figure window.
        self._call_on_main(
            self.window.plot_fig,
            iteration, flows, costs, gp_model, best_min, times,
            spectrum, predicted_flows, final, save_figs,
        )

    def on_experiment_complete(self, message: str) -> None:
        self._run_on_main(self.window.stop_exp, message)

    def should_stop(self) -> bool:
        return self.window.abort_exp_flag or self.window.pump_stop_alert_shown

    def should_pause(self) -> bool:
        return self.window.pause_exp_flag

    def validate_spectrum(self, spectrum) -> bool:
        if self._call_on_main(self.window.manual_spectrum_validation.get):
            return self._call_on_main(self.window.ask_for_spectrum_validation)
        return True

    def validate_cost(self, cost: float) -> bool:
        return self._call_on_main(self.window.ask_for_cost_validation, cost)

    def move_to_next_well(self) -> None:
        self._run_on_main(self.window.move_to_next_well)

    def switch_valve_to_trash(self) -> None:
        self._run_on_main(self.window.switch_valve_to_trash)

    def switch_valve_to_collection(self) -> None:
        self._run_on_main(self.window.switch_valve_to_collection)

    def set_phase(self, phase: str) -> None:
        self._run_on_main(self.window.set_phase, phase)

plt.ioff()
#
# Class to print the logger infos to the console to keep track of things
class PrintHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        # Use print() for all levels so the message goes through sys.stdout /
        # sys.stderr which are reconfigured to UTF-8 in main().  Creating a
        # fresh StreamHandler() each call (old code) bypassed that and used the
        # system encoding (cp1252 on Windows), causing UnicodeEncodeError.
        try:
            print(msg, file=sys.stderr if record.levelno >= logging.WARNING else sys.stdout)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace').decode('ascii'),
                  file=sys.stderr if record.levelno >= logging.WARNING else sys.stdout)

def setup_global_exception_logging(logger):
    """Configure global exception handling to route uncaught errors to the logger."""

    # Save the original exception handlers
    original_excepthook = sys.excepthook
    original_threading_excepthook = getattr(threading, 'excepthook', None)

    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """Global handler for all uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            original_excepthook(exc_type, exc_value, exc_traceback)
            return

        import traceback
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Specialized messages based on error type
        if "main thread is not in the main loop" in str(exc_value):
            logger.error(f"TKINTER THREAD ERROR: {str(exc_value)}")
            logger.debug(f"Full details: {error_msg}")
        elif "RuntimeError" in str(exc_type):
            logger.error(f"RUNTIME ERROR: {str(exc_value)}")
            logger.debug(f"Full details: {error_msg}")
        else:
            logger.error(f"UNHANDLED EXCEPTION: {error_msg}")

    def threading_exception_handler(args):
        """Handler for exceptions in threads"""
        exc_type, exc_value, exc_traceback, thread = args
        import traceback
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error(f"THREAD EXCEPTION '{thread.name}': {error_msg}")

    # Installation
    sys.excepthook = global_exception_handler
    if hasattr(threading, 'excepthook'):
        threading.excepthook = threading_exception_handler

    logger.info("Global exception handler installed")

    return original_excepthook, original_threading_excepthook


class ConfigurationWindow(tk.Toplevel):
    def call_soon(self, func, *args):
        """Run func(*args) on the Tk main thread (safe from any thread)."""
        self.dispatcher.call_soon(func, *args)

    def __init__(self, master, exp, experiment_path, experiment_name, logger):
        super().__init__(master)
        self.master.withdraw()  # Hide main window
        self.lift()
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        if os.path.exists(_ICON_PATH):
            self.iconbitmap(_ICON_PATH)

        self.title("2FAST - Experiment Controls")
        self.exp = exp
        # worker threads hand GUI work to the main thread through this queue (Tkinter is not thread-safe)
        self.dispatcher = MainThreadDispatcher(self)
        self.exp.app = self
        self.exp.callbacks = TkCallbacks(self)  # Wire callbacks for GUI decoupling
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.logger = logger
        self.original_hooks = setup_global_exception_logging(self.logger)
        self.after_ids = []
        self.figure_window = None

        self.pump_stop_alert_shown = False

        # Periodic GC pump on the Tk main thread. Forces matplotlib/Tk objects
        # (FigureCanvasTkAgg PhotoImages, etc.) to be finalized on the main
        # thread instead of being collected later from a worker thread, which
        # would trigger "Tcl_AsyncDelete: async handler deleted by the wrong
        # thread" panics on process exit.
        self._schedule_main_thread_gc()
        self._prime_abort = False
        self.steady_state = False
        self._current_phase = "Idle"
        self.abort_exp_flag = False
        self.pause_exp_flag = False
        self.next_sample_pressed = False

        # ── Apply modern theme ────────────────────────────────────────
        self.configure(bg=COLORS['bg'])
        _configure_ttk_style()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.minsize(1100, 700)
        self.state('zoomed')

        # ══════════════════════════════════════════════════════════════
        # HEADER — dark bar with device status + emergency stop
        # ══════════════════════════════════════════════════════════════
        self._header = tk.Frame(self, bg=COLORS['header_bg'], height=52)
        self._header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._header.grid_columnconfigure(2, weight=1)
        self._header.grid_propagate(False)

        # App title
        tk.Label(self._header, text="2FAST", font=('Segoe UI', 13, 'bold'),
                 bg=COLORS['header_bg'], fg='#ffffff').grid(
            row=0, column=0, padx=(12, 20), sticky="w")

        # Device status indicators
        self._device_status = tk.Frame(self._header, bg=COLORS['header_bg'])
        self._device_status.grid(row=0, column=2, sticky="ew")

        self._header_pump_labels = []
        self._header_fill_vals = [0.0] * self.exp.number_of_pumps
        self._header_flow_vals = [0.0] * self.exp.number_of_pumps
        for i in range(self.exp.number_of_pumps):
            lbl = tk.Label(self._device_status, font=FONTS['header_sm'],
                           bg=COLORS['header_bg'], fg='#94a3b8',
                           text=f"P{i+1}: -- \u00b5L | -- \u00b5L/s")
            lbl.pack(side="left", padx=(0, 12))
            self._header_pump_labels.append(lbl)

        # State badge
        self._header_state_label = tk.Label(
            self._header, text=" Idle ", font=FONTS['header_sm'],
            bg='#334155', fg='#94a3b8', padx=8, pady=2)
        self._header_state_label.grid(row=0, column=3, padx=6)
        self._header_collection_label = tk.Label(
            self._header, text="", font=FONTS['header_sm'],
            bg=COLORS['header_bg'], fg='#94a3b8')
        self._header_collection_label.grid(row=0, column=4, padx=(0, 8))

        # Emergency stop
        self._stop_all_btn = tk.Button(
            self._header, text="\u26A0  STOP ALL", font=('Segoe UI', 10, 'bold'),
            bg=COLORS['danger'], fg='white',
            activebackground='#b91c1c', activeforeground='white',
            relief='flat', bd=0, padx=14, pady=4, cursor='hand2',
            command=self.stop_flow,
        )
        self._stop_all_btn.grid(row=0, column=5, padx=(4, 10), sticky="e")

        # ══════════════════════════════════════════════════════════════
        # LEFT PANEL — persistent experiment controls
        # ══════════════════════════════════════════════════════════════
        self._exp_panel = tk.Frame(self, bg=COLORS['bg'], width=520)
        self._exp_panel.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=(4, 2))
        self._exp_panel.grid_propagate(False)
        self._exp_panel.grid_rowconfigure(0, weight=1)
        self._exp_panel.grid_columnconfigure(0, weight=1)

        self._exp_canvas = tk.Canvas(self._exp_panel, bg=COLORS['bg'],
                                     highlightthickness=0, width=510)
        self._exp_scrollbar = ttk.Scrollbar(self._exp_panel, orient="vertical",
                                            command=self._exp_canvas.yview)
        self._exp_canvas.configure(yscrollcommand=self._exp_scrollbar.set)
        self._exp_scrollbar.grid(row=0, column=1, sticky="ns")
        self._exp_canvas.grid(row=0, column=0, sticky="nsew")
        self._exp_inner = tk.Frame(self._exp_canvas, bg=COLORS['bg'])
        self._exp_canvas.create_window((0, 0), window=self._exp_inner,
                                       anchor="nw", tags="inner")
        self._exp_inner.bind("<Configure>",
            lambda e: self._exp_canvas.configure(scrollregion=self._exp_canvas.bbox("all")))
        self._exp_canvas.bind("<Configure>",
            lambda e: self._exp_canvas.itemconfigure("inner", width=e.width))

        def _on_exp_mousewheel(event):
            self._exp_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_exp_scroll(event):
            self._exp_canvas.bind_all("<MouseWheel>", _on_exp_mousewheel)

        def _unbind_exp_scroll(event):
            self._exp_canvas.unbind_all("<MouseWheel>")

        self._exp_canvas.bind("<Enter>", _bind_exp_scroll)
        self._exp_canvas.bind("<Leave>", _unbind_exp_scroll)

        # ══════════════════════════════════════════════════════════════
        # NOTEBOOK — tabbed content area (right side)
        # ══════════════════════════════════════════════════════════════
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=1, sticky="nsew", padx=6, pady=(4, 2))

        self.tab_pumps = tk.Frame(self._notebook, bg=COLORS['bg'])
        self.tab_spectro = tk.Frame(self._notebook, bg=COLORS['bg'])
        self.tab_collector = tk.Frame(self._notebook, bg=COLORS['bg'])
        self.tab_wash = tk.Frame(self._notebook, bg=COLORS['bg'])

        self._notebook.add(self.tab_pumps, text="  \u2699  Pumps  ")
        self._notebook.add(self.tab_spectro, text="  \u2261  Spectrometer  ")
        self._notebook.add(self.tab_collector, text="  \u25CE  Collector  ")
        self._notebook.add(self.tab_wash, text="  \u2B6E  C3000 / MFC  ")

        for tab in (self.tab_pumps, self.tab_collector, self.tab_wash,
                    self.tab_spectro):
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

        # ══════════════════════════════════════════════════════════════
        # FOOTER — navigation bar
        # ══════════════════════════════════════════════════════════════
        self._footer = tk.Frame(self, bg=COLORS['card_bg'], relief='solid', bd=1,
                                highlightbackground=COLORS['border'],
                                highlightthickness=1)
        self._footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

        self.back_button = tk.Button(self._footer, text="Back", command=self.back)
        _style_button(self.back_button)
        self.back_button.pack(side="right", padx=6, pady=6)

        # ══════════════════════════════════════════════════════════════
        # PUMP TAB
        # ══════════════════════════════════════════════════════════════
        self.pump_frame = tk.Frame(self.tab_pumps, bg=COLORS['bg'])
        self.pump_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # ── Toolbar row: calibrate, faults, prime ─────────────────────
        toolbar = tk.Frame(self.pump_frame, bg=COLORS['bg'])
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.calibrate_button = tk.Button(toolbar, text="Calibrate Pumps", command=self.calibrate_pumps)
        _style_button(self.calibrate_button, 'primary')
        self.calibrate_button.pack(side="left", padx=(0, 6))

        self.calibration_status_label = tk.Label(toolbar, text="Not calibrated",
                                                 fg=COLORS['danger'], font=FONTS['body'],
                                                 bg=COLORS['bg'])
        self.calibration_status_label.pack(side="left", padx=(0, 16))
        self.update_calibration_status()

        self.clear_fault_button = tk.Button(toolbar, text="Clear Faults", command=self.clear_pump_faults)
        _style_button(self.clear_fault_button)
        self.clear_fault_button.pack(side="left", padx=(0, 6))

        self.pump_fault_label = tk.Label(toolbar, text="No faults",
                                         fg=COLORS['success'], font=FONTS['body'],
                                         bg=COLORS['bg'])
        self.pump_fault_label.pack(side="left", padx=(0, 16))
        self.check_for_pump_faults()

        self.prime_pump_cycle_button = tk.Button(toolbar, text="Prime All Pumps", command=self.prime_pump_cycle)
        _style_button(self.prime_pump_cycle_button, 'teal')
        self.prime_pump_cycle_button.pack(side="left", padx=(0, 6))

        # ── Pump cards grid ──────────────────────────────────────────
        self.syringe_frame = tk.Frame(self.pump_frame, bg=COLORS['bg'])
        self.syringe_frame.grid(row=1, column=0, sticky="ew")

        self.syringe_level_label = []
        self.current_flow_label = []
        self.pump_flow_entry = []
        self.pump_refill_entry = []
        self.selected_reagents = []
        self.reagent_concentrations = []
        self._pump_fill_bars = []

        for i in range(self.exp.number_of_pumps):
            self.syringe_frame.grid_columnconfigure(i, weight=1)
            card = _make_card(self.syringe_frame)
            card.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")

            # Pump header
            tk.Label(card, text=f"Syringe {i+1}", font=FONTS['subsection'],
                     bg=COLORS['card_bg']).grid(row=0, column=0, columnspan=2,
                                                 padx=10, pady=(8, 2), sticky="w")

            # Flow display
            flow_lbl = tk.Label(card, text="0.00 µL/s", font=FONTS['value'],
                                fg=COLORS['primary'], bg=COLORS['card_bg'])
            flow_lbl.grid(row=1, column=0, columnspan=2, padx=10, sticky="w")
            self.current_flow_label.append(flow_lbl)

            # Fill level bar
            fill_bar = ttk.Progressbar(card, orient="horizontal", length=130,
                                       mode="determinate",
                                       style='Teal.Horizontal.TProgressbar')
            fill_bar.grid(row=2, column=0, columnspan=2, padx=10, pady=(4, 0), sticky="ew")
            self._pump_fill_bars.append(fill_bar)

            # Fill level label
            fill_lbl = tk.Label(card, text="-- µL", font=FONTS['mono_sm'],
                                fg=COLORS['text_secondary'], bg=COLORS['card_bg'])
            fill_lbl.grid(row=3, column=0, columnspan=2, padx=10, sticky="w")
            self.syringe_level_label.append(fill_lbl)

            # Target entry
            tk.Label(card, text="Target flow (µL/s):", font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=4, column=0, padx=(10, 2), pady=(6, 0), sticky="w")
            target_entry = tk.Entry(card, width=8)
            _style_entry(target_entry)
            target_entry.grid(row=4, column=1, padx=(0, 10), pady=(6, 0), sticky="w")
            self.pump_flow_entry.append(target_entry)

            # Refill entry
            tk.Label(card, text="Refill to (µL):", font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=5, column=0, padx=(10, 2), pady=2, sticky="w")
            refill_entry = tk.Entry(card, width=8)
            _style_entry(refill_entry)
            refill_entry.grid(row=5, column=1, padx=(0, 10), pady=2, sticky="w")
            self.pump_refill_entry.append(refill_entry)

            # Reagent selector
            _default_reagents = ["AgNO3", "PVP", "Acetic Acid", "Ascorbic Acid", "NaBH4"]
            _default_concs = ["1.8", "1.2", "1256", "1.8", "0.036"]
            _reagent_default = _default_reagents[i] if i < len(_default_reagents) else ""
            _conc_default = _default_concs[i] if i < len(_default_concs) else "1.8"
            selected_reagent = tk.StringVar(value=_reagent_default)
            self.selected_reagents.append(selected_reagent)
            reagent_menu = tk.OptionMenu(card, selected_reagent,
                                         "AgNO3", "PVP", "Acetic Acid", "Ascorbic Acid",
                                         "NaBH4", "AgNO3 + PVP")
            reagent_menu.config(font=FONTS['small'], bg=COLORS['card_bg'],
                                relief='solid', bd=1, highlightthickness=0)
            reagent_menu.grid(row=6, column=0, columnspan=2, padx=10, pady=(2, 2), sticky="ew")

            reagent_concentration = tk.StringVar(value=_conc_default)
            self.reagent_concentrations.append(reagent_concentration)
            conc_frame = tk.Frame(card, bg=COLORS['card_bg'])
            conc_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="ew")
            tk.Label(conc_frame, text="mM:", font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
            conc_entry = tk.Entry(conc_frame, textvariable=reagent_concentration, width=6)
            _style_entry(conc_entry)
            conc_entry.pack(side="left", padx=4)

        self.get_syringe_levels()
        self.get_current_flows()

        # ── Flow control buttons + refill flow ───────────────────────
        flow_ctrl = tk.Frame(self.pump_frame, bg=COLORS['bg'])
        flow_ctrl.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.start_flow_button = tk.Button(flow_ctrl, text="\u25B6  Start Flow", command=self.start_flow)
        _style_button(self.start_flow_button, 'success')
        self.start_flow_button.pack(side="left", padx=(0, 6))

        self.start_random_flow_button = tk.Button(flow_ctrl, text="Random Flow", command=self.start_random_flow)
        _style_button(self.start_random_flow_button, 'primary')
        self.start_random_flow_button.pack(side="left", padx=(0, 6))

        self.stop_flow_button = tk.Button(flow_ctrl, text="\u25A0  Stop Flow", command=self.stop_flow)
        _style_button(self.stop_flow_button, 'danger')
        self.stop_flow_button.pack(side="left", padx=(0, 16))

        self.refill_pumps_button = tk.Button(flow_ctrl, text="Refill", command=self.refill)
        _style_button(self.refill_pumps_button, 'teal')
        self.refill_pumps_button.pack(side="left", padx=(0, 6))

        tk.Label(flow_ctrl, text="Refill flow [µL/s]:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['bg']).pack(side="left")
        self.refill_flow_entry = tk.Entry(flow_ctrl, width=6)
        _style_entry(self.refill_flow_entry)
        self.refill_flow_entry.pack(side="left", padx=4)

        # ── Flow mode selector (wash/purge vs synthesis) ────────────
        mode_frame = tk.Frame(self.pump_frame, bg=COLORS['bg'])
        mode_frame.grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.wash_mode_var = tk.IntVar(value=0)
        self._wash_mode_cb = tk.Checkbutton(
            mode_frame, text="Wash / Purge mode",
            variable=self.wash_mode_var, bg=COLORS['bg'],
            font=FONTS['small'], command=self._update_flow_mode_label)
        self._wash_mode_cb.pack(side="left", padx=(0, 8))
        self._flow_mode_label = tk.Label(
            mode_frame, text="  SYNTHESIS  ", font=FONTS['small'],
            fg="white", bg=COLORS['success'], padx=6, pady=1, relief="groove")
        self._flow_mode_label.pack(side="left")

        # ── Valve controls ────────────────────────────────────────
        self.refill_valve_frame = tk.Frame(self.pump_frame, bg=COLORS['bg'])
        self.refill_valve_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self.switch_8_port_valve_button = tk.Button(self.refill_valve_frame,
                                                    text="Switch 8-port Valve",
                                                    command=self.switch_8_port_valve)
        _style_button(self.switch_8_port_valve_button)
        self.switch_8_port_valve_button.pack(side="left", padx=(0, 6))

        self.valve_8_position_label = tk.Label(self.refill_valve_frame, text="8-port:",
                                               font=FONTS['small'], fg=COLORS['text_secondary'],
                                               bg=COLORS['bg'])
        self.valve_8_position_label.pack(side="left")
        self.valve_8_current_position_label = tk.Label(self.refill_valve_frame, text="--",
                                                       font=FONTS['mono_sm'], bg=COLORS['bg'])
        self.valve_8_current_position_label.pack(side="left", padx=(0, 16))
        self.update_8_valve_position()

        self.switch_10_port_valve_button = tk.Button(self.refill_valve_frame,
                                                     text="Switch 10-port Valve",
                                                     command=self.switch_10_port_valve)
        _style_button(self.switch_10_port_valve_button)
        self.switch_10_port_valve_button.pack(side="left", padx=(0, 6))

        self.valve_10_position_label = tk.Label(self.refill_valve_frame, text="10-port:",
                                                font=FONTS['small'], fg=COLORS['text_secondary'],
                                                bg=COLORS['bg'])
        self.valve_10_position_label.pack(side="left")
        self.valve_10_current_position_label = tk.Label(self.refill_valve_frame, text="--",
                                                        font=FONTS['mono_sm'], bg=COLORS['bg'])
        self.valve_10_current_position_label.pack(side="left", padx=(0, 16))
        self.update_10_valve_position()

        self._valve_mode_label = tk.Label(self.refill_valve_frame, text="--",
                                          font=FONTS['mono_sm'], fg=COLORS['primary'],
                                          bg=COLORS['bg'])
        self._valve_mode_label.pack(side="left")

        # ══════════════════════════════════════════════════════════════
        # C3000 / MFC TAB
        # ══════════════════════════════════════════════════════════════

        self.mfc_frame = _make_card(self.tab_wash)
        self.mfc_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.mfc_frame_label = tk.Label(self.mfc_frame, text="Mass Flow Controller",
                                        font=FONTS['section'], bg=COLORS['card_bg'])
        self.mfc_frame_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        self.activate_mfc_label = tk.Label(self.mfc_frame, text="Activate Flow Controller",
                                               font=FONTS['body'], bg=COLORS['card_bg'])
        self.activate_mfc_label.grid(row=1, column=0, padx=10, sticky="w")

        self.activate_mfc_var = tk.IntVar()
        self.activate_mfc_checkbox = tk.Checkbutton(self.mfc_frame, variable=self.activate_mfc_var,
                                                    command=self.toggle_mfc_options, bg=COLORS['card_bg'])
        self.activate_mfc_checkbox.grid(row=1, column=1)

        self.mfc_current_flow_label = tk.Label(self.mfc_frame, text="Current Flow",
                                                   font=FONTS['body'], bg=COLORS['card_bg'])
        self.mfc_current_flow_label.grid(row=2, column=0, padx=10, sticky="w")
        self.mfc_flow_label = tk.Label(self.mfc_frame, text="", font=FONTS['value'],
                                       fg=COLORS['primary'], bg=COLORS['card_bg'])
        self.mfc_set_flow_label = tk.Label(self.mfc_frame, text="Target Flow [\u00b5L/s]",
                                           font=FONTS['body'], bg=COLORS['card_bg'])
        self.mfc_set_flow_entry = tk.Entry(self.mfc_frame)
        _style_entry(self.mfc_set_flow_entry)
        self.mfc_set_flow_button = tk.Button(self.mfc_frame, text="Start Flow", command=self.set_mfc_flow)
        _style_button(self.mfc_set_flow_button, 'success')
        self.mfc_flow_bar = ttk.Progressbar(self.mfc_frame, orient="horizontal", mode="determinate",
                                            length=150, style='Teal.Horizontal.TProgressbar')
        self.mfc_flow_bar.grid(row=3, column=0, columnspan=2, pady=5, padx=10)
        self.mfc_stop_button = tk.Button(self.mfc_frame, text="Stop Flow", command=self.stop_mfc_flow)
        _style_button(self.mfc_stop_button, 'danger')

        # List of after IDs to cancel callbacks
        self.after_ids = []

        # Hide MFC widgets at startup
        self.toggle_mfc_options()

        self.c3000_frame = _make_card(self.tab_wash)
        self.c3000_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 4))
        self.c3000_section_label = tk.Label(self.c3000_frame, text="C3000 Control",
                                            font=FONTS['section'], bg=COLORS['card_bg'])
        self.c3000_section_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")
        self.c3000_status_label = tk.Label(self.c3000_frame, text="C3000 Status",
                                                font=FONTS['body'], bg=COLORS['card_bg'])
        self.c3000_connect_button = tk.Button(self.c3000_frame, text="Connect", command=self.connect_c3000)
        _style_button(self.c3000_connect_button, 'danger')
        self.c3000_connect_button.grid(row=0, column=1, padx=4)

        self.c3000a_frame = tk.Frame(self.c3000_frame, bg=COLORS['card_bg'], bd=1,
                                     relief='solid', highlightbackground=COLORS['border'])
        self.c3000a_frame.grid(row=2, column=1, sticky="nsew", padx=6, pady=4)
        self.c3000a_frame.grid_rowconfigure(0, weight=1)

        self.c3000a_section_label = tk.Label(self.c3000a_frame, text="Water Pump",
                                             font=FONTS['subsection'], bg=COLORS['card_bg'])
        self.c3000a_section_label.grid(row=0, column=0, columnspan=5, padx=8, pady=(6, 2), sticky="w")

        self.c3000b_frame = tk.Frame(self.c3000_frame, bg=COLORS['card_bg'], bd=1,
                                     relief='solid', highlightbackground=COLORS['border'])
        self.c3000b_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)
        self.c3000b_frame.grid_rowconfigure(0, weight=1)
        self.c3000b_section_label = tk.Label(self.c3000b_frame, text="HNO\u2083 Pump",
                                             font=FONTS['subsection'], bg=COLORS['card_bg'])
        self.c3000b_section_label.grid(row=0, column=0, columnspan=5, padx=8, pady=(6, 2), sticky="w")

        self.c3000a_status_label = tk.Label(self.c3000a_frame, text="", bg=COLORS['card_bg'])
        self.c3000b_status_label = tk.Label(self.c3000b_frame, text="", bg=COLORS['card_bg'])

        for frame, prefix, init_cmd, force_cmd, refill_cmd, start_cmd, stop_cmd in [
            (self.c3000a_frame, 'c3000a', self.initialize_c3000_1, self.force_initialize_c3000_1,
             self.refill_c3000_1, self.start_c3000_pump_1, self.stop_c3000_pump_1),
            (self.c3000b_frame, 'c3000b', self.initialize_c3000_2, self.force_initialize_c3000_2,
             self.refill_c3000_2, self.start_c3000_pump_2, self.stop_c3000_pump_2),
        ]:
            init_btn = tk.Button(frame, text="Initialize", command=init_cmd)
            _style_button(init_btn, 'primary')
            init_btn.grid(row=1, column=1, padx=2, pady=2)

            force_btn = tk.Button(frame, text="Force Init", command=force_cmd)
            _style_button(force_btn, 'warning')
            force_btn.grid(row=1, column=2, padx=2, pady=2)

            tk.Label(frame, text="Refill [\u00b5L/s]:", font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=2, column=0, padx=8, sticky="w")
            refill_entry = tk.Entry(frame, width=8)
            _style_entry(refill_entry)
            refill_entry.grid(row=3, column=0, padx=8, sticky="w")
            refill_btn = tk.Button(frame, text="Refill", command=refill_cmd)
            _style_button(refill_btn, 'teal')
            refill_btn.grid(row=3, column=1, padx=2, pady=2)

            tk.Label(frame, text="Target Flow (µL/s):", font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=4, column=0, padx=8, sticky="w")
            flow_entry = tk.Entry(frame, width=8)
            _style_entry(flow_entry)
            flow_entry.grid(row=4, column=1, padx=2, sticky="w")
            start_btn = tk.Button(frame, text="Start", command=start_cmd)
            _style_button(start_btn, 'success')
            start_btn.grid(row=4, column=2, padx=2, pady=2)
            stop_btn = tk.Button(frame, text="Stop", command=stop_cmd)
            _style_button(stop_btn, 'danger')
            stop_btn.grid(row=4, column=3, padx=2, pady=2)

            setattr(self, f'{prefix}_init_button', init_btn)
            setattr(self, f'{prefix}_force_init_button', force_btn)
            setattr(self, f'{prefix}_refill_button', refill_btn)
            setattr(self, f'{prefix}_refill_flow_entry', refill_entry)
            setattr(self, f'{prefix}_flow_label', tk.Label(frame))
            setattr(self, f'{prefix}_flow_entry', flow_entry)
            setattr(self, f'{prefix}_flow_button', start_btn)
            setattr(self, f'{prefix}_stop_button', stop_btn)


        # Valve switch buttons
        # create 3 radiobuttons left, center, right and a switch valve button
        tk.Label(self.c3000a_frame, text="Valve position:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=5, column=0, sticky="w", padx=4)
        self.c3000_1_valve_position = tk.StringVar(value="left")
        tk.Radiobutton(self.c3000a_frame, text="Left (Refill)", variable=self.c3000_1_valve_position,
                       value="left", command=self.switch_c3000_1_valve).grid(row=6, column=0)
        tk.Radiobutton(self.c3000a_frame, text="Center (Z-flow cell)", variable=self.c3000_1_valve_position,
                       value="center", command=self.switch_c3000_1_valve).grid(row=6, column=1)
        tk.Radiobutton(self.c3000a_frame, text="Right (Chip)", variable=self.c3000_1_valve_position,
                       value="right", command=self.switch_c3000_1_valve).grid(row=6, column=2)

        tk.Label(self.c3000b_frame, text="Valve position:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=5, column=0, sticky="w", padx=4)
        self.c3000_2_valve_position = tk.StringVar(value="left")
        tk.Radiobutton(self.c3000b_frame, text="Left (Refill)", variable=self.c3000_2_valve_position,
                       value="left", command=self.switch_c3000_2_valve).grid(row=6, column=0)
        tk.Radiobutton(self.c3000b_frame, text="Center (Z-flow cell)", variable=self.c3000_2_valve_position,
                       value="center", command=self.switch_c3000_2_valve).grid(row=6, column=1)
        tk.Radiobutton(self.c3000b_frame, text="Right (Chip)", variable=self.c3000_2_valve_position,
                       value="right", command=self.switch_c3000_2_valve).grid(row=6, column=2)

        # Nitric acid flush volume
        self.nitric_acid_volume_label = tk.Label(self.c3000b_frame, text="Nitric Acid Flush Volume [µL]")
        self.nitric_acid_volume_label.grid(row=7, column=0)
        self.nitric_acid_volume_entry = tk.Entry(self.c3000b_frame)
        self.nitric_acid_volume_entry.grid(row=7, column=1)
        self.nitric_acid_volume_entry.insert(0, "100")

        # Water flush volume
        self.water_volume_label = tk.Label(self.c3000a_frame, text="Water Flush Volume [µL]")
        self.water_volume_label.grid(row=7, column=0)
        self.water_volume_entry = tk.Entry(self.c3000a_frame)
        self.water_volume_entry.grid(row=7, column=1)
        self.water_volume_entry.insert(0, "600")

        # Back pressure flow rate
        self.back_pressure_label = tk.Label(self.c3000_frame, text="Back Pressure flow [µL.s-1]")
        self.back_pressure_label.grid(row=3, column=0)
        self.back_pressure_entry = tk.Entry(self.c3000_frame)
        self.back_pressure_entry.grid(row=3, column=1)
        self.back_pressure_entry.insert(0, "0.6")

        # ── Wash options card ─────────────────────────────────────────
        self.wash_options_frame = _make_card(self.tab_wash)
        self.wash_options_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 4))
        tk.Label(self.wash_options_frame, text="Wash Options",
                 font=FONTS['section'], bg=COLORS['card_bg']).grid(
            row=0, column=0, columnspan=4, padx=10, pady=(8, 4), sticky="w")

        # Manual wash button
        self.manual_wash_button = tk.Button(
            self.wash_options_frame, text="Run intensive wash now",
            command=self.manual_wash)
        _style_button(self.manual_wash_button, 'warning')
        self.manual_wash_button.grid(row=1, column=0, columnspan=2, padx=10, pady=(4, 8), sticky="w")

        # Regular wash checkbox
        self.regular_wash_var = tk.IntVar(value=0)
        tk.Checkbutton(
            self.wash_options_frame, text="Regular intensive wash (every 9 samples)",
            variable=self.regular_wash_var,
            bg=COLORS['card_bg'], font=FONTS['body']
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=2, sticky="w")

        # Counter display
        self.wash_counter_label = tk.Label(
            self.wash_options_frame,
            text="Samples since last intensive wash: 0",
            font=FONTS['small'], fg=COLORS['text_secondary'], bg=COLORS['card_bg'])
        self.wash_counter_label.grid(row=3, column=0, columnspan=4, padx=10, pady=(0, 4), sticky="w")

        # PVP threshold wash checkbox + entry
        self.pvp_threshold_wash_var = tk.IntVar(value=0)
        tk.Checkbutton(
            self.wash_options_frame, text="Intensive wash if Ag/PVP >",
            variable=self.pvp_threshold_wash_var,
            bg=COLORS['card_bg'], font=FONTS['body']
        ).grid(row=4, column=0, padx=(10, 2), pady=2, sticky="w")
        self.pvp_threshold_entry = tk.Entry(self.wash_options_frame, width=6)
        _style_entry(self.pvp_threshold_entry)
        self.pvp_threshold_entry.insert(0, "3.0")
        self.pvp_threshold_entry.grid(row=4, column=1, padx=(0, 10), pady=2, sticky="w")

        # ── Intensive wash parameters ──
        sep = tk.Frame(self.wash_options_frame, bg=COLORS['border'], height=1)
        sep.grid(row=5, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 4))
        tk.Label(self.wash_options_frame,
                 text="Intensive wash parameters",
                 font=FONTS['small'], fg=COLORS['text_secondary'], bg=COLORS['card_bg']
                 ).grid(row=6, column=0, columnspan=4, padx=10, pady=(0, 4), sticky="w")

        tk.Label(self.wash_options_frame, text="Acid [µL]:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
            row=7, column=0, padx=(10, 2), sticky="w")
        self.intensive_acid_volume_entry = tk.Entry(self.wash_options_frame, width=7)
        _style_entry(self.intensive_acid_volume_entry)
        self.intensive_acid_volume_entry.insert(0, "150")
        self.intensive_acid_volume_entry.grid(row=7, column=1, padx=(0, 10), pady=2, sticky="w")

        tk.Label(self.wash_options_frame, text="Total water [µL]:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
            row=8, column=0, padx=(10, 2), sticky="w")
        self.intensive_water_volume_entry = tk.Entry(self.wash_options_frame, width=7)
        _style_entry(self.intensive_water_volume_entry)
        self.intensive_water_volume_entry.insert(0, "900")
        self.intensive_water_volume_entry.grid(row=8, column=1, padx=(0, 10), pady=2, sticky="w")

        tk.Label(self.wash_options_frame, text="Stop back pressure after [µL water]:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
            row=9, column=0, padx=(10, 2), pady=(2, 8), sticky="w")
        self.intensive_water_bp_stop_entry = tk.Entry(self.wash_options_frame, width=7)
        _style_entry(self.intensive_water_bp_stop_entry)
        self.intensive_water_bp_stop_entry.insert(0, "450")
        self.intensive_water_bp_stop_entry.grid(row=9, column=1, padx=(0, 10), pady=(2, 8), sticky="w")

        # ════════════════════════════════════════════════════════════════
        # SPECTROMETER TAB
        # ════════════════════════════════════════════════════════════════
        self.spectro_frame = _make_card(self.tab_spectro)
        self.spectro_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.spectro_control_label = tk.Label(self.spectro_frame, text="Spectrometer Control",
                                              font=FONTS['section'], bg=COLORS['card_bg'])
        self.spectro_control_label.grid(row=0, columnspan=6, padx=10, pady=(8, 4), sticky="w")

        # ── Action buttons row ────────────────────────────────────
        btn_row = tk.Frame(self.spectro_frame, bg=COLORS['card_bg'])
        btn_row.grid(row=1, column=0, columnspan=5, sticky="w", padx=10, pady=(4, 8))

        self.measure_reference_frame = tk.Frame(btn_row, highlightbackground=COLORS['danger'],
                                                highlightcolor=COLORS['danger'], highlightthickness=3, bd=0)
        self.measure_reference_button = tk.Button(self.measure_reference_frame, text="Reference",
                                                  command=self.measure_reference)
        _style_button(self.measure_reference_button, 'danger')
        self.measure_reference_frame.pack(side="left", padx=(0, 6))
        self.measure_reference_button.pack(padx=2, pady=2)

        plot_spectrum = tk.Button(btn_row, text="Live Plot", command=self.plot_spectrum_command)
        _style_button(plot_spectrum, 'primary')
        plot_spectrum.pack(side="left", padx=(0, 6))

        single_plot_spectrum = tk.Button(btn_row, text="Single Plot", command=self.measure_spectrum)
        _style_button(single_plot_spectrum)
        single_plot_spectrum.pack(side="left", padx=(0, 6))

        smart_plot_spectrum = tk.Button(btn_row, text="Smart Plot", command=self.smart_plot_spectrum)
        _style_button(smart_plot_spectrum)
        smart_plot_spectrum.pack(side="left", padx=(0, 16))

        self.open_lamp_shutter_button = tk.Button(btn_row, text="Open Shutter",
                                                  command=self.open_lamp_shutter)
        _style_button(self.open_lamp_shutter_button, 'teal')
        self.open_lamp_shutter_button.pack(side="left", padx=(0, 6))
        self.close_lamp_shutter_button = tk.Button(btn_row, text="Close Shutter",
                                                   command=self.close_lamp_shutter)
        _style_button(self.close_lamp_shutter_button)
        self.close_lamp_shutter_button.pack(side="left")

        # ── Parameters grid ──────────────────────────────────────
        param_labels = ["Integration time (ms)", "Nb Averages", "Smoothing (px)",
                        "Min wavelength (nm)", "Max wavelength (nm)"]
        param_defaults = [exp.spectro.integration_time, exp.spectro.number_average,
                          exp.spectro.smoothing,
                          "310.0", "1000.0"]
        for c, (lbl_text, default) in enumerate(zip(param_labels, param_defaults)):
            tk.Label(self.spectro_frame, text=lbl_text, font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=2, column=c, padx=10, pady=(0, 2), sticky="w")

        self.int_time_entry = tk.Entry(self.spectro_frame, width=10, justify="center")
        _style_entry(self.int_time_entry)
        self.int_time_entry.insert(0, param_defaults[0])
        self.int_time_entry.grid(row=3, column=0, padx=10, sticky="w")

        self.average_number_entry = tk.Entry(self.spectro_frame, width=10, justify="center")
        _style_entry(self.average_number_entry)
        self.average_number_entry.insert(0, param_defaults[1])
        self.average_number_entry.grid(row=3, column=1, padx=10, sticky="w")

        self.smoothing_entry = tk.Entry(self.spectro_frame, width=10, justify="center")
        _style_entry(self.smoothing_entry)
        self.smoothing_entry.insert(0, param_defaults[2])
        self.smoothing_entry.grid(row=3, column=2, padx=10, sticky="w")

        self.min_wl_entry = tk.Entry(self.spectro_frame, width=10, justify="center")
        _style_entry(self.min_wl_entry)
        self.min_wl_entry.insert(0, param_defaults[3])
        self.min_wl_entry.grid(row=3, column=3, padx=10, sticky="w")

        self.max_wl_entry = tk.Entry(self.spectro_frame, width=10, justify="center")
        _style_entry(self.max_wl_entry)
        self.max_wl_entry.insert(0, param_defaults[4])
        self.max_wl_entry.grid(row=3, column=4, padx=10, sticky="w")

        self.update_spectro_parameters_button = tk.Button(self.spectro_frame, text="Update Parameters",
                                                          command=self.update_spectro_parameters)
        _style_button(self.update_spectro_parameters_button, 'primary')
        self.update_spectro_parameters_button.grid(row=4, column=0, columnspan=2, padx=10, pady=8, sticky="w")

        # ══════════════════════════════════════════════════════════════
        # COLLECTOR TAB
        # ══════════════════════════════════════════════════════════════
        self.collector_frame = _make_card(self.tab_collector)
        self.collector_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.collector_label = tk.Label(self.collector_frame, text="Fraction Collector",
                                        font=FONTS['section'], bg=COLORS['card_bg'])
        self.collector_label.grid(row=0, column=0, columnspan=5, padx=10, pady=(8, 4), sticky="w")

        # ── Status row ───────────────────────────────────────────
        status_row = tk.Frame(self.collector_frame, bg=COLORS['card_bg'])
        status_row.grid(row=1, column=0, columnspan=5, sticky="w", padx=10, pady=(0, 6))

        tk.Label(status_row, text="Status:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
        self.collector_status_label = tk.Label(status_row, text="", font=FONTS['body'],
                                               bg=COLORS['card_bg'])
        self.collector_status_label.pack(side="left", padx=(4, 16))

        self.collector_homing_button = tk.Button(status_row, text="Homing", command=self.homing)
        _style_button(self.collector_homing_button, 'primary')
        self.collector_homing_button.pack(side="left", padx=(0, 16))
        self.update_collector_status()

        tk.Label(status_row, text="Connection:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
        self.collector_connection_status_label = tk.Label(status_row, text="", font=FONTS['body'],
                                                          bg=COLORS['card_bg'])
        self.collector_connection_status_label.pack(side="left", padx=(4, 0))

        # ── Control row ──────────────────────────────────────────
        ctrl_row = tk.Frame(self.collector_frame, bg=COLORS['card_bg'])
        ctrl_row.grid(row=2, column=0, columnspan=5, sticky="w", padx=10, pady=(0, 4))

        tk.Label(ctrl_row, text="Position:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
        self.collector_position_entry = tk.Entry(ctrl_row, width=6)
        _style_entry(self.collector_position_entry)
        self.collector_position_entry.pack(side="left", padx=4)

        self.collector_move_button = tk.Button(ctrl_row, text="Move", command=self.move_to_position)
        _style_button(self.collector_move_button)
        self.collector_move_button.pack(side="left", padx=(0, 6))
        self.next_well_button = tk.Button(ctrl_row, text="Next Well", command=self.move_to_next_well)
        _style_button(self.next_well_button, 'teal')
        self.next_well_button.pack(side="left", padx=(0, 16))

        self.collector_valve_switch_button_0 = tk.Button(ctrl_row, text="Valve: Trash",
                                                         command=self.switch_valve_to_trash)
        _style_button(self.collector_valve_switch_button_0, 'warning')
        self.collector_valve_switch_button_0.pack(side="left", padx=(0, 6))
        self.collector_valve_switch_button_1 = tk.Button(ctrl_row, text="Valve: Collect",
                                                         command=self.switch_valve_to_collection)
        _style_button(self.collector_valve_switch_button_1, 'success')
        self.collector_valve_switch_button_1.pack(side="left", padx=(0, 8))

        tk.Label(ctrl_row, text="Valve:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
        self.collector_valve_current_position_label = tk.Label(ctrl_row, text="trash",
                                                               font=FONTS['mono'], bg=COLORS['card_bg'])
        self.collector_valve_current_position_label.pack(side="left", padx=4)
        self.update_valve_position()

        # ── Well plate ───────────────────────────────────────────
        self.collector_plate = tk.Canvas(self.collector_frame, bg="white", width=340, height=220,
                                         highlightthickness=1, highlightbackground=COLORS['border'])
        self.collector_plate.grid(row=3, column=0, columnspan=5, padx=10, pady=6)

        self.ROWS = 6
        self.COLUMNS = 9
        self.WELL_RADIUS = 11
        self.PADDING = 14
        self.current_well = None
        self.filled_wells = []
        self.wells = []
        self.well_coords = {}

        total_width = self.COLUMNS * (2 * self.WELL_RADIUS + self.PADDING) + self.PADDING
        total_height = self.ROWS * (2 * self.WELL_RADIUS + self.PADDING) + self.PADDING

        offset_x = (360 - total_width) // 2
        offset_y = (230 - total_height) // 2

        for col in range(self.COLUMNS):
            col_label = chr(73 - col)
            self.collector_plate.create_text(
                offset_x + self.PADDING + col * (2 * self.WELL_RADIUS + self.PADDING) + self.WELL_RADIUS,
                offset_y + self.PADDING // 2,
                text=col_label)

        for row in range(self.ROWS):
            row_label = str(row + 1)
            self.collector_plate.create_text(offset_x + self.PADDING // 2, offset_y + self.PADDING + row * (
                    2 * self.WELL_RADIUS + self.PADDING) + self.WELL_RADIUS,
                                             text=row_label)

            row_wells = []
            for col in range(self.COLUMNS):
                x = (col + 1) * self.PADDING + (2 * col + 1) * self.WELL_RADIUS
                y = (row + 1) * self.PADDING + (2 * row + 1) * self.WELL_RADIUS
                well = self.collector_plate.create_oval(x - self.WELL_RADIUS, y - self.WELL_RADIUS,
                                                        x + self.WELL_RADIUS,
                                                        y + self.WELL_RADIUS, fill="white")
                row_wells.append(well)
                self.well_coords[(row, col)] = (x, y)
                self.collector_plate.tag_bind(well, "<Button-1>",
                                              lambda event, row=row, col=col: self.click_to_position(row, col))
            self.wells.append(row_wells)

        # ── Z axis control ───────────────────────────────────────
        z_row = tk.Frame(self.collector_frame, bg=COLORS['card_bg'])
        z_row.grid(row=4, column=0, columnspan=5, sticky="w", padx=10, pady=(0, 8))
        self.z_axis_frame = z_row

        tk.Label(z_row, text="Z-axis:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side="left")
        self.z_axis_position_entry = tk.Entry(z_row, width=6)
        _style_entry(self.z_axis_position_entry)
        self.z_axis_position_entry.pack(side="left", padx=4)
        self.z_axis_move_button = tk.Button(z_row, text="Move", command=self.move_z_axis)
        _style_button(self.z_axis_move_button)
        self.z_axis_move_button.pack(side="left", padx=(0, 8))
        self.z_axis_current_position_label = tk.Label(z_row, text="", font=FONTS['mono'],
                                                       bg=COLORS['card_bg'])
        self.z_axis_current_position_label.pack(side="left")
        self.update_z_axis_position()

        # ══════════════════════════════════════════════════════════════
        # EXPERIMENT PANEL (persistent left sidebar)
        # ══════════════════════════════════════════════════════════════
        self.exp_frame = _make_card(self._exp_inner)
        self.exp_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        _cur_row = 0
        self.exp_control_label = tk.Label(self.exp_frame, text="Experiment Control",
                                          font=FONTS['section'], bg=COLORS['card_bg'])
        self.exp_control_label.grid(row=_cur_row, column=0, columnspan=2, padx=10, pady=(8, 6), sticky="w")

        # ── Parameters (2-column: label | entry) ────────────────
        exp_params = [
            ("Residence time (s)", self.exp.residence_time),
            ("Number of samples", "2"),
            ("Sample volume (\u00b5L)", "100"),
            ("Spectro rate (s)", "5"),
        ]
        _cur_row += 1
        for c, (lbl, default) in enumerate(exp_params):
            tk.Label(self.exp_frame, text=lbl, font=FONTS['small'],
                     fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(
                row=_cur_row + c, column=0, padx=(10, 4), pady=1, sticky="w")

        self.residence_time_entry = tk.Entry(self.exp_frame, width=10)
        _style_entry(self.residence_time_entry)
        self.residence_time_entry.insert(0, self.exp.residence_time)
        self.residence_time_entry.grid(row=_cur_row, column=1, padx=(0, 10), pady=1, sticky="w")

        self.number_of_samples_entry = tk.Entry(self.exp_frame, width=10)
        _style_entry(self.number_of_samples_entry)
        self.number_of_samples_entry.insert(0, "2")
        self.number_of_samples_entry.grid(row=_cur_row + 1, column=1, padx=(0, 10), pady=1, sticky="w")

        self.sample_volume_entry = tk.Entry(self.exp_frame, width=10)
        _style_entry(self.sample_volume_entry)
        self.sample_volume_entry.insert(0, "100")
        self.sample_volume_entry.grid(row=_cur_row + 2, column=1, padx=(0, 10), pady=1, sticky="w")

        self.spectro_sampling_rate_entry = tk.Entry(self.exp_frame, width=10)
        _style_entry(self.spectro_sampling_rate_entry)
        self.spectro_sampling_rate_entry.insert(0, "1")
        self.spectro_sampling_rate_entry.grid(row=_cur_row + 3, column=1, padx=(0, 10), pady=1, sticky="w")
        _cur_row += 4

        # ── Separator ────────────────────────────────────────────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        # ── Checkboxes (stacked vertically) ──────────────────────
        self.inline_spectro_var = tk.IntVar(value=1)
        self.inline_spectro_checkbox = tk.Checkbutton(self.exp_frame, text="Inline Spectro",
                                                      variable=self.inline_spectro_var,
                                                      bg=COLORS['card_bg'], font=FONTS['body'])
        self.inline_spectro_checkbox.grid(row=_cur_row, column=0, columnspan=2, padx=10, sticky="w")
        _cur_row += 1

        self.stop_flow_exp = tk.IntVar(value=1)
        self.stop_flow_checkbox = tk.Checkbutton(self.exp_frame, text="Stop-flow Mode",
                                                  variable=self.stop_flow_exp,
                                                  bg=COLORS['card_bg'], font=FONTS['body'])
        self.stop_flow_checkbox.grid(row=_cur_row, column=0, columnspan=2, padx=10, sticky="w")
        _cur_row += 1

        self.sample_collection_var = tk.IntVar()
        self.sample_collection_checkbox = tk.Checkbutton(self.exp_frame, text="Sample Collection",
                                                          variable=self.sample_collection_var,
                                                          bg=COLORS['card_bg'], font=FONTS['body'])
        self.sample_collection_checkbox.grid(row=_cur_row, column=0, columnspan=2, padx=10, sticky="w")
        _cur_row += 1

        self.manual_spectrum_validation = tk.BooleanVar(self.exp_frame)
        self.manual_spectrum_validation_button = tk.Checkbutton(self.exp_frame, text="Manual Validation",
                                                                variable=self.manual_spectrum_validation,
                                                                bg=COLORS['card_bg'], font=FONTS['body'])
        self.manual_spectrum_validation_button.grid(row=_cur_row, column=0, columnspan=2, padx=10, sticky="w")
        _cur_row += 1

        # ── Separator ────────────────────────────────────────────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        # ── Experiment type (stacked radios) ─────────────────────
        tk.Label(self.exp_frame, text="Experiment type", font=FONTS['subsection'],
                 bg=COLORS['card_bg']).grid(row=_cur_row, column=0, columnspan=2,
                                            padx=10, sticky="w")
        _cur_row += 1
        self.exp_type_list = ["Fixed", "Random", "LHS Flows", "Projected LHS Flows", "CSV Flows", "Bayesian Optimization"]
        self.exp_type_var = tk.StringVar(self.exp_frame)
        self.exp_type_var.set("Fixed")
        for exp_type in self.exp_type_list:
            tk.Radiobutton(self.exp_frame, text=exp_type, variable=self.exp_type_var,
                           value=exp_type, bg=COLORS['card_bg'], font=FONTS['body']).grid(
                row=_cur_row, column=0, columnspan=2, padx=20, sticky="w")
            _cur_row += 1

        # CSV flow file picker
        self.csv_flow_file = None
        csv_row = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        csv_row.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 4))
        _cur_row += 1
        self.load_csv_button = tk.Button(csv_row, text="Load CSV\u2026",
                                         command=self.load_csv_flow_file)
        _style_button(self.load_csv_button)
        self.load_csv_button.pack(side="left", padx=(0, 6))
        self.csv_file_label = tk.Label(csv_row, text="No file", font=FONTS['small'],
                                       fg=COLORS['text_light'], bg=COLORS['card_bg'])
        self.csv_file_label.pack(side="left")

        # BO parameters window
        def open_bo_window():
            self.bo_params_button['state'] = 'disabled'
            self.bo_params_window.deiconify()

        def on_closing():
            self.bo_params_window.withdraw()
            self.bo_params_button['state'] = 'normal'

        def _on_exp_type_change(*_args):
            if self.exp_type_var.get() == 'Bayesian Optimization':
                open_bo_window()
            else:
                on_closing()

        self.exp_type_var.trace_add('write', _on_exp_type_change)

        self.bo_params_button = tk.Button(self.exp_frame, text='BO Parameters\u2026', command=open_bo_window)
        _style_button(self.bo_params_button, 'primary')

        self.bo_params_window = tk.Toplevel(self)
        self.bo_params_window.protocol("WM_DELETE_WINDOW", on_closing)
        self.bo_params_window.title('BO parameters control')
        if os.path.exists(_ICON_PATH):
            self.bo_params_window.iconbitmap(_ICON_PATH)
        self.bo_params_window.transient(self)
        on_closing()

        # ── Separator ────────────────────────────────────────────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        # ── Action buttons (2-column grid) ───────────────────────
        action_row = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        action_row.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 4))
        _cur_row += 1

        self.start_experiment_button = tk.Button(action_row, text="\u25B6  Start",
                                                  command=self.start_exp)
        _style_button(self.start_experiment_button, 'success')
        self.start_experiment_button.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="ew")

        self.bo_params_button.grid(in_=action_row, row=0, column=1, padx=(0, 4), pady=2, sticky="ew")

        self.prime_pump_button = tk.Button(action_row, text="Prime Pump", command=self.prime_pump)
        _style_button(self.prime_pump_button, 'teal')
        self.prime_pump_button.grid(row=0, column=2, pady=2, sticky="ew")

        self.save_parameters_button = tk.Button(action_row, text="Save Params",
                                                 command=self.save_parameters)
        _style_button(self.save_parameters_button)
        self.save_parameters_button.grid(row=1, column=0, padx=(0, 4), pady=2, sticky="ew")

        self.load_previous_exp_button = tk.Button(action_row, text="Load Previous",
                                                   command=self.load_previous_experiment)
        _style_button(self.load_previous_exp_button)
        self.load_previous_exp_button.grid(row=1, column=1, columnspan=2, pady=2, sticky="ew")

        for col in range(3):
            action_row.grid_columnconfigure(col, weight=1)

        # ── Experiment controls (pause / continue / abort) ───────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        ctrl_row = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        ctrl_row.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 4))
        _cur_row += 1

        self.continue_exp_button = tk.Button(ctrl_row, text="\u25B6  Continue",
                                              command=self.continue_exp, state="disabled")
        _style_button(self.continue_exp_button, 'success')
        self.continue_exp_button.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="ew")

        self.pause_exp_button = tk.Button(ctrl_row, text="\u23F8  Pause",
                                           command=self.pause_exp, state="disabled")
        _style_button(self.pause_exp_button, 'warning')
        self.pause_exp_button.grid(row=0, column=1, padx=(0, 4), pady=2, sticky="ew")

        self.abort_exp_button = tk.Button(ctrl_row, text="\u2716  Abort",
                                           command=self.abort_exp, state="disabled")
        _style_button(self.abort_exp_button, 'danger')
        self.abort_exp_button.grid(row=0, column=2, pady=2, sticky="ew")

        # Stop All: always enabled, brings every device to a safe state.
        self.stop_all_button = tk.Button(ctrl_row, text="\u26D4  STOP ALL",
                                          command=self.stop_all_hardware)
        _style_button(self.stop_all_button, 'danger')
        self.stop_all_button.grid(row=1, column=0, columnspan=3, pady=(2, 2), sticky="ew")

        self.next_sample_button = tk.Button(ctrl_row, text="Next Sample \u25B8",
                                             command=self.next_sample)
        _style_button(self.next_sample_button, 'teal')
        self.next_sample_button.grid(row=2, column=0, columnspan=3, pady=2, sticky="ew")

        for col in range(3):
            ctrl_row.grid_columnconfigure(col, weight=1)

        # ── Status indicators ────────────────────────────────────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        status_grid = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        status_grid.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        _cur_row += 1

        tk.Label(status_grid, text="Collection:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=0, column=0, sticky="w")
        self.collection_status_label = tk.Label(status_grid, text="", font=FONTS['body'],
                                                bg=COLORS['card_bg'])
        self.collection_status_label.grid(row=0, column=1, padx=(4, 0), sticky="w")
        self.update_collection_status()

        tk.Label(status_grid, text="State:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=1, column=0, sticky="w")
        self.state_label = tk.Label(status_grid, text="", font=FONTS['body'],
                                    bg=COLORS['card_bg'])
        self.state_label.grid(row=1, column=1, padx=(4, 0), sticky="w")
        self.update_state_status()

        tk.Label(status_grid, text="Sample:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=2, column=0, sticky="w")
        self.sample_level = tk.Label(status_grid, text="0", font=FONTS['mono'],
                                     bg=COLORS['card_bg'])
        self.sample_level.grid(row=2, column=1, padx=(4, 0), sticky="w")
        self.get_sample_level()

        # ── Log entry ────────────────────────────────────────────
        log_row = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        log_row.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        _cur_row += 1

        self.add_to_log_entry = tk.Entry(log_row)
        _style_entry(self.add_to_log_entry)
        self.add_to_log_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.add_to_log_button = tk.Button(log_row, text="Log", command=self.add_to_log)
        _style_button(self.add_to_log_button)
        self.add_to_log_button.pack(side="left")

        # ── Advanced Parameters (collapsible) ────────────────────
        ttk.Separator(self.exp_frame, orient='horizontal').grid(
            row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        _cur_row += 1

        self._adv_visible = tk.BooleanVar(value=False)
        adv_toggle = tk.Button(self.exp_frame, text="\u25B8 Advanced Parameters",
                               command=self._toggle_advanced,
                               anchor="w", relief='flat', bd=0,
                               bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                               font=FONTS['subsection'], cursor='hand2')
        adv_toggle.grid(row=_cur_row, column=0, columnspan=2, padx=10, sticky="w")
        self._adv_toggle_btn = adv_toggle
        _cur_row += 1

        self._adv_frame = tk.Frame(self.exp_frame, bg=COLORS['card_bg'])
        self._adv_frame.grid(row=_cur_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self._adv_frame_row = _cur_row
        _cur_row += 1
        self._adv_frame.grid_remove()

        ar = 0
        # -- Termination criteria --
        tk.Label(self._adv_frame, text="Termination criteria", font=FONTS['subsection'],
                 bg=COLORS['card_bg']).grid(row=ar, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ar += 1

        tk.Label(self._adv_frame, text="Correlation threshold:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=ar, column=0, sticky="w")
        self._adv_corr_entry = tk.Entry(self._adv_frame, width=8)
        _style_entry(self._adv_corr_entry)
        self._adv_corr_entry.insert(0, "0.998")
        self._adv_corr_entry.grid(row=ar, column=1, sticky="w", pady=1)
        ar += 1

        tk.Label(self._adv_frame, text="Amplitude threshold:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=ar, column=0, sticky="w")
        self._adv_amp_entry = tk.Entry(self._adv_frame, width=8)
        _style_entry(self._adv_amp_entry)
        self._adv_amp_entry.insert(0, "0.02")
        self._adv_amp_entry.grid(row=ar, column=1, sticky="w", pady=1)
        ar += 1

        tk.Label(self._adv_frame, text="Min spectra:", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=ar, column=0, sticky="w")
        self._adv_min_spectra_entry = tk.Entry(self._adv_frame, width=8)
        _style_entry(self._adv_min_spectra_entry)
        self._adv_min_spectra_entry.insert(0, "10")
        self._adv_min_spectra_entry.grid(row=ar, column=1, sticky="w", pady=1)
        ar += 1

        tk.Label(self._adv_frame, text="Timeout (s):", font=FONTS['small'],
                 fg=COLORS['text_secondary'], bg=COLORS['card_bg']).grid(row=ar, column=0, sticky="w")
        self._adv_timeout_entry = tk.Entry(self._adv_frame, width=8)
        _style_entry(self._adv_timeout_entry)
        self._adv_timeout_entry.insert(0, "300")
        self._adv_timeout_entry.grid(row=ar, column=1, sticky="w", pady=1)
        ar += 1

        # -- Apply button --
        self._adv_apply_btn = tk.Button(self._adv_frame, text="Apply Advanced Settings",
                                         command=self._apply_advanced_params)
        _style_button(self._adv_apply_btn, 'primary')
        self._adv_apply_btn.grid(row=ar, column=0, columnspan=2, sticky="ew", pady=(6, 4))

        ## BO parameters
        self.bo_frame = tk.Frame(self.bo_params_window, borderwidth=2, relief="groove")
        self.bo_frame.grid(sticky="nsew", padx=3, pady=3)
        row = 0
        self.bo_label = tk.Label(self.bo_frame, text="Self Optimizing Experiment", font=FONTS['section'])
        self.bo_label.grid(row=row, columnspan=5)
        row += 1

        # Maxiter
        self.maxiter_label = tk.Label(self.bo_frame, text='Maximum iteration :')
        self.maxiter_label.grid(row=row, column=0, sticky='w')
        self.maxiter_entry = tk.Entry(self.bo_frame)
        self.maxiter_entry.grid(row=row, column=1)
        self.maxiter_entry.insert(0, self.exp.maxiter)
        self.tol_stop_var = tk.BooleanVar(self.bo_frame)
        self.tol_stop_var.set(True)
        self.tol_stop_checkbox = tk.Checkbutton(self.bo_frame, text='Stop when tolerance is reached',
                                                var=self.tol_stop_var, onvalue=True, offvalue=False)
        self.tol_stop_checkbox.grid(row=row, column=2, columnspan=2)
        row += 1

        # Objective selection
        self.objective_label = tk.Label(self.bo_frame, text="Objective", font=("Arial", 9, "bold"))
        self.objective_label.grid(row=row, column=0, sticky='w')
        row += 1
        self.objective_list = ["single_point", "full_spectrum", "lambda_max", "pca"]
        self.obj_var = tk.StringVar(self.bo_frame)
        self.obj_var.set(self.exp.target_style)
        self.previous_obj = self.obj_var.get()
        for i, obj in enumerate(self.objective_list):
            tk.Radiobutton(self.bo_frame, text=obj, variable=self.obj_var, value=obj,
                           command=self.obj_switch).grid(row=row, column=i)
        # Single point target
        self.single_point_wl_label = tk.Label(self.bo_frame, text='Target wavelength')
        self.single_point_wl_entry = tk.Entry(self.bo_frame, width=7)
        self.single_point_wl_entry.insert(0, self.exp.tg_wl)
        self.single_point_wl_label.grid(row=row - 1, column=2)
        self.single_point_wl_entry.grid(row=row, column=2)
        self.single_point_abs_label = tk.Label(self.bo_frame, text='Target absorbance')
        self.single_point_abs_entry = tk.Entry(self.bo_frame, width=7)
        self.single_point_abs_entry.insert(0, self.exp.tg_abs)
        self.single_point_abs_label.grid(row=row - 1, column=3)
        self.single_point_abs_entry.grid(row=row, column=3)
        # Target spectrum measurement
        self.full_spectrum_tg_flows_labels = [tk.Label(self.bo_frame, text=f'Target flow {i + 1}') for i in
                                              range(self.exp.number_of_pumps)]
        self.full_spectrum_tg_flows_entries = [tk.Entry(self.bo_frame) for _ in range(self.exp.number_of_pumps)]
        self.meas_target_button_frame = tk.Frame(self.bo_frame, highlightbackground="red",
                                                 highlightcolor="red", highlightthickness=2, bd=0)
        self.meas_target_button = tk.Button(self.meas_target_button_frame, text='Measure target',
                                            command=self.measure_target)
        self.meas_target_button.grid()
        # Target loading
        self.target_spectrum_file = None  # Initialize file path variable
        self.target_path_label = ""
        self.full_spectrum_load_button = tk.Button(self.bo_frame, text='Load a target spectrum',
                                                   command=self.browse_file)
        self.target_path_label = tk.Label(self.bo_frame, text=self.target_path_label)
        # Build a PHYSICS-defined target (lambda_max/FWHM/amplitude) instead of
        # picking a generated spectrum — see target_builder / paper discussion.
        self.build_target_button = tk.Button(self.bo_frame, text='Build target (physics)',
                                             command=self.build_parametric_target_dialog)

        # --- PCA target widgets ---
        # 'pca' style reuses the target-spectrum loader above and adds a frozen
        # PCA basis (.npz) plus an optional warm-start from a folder of .h5.
        self.pca_model_file = None
        self.warmstart_dir = None
        self.pca_variance_weighted_var = tk.BooleanVar(self.bo_frame, value=False)
        self.pca_load_button = tk.Button(self.bo_frame, text='Load PCA basis (.npz)',
                                         command=self.browse_pca_file)
        self.pca_path_label = tk.Label(self.bo_frame, text='')
        self.pca_varweight_check = tk.Checkbutton(self.bo_frame, text='variance-weighted',
                                                  variable=self.pca_variance_weighted_var)
        # Score-space distance metric + number of leading PCs. 'cosine' (k~2-4) is
        # bounded/smooth -> low GP uncertainty; 'sqdist' = legacy behaviour.
        self.pca_metric_var = tk.StringVar(self.bo_frame, value='sqdist')
        self.pca_metric_label = tk.Label(self.bo_frame, text='cost metric')
        self.pca_metric_menu = tk.OptionMenu(self.bo_frame, self.pca_metric_var,
                                             'sqdist', 'dist', 'cosine')
        self.pca_ncomp_label = tk.Label(self.bo_frame, text='n_comp (0=all)')
        self.pca_ncomp_entry = tk.Entry(self.bo_frame, width=6)
        self.pca_ncomp_entry.insert(0, '0')
        # Optional hybrid term: add a lambda_max penalty to the (non-normalized)
        # PCA cost so peak amplitude is kept while peak position stays sharp.
        self.pca_lambda_var = tk.BooleanVar(self.bo_frame, value=False)
        self.pca_lambda_check = tk.Checkbutton(self.bo_frame, text='hybrid λmax',
                                               variable=self.pca_lambda_var)
        self.pca_lambda_weight_label = tk.Label(self.bo_frame, text='λ weight')
        self.pca_lambda_weight_entry = tk.Entry(self.bo_frame, width=6)
        self.pca_lambda_weight_entry.insert(0, '1.0')
        self.pca_lambda_scale_label = tk.Label(self.bo_frame, text='λ scale (nm)')
        self.pca_lambda_scale_entry = tk.Entry(self.bo_frame, width=6)
        self.pca_lambda_scale_entry.insert(0, '30')
        # Optional amplitude term: discriminate good-shape compositions by
        # yield/concentration on the (max-normalized) PCA cost. Keep small.
        self.pca_amp_var = tk.BooleanVar(self.bo_frame, value=False)
        self.pca_amp_check = tk.Checkbutton(self.bo_frame, text='hybrid amplitude',
                                            variable=self.pca_amp_var)
        self.pca_amp_weight_label = tk.Label(self.bo_frame, text='amp weight')
        self.pca_amp_weight_entry = tk.Entry(self.bo_frame, width=6)
        self.pca_amp_weight_entry.insert(0, '0.3')
        self.warmstart_button = tk.Button(self.bo_frame, text='Warm-start from H5 folder',
                                          command=self.browse_warmstart_dir)
        self.warmstart_path_label = tk.Label(self.bo_frame, text='')
        self.warmstart_clear_button = tk.Button(self.bo_frame, text='Clear warm-start',
                                                command=self.clear_warmstart)

        # Cost method selection for full spectrum
        self.cost_method_label = tk.Label(self.bo_frame, text='Cost method:')
        self.cost_method_var = tk.StringVar(self.bo_frame)
        self.cost_method_var.set('L2')  # Default to L2
        self.cost_method_l2 = tk.Radiobutton(self.bo_frame, text='L2',
                                             variable=self.cost_method_var, value='L2')
        self.cost_method_cosine = tk.Radiobutton(self.bo_frame, text='Cosine similarity',
                                                 variable=self.cost_method_var, value='cosine')
        row += 1

        # Tolerance
        self.tol_label = tk.Label(self.bo_frame, text='tolerance:', justify='right')
        self.tol_label.grid(row=row, column=0)
        self.tol_entry = tk.Entry(self.bo_frame, width=4)
        self.tol_entry.insert(0, self.exp.tol)
        self.tol_entry.grid(row=row, column=1)
        self.relative_tol_var = tk.BooleanVar(self.bo_frame)
        self.relative_tol_var.set(False)
        self.relative_tol_checkbox = tk.Checkbutton(self.bo_frame, text='Relative tolerance (%)',
                                                    var=self.relative_tol_var, onvalue=True, offvalue=False)
        self.relative_tol_checkbox.grid(row=row, column=2)

        self.obj_switch()
        row += 1

        # Experiment initialization strategy
        self.init_strategy_label = tk.Label(self.bo_frame, text="Initialization strategy",
                                            font=("Arial", 9, "bold"))
        self.init_strategy_label.grid(row=row, column=0, sticky='w')
        row += 1

        # Define number of init points
        self.nb_init_samples_label = tk.Label(self.bo_frame, text="initial points")
        self.nb_init_samples_label.grid(row=row, column=1, sticky='w')
        self.nb_init_samples_entry = tk.Entry(self.bo_frame, justify="right", width=3)
        self.nb_init_samples_entry.insert(0, self.exp.nb_init_samples)
        self.nb_init_samples_entry.bind('<Return>', lambda _: self.update_bo_grid())
        self.nb_init_samples_entry.bind('<FocusOut>', lambda _: self.update_bo_grid())
        self.nb_init_samples_entry.grid(row=row, column=0, sticky='e')
        # Latin hypercube = LHS on the simplex of compositions; Projected LHS = toofast 1.0 "LatinHyperCube"
        self.init_strategy_list = ["Random", "Latin hypercube", "Projected LHS", "User defined"]
        self.init_val = ["random", "latinhypercube", "projected_lhs", "fixed"]
        self.init_var = tk.StringVar(self.bo_frame)
        self.init_var.set(self.exp.init_option)
        init_options_frame = tk.Frame(self.bo_frame)       # the four options side by side, compact
        init_options_frame.grid(row=row, column=2, columnspan=4, sticky='w')
        for strategy, opt in zip(self.init_strategy_list, self.init_val):
            tk.Radiobutton(init_options_frame, text=strategy, variable=self.init_var, value=opt,
                           command=self.init_flows_switch).pack(side='left', padx=(0, 12))
        self.init_pump_labels = [tk.Label(self.bo_frame, text=f'Pump {i + 1}') for i in range(self.exp.number_of_pumps)]
        self.init_samples_labels = [tk.Label(self.bo_frame, text=f'Sample {i + 1}') for i in
                                    range(self.exp.nb_init_samples)]
        self.init_flows_entries = []
        for _ in range(int(self.nb_init_samples_entry.get())):
            self.init_flows_entries.append([tk.Entry(self.bo_frame) for _ in range(self.exp.number_of_pumps)])
            [entry.bind('<Return>', lambda _: self.check_flow_sum()) for entry in self.init_flows_entries[-1]]
            [entry.bind('<FocusOut>', lambda _: self.check_flow_sum()) for entry in self.init_flows_entries[-1]]
        self.check_flows_labels = [tk.Label(self.bo_frame, text=f'Sum of flows must be {self.exp.total_flow}.'
                                            ) for _ in range(int(self.nb_init_samples_entry.get()))]
        self.init_flows_switch()
        row += 1

        # Acquisition strategy
        self.acquisition_strategy_label = tk.Label(self.bo_frame, text="Acquisition strategy",
                                                   font=("Arial", 9, "bold"))
        self.acquisition_strategy_label.grid(row=row, column=0, sticky='w')
        row += 1
        self.acq_func = ['ExpectedImprovement', 'LowerConfidenceBound', 'Thompson Sampling']
        self.acq_func_val = ['EI', 'LCB', 'TS']
        self.acq_func_var = tk.StringVar(self.bo_frame)
        self.acq_func_var.set(self.exp.acq_func)
        for i, (acq_func, val) in enumerate(zip(self.acq_func, self.acq_func_val)):
            tk.Radiobutton(self.bo_frame, text=acq_func, variable=self.acq_func_var, value=val
                           ).grid(row=row, column=2 * i, columnspan=2 if i != 2 else 1)
        row += 1

        # BO preset \u2014 atomic switch: kernel + cost combo
        tk.Label(self.bo_frame, text="BO preset", font=("Arial", 9, "bold")).grid(
            row=row, column=0, sticky='w')
        row += 1

        self._bo_preset_var = tk.StringVar(value="default")
        tk.Radiobutton(self.bo_frame, text="Default (RBF + quadratic cost)",
                       variable=self._bo_preset_var, value="default").grid(
            row=row, column=0, columnspan=2, sticky="w")
        tk.Radiobutton(self.bo_frame, text="Robust (Matern 2.5 iso + Huber cost)",
                       variable=self._bo_preset_var, value="robust").grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1

        self._bo_preset_info = tk.Label(
            self.bo_frame,
            text="[i] Robust uses Huber loss (delta=30 nm) on lambda_max and "
                 "Matern 2.5 isotropic with length-scale bounds (0.5, 10).",
            font=("Arial", 8), fg=COLORS['text_secondary'],
            wraplength=520, justify='left')
        self._bo_preset_info.grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        # Flat-spectrum (no-particle) handling at GP-fit time
        tk.Label(self.bo_frame, text="Flat-spectrum strategy",
                 font=("Arial", 9, "bold")).grid(row=row, column=0, sticky='w')
        row += 1
        self._bo_flat_var = tk.StringVar(value="nearest_high")
        for ci, (val, lbl) in enumerate([
            ("nearest_high", "Nearest-high (default)"),
            ("penalty_max", "Penalty max"),
            ("penalty_moderate", "Penalty moderate"),
            ("exclude", "Exclude"),
        ]):
            tk.Radiobutton(self.bo_frame, text=lbl, variable=self._bo_flat_var,
                           value=val).grid(row=row + ci // 2, column=2 * (ci % 2),
                                           columnspan=2, sticky="w")
        row += 2
        self._bo_feas_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.bo_frame,
                       text="Feasibility-aware EI (P(particles) classifier — needs ≥4 flat pts)",
                       variable=self._bo_feas_var).grid(
            row=row, column=0, columnspan=4, sticky="w")
        row += 1
        self._bo_ghosts_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.bo_frame,
                       text="Physics ghost points (seed GP with no-Ag / no-NaBH4 = flat penalty)",
                       variable=self._bo_ghosts_var).grid(
            row=row, column=0, columnspan=4, sticky="w")
        row += 1
        tk.Label(self.bo_frame, text="Ghost noise (alpha):").grid(
            row=row, column=0, sticky='w')
        self._bo_ghost_noise_entry = tk.Entry(self.bo_frame, width=10)
        self._bo_ghost_noise_entry.insert(0, "auto")
        self._bo_ghost_noise_entry.grid(row=row, column=1, sticky='w')
        tk.Label(self.bo_frame,
                 text="normalized units ; 'auto' = 0.05 ; or a fixed float",
                 fg=COLORS['text_light']).grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1
        tk.Label(self.bo_frame, text="Ghost penalty lift:").grid(
            row=row, column=0, sticky='w')
        self._bo_ghost_lift_entry = tk.Entry(self.bo_frame, width=10)
        self._bo_ghost_lift_entry.insert(0, "1.5")
        self._bo_ghost_lift_entry.grid(row=row, column=1, sticky='w')
        tk.Label(self.bo_frame,
                 text="ghosts sit this × above neighbours (1.0 = off) → soft slope",
                 fg=COLORS['text_light']).grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1

        # Hyperparameter prior: load an offline-tuned kernel (.pkl) as a warm
        # start for the live GP's hyperparameter optimisation (re-optimised as
        # points arrive). Produced by notebooks/kernel_benchmark_pca.ipynb.
        self.prior_kernel_file = None
        tk.Label(self.bo_frame, text="Hyperparameter prior (warm-start kernel)",
                 font=("Arial", 9, "bold")).grid(row=row, column=0, columnspan=4, sticky='w')
        row += 1
        self.prior_kernel_button = tk.Button(
            self.bo_frame, text="Load prior kernel (.pkl)",
            command=self.browse_prior_kernel)
        self.prior_kernel_button.grid(row=row, column=0, columnspan=2, sticky="w")
        self.prior_kernel_clear_button = tk.Button(
            self.bo_frame, text="Clear", command=self.clear_prior_kernel)
        self.prior_kernel_label = tk.Label(self.bo_frame, text="",
                                           fg=COLORS['text_light'])
        self.prior_kernel_label.grid(row=row, column=2, columnspan=2, sticky="w")
        row += 1

        # Cost shaping: log-warp + noise ceiling. The log warp fits the GP on
        # log(cost) so the low-cost valley stays resolvable (push toward ~0.1)
        # while the bad region compresses into a steep plateau. The noise ceiling
        # caps how much variance the GP may call 'noise' (normalized units).
        tk.Label(self.bo_frame, text="Cost shaping",
                 font=("Arial", 9, "bold")).grid(row=row, column=0, columnspan=4, sticky='w')
        row += 1
        self._bo_logcost_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.bo_frame,
                       text="Log cost warp (resolve the low-cost valley → push toward ~0.1)",
                       variable=self._bo_logcost_var).grid(
            row=row, column=0, columnspan=4, sticky="w")
        row += 1
        tk.Label(self.bo_frame, text="Noise floor (norm. var):").grid(
            row=row, column=0, sticky='w')
        self._bo_noise_floor_entry = tk.Entry(self.bo_frame, width=10)
        self._bo_noise_floor_entry.insert(0, "1e-3")
        self._bo_noise_floor_entry.grid(row=row, column=1, sticky='w')
        tk.Label(self.bo_frame,
                 text="WhiteKernel noise lower bound ; ~3% of signal std at 1e-3 (fine for cosine)",
                 fg=COLORS['text_light']).grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1
        tk.Label(self.bo_frame, text="Noise ceiling (norm. var):").grid(
            row=row, column=0, sticky='w')
        self._bo_noise_ceiling_entry = tk.Entry(self.bo_frame, width=10)
        self._bo_noise_ceiling_entry.insert(0, "10")
        self._bo_noise_ceiling_entry.grid(row=row, column=1, sticky='w')
        tk.Label(self.bo_frame,
                 text="cap on fitted WhiteKernel noise ; lower = trust the valley more",
                 fg=COLORS['text_light']).grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1
        tk.Label(self.bo_frame, text="Min length scale:").grid(
            row=row, column=0, sticky='w')
        self._bo_ls_min_entry = tk.Entry(self.bo_frame, width=10)
        self._bo_ls_min_entry.insert(0, "1.0")
        self._bo_ls_min_entry.grid(row=row, column=1, sticky='w')
        tk.Label(self.bo_frame,
                 text="floor on GP length scale (flow units) ; raise to kill a 'bed of nails'",
                 fg=COLORS['text_light']).grid(
            row=row, column=2, columnspan=2, sticky="w")
        row += 1

        # GP Kernel (used only when preset == "default")
        tk.Label(self.bo_frame, text="GP Kernel (default preset only)",
                 font=("Arial", 9, "bold")).grid(row=row, column=0, sticky='w')
        row += 1

        self._bo_kernel_var = tk.StringVar(value="RBF")
        self._bo_kernel_rbf = tk.Radiobutton(
            self.bo_frame, text="RBF (shared length scale)",
            variable=self._bo_kernel_var, value="RBF")
        self._bo_kernel_rbf.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bo_kernel_ard = tk.Radiobutton(
            self.bo_frame, text="ARD (per-dimension)",
            variable=self._bo_kernel_var, value="ARD")
        self._bo_kernel_ard.grid(row=row, column=2, columnspan=2, sticky="w")
        row += 1

        self._bo_kernel_warning = tk.Label(self.bo_frame, text="",
                                            font=("Arial", 8), fg=COLORS['warning'])
        self._bo_kernel_warning.grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        def _refresh_kernel_warning():
            preset = self._bo_preset_var.get()
            if preset == "robust":
                self._bo_kernel_warning.config(
                    text="\u26A0 Robust preset overrides kernel choice (Matern 2.5 isotropic).")
                self._bo_kernel_rbf.config(state='disabled')
                self._bo_kernel_ard.config(state='disabled')
            else:
                self._bo_kernel_rbf.config(state='normal')
                self._bo_kernel_ard.config(state='normal')
                if self._bo_kernel_var.get() == "ARD":
                    self._bo_kernel_warning.config(
                        text="\u26A0 ARD requires more data to avoid overfitting")
                else:
                    self._bo_kernel_warning.config(text="")

        # Length scale + Noise level: ignored when preset == "robust"
        self._bo_ls_label = tk.Label(self.bo_frame, text="Length scale:")
        self._bo_ls_label.grid(row=row, column=0, sticky='w')
        self._bo_ls_entry = tk.Entry(self.bo_frame, width=7)
        self._bo_ls_entry.insert(0, "1.0")
        self._bo_ls_entry.grid(row=row, column=1, sticky='w')
        row += 1

        self._bo_noise_label = tk.Label(self.bo_frame, text="Noise level:")
        self._bo_noise_label.grid(row=row, column=0, sticky='w')
        self._bo_noise_entry = tk.Entry(self.bo_frame, width=7)
        self._bo_noise_entry.insert(0, "0.001")
        self._bo_noise_entry.grid(row=row, column=1, sticky='w')
        row += 1

        # Huber delta: only used when preset == "robust"
        self._bo_huber_label = tk.Label(self.bo_frame, text="Huber delta (nm):")
        self._bo_huber_label.grid(row=row, column=0, sticky='w')
        self._bo_huber_entry = tk.Entry(self.bo_frame, width=7)
        self._bo_huber_entry.insert(0, "30.0")
        self._bo_huber_entry.grid(row=row, column=1, sticky='w')
        row += 1

        # GP restarts: shared by both presets (CPU-time vs. quality tradeoff)
        tk.Label(self.bo_frame, text="GP restarts:").grid(row=row, column=0, sticky='w')
        self._bo_restarts_entry = tk.Entry(self.bo_frame, width=7)
        self._bo_restarts_entry.insert(0, "100")
        self._bo_restarts_entry.grid(row=row, column=1, sticky='w')
        row += 1

        def _refresh_preset_widgets():
            preset = self._bo_preset_var.get()
            if preset == "robust":
                # length scale and noise are forced internally → grey out
                self._bo_ls_entry.config(state='disabled')
                self._bo_ls_label.config(fg=COLORS['text_light'])
                self._bo_noise_entry.config(state='disabled')
                self._bo_noise_label.config(fg=COLORS['text_light'])
                # Huber delta active
                self._bo_huber_entry.config(state='normal')
                self._bo_huber_label.config(fg=COLORS['text'])
            else:
                self._bo_ls_entry.config(state='normal')
                self._bo_ls_label.config(fg=COLORS['text'])
                self._bo_noise_entry.config(state='normal')
                self._bo_noise_label.config(fg=COLORS['text'])
                self._bo_huber_entry.config(state='disabled')
                self._bo_huber_label.config(fg=COLORS['text_light'])

        def _on_bo_kernel_change(*_):
            _refresh_kernel_warning()

        def _on_bo_preset_change(*_):
            _refresh_kernel_warning()
            _refresh_preset_widgets()

        self._bo_kernel_var.trace_add('write', _on_bo_kernel_change)
        self._bo_preset_var.trace_add('write', _on_bo_preset_change)
        _refresh_kernel_warning()
        _refresh_preset_widgets()

        # Pump bounds
        self.bounds_label = tk.Label(self.bo_frame, text="Pump bounds", font=("Arial", 9, "bold"))
        self.bounds_label.grid(row=row, column=0, sticky='w')
        [tk.Label(self.bo_frame, text='min' if i == 1 else 'max').grid(row=row, column=i) for i in [1, 2]]
        tk.Label(self.bo_frame, text='Fixed flows').grid(row=row, column=4)
        row += 1
        self.min_bound_entry = []
        self.max_bound_entry = []
        self.fixed_entry = []
        self.fixed_pumps_vars = [tk.BooleanVar(self.bo_frame) for _ in range(self.exp.number_of_pumps)]
        [self.fixed_pumps_vars[i].set(False) for i in range(self.exp.number_of_pumps)]
        for i in range(self.exp.number_of_pumps):
            tk.Label(self.bo_frame, text=f'Pump {i + 1}').grid(row=row + i, column=0)
            self.min_bound_entry.append(tk.Entry(self.bo_frame))
            self.min_bound_entry[-1].grid(row=row + i, column=1)
            self.min_bound_entry[-1].insert(0, self.exp.bounds[i][0])
            self.max_bound_entry.append(tk.Entry(self.bo_frame))
            self.max_bound_entry[-1].grid(row=row + i, column=2)
            self.max_bound_entry[-1].insert(0, self.exp.bounds[i][1])
            tk.Checkbutton(self.bo_frame, text='Fix pump', var=self.fixed_pumps_vars[i],
                           command=self.switch_fixed, onvalue=True, offvalue=False).grid(row=row + i, column=3)
            self.fixed_entry.append(tk.Entry(self.bo_frame))
            self.fixed_entry[-1].grid(row=row + i, column=4)
            self.fixed_entry[-1]['state'] = ['disabled']

        # Apply theme to all child widgets that weren't explicitly styled
        self._apply_bulk_theme()

        # Push the default spectrometer parameters shown in the GUI to the
        # hardware so that the user does not need to click "Update Parameters"
        # before measuring.
        try:
            self.update_spectro_parameters()
        except Exception as e:
            self.logger.warning(f"Could not apply default spectrometer parameters: {e}")

    def _apply_bulk_theme(self):
        """Walk all child widgets and apply uniform fonts / bg to un-styled ones."""
        skip_parents = {self._header, self._footer, self._exp_panel}
        for widget in self._all_descendants(self):
            if any(self._is_descendant_of(widget, p) for p in skip_parents):
                continue
            cls = widget.winfo_class()
            try:
                parent_bg = str(widget.master.cget('bg')) if hasattr(widget.master, 'cget') else COLORS['bg']
                if cls == 'Label':
                    current_font = str(widget.cget('font'))
                    if 'bold' not in current_font:
                        widget.config(font=FONTS['body'])
                    if str(widget.cget('bg')) not in (
                            COLORS['card_bg'], COLORS['header_bg'], COLORS['bg']):
                        widget.config(bg=parent_bg)
                elif cls == 'Entry':
                    _style_entry(widget)
                elif cls == 'Button':
                    current_font = str(widget.cget('font'))
                    if current_font in ('TkDefaultFont', ''):
                        widget.config(font=FONTS['body'])
                elif cls in ('Checkbutton', 'Radiobutton'):
                    widget.config(font=FONTS['body'], bg=parent_bg)
                elif cls == 'Frame':
                    cur_bg = str(widget.cget('bg'))
                    if cur_bg not in (COLORS['card_bg'], COLORS['header_bg'],
                                      COLORS['bg'], COLORS['card_border']):
                        widget.config(bg=parent_bg)
            except tk.TclError:
                pass

    @staticmethod
    def _is_descendant_of(widget, ancestor):
        w = widget
        while w is not None:
            if w is ancestor:
                return True
            w = w.master
        return False

    @staticmethod
    def _all_descendants(widget):
        """Recursively yield every child widget."""
        for child in widget.winfo_children():
            yield child
            yield from ConfigurationWindow._all_descendants(child)

    def _refresh_header_pump(self, i):
        fill = self._header_fill_vals[i]
        flow = self._header_flow_vals[i]
        self._header_pump_labels[i].config(
            text=f"P{i+1}: {round(fill)} \u00b5L | {flow:.2f} \u00b5L/s")

    def flush_nitric_acid(self):
        self.exp.flush_nitric_acid()

    def obj_switch(self):
        row = self.objective_label.grid_info().get('row') + 1
        # Hide all widgets
        # Single point target
        self.single_point_wl_label.grid_forget()
        self.single_point_abs_label.grid_forget()
        self.single_point_wl_entry.grid_forget()
        self.single_point_abs_entry.grid_forget()
        # Full spectrum target
        [label.grid_forget() for i, label in enumerate(self.full_spectrum_tg_flows_labels)]
        [entry.grid_forget() for entry in self.full_spectrum_tg_flows_entries]
        self.meas_target_button_frame.grid_forget()
        self.full_spectrum_load_button.grid_forget()
        self.target_path_label.grid_forget()
        self.build_target_button.grid_forget()
        self.cost_method_label.grid_forget()
        self.cost_method_l2.grid_forget()
        self.cost_method_cosine.grid_forget()
        if self.target_path_label is not None:
            self.target_path_label.grid_forget()
        # PCA widgets
        self.pca_load_button.grid_forget()
        self.pca_path_label.grid_forget()
        self.pca_varweight_check.grid_forget()
        self.pca_metric_label.grid_forget()
        self.pca_metric_menu.grid_forget()
        self.pca_ncomp_label.grid_forget()
        self.pca_ncomp_entry.grid_forget()
        self.pca_lambda_check.grid_forget()
        self.pca_lambda_weight_label.grid_forget()
        self.pca_lambda_weight_entry.grid_forget()
        self.pca_lambda_scale_label.grid_forget()
        self.pca_lambda_scale_entry.grid_forget()
        self.pca_amp_check.grid_forget()
        self.pca_amp_weight_label.grid_forget()
        self.pca_amp_weight_entry.grid_forget()
        self.warmstart_button.grid_forget()
        self.warmstart_path_label.grid_forget()
        self.warmstart_clear_button.grid_forget()

        # Show the relevant widgets
        if self.obj_var.get() == 'full_spectrum':
            [label.grid(row=row - 1, column=i + len(self.objective_list)) for i, label in
             enumerate(self.full_spectrum_tg_flows_labels)]
            [entry.grid(row=row, column=i + len(self.objective_list)) for i, entry in
             enumerate(self.full_spectrum_tg_flows_entries)]
            self.meas_target_button_frame.grid(row=row, column=self.exp.number_of_pumps + len(self.objective_list))
            self.full_spectrum_load_button.grid(row=row, column=self.exp.number_of_pumps + len(self.objective_list) + 1)
            if self.target_path_label != "":
                self.target_path_label.grid(row=row, column=self.exp.number_of_pumps + len(self.objective_list) + 2)
            # Show cost method selection
            self.cost_method_label.grid(row=row + 1, column=len(self.objective_list))
            self.cost_method_l2.grid(row=row + 1, column=len(self.objective_list) + 1)
            self.cost_method_cosine.grid(row=row + 1, column=len(self.objective_list) + 2)
        elif self.obj_var.get() == 'single_point':
            self.single_point_wl_label.grid(row=row - 1, column=len(self.objective_list))
            self.single_point_wl_entry.grid(row=row, column=len(self.objective_list))
            self.single_point_abs_label.grid(row=row - 1, column=len(self.objective_list) + 1)
            self.single_point_abs_entry.grid(row=row, column=len(self.objective_list) + 1)
        elif self.obj_var.get() == 'lambda_max':
            self.single_point_wl_label.grid(row=row - 1, column=len(self.objective_list))
            self.single_point_wl_entry.grid(row=row, column=len(self.objective_list))
        elif self.obj_var.get() == 'pca':
            n = len(self.objective_list)
            # Target spectrum to match (load a file or build one from physics)
            self.full_spectrum_load_button.grid(row=row, column=n)
            self.build_target_button.grid(row=row, column=n + 4)
            if self.target_spectrum_file:
                self.target_path_label.grid(row=row, column=n + 1)
            # Frozen PCA basis + options
            self.pca_load_button.grid(row=row, column=n + 2)
            self.pca_path_label.grid(row=row, column=n + 3)
            self.pca_varweight_check.grid(row=row + 1, column=n + 2)
            # Cost metric + number of components (own rows, no overlap with hybrids)
            self.pca_metric_label.grid(row=row + 1, column=n + 3)
            self.pca_metric_menu.grid(row=row + 1, column=n + 4)
            self.pca_ncomp_label.grid(row=row + 2, column=n + 3)
            self.pca_ncomp_entry.grid(row=row + 2, column=n + 4)
            # Optional hybrid λmax term (keeps amplitude + sharp peak position)
            self.pca_lambda_check.grid(row=row + 3, column=n + 2)
            self.pca_lambda_weight_label.grid(row=row + 3, column=n + 3)
            self.pca_lambda_weight_entry.grid(row=row + 3, column=n + 4)
            self.pca_lambda_scale_label.grid(row=row + 4, column=n + 3)
            self.pca_lambda_scale_entry.grid(row=row + 4, column=n + 4)
            # Optional hybrid amplitude term (discriminate by yield/concentration)
            self.pca_amp_check.grid(row=row + 5, column=n + 2)
            self.pca_amp_weight_label.grid(row=row + 5, column=n + 3)
            self.pca_amp_weight_entry.grid(row=row + 5, column=n + 4)
            # Optional warm-start from a folder of historical .h5
            self.warmstart_button.grid(row=row + 1, column=n)
            self.warmstart_path_label.grid(row=row + 1, column=n + 1)
            if self.warmstart_dir:
                self.warmstart_clear_button.grid(row=row + 2, column=n)

        if float(self.tol_entry.get()) == self.exp.default_tol[self.previous_obj]:
            self.tol_entry.delete(0, 'end')
            self.tol_entry.insert(0, self.exp.default_tol[self.obj_var.get()])
        self.previous_obj = self.obj_var.get()

    def next_sample(self):
        self.next_sample_pressed = True
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select target spectrum CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.target_spectrum_file = file_path
            # Show just the filename in the label for readability
            filename = file_path.split('/')[-1]
            self.target_path_label.config(text=f"Loaded: {filename}")
            row = self.full_spectrum_load_button.grid_info().get('row')
            column = self.full_spectrum_load_button.grid_info().get('column')
            self.target_path_label.grid(row=row, column=column + 1)
            if self.obj_var.get() == 'pca':
                self.obj_switch()  # refresh layout so the label shows for pca

    def _target_wl_grid(self):
        """Wavelength grid for a built target (spectrometer grid, else default)."""
        try:
            wl = np.asarray(self.exp.spectro.wavelengths, dtype=float)
            if wl.size > 10:
                return wl
        except Exception:
            pass
        return np.linspace(315.0, 750.0, 1024)

    def build_parametric_target_dialog(self):
        """Popup to build a physics-defined target (lambda_max / FWHM / amplitude)."""
        win = tk.Toplevel(self.bo_frame)
        win.title("Build physics target")
        fields = [("lambda_max (nm) — size", "420"),
                  ("FWHM (nm) — dispersity", "60"),
                  ("amplitude — yield", "1.0")]
        entries = []
        for r, (lab, default) in enumerate(fields):
            tk.Label(win, text=lab).grid(row=r, column=0, sticky="w", padx=6, pady=3)
            e = tk.Entry(win, width=10); e.insert(0, default)
            e.grid(row=r, column=1, padx=6, pady=3); entries.append(e)
        shape_var = tk.StringVar(value="lorentzian")
        tk.Label(win, text="lineshape").grid(row=3, column=0, sticky="w", padx=6)
        tk.OptionMenu(win, shape_var, "lorentzian", "gaussian").grid(
            row=3, column=1, sticky="w", padx=6)

        def create():
            try:
                lam, fwhm, amp = (float(e.get()) for e in entries)
            except ValueError:
                messagebox.showerror("Build target", "Invalid numeric values.")
                return
            self._build_parametric_target(lam, fwhm, amp, shape_var.get())
            win.destroy()
        tk.Button(win, text="Create target", command=create).grid(
            row=4, column=0, columnspan=2, pady=8)

    def _build_parametric_target(self, lambda_max, fwhm, amplitude, lineshape):
        import tempfile
        from toofast.core.target_builder import save_parametric_target_csv
        wl = self._target_wl_grid()
        path = os.path.join(tempfile.gettempdir(), "toofast_parametric_target.csv")
        save_parametric_target_csv(path, wl, lambda_max, fwhm, amplitude, lineshape)
        self.target_spectrum_file = path
        self.target_path_label.config(
            text=f"Target: λmax={lambda_max:g} FWHM={fwhm:g} A={amplitude:g} ({lineshape})")
        self.logger.info(
            "Built physics target: lambda_max=%s FWHM=%s amp=%s %s -> %s",
            lambda_max, fwhm, amplitude, lineshape, path)
        if self.obj_var.get() in ('pca', 'full_spectrum'):
            self.obj_switch()

    def browse_pca_file(self):
        file_path = filedialog.askopenfilename(
            title="Select frozen PCA basis (.npz)",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")]
        )
        if file_path:
            self.pca_model_file = file_path
            self.pca_path_label.config(text=f"Basis: {os.path.basename(file_path)}")

    def browse_prior_kernel(self):
        file_path = filedialog.askopenfilename(
            title="Select offline-tuned kernel (.pkl)",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            self.prior_kernel_file = file_path
            self.prior_kernel_label.config(
                text=f"Prior: {os.path.basename(file_path)}")
            self.prior_kernel_clear_button.grid(
                row=self.prior_kernel_button.grid_info().get('row'), column=4)

    def clear_prior_kernel(self):
        self.prior_kernel_file = None
        self.prior_kernel_label.config(text="")
        self.prior_kernel_clear_button.grid_forget()

    def browse_warmstart_dir(self):
        dir_path = filedialog.askdirectory(title="Select folder of historical .h5 samples")
        if not dir_path:
            return
        self.warmstart_dir = dir_path
        self.warmstart_path_label.config(text=f"Warm-start: {os.path.basename(dir_path)}")
        # Build the PCA target now, then seed the BO from the historical samples.
        try:
            self.update_bo_params()
            flows, costs = self.exp.warm_start_from_h5(dir_path)
            # Warm-start supplies the initial design: no init phase runs, and
            # maxiter now counts NEW BO iterations. Disable the init-points input
            # so the user isn't asked for something that is ignored.
            try:
                self.nb_init_samples_entry.config(state='disabled')
                self.nb_init_samples_label.config(
                    text='initial points (from warm-start)')
            except Exception:
                pass
            messagebox.showinfo(
                "Warm-start",
                f"Seeded BO with {len(costs)} historical points\n"
                f"Best PCA cost: {costs.min():.5f}\n"
                f"Init phase skipped — 'maxiter' now counts new BO iterations.",
            )
        except Exception as e:
            messagebox.showerror("Warm-start failed", str(e))
        else:
            # Reveal the 'Clear warm-start' button now that a folder is set.
            try:
                row = self.warmstart_button.grid_info().get('row')
                col = self.warmstart_button.grid_info().get('column')
                if row is not None:
                    self.warmstart_clear_button.grid(row=int(row) + 1, column=int(col))
            except Exception:
                pass

    def clear_warmstart(self):
        """Deselect the warm-start folder and revert to a cold start.

        Clears the seeded data on the experiment, re-enables the init-points
        input and hides the clear button. Safe to call even if nothing was set.
        """
        self.warmstart_dir = None
        self.exp.previous_flows = None
        self.exp.previous_costs = None
        self.exp._n_warmstart = 0
        try:
            self.warmstart_path_label.config(text='')
            self.nb_init_samples_entry.config(state='normal')
            self.nb_init_samples_label.config(text='initial points')
            self.warmstart_clear_button.grid_forget()
        except Exception:
            pass
        self.logger.info("Warm-start folder deselected — back to cold start.")

    def init_flows_switch(self):
        if self.init_var.get() == 'fixed':
            self.update_bo_grid(from_entry=False)
            row = self.nb_init_samples_label.grid_info().get('row')
            [label.grid(row=row - 1, column=i + 3 + len(self.init_val)
                        ) for i, label in enumerate(self.init_pump_labels)]
            [label.grid(row=row + i, column=2 + len(self.init_val)
                        ) for i, label in enumerate(self.init_samples_labels)]
            for i, entry_set in enumerate(self.init_flows_entries):
                for j, entry in enumerate(entry_set):
                    entry.grid(row=row + i, column=j + 3 + len(self.init_val))
            self.check_flow_sum()
        else:
            [label.grid_forget() for label in self.init_pump_labels]
            [label.grid_forget() for label in self.init_samples_labels]
            [label.grid_forget() for label in self.check_flows_labels]
            for entry in [item for innerlist in self.init_flows_entries for item in innerlist]:
                entry.grid_forget()

    def update_bo_grid(self, from_entry=True):
        nb_samples = int(self.nb_init_samples_entry.get())
        if len(self.init_samples_labels) < nb_samples:
            while nb_samples != len(self.init_samples_labels):
                self.init_samples_labels.append(
                    tk.Label(self.bo_frame, text=f'Sample {len(self.init_samples_labels) + 1}'))
                self.init_flows_entries.append([tk.Entry(self.bo_frame) for _ in range(self.exp.number_of_pumps)])
                self.check_flows_labels.append(
                    tk.Label(self.bo_frame, text=f'Sum of flows must be {self.exp.total_flow}.'))
            self.check_flow_sum()
            if from_entry:
                self.init_flows_switch()
        else:
            while nb_samples != len(self.init_samples_labels):
                self.init_samples_labels[-1].grid_forget()
                [entry.grid_forget() for entry in self.init_flows_entries[-1]]
                self.check_flows_labels[-1].grid_forget()
                self.init_samples_labels.pop()
                self.init_flows_entries.pop()
                self.check_flows_labels.pop()


    def check_flow_sum(self):
        [label.grid_forget() for label in self.check_flows_labels]
        row = self.nb_init_samples_label.grid_info().get('row')
        for i, set in enumerate(self.init_flows_entries):
            flows = [0 if entry.get() == "" else float(entry.get())for entry in set]
            if self.exp.total_flow != round(sum(flows),2) or not all([0 <= flow <= self.exp.total_flow for flow in flows]):
                self.check_flows_labels[i].grid(row=row + i, column=self.exp.number_of_pumps + 6)
    # Known issue: rounding errors can cause sums equal to total_flow to be rejected
    
    def switch_fixed(self):
        for i in range(self.exp.number_of_pumps):
            if self.fixed_pumps_vars[i].get():
                self.min_bound_entry[i]['state'] = 'disabled'
                self.max_bound_entry[i]['state'] = 'disabled'
                self.fixed_entry[i]['state'] = 'normal'
            else:
                self.min_bound_entry[i]['state'] = 'normal'
                self.max_bound_entry[i]['state'] = 'normal'
                self.fixed_entry[i]['state'] = 'disabled'

    def check_for_pump_faults(self):
        if not hasattr(self, 'fault_displayed'):
            self.fault_displayed = False

        if self.exp.pump_syst.check_for_faults():
            self.pump_fault_label.config(text="Pump Fault Status: Faults detected", fg=COLORS['danger'])
            self.logger.error("Pump fault detected")
            self.stop_flow()
            if not self.fault_displayed:
                tk.messagebox.showerror("Pump Fault", "Pump fault detected, all pumps have been stopped.")
                self.fault_displayed = True
        else:
            self.pump_fault_label.config(text="Pump Fault Status: No faults", fg=COLORS['success'])
            self.fault_displayed = False
        after_id = self.after(1000, self.check_for_pump_faults)
        self.after_ids.append(after_id)

    def clear_pump_faults(self):
        self.exp.pump_syst.clear_all_faults()
        self.logger.info("Pump faults cleared")

    def update_calibration_status(self):
        status_list = []
        for pump in self.exp.pump_syst.pumps:
            if pump.is_simulation:
                status_list.append(True)
            else:
                status_list.append(pump.is_position_sensing_initialized())
        if all(status_list):
            self.calibration_status_label.config(text="Calibration Status: OK", fg=COLORS['success'])
        else:
            self.calibration_status_label.config(text="Calibration Status: Not calibrated", fg=COLORS['danger'])
        after_id = self.after(3000, self.update_calibration_status)
        self.after_ids.append(after_id)

    def calibrate_pumps(self):
        response = tk.messagebox.askokcancel("Calibration Warning",
                                             "Pumps are about to be calibrated. "
                                             "Please remove any syringe from pump to avoid damage.")
        if response:
            self.exp.pump_syst.calibration_all()
            self.logger.info("Pumps have been calibrated.")
        self.logger.info("Pumps calibration")

    def _schedule_main_thread_gc(self, interval_ms: int = 2000):
        """Run gc.collect() on the Tk main thread on a fixed cadence.

        Worker threads create no FigureCanvasTkAgg PhotoImages directly, but
        references to those Tk-bound objects can sit in unreachable cycles
        until Python's GC walks them. If GC runs while a worker thread is
        active, finalizers fire on the worker thread and Tcl panics. Calling
        gc.collect() here keeps that finalization on the main thread.
        """
        try:
            gc.collect()
        except Exception:
            pass
        try:
            after_id = self.after(interval_ms, self._schedule_main_thread_gc, interval_ms)
            self.after_ids.append(after_id)
        except Exception:
            # Tk is being torn down; nothing to do.
            pass

    def get_syringe_levels(self):
        fill_levels = self.exp.pump_syst.get_fill_levels()
        for i, fill_level in enumerate(fill_levels):
            self.syringe_level_label[i].config(text=f"{round(fill_level)} \u00b5L")
            self._header_fill_vals[i] = fill_level
            self._refresh_header_pump(i)
            max_vol = self.exp.pump_syst.pumps[i].max_volume if hasattr(
                self.exp.pump_syst.pumps[i], 'max_volume') else 5000
            pct = max(0, min(100, fill_level / max_vol * 100))
            self._pump_fill_bars[i]['value'] = pct
            bar_style = 'Danger.Horizontal.TProgressbar' if pct < 25 else 'Teal.Horizontal.TProgressbar'
            self._pump_fill_bars[i].config(style=bar_style)
            if fill_level <= 100 and not self.pump_stop_alert_shown:
                self.pump_stop_alert_shown = True
                self.logger.warning(f"Syringe {i + 1} low fill level: {round(fill_level)} µL")
            elif all(level > 100 for level in fill_levels):
                self.pump_stop_alert_shown = False
        after_id = self.after(100, self.get_syringe_levels)
        self.after_ids.append(after_id)

    def refill(self):
        # Get current syringe levels
        syringe_levels = self.exp.pump_syst.get_fill_levels()

        # Get target fill levels from user
        target_fill_levels = [float(entry.get()) for entry in self.pump_refill_entry]
        refill_flow = float(self.refill_flow_entry.get())

        # Check if refill flow is too high
        if refill_flow > 100:
            message = tk.messagebox.askokcancel("Flow rate warning",
                                                "Flow rate is too high for direct refill. Do you want to proceed?")
            if not message:
                return
        self.switch_refill_valve_to_reagents()
        time.sleep(3)  # Let the valve switch
        # Set fill level for each pump if target level is higher than current level
        for i, level in enumerate(syringe_levels):
            if target_fill_levels[i] > level:
                self.exp.pump_syst.pumps[i].set_fill_level(target_fill_levels[i], refill_flow)
        self.logger.info(f"Pumps refill to levels: {target_fill_levels} at flow rate: {refill_flow} µL.s-1")

    def get_current_flows(self):
        current_flows = self.exp.pump_syst.get_flows()
        for i in range(self.exp.number_of_pumps):
            flow_str = f"{round(current_flows[i], 2)} \u00b5L/s"
            self.current_flow_label[i].config(text=flow_str)
            self._header_flow_vals[i] = current_flows[i]
            self._refresh_header_pump(i)
        after_id = self.after(100, self.get_current_flows)
        self.after_ids.append(after_id)

    def _update_flow_mode_label(self):
        if self.wash_mode_var.get():
            self._flow_mode_label.config(
                text="  WASH / PURGE  ", bg=COLORS['danger'])
        else:
            self._flow_mode_label.config(
                text="  SYNTHESIS  ", bg=COLORS['success'])

    def start_flow(self):
        flows = [float(entry.get()) for entry in self.pump_flow_entry]
        total = sum(flows)
        if not self.wash_mode_var.get():
            max_flow = float(self.max_flow_entry.get()) if hasattr(self, 'max_flow_entry') else 14.0
            if total > max_flow:
                proceed = tk.messagebox.askokcancel(
                    "Flow rate warning",
                    f"Total flow ({total:.1f} µL/s) exceeds synthesis limit "
                    f"({max_flow:.0f} µL/s).\n"
                    "Enable Wash/Purge mode for higher flows.\n\n"
                    "Proceed anyway?")
                if not proceed:
                    return
        self.exp.pump_syst.set_flows(flows)
        self.logger.info("Pumps started at the following rates: " + str(flows))

    def start_random_flow(self, is_exp=False):
        rd_flows = self.exp.pump_syst.generate_random_flows()
        self.exp.pump_syst.set_flows(rd_flows)
        if is_exp:
            path = self.exp.experiment_path
        else:
            path=self.experiment_path
        self.exp.pump_syst.write_flows(path)
        self.wait_for_steady_state()
        self.logger.info("Random flows: " + str(self.exp.pump_syst.get_flows()))

    def stop_flow(self):
        self._prime_abort = True
        self.exp.pump_syst.stop_all_pumps()

    def measure_reference(self):
        file_path = os.path.join(self.experiment_path, "reference_spectrum.csv")
        overwrite = False
        if os.path.exists(file_path):
            overwrite = tk.messagebox.askokcancel("Reference spectrum",
                                                  "A reference spectrum has already been measured. "
                                                  "Do you want to measure a new one?")
        self.exp.measure_reference(overwrite)
        self.measure_reference_frame.config(highlightbackground="green", highlightcolor="green")
        self.logger.info("Reference spectrum measured")

    def measure_target(self):
        tg_flows = np.array([0 if entry.get() == '' else float(entry.get()) for entry in
                             self.full_spectrum_tg_flows_entries])
        if sum(tg_flows) == 8 \
                and all([0 <= flow <= self.exp.total_flow for flow in tg_flows]) \
                and self.exp.spectro.reference is not None:
            proceed = True
            if self.exp.tg_spectrum is not None:
                proceed = tk.messagebox.askokcancel("Target warning",
                                                    "A target spectrum is already loaded. "
                                                    "Do you want to overwrite it ?")
            if proceed:
                threading.Thread(target=lambda: self.exp.make_flow_and_measure(tg_flows, meas_type='target'),
                                 daemon=True).start()
                self.meas_target_button_frame.config(highlightbackground="green", highlightcolor="green")
        else:
            tk.messagebox.showwarning("Cannot measure target",
                                      f"The flows don't add up to {self.exp.total_flow} "
                                      f"or reference spectrum missing")

    def update_spectro_parameters(self):
        integration_time = float(self.int_time_entry.get())
        number_average = int(self.average_number_entry.get())
        smoothing = int(self.smoothing_entry.get())
        min_wl = float(self.min_wl_entry.get())
        max_wl = float(self.max_wl_entry.get())
        self.exp.spectro.update_parameters(integration_time=integration_time, number_average=number_average, smoothing=smoothing,
                                           start_wl = min_wl, end_wl = max_wl)
        self.logger.info(f"Spectrometer parameters updated with integration time: {integration_time} ms "
                         f"and number of averages: {number_average} between {min_wl} and {max_wl} nm")

    def UMI(self, text):
        self.ax.set_ylim([0, float(text)])

    def animate(self, frame):
        try:
            # Use spectrum-specific variables
            if hasattr(self, 'spectrum_line'):
                self.spectrum_line.set_data(self.spectrum_xs, self.spectrum_ys)
                return self.spectrum_line,
        except Exception as e:
            self.logger.error(f"Error in animate: {e}")
            return []

    def plot_spectrum_command(self):
        # Create local variables instead of using class attributes
        spectrum_fig, spectrum_ax = plt.subplots()
        spectrum_ax.set_title('SCOPE')
        spectrum_ax.set_ylabel('Intensity')
        spectrum_ax.set_xlabel('Wavelengths')
        spectrum_ax.set_xlim([min(self.exp.spectro.wavelengths), max(self.exp.spectro.wavelengths)])
        spectrum_ax.set_ylim([-0.5, 1.5])

        # Store references locally for this function
        self.spectrum_fig = spectrum_fig
        self.spectrum_ax = spectrum_ax
        self.spectrum_xs = []
        self.spectrum_ys = []
        self.spectrum_line, = spectrum_ax.plot(self.spectrum_xs, self.spectrum_ys)

        spectrum_fig.canvas.mpl_disconnect(spectrum_fig.canvas.manager.key_press_handler_id)
        textbox_ax = plt.axes([0.3, 0.05, 0.15, 0.03])
        textbox = TextBox(textbox_ax, 'Integration Time(ms)  ', initial=str(self.int_time_entry.get()))
        textbox.on_submit(self.update_spectro_parameters)

        textbox2_ax = plt.axes([0.6, 0.05, 0.15, 0.03])
        textbox2 = TextBox(textbox2_ax, 'Max Intensity  ', initial=str(1.5))
        textbox2.on_submit(self.UMI)

        # Use local figure for animation. 100 ms is plenty fast for live
        # spectrum display and avoids hammering the Tk main loop / matplotlib
        # backend, which previously contributed to threading races.
        self.spectrum_animation = FuncAnimation(spectrum_fig, self.animate, interval=100)

    def plot_fig(self, i, flows, costs, gaussian_process, best_min, times, spectrum,
                 predicted_flows=None, final_fig=False, save_figs=True):
        '''
        Render a full figure for the bayesian experimentation
        '''
        # Close previous iteration figure to avoid accumulation
        if hasattr(self, '_bo_fig') and self._bo_fig is not None:
            try:
                plt.close(self._bo_fig)
            except Exception:
                pass
            self._bo_fig = None

        fig = plot_fig(self.exp, i, flows, costs, gaussian_process, best_min, times, spectrum,
                       predicted_flows, final_fig, save_figs)
        if fig is not None:
            self._bo_fig = fig
            # If the user closes the BO figure window, drop our reference so
            # we don't try to plt.close() an already-destroyed figure later.
            try:
                fig.canvas.mpl_connect('close_event', lambda _e: setattr(self, '_bo_fig', None))
            except Exception:
                pass
            plt.show(block=False)


    def plot_spectrum(self, spectrum, save_dir=None, title='spectrum'):
        '''
        for a single spectrum plot
        '''
        spectrum.plot_spectrum(save_dir=save_dir, title=title)

    def measure_spectrum(self, sample_id=""):
        self.exp.measure_spectrum(sample_id)
        self.logger.info("Spectrum measured")

    def smart_plot_spectrum(self, sample_id=""):
        smart_spectrum = self.exp.smart_plot_spectrum(sample_id)
        self.logger.info("Smart Spectrum measured")
        smart_spectrum.save_data(self.experiment_path, sample_id)
        self.logger.info("Spectrum saved")

    def get_sample_level(self):
        if self.exp.pump_syst.pumps[0].is_simulation:
            vol = 0
        else:
            vol = round(sum([pump.get_dosed_volume() for pump in self.exp.pump_syst.pumps]), 1)
            self.sample_level.config(text=vol)
            after_id = self.after(1000, self.get_sample_level)
            self.after_ids.append(after_id)
        return vol

    def check_sample_level(self):
        level_max = float(self.sample_volume_entry.get())
        if self.get_sample_level() >= level_max:
            self.logger.info("Sample filled")
            # Proceed with other tasks
            # Set a flag or call other functions as needed
            self.exp.collection_status = False
            self.logger.info("Sample collection complete")
        else:
            # If condition not met, schedule the check again after 5 seconds
            after_id = self.after(500, self.check_sample_level)
            self.after_ids.append(after_id)

    def prime_pump(self, pump_index=0):
        self.stop_flow()
        if pump_index < self.exp.number_of_pumps:
            pump = self.exp.pump_syst.pumps[pump_index]
            pump.generate_flow(pump.max_flow)
            after_id = self.after(5000, self.prime_pump, pump_index + 1)  # Schedule the next pump action after 5 seconds
            self.after_ids.append(after_id)
        else:
            self.exp.prime_status = True
            self.logger.info("All pumps primed")

    def prime_pump_cycle(self):
        self.prime_pump_cycle_button.config(state='disabled')
        self._prime_abort = False
        threading.Thread(target=self._prime_pump_cycle_worker, daemon=True).start()

    def _prime_pump_cycle_worker(self):
        try:
            self.exp.pump_syst.stop_all_pumps()
            self.switch_refill_valve_to_reagents()
            time.sleep(2)
            for cycle in range(3):
                if self._prime_abort:
                    break
                self.logger.info(f"Prime cycle {cycle + 1}/3: aspirating 1200 µL")
                for pump in self.exp.pump_syst.pumps:
                    pump.set_fill_level(1200, 100)
                while any(pump.is_pumping() for pump in self.exp.pump_syst.pumps):
                    if self._prime_abort:
                        self.stop_flow()
                        return
                    time.sleep(0.2)
                time.sleep(3)
                if self._prime_abort:
                    break
                self.logger.info(f"Prime cycle {cycle + 1}/3: dispensing to 0")
                for pump in self.exp.pump_syst.pumps:
                    pump.set_fill_level(0, 100)
                while any(pump.is_pumping() for pump in self.exp.pump_syst.pumps):
                    if self._prime_abort:
                        self.stop_flow()
                        return
                    time.sleep(0.2)
            self.stop_flow()
            self.exp.prime_status = True
            self.logger.info("Pumps primed")
        except Exception as e:
            self.logger.error(f"Prime cycle error: {e}")
            self.stop_flow()
        finally:
            self.call_soon(lambda: self.prime_pump_cycle_button.config(state='normal'))

    def wait_for_steady_state(self):
        start_time = time.time()
        self._current_phase = "Stabilizing"
        while time.time() - start_time < 2.5 * self.exp.residence_time:
            self.update()
            time.sleep(0.01)
        self._current_phase = "Measuring"

    def collect_sample(self):
        self.logger.info("start collection")
        time.sleep(6)
        self.logger.info("Sample collected")

    def update_collection_status(self):
        if self.exp.collection_status:
            self.collection_status_label.config(text="Collecting", fg=COLORS['success'])
            self._header_collection_label.config(text="Collecting", fg=COLORS['success'])
        else:
            self.collection_status_label.config(text="Not collecting", fg=COLORS['danger'])
            self._header_collection_label.config(text="Not collecting", fg=COLORS['text_light'])
        after_id = self.after(500, self.update_collection_status)
        self.after_ids.append(after_id)

    def update_state_status(self):
        phase = self._current_phase
        phase_colors = {
            'Idle': COLORS['text_secondary'],
            'Pumping': COLORS['primary'],
            'Stabilizing': COLORS['warning'],
            'Measuring': COLORS['success'],
            'Cleaning': COLORS['teal'],
            'Collecting': COLORS['success'],
            'Paused': COLORS['warning'],
        }
        color = phase_colors.get(phase, COLORS['text_secondary'])
        self.state_label.config(text=phase, fg=color)
        self._header_state_label.config(text=phase, fg=color,
                                        bg='#334155')
        after_id = self.after(500, self.update_state_status)
        self.after_ids.append(after_id)

    def set_phase(self, phase):
        self._current_phase = phase

    def _toggle_advanced(self):
        if self._adv_visible.get():
            self._adv_frame.grid_remove()
            self._adv_toggle_btn.config(text="\u25B8 Advanced Parameters")
            self._adv_visible.set(False)
        else:
            self._adv_frame.grid()
            self._adv_toggle_btn.config(text="\u25BE Advanced Parameters")
            self._adv_visible.set(True)

    def _apply_advanced_params(self):
        from toofast.config import TerminationConfig
        try:
            self.exp.termination = TerminationConfig(
                correlation_threshold=float(self._adv_corr_entry.get()),
                amplitude_threshold=float(self._adv_amp_entry.get()),
                min_spectra=int(self._adv_min_spectra_entry.get()),
                monitoring_timeout_s=float(self._adv_timeout_entry.get()),
            )

            self.logger.info(
                f"Advanced params applied: corr={self.exp.termination.correlation_threshold}, "
                f"amp={self.exp.termination.amplitude_threshold}"
            )
        except ValueError as e:
            tk.messagebox.showerror("Invalid parameter", str(e))

    def update_bo_params(self):
        init_flows = np.array([[0 if entry.get() == '' else float(entry.get()) for entry in
                                entry_set] for entry_set in self.init_flows_entries])
        target_init = np.array([0 if entry.get() == '' else float(entry.get()) for entry in
                                self.full_spectrum_tg_flows_entries])
        bounds = np.array([(float(self.min_bound_entry[i].get()), float(self.max_bound_entry[i].get())) for i in
                           range(self.exp.number_of_pumps)])
        fixed_flows = np.array([0 if flow.get() == '' else float(flow.get()) for flow in self.fixed_entry])

        self.exp.update_optim_params(maxiter=int(self.maxiter_entry.get()),
                                     residence_time=float(self.residence_time_entry.get()),
                                     reagents=np.array([reagent.get() for reagent in self.selected_reagents]),
                                     nb_init_samples=int(self.nb_init_samples_entry.get()),
                                     init_option=self.init_var.get(),
                                     init_flows=init_flows,
                                     target_style=self.obj_var.get(),
                                     target_init=target_init,
                                     tg_wl=float(self.single_point_wl_entry.get()),
                                     tg_abs=float(self.single_point_abs_entry.get()),
                                     tolerance=float(self.tol_entry.get()), rel_tol=self.relative_tol_var.get(),
                                     tol_stop=self.tol_stop_var.get(),
                                     acq_func=self.acq_func_var.get(),
                                     bounds=bounds,
                                     fixed_pumps=np.array([fix.get() for fix in self.fixed_pumps_vars]),
                                     fixed_flows=fixed_flows,
                                     target_spectrum_file=self.target_spectrum_file,
                                     cost_method=self.cost_method_var.get(),
                                     pca_model_file=self.pca_model_file,
                                     pca_variance_weighted=bool(self.pca_variance_weighted_var.get()),
                                     pca_lambda_weight=(float(self.pca_lambda_weight_entry.get())
                                                        if self.pca_lambda_var.get() else 0.0),
                                     pca_lambda_scale_nm=float(self.pca_lambda_scale_entry.get()),
                                     pca_amp_weight=(float(self.pca_amp_weight_entry.get())
                                                     if self.pca_amp_var.get() else 0.0),
                                     pca_cost_metric=self.pca_metric_var.get(),
                                     pca_n_components=int(float(self.pca_ncomp_entry.get() or 0)),
                                     )

        bo_preset = self._bo_preset_var.get()
        self.exp.bo_preset_override = bo_preset
        # Cost shaping: log warp + noise ceiling (see Experiment.cost_transform).
        self.exp.cost_transform = 'log' if self._bo_logcost_var.get() else 'none'
        try:
            self.exp.white_noise_floor = float(self._bo_noise_floor_entry.get())
        except ValueError:
            self.logger.warning("Invalid noise floor; keeping %s",
                                self.exp.white_noise_floor)
        try:
            self.exp.white_noise_ceiling = float(self._bo_noise_ceiling_entry.get())
        except ValueError:
            self.logger.warning("Invalid noise ceiling; keeping %s",
                                self.exp.white_noise_ceiling)
        try:
            self.exp.length_scale_min = float(self._bo_ls_min_entry.get())
        except ValueError:
            self.logger.warning("Invalid min length scale; keeping %s",
                                self.exp.length_scale_min)
        # Hyperparameter prior: warm-start the live kernel from an offline-tuned
        # one (re-optimised live; see Experiment.load_prior_kernel).
        if self.prior_kernel_file:
            try:
                self.exp.load_prior_kernel(self.prior_kernel_file)
            except Exception as e:
                messagebox.showwarning(
                    "Prior kernel",
                    f"Could not load prior kernel:\n{e}\nContinuing without it.")
        self.exp.flat_strategy_override = self._bo_flat_var.get()
        self.exp.feasibility_aware_override = bool(self._bo_feas_var.get())
        self.exp.use_physics_ghosts = bool(self._bo_ghosts_var.get())
        _gn_raw = self._bo_ghost_noise_entry.get().strip()
        if _gn_raw.lower() in ("", "auto"):
            self.exp.ghost_noise_level = "auto"
        else:
            try:
                self.exp.ghost_noise_level = float(_gn_raw)
            except ValueError:
                self.logger.warning(
                    "Invalid ghost noise '%s' — falling back to 'auto'.", _gn_raw)
                self.exp.ghost_noise_level = "auto"
        try:
            self.exp.ghost_penalty_lift = float(self._bo_ghost_lift_entry.get())
        except ValueError:
            self.logger.warning("Invalid ghost penalty lift; keeping %s",
                                self.exp.ghost_penalty_lift)
        self.logger.info(
            "Flat-spectrum strategy: %s | feasibility-aware EI: %s | "
            "physics ghosts: %s | ghost noise: %s",
            self.exp.flat_strategy_override, self.exp.feasibility_aware_override,
            self.exp.use_physics_ghosts, self.exp.ghost_noise_level)
        try:
            self.exp.huber_delta_nm_override = float(self._bo_huber_entry.get())
        except (ValueError, AttributeError):
            self.exp.huber_delta_nm_override = 30.0
        kernel_type = self._bo_kernel_var.get()
        if bo_preset == "robust":
            self.logger.info(
                "Robust BO preset enabled: Matern 2.5 isotropic + Huber loss "
                "(delta=%.1f nm). The GP Kernel / Length scale / Noise level "
                "entries are ignored in this mode.",
                self.exp.huber_delta_nm_override,
            )
        elif kernel_type != "RBF":
            self.logger.warning(f"Kernel changed to {kernel_type}. Non-default kernels may require more data.")
        self.exp.kernel_type_override = kernel_type
        self.exp.rbf_length_scale_override = float(self._bo_ls_entry.get())
        self.exp.white_noise_override = float(self._bo_noise_entry.get())
        self.exp.gp_n_restarts_override = int(self._bo_restarts_entry.get())

    def start_exp(self):
        exp_type = self.exp_type_var.get()
        self.logger.debug("Starting experiment type: %s", exp_type)
        if self.exp.spectro.reference == None:
            tk.messagebox.showwarning("Experiment warning", "No reference spectrum. Aborting experiment")
            return

        self.update_spectro_parameters()

        if hasattr(self.exp, 'c3000_1') and self.exp.c3000_1.con_status:
            self.exp.c3000_1.switch_valve_to_position("left", verify=False)
        if hasattr(self.exp, 'c3000_2') and self.exp.c3000_2.con_status:
            self.exp.c3000_2.switch_valve_to_position("left", verify=False)

        if exp_type == 'Bayesian Optimization':
            self.update_bo_params()
            is_ready, fault_reasons = self.exp.check_if_ready()
            if not is_ready:
                tk.messagebox.showwarning("Bayesian experiment", "BO's initialisation went wrong:" + fault_reasons)
                return

        else:
            if not self.exp.prime_status:
                response = tk.messagebox.askokcancel("Prime warning",
                                                     "Pumps are not primed. Are you sure you want to proceed?")
                if not response:
                    return
            # check for collector status
            if not (self.exp.collector.con_status and self.exp.collector.init_status):
                response = tk.messagebox.askokcancel("Collector warning",
                                                     "Fraction collector is not ready. "
                                                     "Are you sure you want to proceed?")
                if not response:
                    return

        # Create a sub-folder
        i = 0
        exp_name = f'{self.experiment_name}_{i}'
        exp_path = os.path.join(self.experiment_path, exp_name)
        while os.path.exists(exp_path):
            i += 1
            exp_name = f'{self.experiment_name}_{i}'
            exp_path = os.path.join(self.experiment_path, exp_name)
        os.makedirs(exp_path + ('/figures' if exp_type == 'Bayesian Optimization' else ''))
        self.exp.experiment_name = exp_name
        self.exp.experiment_path = exp_path
        self.save_parameters()
        self.data_manager = DataManager(experiment_path=exp_path, experiment_name=exp_name,
                                        logger=self.logger)

        self.logger.info(f'Start {exp_type} experiment: {self.exp.experiment_name}_{i}')
        self.abort_exp_flag = False
        self.exp._abort_event.clear()
        self.pause_exp_button['state'] = 'normal'
        self.abort_exp_button['state'] = 'normal'
        _style_button(self.abort_exp_button, 'danger')
        if exp_type == 'Fixed':
            threading.Thread(target=self.exp.start_exp_fixed, daemon=True).start()
        elif exp_type in ['Random', 'LHS Flows', 'Projected LHS Flows']:
            threading.Thread(target=self.exp.start_exp_random, daemon=True).start()
        elif exp_type == 'CSV Flows':
            if not self.csv_flow_file:
                tk.messagebox.showwarning("CSV Flows", "Please load a CSV flow file first.")
                return
            threading.Thread(target=self.exp.start_exp_from_csv,
                             args=(self.csv_flow_file,), daemon=True).start()
        elif exp_type == 'Bayesian Optimization':
            self.logger.debug("BO")
            threading.Thread(target=self.exp.stop_flow, daemon=True).start()
        else:
            self.logger.error(f'No matching experiment type: {exp_type}')

    def pause_exp(self):
        self.logger.warning("Pause button pressed")
        self.pause_exp_flag = True
        self.pause_exp_button['state'] = 'disabled'
        _style_button(self.pause_exp_button, 'warning')
        self.continue_exp_button['state'] = 'normal'
        _style_button(self.continue_exp_button, 'success')

    def continue_exp(self):
        self.logger.warning("Continue button pressed")
        self.pause_exp_flag = False
        self.continue_exp_button['state'] = 'disabled'
        _style_button(self.continue_exp_button, 'success')
        self.pause_exp_button['state'] = 'normal'
        _style_button(self.pause_exp_button, 'warning')

    def abort_exp(self):
        """Abort the running experiment and bring hardware to safe state.

        Sets the abort flag/event so cooperative loops exit ASAP, then performs
        the hardware stop on a daemon thread so the GUI stays responsive.
        """
        self.logger.warning("Abort button pressed")
        self.pause_exp_flag = False
        self.abort_exp_flag = True
        self.exp._abort_event.set()
        # Stop Nemesys immediately (SDK is thread-safe), even if the experiment
        # thread is still mid-flush — that frees pressure right away.
        try:
            self.exp.pump_syst.stop_all_pumps()
        except Exception as e:
            self.logger.error(f"Abort: Nemesys stop failed: {e}")
        # Hardware stop runs on a worker so we don't block the Tk main loop.
        threading.Thread(target=self.exp.trigger_emergency_stop, daemon=True).start()

    def stop_all_hardware(self):
        """Always-available emergency stop. Same hardware actions as abort, but
        also usable outside a running experiment (e.g. during manual priming)."""
        self.logger.warning("STOP ALL pressed")
        self.pause_exp_flag = False
        self.abort_exp_flag = True
        self.exp._abort_event.set()
        try:
            self.exp.pump_syst.stop_all_pumps()
        except Exception as e:
            self.logger.error(f"Stop All: Nemesys stop failed: {e}")
        threading.Thread(target=self.exp.trigger_emergency_stop, daemon=True).start()

    def stop_exp(self, message=""):
        self.logger.info('Ending experiment...')
        self._current_phase = "Idle"
        self.exp.collection_status = False
        self.stop_flow()
        plt.close('all')
        self.start_experiment_button['state'] = 'normal'
        self.abort_exp_button['state'] = 'disabled'
        _style_button(self.abort_exp_button, 'danger')
        self.pause_exp_button['state'] = 'disabled'
        _style_button(self.pause_exp_button, 'warning')
        self.continue_exp_button['state'] = 'disabled'
        _style_button(self.continue_exp_button, 'success')
        self.logger.info("End of experiment: " + message)




    def ask_for_spectrum_validation(self):
        return self._show_validation_dialog(
            title="Spectrum validation",
            message="Is the spectrum valid?",
        )

    def ask_for_cost_validation(self, cost_at_predict):
        return self._show_validation_dialog(
            title="Cost Confirmation",
            message=(f"Measured cost is {cost_at_predict}. Do you want to proceed?\n"
                     f"If no, the spectrum will be done again."),
        )

    def _show_validation_dialog(self, title, message):
        """Topmost validation dialog. Disables the Start button while shown so
        the user cannot accidentally launch a new experiment in parallel.
        Manual pump/valve controls remain accessible (no global grab_set).
        """
        prev_start_state = None
        try:
            prev_start_state = str(self.start_experiment_button['state'])
            self.start_experiment_button['state'] = 'disabled'
        except Exception:
            pass

        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        try:
            dialog.attributes('-topmost', True)
        except tk.TclError:
            pass
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent closing via X

        tk.Label(dialog, text=message, padx=20, pady=20,
                 wraplength=360, justify="left").pack()

        result = {'value': False}

        def _ok():
            result['value'] = True
            dialog.destroy()

        def _cancel():
            result['value'] = False
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        ok_btn = tk.Button(btn_frame, text="OK", width=10, command=_ok)
        _style_button(ok_btn, 'success')
        ok_btn.pack(side="left", padx=8)
        cancel_btn = tk.Button(btn_frame, text="Cancel", width=10, command=_cancel)
        _style_button(cancel_btn, 'danger')
        cancel_btn.pack(side="left", padx=8)

        dialog.update_idletasks()
        # Center on parent
        try:
            x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
            y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            pass

        dialog.lift()
        dialog.focus_force()
        ok_btn.focus_set()
        self.wait_window(dialog)

        try:
            if prev_start_state is not None:
                self.start_experiment_button['state'] = prev_start_state
        except Exception:
            pass

        return result['value']
    def ask_for_cost_confirmation(self):
        response = tk.messagebox.askokcancel("Cost confirmation",
                                             "Flat spectra detected, cost will be set to 10000. Proceed ?")
        return response

    def click_to_position(self, row, col):
        well_letter = chr(ord('A') + (8 - col))
        well_number = row + 1
        self.collector_position_entry.delete(0, tk.END)
        self.collector_position_entry.insert(0, well_letter + str(well_number))
        if self.current_well:
            if (row, 8 - col) in self.filled_wells:
                self.collector_plate.itemconfig(self.current_well, fill="blue")
            else:
                self.collector_plate.itemconfig(self.current_well, fill="white")

        # Update the current well
        self.current_well = self.wells[row][col]

        # Change the fill color of the selected well to grey
        self.collector_plate.itemconfig(self.current_well, fill="grey")

    def move_to_next_well(self):
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):
                if (row, 8 - col) not in self.filled_wells:
                    self.filled_wells.append((row, 8 - col))
                    self.logger.debug("Filled wells: %s", self.filled_wells)
                    self.collector_plate.itemconfig(self.wells[row][8 - col], fill="blue")
                    self.click_to_position(row, 8 - col)
                    self.move_to_position()
                    return
        self.logger.warning("All wells are filled.")

    def move_to_position(self):
        if not self.exp.collector.init_status:
            tk.messagebox.showwarning("Collector warning",
                                      "Fraction collector is not ready. Please check connection and homing.")
            return
        position = self.collector_position_entry.get()
        if len(position) == 2 and position[0].isalpha() and position[1].isdigit():
            row = ord(position[0].upper()) - ord('A')
            col = int(position[1]) - 1
            if row < 9 and col < 6:
                self.exp.collector.move_to_position(position)
                self.logger.info(f"Fraction collector moved to position {position}")
        else:
            self.logger.warning("Invalid position")

    def move_z_axis(self):
        position = self.z_axis_position_entry.get()
        if position.isdigit() and 0 < int(position) < 1000:
            self.exp.collector.move_z_axis(int(position))
            self.logger.info(f"Z axis moved to position {position}")
        else:
            self.logger.warning("Invalid position")

    def update_z_axis_position(self):
        self.z_axis_current_position_label.config(text=str(self.exp.collector.current_z_position))
        after_id = self.after(500, self.update_z_axis_position)
        self.after_ids.append(after_id)
    def open_lamp_shutter(self):
        self.exp.collector.open_lamp_shutter()
    def close_lamp_shutter(self):
        self.exp.collector.close_lamp_shutter()

    def switch_valve_to_trash(self):
        self.exp.collector.switch_collection_valve(0)
        self.logger.info("Collection Valve switched to position 0")

    def switch_valve_to_collection(self):
        self.exp.collector.switch_collection_valve(1)
        self.logger.info("Collection Valve switched to position 1")

    def switch_8_port_valve(self):
        if self.exp.collector.current_8_port_valve_position == 0:
            self.exp.collector.switch_8_port_valve(1)
            self.exp.collector.current_8_port_valve_position = 1
            self.logger.info("8 port valve switched to position 1")
        else:
            self.exp.collector.switch_8_port_valve(0)
            self.exp.collector.current_8_port_valve_position = 0
            self.logger.info("8 port valve switched to position 0")
    def switch_10_port_valve(self):
        if self.exp.collector.current_10_port_valve_position == 0:
            self.exp.collector.switch_10_port_valve(1)
            self.exp.collector.current_10_port_valve_position = 1
            self.logger.info("10 port valve switched to position 1")
        else:
            self.exp.collector.switch_10_port_valve(0)
            self.exp.collector.current_10_port_valve_position = 0
            self.logger.info("10 port valve switched to position 0")

    def switch_refill_valve_to_reagents(self):
        pass


    def update_refill_valve_position(self):
        pass

    def update_8_valve_position(self):
        pos = self.exp.collector.current_8_port_valve_position
        self.valve_8_current_position_label.config(text=f"Position {pos}")
        self._update_valve_mode_label()
        self.after(500, self.update_8_valve_position)

    def update_10_valve_position(self):
        pos = self.exp.collector.current_10_port_valve_position
        self.valve_10_current_position_label.config(text=f"Position {pos}")
        self._update_valve_mode_label()
        self.after(500, self.update_10_valve_position)

    def _update_valve_mode_label(self):
        p8 = self.exp.collector.current_8_port_valve_position
        p10 = self.exp.collector.current_10_port_valve_position
        modes = {
            (0, 0): "Lost flow",
            (1, 1): "Stop flow / Chip wash",
            (1, 0): "Z-flow cell wash",
            (0, 1): "Unknown",
        }
        mode = modes.get((p8, p10), "Unknown")
        if hasattr(self, '_valve_mode_label'):
            self._valve_mode_label.config(text=mode)

    def homing(self):
        self.exp.collector.homing()
        self.logger.info("Fraction collector homing")

    def update_collector_status(self):
        if self.exp.collector.init_status:
            self.collector_status_label.config(text="Ready", fg=COLORS['success'])
        else:
            self.collector_status_label.config(text="Not ready", fg=COLORS['danger'])
        after_id = self.after(500, self.update_collector_status)
        self.after_ids.append(after_id)

    def update_valve_position(self):
        self.collector_valve_current_position_label.config(text=self.exp.collector.current_valve_position)
        after_id = self.after(400, self.update_valve_position)
        self.after_ids.append(after_id)

    def add_to_log(self):
        log_text = self.add_to_log_entry.get()
        self.logger.info(log_text)

    def save_parameters(self):
        parameters = {
            "Residence Time": self.residence_time_entry.get(),
            "Number of Samples": self.number_of_samples_entry.get(),
            "Sample Volume": self.sample_volume_entry.get(),
            "Spectro Sampling Rate": self.spectro_sampling_rate_entry.get(),
            "Inline Spectro": self.inline_spectro_var.get(),
            "Integration Time": self.int_time_entry.get(),
            "Number of Averages": self.average_number_entry.get()
        }
        for i in range(self.exp.number_of_pumps):
            parameters[f"Pump {i + 1} reagent"] = self.selected_reagents[i].get()
            parameters[f"Pump {i + 1} concentration"] = self.reagent_concentrations[i].get()

        with open(os.path.join(self.experiment_path, "metadata.txt"), "a") as file:
            for key, value in parameters.items():
                file.write(f"{key}: {value}\n")

    def load_previous_experiment(self):
        """Load previous experiment data for warm-starting BO"""
        try:
            # Ask user to select previous experiment directory
            previous_exp_path = filedialog.askdirectory(
                title="Select Previous Experiment Directory",
                initialdir=self.experiment_path
            )

            if not previous_exp_path:
                return  # User cancelled

            # Load the data using the Experiment method
            flows, costs = self.exp.load_previous_experiment(previous_exp_path)

            if flows is not None and costs is not None:
                # Store the loaded data
                self.exp.previous_flows = flows
                self.exp.previous_costs = costs

                # Refresh GUI fields from loaded config
                self._sync_gui_from_loaded_config()

                messagebox.showinfo(
                    "Success",
                    f"Successfully loaded {len(costs)} data points from previous experiment.\n"
                    f"Best cost: {costs.min():.4f}\n"
                    f"The next Bayesian Optimization will start from this data."
                )

                self.logger.info(f"Loaded previous experiment from: {previous_exp_path}")
            else:
                # Check if target mismatch was the reason for failure
                if hasattr(self.exp, 'prev_target_info'):
                    prev_info = self.exp.prev_target_info

                    # Build message about the mismatch
                    msg = "Target parameters mismatch!\n\n"
                    msg += f"Previous experiment target:\n"
                    if prev_info.get('style'):
                        msg += f"  - Style: {prev_info['style']}\n"
                    if prev_info.get('wavelength'):
                        msg += f"  - Wavelength: {prev_info['wavelength']} nm\n"
                    if prev_info.get('absorbance'):
                        msg += f"  - Absorbance: {prev_info['absorbance']}\n"

                    msg += f"\nCurrent experiment target:\n"
                    msg += f"  - Style: {self.exp.target_style}\n"
                    msg += f"  - Wavelength: {self.exp.tg_wl} nm\n"
                    if self.exp.target_style == 'single_point':
                        msg += f"  - Absorbance: {self.exp.tg_abs}\n"

                    msg += "\nThe costs were calculated for a different target.\n"
                    msg += "Do you want to update your current target to match the previous experiment?"

                    # Ask user if they want to update the target
                    if messagebox.askyesno("Target Mismatch", msg):
                        # Update the experiment parameters
                        if prev_info.get('style'):
                            self.exp.target_style = prev_info['style']
                            self.obj_var.set(prev_info['style'])

                        if prev_info.get('wavelength'):
                            self.exp.tg_wl = prev_info['wavelength']
                            self.single_point_wl_entry.delete(0, tk.END)
                            self.single_point_wl_entry.insert(0, str(prev_info['wavelength']))

                        if prev_info.get('absorbance') is not None:
                            self.exp.tg_abs = prev_info['absorbance']
                            self.single_point_abs_entry.delete(0, tk.END)
                            self.single_point_abs_entry.insert(0, str(prev_info['absorbance']))

                        # Update the UI to show correct widgets
                        # self.show_hide_obj_params()

                        self.logger.info("Updated target parameters to match previous experiment")
                        messagebox.showinfo("Target Updated",
                                          "Target parameters have been updated.\n"
                                          "Please click 'Load Previous Experiment' again to load the data.")
                    else:
                        self.logger.info("User declined to update target parameters")

                    # Clean up the stored info
                    delattr(self.exp, 'prev_target_info')
                else:
                    messagebox.showerror(
                        "Load Failed",
                        "Failed to load previous experiment data. Check the log for details."
                    )

        except Exception as e:
            self.logger.error(f"Error loading previous experiment: {e}")
            messagebox.showerror("Load Error", f"An error occurred: {e}")

    def _sync_gui_from_loaded_config(self):
        """Update GUI fields to reflect parameters loaded from a previous experiment."""
        cfg = getattr(self.exp, 'loaded_config', None)
        if cfg is None:
            return

        def _set(widget, value):
            if value is None:
                return
            widget.delete(0, tk.END)
            widget.insert(0, str(value))

        # BO parameters
        _set(self.maxiter_entry, cfg.get('maxiter'))
        _set(self.nb_init_samples_entry, cfg.get('nb_init_samples'))
        if cfg.get('acq_func'):
            self.acq_func_var.set(cfg['acq_func'])
        if cfg.get('init_option'):
            from toofast.core.run_config import init_option_from_config
            self.init_var.set(init_option_from_config(cfg))

        # Target
        if cfg.get('target_style'):
            self.obj_var.set(cfg['target_style'])
        _set(self.single_point_wl_entry, cfg.get('tg_wl'))
        _set(self.single_point_abs_entry, cfg.get('tg_abs'))
        _set(self.tol_entry, cfg.get('tol'))

        # Timing
        _set(self.residence_time_entry, cfg.get('residence_time'))
        _set(self.spectro_sampling_rate_entry, cfg.get('spectro_sampling_rate'))

        # Wash / flush
        _set(self.nitric_acid_volume_entry, cfg.get('wash_acid_volume'))
        _set(self.water_volume_entry, cfg.get('wash_water_volume'))
        _set(self.back_pressure_entry, cfg.get('wash_backpressure_flow'))

        # GP kernel overrides
        if cfg.get('bo_preset_override'):
            self._bo_preset_var.set(cfg['bo_preset_override'])
        if cfg.get('flat_strategy_override'):
            self._bo_flat_var.set(cfg['flat_strategy_override'])
        if 'feasibility_aware_override' in cfg:
            self._bo_feas_var.set(bool(cfg['feasibility_aware_override']))
        if 'use_physics_ghosts' in cfg:
            self._bo_ghosts_var.set(bool(cfg['use_physics_ghosts']))
        if 'ghost_noise_level' in cfg:
            _gn = cfg['ghost_noise_level']
            self._bo_ghost_noise_entry.delete(0, tk.END)
            self._bo_ghost_noise_entry.insert(
                0, 'auto' if _gn in (None, 'auto') else str(_gn))
        if 'ghost_penalty_lift' in cfg:
            self._bo_ghost_lift_entry.delete(0, tk.END)
            self._bo_ghost_lift_entry.insert(0, str(cfg['ghost_penalty_lift']))
        if cfg.get('kernel_type_override'):
            self._bo_kernel_var.set(cfg['kernel_type_override'])
        _set(self._bo_ls_entry, cfg.get('rbf_length_scale_override'))
        _set(self._bo_noise_entry, cfg.get('white_noise_override'))
        _set(self._bo_huber_entry, cfg.get('huber_delta_nm_override'))
        _set(self._bo_restarts_entry, cfg.get('gp_n_restarts_override'))

        # Pump bounds and fixed pumps / fixed flows
        bounds = cfg.get('bounds')
        if bounds is not None:
            for i, (mn, mx) in enumerate(bounds):
                if i < len(self.min_bound_entry):
                    _set(self.min_bound_entry[i], mn)
                    _set(self.max_bound_entry[i], mx)
        fixed_pumps_cfg = cfg.get('fixed_pumps')
        fixed_flows_cfg = cfg.get('fixed_flows')
        if fixed_pumps_cfg is not None:
            for i, fixed in enumerate(fixed_pumps_cfg):
                if i < len(self.fixed_pumps_vars):
                    self.fixed_pumps_vars[i].set(bool(fixed))
            # Update entry enable/disable states to match checkboxes
            self.switch_fixed()
        if fixed_flows_cfg is not None:
            for i, val in enumerate(fixed_flows_cfg):
                if i < len(self.fixed_entry) and val is not None:
                    # The entry is only writable if pump is fixed; temporarily enable
                    state = self.fixed_entry[i].cget('state')
                    self.fixed_entry[i]['state'] = 'normal'
                    _set(self.fixed_entry[i], val)
                    self.fixed_entry[i]['state'] = state

        # Spectrometer: only warn on mismatch, do NOT force re-acquisition of reference
        prev_int = cfg.get('spectro_integration_time')
        if prev_int is not None:
            spectro = self.exp.spectro
            cur = (spectro.integration_time, spectro.number_average, spectro.smoothing)
            prev = (prev_int,
                    cfg.get('spectro_number_average', cur[1]),
                    cfg.get('spectro_smoothing', cur[2]))
            if cur != prev:
                self.logger.warning(
                    f"Spectrometer parameters differ from previous experiment: "
                    f"current (int={cur[0]}, avg={cur[1]}, smooth={cur[2]}) vs "
                    f"previous (int={prev[0]}, avg={prev[1]}, smooth={prev[2]}). "
                    f"Reference NOT touched — update spectro params and re-acquire "
                    f"reference manually if needed.")
            else:
                self.logger.info("Spectrometer parameters match previous experiment.")

        # Termination (advanced)
        _set(self._adv_corr_entry, cfg.get('termination_correlation'))
        _set(self._adv_amp_entry, cfg.get('termination_amplitude'))
        _set(self._adv_min_spectra_entry, cfg.get('termination_min_spectra'))
        _set(self._adv_timeout_entry, cfg.get('termination_timeout'))

        self.logger.info("GUI fields updated from previous experiment config")

    def load_csv_flow_file(self):
        """Open a file dialog to select a CSV flow file."""
        file_path = filedialog.askopenfilename(
            title="Select flow file (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.csv_flow_file = file_path
            self.csv_file_label.config(text=os.path.basename(file_path), fg=COLORS['success'])
            self.logger.info(f"CSV flow file loaded: {file_path}")

    # create and save log file with each action on the pumps and spectro
    # Associated methods
    def toggle_mfc_options(self):
        """Show or hide flow controller options."""
        if self.activate_mfc_var.get():
            try:
                # Activate controller
                self.exp.connect_mfc()

                # Show widgets
                self.mfc_current_flow_label.grid(row=1, column=0)
                self.mfc_flow_label.grid(row=1, column=1)
                self.mfc_flow_bar.grid(row=2, column=0, columnspan=2, pady=5)
                self.mfc_set_flow_label.grid(row=3, column=0)
                self.mfc_set_flow_entry.grid(row=3, column=1)
                self.mfc_set_flow_button.grid(row=3, column=2)
                self.mfc_stop_button.grid(row=3, column=3)
                # Start flow rate update
                self.get_mfc_flow()

            except Exception as e:
                self.logger.error(f"Error activating MFC: {e}")
        else:
        # Hide widgets
            self.mfc_current_flow_label.grid_remove()
            self.mfc_flow_label.grid_remove()
            self.mfc_flow_bar.grid_remove()
            self.mfc_set_flow_label.grid_remove()
            self.mfc_set_flow_entry.grid_remove()
            self.mfc_set_flow_button.grid_remove()
            self.mfc_stop_button.grid_remove()

        # Cancel existing callbacks
        #     self.cancel_after_callbacks()

    def get_mfc_flow(self):
        """Update displayed flow rate and progress bar."""
        try:
            # Measure and display current flow rate
            flow = self.exp.mfc.measure_flow()  # Returns current flow in L/min
            self.mfc_flow_label.config(text=f"{flow:.2f} uln/s")

            # Update progress bar (adapted for 0-100 L/min range)
            max_flow = 20  # Maximum range (adjust according to your controller)
            percentage = min((flow / max_flow) * 100, 100)  # Convert to percentage
            self.mfc_flow_bar["value"] = percentage

            # Reschedule after 1 second
            after_id = self.after(1000, self.get_mfc_flow)
            self.after_ids.append(after_id)

        except Exception as e:
            self.logger.error(f"Error measuring flow rate: {e}")

    def set_mfc_flow(self):
        """Set a new target flow rate."""
        try:
            flow_value = float(self.mfc_set_flow_entry.get())
            self.exp.mfc.set_flow(flow_value)
            self.logger.info(f"Target flow set to {flow_value} L/min")

        except ValueError:
            self.logger.warning("Invalid input for target flow rate.")

        except Exception as e:
            self.logger.error(f"Error setting flow rate: {e}")

    def stop_mfc_flow(self):
        self.mfc_set_flow_entry.delete(0, tk.END)
        self.mfc_set_flow_entry.insert(0, "0")
        self.set_mfc_flow()

    def cancel_after_callbacks(self):
        """Cancel all scheduled after callbacks."""
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.after_ids.clear()
        return

    def connect_c3000(self):
        try:
            self.exp.connect_c3000()
            self.c3000_connect_button.config(text="Connected", bg=COLORS['success'], fg='white')
            self.update_c3000_valve_positions()
        except Exception as e:
            self.logger.error(f"C3000 connection error: {e}")
            self.c3000_connect_button.config(text="Error", bg=COLORS['danger'], fg='white')


    def initialize_c3000_1(self):
        self.exp.c3000_1.initialize()
        self.logger.info("C3000_1 initialized")
    def initialize_c3000_2(self):
        self.exp.c3000_2.initialize()
        self.logger.info("C3000_2 initialized")

    def force_initialize_c3000_1(self):
        self.exp.c3000_1.force_initialize()
        self.logger.info("C3000_1 force initialized")
    def force_initialize_c3000_2(self):
        self.exp.c3000_2.force_initialize()
        self.logger.info("C3000_2 force initialized")

    def refill_c3000_1(self):
        refill_flow = float(self.c3000a_refill_flow_entry.get())
        self.exp.c3000_1.refill(flow=refill_flow)
        self.logger.info(f"C3000_1 refilled with {refill_flow} uln/s")
    def refill_c3000_2(self):
        refill_flow = float(self.c3000b_refill_flow_entry.get())
        self.exp.c3000_2.refill(flow=refill_flow)
        self.logger.info(f"C3000_2 refilled with {refill_flow} uln/s")

    def start_c3000_pump_1(self):
        # get flow value
        flow_value = float(self.c3000a_flow_entry.get())
        self.exp.c3000_1.generate_flow(flow_value)
        self.logger.info(f"C3000_1 started with {flow_value} uln/s")
    def start_c3000_pump_2(self):
        # get flow value
        flow_value = float(self.c3000b_flow_entry.get())
        self.exp.c3000_2.generate_flow(flow_value)
        self.logger.info(f"C3000_2 started with {flow_value} uln/s")

    def stop_c3000_pump_1(self):
        self.exp.c3000_1.stop_pump()
        self.logger.info("C3000_1 stopped")
    def stop_c3000_pump_2(self):
        self.exp.c3000_2.stop_pump()
        self.logger.info("C3000_2 stopped")

    def manual_wash(self):
        threading.Thread(target=self.exp.trigger_manual_wash, daemon=True).start()

    def update_wash_counter_display(self):
        count = self.exp._wash_sample_counter
        self.wash_counter_label.config(
            text=f"Samples since last intensive wash: {count}")

    def switch_c3000_1_valve(self):
        valve_position = self.c3000_1_valve_position.get()
        self.exp.c3000_1.switch_valve_to_position(valve_position)
        self.logger.info(f"C3000_1 valve switched to position {valve_position}")

    def switch_c3000_2_valve(self):
        valve_position = self.c3000_2_valve_position.get()
        self.exp.c3000_2.switch_valve_to_position(valve_position)
        self.logger.info(f"C3000_2 valve switched to position {valve_position}")

    def update_c3000_valve_positions(self):
        """Read real C3000 valve positions and update radio buttons (one-shot, no polling).

        Called once after connect. Periodic polling is intentionally removed —
        get_valve_position() blocks the Tk main thread via serial I/O causing
        freezes. Will be reworked on a dedicated branch.
        """
        try:
            pos1 = self.exp.c3000_1.get_valve_position()
            if pos1 in ("left", "center", "right"):
                self.c3000_1_valve_position.set(pos1)
        except Exception:
            pass
        try:
            pos2 = self.exp.c3000_2.get_valve_position()
            if pos2 in ("left", "center", "right"):
                self.c3000_2_valve_position.set(pos2)
        except Exception:
            pass

    def back(self):
        self.stop_flow()
        self.logger.info('End of log -')
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.master.deiconify()
        self.destroy()

    def on_closing(self):
        self.stop_flow()
        plt.close('all')
        self.logger.info('End of log -')
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.master.quit()

    def cleanup_on_exit(self):
        """Called when closing the application"""
        if hasattr(self, 'original_hooks'):
            sys.excepthook, threading.excepthook = self.original_hooks
            self.logger.info("Exception handlers restored")


class MainWindow:
    """
    Main application window: allows user to choose experiment number and folder.
    Asks user for number of pumps + syringe sizes.
    Initializes pumps + spectrometer.
    """
    def __init__(self, master):
        self.master = master
        self.master.lift()
        self.master.focus_force()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        if os.path.exists(_ICON_PATH):
            self.master.iconbitmap(_ICON_PATH)
        self.master.title("2FAST - New Experiment")

        row = 0
        # Experiment name
        self.experiment_name_label = tk.Label(master, text="Experiment name:", font=FONTS['body'])
        self.experiment_name_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.experiment_name_entry = tk.Entry(master)
        _style_entry(self.experiment_name_entry)
        self.experiment_name_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.experiment_name_entry.insert(0, "test")
        row += 1

        # Experiment folder
        self.data_folder_label = tk.Label(master, text="Experiment folder:", font=FONTS['body'])
        self.data_folder_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.data_folder_entry = tk.Entry(master)
        _style_entry(self.data_folder_entry)
        self.data_folder_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        _default_data = os.path.join(os.path.expanduser("~"), "Documents", "2FAST_Data")
        self.data_folder_entry.insert(0, _default_data)
        self.browse_button = tk.Button(master, text="Browse", command=self.browse_folder)
        _style_button(self.browse_button)
        self.browse_button.grid(row=row, column=2, padx=4, pady=3)
        row += 1

        # Chip ID
        self.chip_id_label = tk.Label(master, text="Chip ID:", font=FONTS['body'])
        self.chip_id_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.chip_id_entry = tk.Entry(master)
        _style_entry(self.chip_id_entry)
        self.chip_id_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        row += 1

        # Pumps section header
        self.pump_label = tk.Label(master, text="Pumps", font=FONTS['section'])
        self.pump_label.grid(row=row, columnspan=2)
        row += 1

        # Number of pumps
        self.pump_number_label = tk.Label(master, text="Number of pumps:", font=FONTS['body'])
        self.pump_number_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.pump_number_entry = tk.Entry(master)
        _style_entry(self.pump_number_entry)
        self.pump_number_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.pump_number_entry.insert(0, "5")
        row += 1

        # Syringe section header
        self.syringe_label = tk.Label(master, text="Syringe Configuration", font=FONTS['section'])
        self.syringe_label.grid(row=row, columnspan=2)
        row += 1

        self.inner_diameter_label = tk.Label(master, text="Inner Diameter (mm):", font=FONTS['body'])
        self.inner_diameter_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.inner_diameter_entry = tk.Entry(master)
        _style_entry(self.inner_diameter_entry)
        self.inner_diameter_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.inner_diameter_entry.insert(0, "10.301")
        self.inner_diameter_entry.bind("<KeyRelease>", self.update_volume)
        row += 1

        self.max_piston_stroke_label = tk.Label(master, text="Max Piston Stroke (mm):", font=FONTS['body'])
        self.max_piston_stroke_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.max_piston_stroke_entry = tk.Entry(master)
        _style_entry(self.max_piston_stroke_entry)
        self.max_piston_stroke_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.max_piston_stroke_entry.insert(0, "60")
        self.max_piston_stroke_entry.bind("<KeyRelease>", self.update_volume)
        row += 1

        volume = round((math.pi * math.pow(float(self.inner_diameter_entry.get()) / 2, 2) * float(
            self.max_piston_stroke_entry.get()) / 1000), 1)
        self.volume_label = tk.Label(master, text="Volume:", font=FONTS['body'])
        self.volume_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.volume_value_label = tk.Label(master, text="", font=FONTS['mono'])
        self.volume_value_label.grid(row=row, column=1, sticky='w', padx=4, pady=3)
        self.volume_value_label.config(text=str(volume) + " mL")
        row += 1

        # Max flow
        self.max_flow_label = tk.Label(master, text="Max Flow (ul/s):", font=FONTS['body'])
        self.max_flow_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.max_flow_entry = tk.Entry(master)
        _style_entry(self.max_flow_entry)
        self.max_flow_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.max_flow_entry.insert(0, "14")
        row += 1

        # Serial port settings
        tk.Label(master, text="Collector port:", font=FONTS['body']).grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.collector_port_entry = tk.Entry(master)
        _style_entry(self.collector_port_entry)
        self.collector_port_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.collector_port_entry.insert(0, "COM8")
        row += 1

        tk.Label(master, text="C3000 port:", font=FONTS['body']).grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.c3000_port_entry = tk.Entry(master)
        _style_entry(self.c3000_port_entry)
        self.c3000_port_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.c3000_port_entry.insert(0, "COM19")
        row += 1

        tk.Label(master, text="MFC port:", font=FONTS['body']).grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self.mfc_port_entry = tk.Entry(master)
        _style_entry(self.mfc_port_entry)
        self.mfc_port_entry.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
        self.mfc_port_entry.insert(0, "COM9")
        row += 1

        # Simulation checkbox
        self.simulation_var = tk.BooleanVar(value=False)
        self.simulation_checkbutton = tk.Checkbutton(master, text='Simulate experiment',
                                                     variable=self.simulation_var, onvalue=True, offvalue=False,
                                                     font=FONTS['body'])
        self.simulation_checkbutton.grid(row=row, column=0, padx=4, pady=2)

        # Start experiment button
        self.start_experiment_button = tk.Button(master, text="Start Experiment", command=self.start_experiment)
        _style_button(self.start_experiment_button)
        self.start_experiment_button.grid(row=row, column=1, columnspan=2, padx=4, pady=2)

    def update_volume(self, event):
        try:
            inner_diameter_mm = float(self.inner_diameter_entry.get())
            max_piston_stroke_mm = float(self.max_piston_stroke_entry.get())
            volume = round((math.pi * math.pow(inner_diameter_mm / 2, 2) * max_piston_stroke_mm / 1000), 1)
            self.volume_value_label.config(text=str(volume) + " mL")
        except ValueError:
            self.volume_value_label.config(text="Invalid input")

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.data_folder_entry.delete(0, tk.END)
            self.data_folder_entry.insert(tk.END, folder_path)

    def start_experiment(self):
        experiment_name = self.experiment_name_entry.get()
        data_folder = self.data_folder_entry.get()
        number_of_pumps = int(self.pump_number_entry.get())
        is_simulation = self.simulation_var.get()
        # Configurations folder lives at the project root, regardless of cwd
        _project_root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'
        ))
        # vendor device configurations (not distributed): TOOFAST_DEVICE_CONFIG_DIR, default <project>/Configurations;
        # passed through to the pump driver as `config_file`
        config_folder = os.environ.get("TOOFAST_DEVICE_CONFIG_DIR", os.path.join(_project_root, "Configurations"))
        config_file = os.path.join(config_folder, f"{number_of_pumps}syringes")
        inner_diameter_mm = float(self.inner_diameter_entry.get())
        max_piston_stroke_mm = float(self.max_piston_stroke_entry.get())
        chip_id = self.chip_id_entry.get()
        max_flow = float(self.max_flow_entry.get())
        collector_port = self.collector_port_entry.get().strip()
        c3000_port = self.c3000_port_entry.get().strip()
        mfc_port = self.mfc_port_entry.get().strip()

        if experiment_name and data_folder:
            self.experiment_path = os.path.join(data_folder, experiment_name)
            self.experiment_name = experiment_name
            if os.path.exists(self.experiment_path):
                overwrite = tk.messagebox.askokcancel("Overwriting  Warning",
                                                      "An experiment with this name already exists. "
                                                      "Do you want to overwrite it ? ")
                if overwrite:
                    shutil.rmtree(self.experiment_path)
                    time.sleep(0.25)
                else:
                    return
            os.makedirs(self.experiment_path)

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            log_file_path = os.path.join(self.experiment_path, "experiment_log.log")
            file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # Adding a console handler to keep track of logger during the experiment.
            # Console stays at INFO; DEBUG details only go to the file log.
            print_handler = PrintHandler()
            print_handler.setLevel(logging.INFO)
            print_handler.setFormatter(formatter)
            logger.addHandler(print_handler)


            # Simulation: the experiment runs on a virtual clock, TOOFAST_SIM_SPEED x real time (default 0.1:
            # a 300 s stop-flow takes 30 s). The instrument keeps the wall clock.
            if is_simulation:
                from toofast import clock
                clock.set_clock(clock.VirtualClock(speed=float(os.environ.get("TOOFAST_SIM_SPEED", "0.1"))))
            # Initialisation of the experiment, pumps and spectrometer
            exp = Experiment(self.experiment_name, self.experiment_path, logger, number_of_pumps,
                             inner_diameter_mm, max_piston_stroke_mm, config_file,
                             chip_id, max_flow, is_simulation,
                             collector_port=collector_port, c3000_port=c3000_port,
                             mfc_port=mfc_port)
            configuration_window = ConfigurationWindow(self.master, exp, self.experiment_path, self.experiment_name, logger)
            self.create_metadata()
            self.master.withdraw()  # Hide main window
        else:
            tk.messagebox.showerror("Error", "Please enter experiment name and data folder.")

    # Create metadata file
    def create_metadata(self):
        metadata = {
            "Date": time.strftime("%d/%m/%Y"),
            "Time": time.strftime("%H:%M:%S"),
            "Experiment Name": self.experiment_name,
            "Data Folder": self.data_folder_entry.get(),
            "Number of Pumps": self.pump_number_entry.get(),
            "chip ID": self.chip_id_entry.get(),
            "Inner Diameter": self.inner_diameter_entry.get(),
            "Max Piston Stroke": self.max_piston_stroke_entry.get(),
            "Syringe Volume": self.volume_value_label.cget("text"),
            "Max Flow": self.max_flow_entry.get(),
            "Collector Port": self.collector_port_entry.get(),
            "C3000 Port": self.c3000_port_entry.get(),
            "MFC Port": self.mfc_port_entry.get(),
        }
        with open(os.path.join(self.experiment_path, "metadata.txt"), "w") as file:
            for key, value in metadata.items():
                file.write(f"{key}: {value}\n")

    def on_closing(self):
        self.master.destroy()


def main():
    # Force UTF-8 on stdout/stderr so Unicode characters in log messages
    # (arrows, Greek letters, box-drawing, emoji) don't crash on Windows
    # consoles that default to cp1252.  reconfigure() is Python 3.7+ only;
    # the try/except is a safety net for edge cases (e.g. redirected pipes).
    import io as _io
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, 'reconfigure'):
            try:
                _stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    myappid = '2FAST'  # arbitrary string
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except AttributeError:
        pass  # Not on Windows — skip taskbar app-id

    # Enable DPI awareness on Windows so the UI renders at native resolution
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
