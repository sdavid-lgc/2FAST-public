"""
Headless operation: run a stop-flow Bayesian-optimisation campaign without GUI and without hardware.

`HeadlessApp` stands in for the Tk configuration window: it exposes the settings the experiment reads (wash volumes,
sampling rate, …) as read-only values and turns the GUI actions (plots, valve buttons, dialogs) into no-ops or
log lines. `run_campaign` wires an `Experiment` in simulation mode (virtual reactor, virtual clock) to it and runs
the same `Experiment.stop_flow()` loop as the instrument.

    python -m toofast.simulation --target-lambda 550 --iterations 6 --out sim_runs/demo
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from toofast import clock
from toofast.data.manager import DataManager
from toofast.simulation.reactor import DEFAULT_REAGENTS, DEFAULT_STOCKS_mM, VirtualReactor


class Value:
    """Read-only stand-in for a Tk Entry / Variable (`.get()`)."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


# Settings read by the experiment; defaults = the paper's 2026 sessions (RG runs, _config.json)
DEFAULT_SETTINGS = dict(
    spectro_sampling_rate=1.0,          # s between spectra in stop-flow
    residence_time=12.0,                # s (chip residence time, waits 2.5x before stopping the flow)
    nitric_acid_volume=100.0,           # µL, wash
    water_volume=600.0,                 # µL, wash
    back_pressure=0.6,                  # µL/s per reagent pump during the wash
    intensive_acid_volume=200.0,
    intensive_water_volume=1000.0,
    intensive_water_bp_stop=500.0,
    pvp_threshold=3.0,
    number_of_samples=1,
)


class HeadlessApp:
    """The attributes of the configuration window that `Experiment` uses, without Tk."""

    def __init__(self, experiment, reagents: Sequence[str], concentrations_mM: Sequence[float],
                 settings: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        s = dict(DEFAULT_SETTINGS, **(settings or {}))
        self.exp = experiment
        self.logger = logger or experiment.logger
        self.experiment_path = experiment.experiment_path
        self.data_manager = DataManager(experiment_path=experiment.experiment_path,
                                        experiment_name=experiment.experiment_name, logger=self.logger)
        # flags polled by the experiment
        self.abort_exp_flag = False
        self.next_sample_pressed = False
        # settings (Tk entries / variables in the GUI)
        self.spectro_sampling_rate_entry = Value(str(s["spectro_sampling_rate"]))
        self.residence_time_entry = Value(str(s["residence_time"]))
        self.number_of_samples_entry = Value(str(s["number_of_samples"]))
        self.nitric_acid_volume_entry = Value(str(s["nitric_acid_volume"]))
        self.water_volume_entry = Value(str(s["water_volume"]))
        self.back_pressure_entry = Value(str(s["back_pressure"]))
        self.intensive_acid_volume_entry = Value(str(s["intensive_acid_volume"]))
        self.intensive_water_volume_entry = Value(str(s["intensive_water_volume"]))
        self.intensive_water_bp_stop_entry = Value(str(s["intensive_water_bp_stop"]))
        self.pvp_threshold_entry = Value(str(s["pvp_threshold"]))
        self.exp_type_var = Value("Bayesian Optimization")
        self.sample_collection_var = Value(False)     # no fraction collection
        self.stop_flow_exp = Value(True)
        self.inline_spectro_var = Value(False)
        self.manual_spectrum_validation = Value(False)
        self.regular_wash_var = Value(False)
        self.pvp_threshold_wash_var = Value(False)
        self.selected_reagents = [Value(r) for r in reagents]
        self.reagent_concentrations = [Value(str(c)) for c in concentrations_mM]
        self.pump_flow_entry = [Value("0") for _ in reagents]
        self.messages = []

    # -- Tk scheduling: run immediately (no main-thread constraint without Tk) --------------------------------
    def call_soon(self, func, *args):
        func(*args)

    def after(self, _ms, func, *args):
        func(*args)

    def update(self):
        pass

    # -- GUI actions ------------------------------------------------------------------------------------------
    def plot_spectrum(self, *args, **kwargs):
        pass

    def measure_spectrum(self, sample_id=""):
        self.exp.measure_spectrum(sample_id=sample_id)

    def stop_exp(self, message=""):
        self.messages.append(message)
        self.logger.info(f"Experiment stopped: {message}")
        self.abort_exp_flag = True

    def stop_flow(self):
        self.exp.pump_syst.stop_all_pumps()

    def start_flow(self):
        flows = [float(v.get()) for v in self.pump_flow_entry]
        self.exp.pump_syst.set_flows(flows)

    def switch_valve_to_trash(self):
        self.exp.collector.switch_collection_valve(0)

    def switch_valve_to_collection(self):
        self.exp.collector.switch_collection_valve(1)

    def move_to_next_well(self):
        pass

    def check_sample_level(self):
        self.exp.collection_status = False

    def update_wash_counter_display(self):
        pass

    def ask_for_spectrum_validation(self):
        return True


@dataclass
class CampaignResult:
    flows: np.ndarray
    costs: np.ndarray
    lambda_max: np.ndarray
    experiment_path: str
    virtual_seconds: float
    best_index: int = field(init=False)

    def __post_init__(self):
        self.best_index = int(np.argmin(self.costs))


def make_logger(path: str, name: str = "toofast.simulation", level=logging.INFO) -> logging.Logger:
    os.makedirs(path, exist_ok=True)
    # one logger per run folder (the name must not collide between folders with the same base name)
    logger = logging.getLogger(f"{name}.{os.path.abspath(path).replace('.', '_')}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(path, "experiment_log.log"), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def run_campaign(out_dir: str, target_lambda_max: float = 550.0, iterations: int = 6, n_init: int = 6,
                 reagents: Sequence[str] = DEFAULT_REAGENTS, total_flow: float = 14.0,
                 stocks_mM: Optional[Dict[str, float]] = None, bounds=None, acq_func: str = "EI",
                 seed: int = 0, speed: float = 0.0, settings: Optional[Dict] = None,
                 init_option: str = "latinhypercube",
                 logger: Optional[logging.Logger] = None, reactor: Optional[VirtualReactor] = None,
                 **optim_params) -> CampaignResult:
    """Stop-flow BO toward a target λmax on the simulated platform; writes the usual run folder in `out_dir`.

    `speed` is the real-time factor of the virtual clock (0 = as fast as possible). Extra keyword arguments go to
    `Experiment.update_optim_params` (e.g. ``target_style="full_spectrum"``, ``target_spectrum_file=...``).
    """
    from toofast.core.experiment import Experiment
    from toofast.hardware.simulated import SimulatedRig

    stocks = dict(DEFAULT_STOCKS_mM, **(stocks_mM or {}))
    np.random.seed(seed)
    out_dir = os.path.abspath(out_dir)
    name = os.path.basename(out_dir)
    logger = logger or make_logger(out_dir)
    reactor = reactor or VirtualReactor(reagents=tuple(reagents), stocks_mM=stocks, seed=seed)
    import threading
    vclock = clock.VirtualClock(speed=speed, owner=threading.current_thread())   # reproducible timeline
    with clock.use(vclock):
        reactor.set_reagent_names(reagents)       # also restarts the reactor's timers on the virtual clock
        exp = Experiment(name, out_dir, logger, number_of_pumps=len(reagents), inner_diameter_mm=10.301,
                         max_piston_stroke_mm=60, config_file=None, chip_id="SIM", total_flow=total_flow,
                         is_simulation=True, rig=SimulatedRig(reactor))
        app = HeadlessApp(exp, reagents, [stocks.get(r, 0.0) for r in reagents], settings, logger)
        exp.app = app
        exp.connect_c3000()
        exp.connect_mfc()
        exp.measure_reference()
        params = dict(maxiter=iterations, residence_time=float(app.residence_time_entry.get()),
                      reagents=np.array(reagents), nb_init_samples=n_init, init_option=init_option,
                      target_style="lambda_max", tg_wl=target_lambda_max, acq_func=acq_func,
                      bounds=None if bounds is None else np.asarray(bounds, dtype=float))
        params.update(optim_params)
        exp.update_optim_params(**params)
        flows, costs, _ = exp.stop_flow(plot_figs=False, save_figs=False)
    lam_file = os.path.join(out_dir, "lambda_max.csv")
    lam = np.loadtxt(lam_file, ndmin=1) if os.path.exists(lam_file) else np.full(len(costs), np.nan)
    return CampaignResult(np.asarray(flows), np.asarray(costs), lam, out_dir, vclock.elapsed())
