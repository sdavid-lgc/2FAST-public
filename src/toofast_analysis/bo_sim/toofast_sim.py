# [vendored] verbatim copy of paperAnalysis/bo_sim/toofast_sim.py (md5 096b0ea4bb539356a7bee6b334369b50), tools/vendor_bo_sim.py
"""
bo_sim/toofast_sim.py — la vraie BO `toofast` pilotée par l'oracle.

`toofast.core.experiment.Experiment.optimise()` est réutilisé tel quel
(kernel/preset/prior, ghosts, flat strategy, EI + ExplorationWrapper, DE,
random-fallback si GP peu fiable, coût `CosineSpectrumCost`…). Seules les
deux méthodes qui touchent au matériel sont remplacées : la "mesure" renvoie
le spectre de l'oracle sur la grille de la cible.

Variantes d'hyperparamètres (argument `variant`) :
  robust        preset robuste (Matérn 2.5 iso, bornes ls/bruit du _config.json), pas de prior
  prior_warm    kernel prior (pkl) en warm-start, ré-optimisé à chaque itération
  prior_frozen  kernel prior figé (bornes -> 'fixed')
  frozen_ls<x>  CONTRÔLE : preset robuste avec la length scale figée à x (bruit libre,
                dans les bornes robustes) — sans prior. Sépare "information du prior"
                et "length scale courte" (prior_frozen garde ls = 1.1).
"""
import json
import logging
import os
import re
import sys
import types

import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from . import config as C

# --- stub de la DLL Avantes : toofast.hardware.spectrometer l'importe au niveau
#     module même en simulation (avaspec.py charge avaspecx64.dll à l'import).
if "toofast.hardware.avaspec" not in sys.modules:
    sys.modules["toofast.hardware.avaspec"] = types.ModuleType("toofast.hardware.avaspec")
# [vendored] the Cetoni SDK is not distributed: toofast.hardware.pump imports it at module level, the
# simulated PumpSystem never calls it -> empty stand-ins when it is not installed
try:
    import qmixsdk  # noqa: F401
except ImportError:
    for _m in ("qmixsdk", "qmixsdk.qmixbus", "qmixsdk.qmixpump"):
        sys.modules[_m] = types.ModuleType(_m)
    sys.modules["qmixsdk.qmixpump"].Pump = type("Pump", (), {})   # MyPump subclasses it
    sys.modules["qmixsdk"].qmixbus = sys.modules["qmixsdk.qmixbus"]
    sys.modules["qmixsdk"].qmixpump = sys.modules["qmixsdk.qmixpump"]
if str(C.TOOFAST_SRC) not in sys.path:
    sys.path.insert(0, str(C.TOOFAST_SRC))

from toofast.core.experiment import Experiment          # noqa: E402
from toofast.core.pca_model import SpectralPCA          # noqa: E402  (ré-exporté pour cost.py)
from toofast.hardware.spectrometer import Spectrum      # noqa: E402
from toofast.hardware.pump import PumpSystem            # noqa: E402

VARIANTS = ("robust", "prior_warm", "prior_frozen")
_FROZEN_LS = re.compile(r"^frozen_ls([0-9.]+)$")


def is_variant(v):
    return v in VARIANTS or _FROZEN_LS.match(v) is not None

_quiet = logging.getLogger("bo_sim.toofast")
_quiet.setLevel(logging.WARNING)
_quiet.addHandler(logging.NullHandler())
_quiet.propagate = False


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "_config.json"), encoding="utf-8") as f:
        return json.load(f)


def make_pump_system(n_pumps=5, total_flow=14.0):
    """PumpSystem simulé — sert aux générateurs random / LHS de toofast."""
    return PumpSystem(_quiet, number_of_pumps=n_pumps, inner_diameter_mm=4.6,
                      max_piston_stroke_mm=60, config_file=None,
                      is_simulation=True, max_flow=total_flow)


class SimulatedExperiment(Experiment):
    """Experiment toofast dont la mesure est l'oracle.

    measure_fn(Q5) -> absorbance sur grid_wl, ou None si le point est infaisable
    (gate de l'oracle) : on renvoie alors un spectre nul, que `cost_function`
    traite comme un plat (sentinelle -> remappée par la flat strategy au fit).
    """

    def __init__(self, measure_fn, grid_wl, workdir, name="sim", n_pumps=5,
                 total_flow=14.0):
        os.makedirs(workdir, exist_ok=True)
        super().__init__(name, str(workdir), _quiet, n_pumps, 4.6, 60, None,
                         "sim", total_flow=total_flow, is_simulation=True)
        self._measure_fn = measure_fn          # Q5 -> absorbance sur grid_wl | None
        self._grid_wl = np.asarray(grid_wl, float)
        self.sim_spectra = []                  # spectres "mesurés", ordre d'évaluation
        self.sim_infeasible = []               # flag plat par évaluation

    def _measure(self, flows):
        flows = np.asarray(flows, float).copy()
        ab = self._measure_fn(flows)
        flat = ab is None
        ab = np.zeros_like(self._grid_wl) if flat else np.asarray(ab, float).ravel()
        sp = Spectrum(self._grid_wl, None, "measurement", absorbance=ab)
        self.sim_spectra.append(ab)
        self.sim_infeasible.append(flat)
        return sp, flows

    def cost_function(self, spectrum):
        """Comme toofast, plus la détection de plat pour TOUS les styles de cible.

        toofast ne teste `std(abs) < threshold` qu'en 'pca' et 'lambda_max' ; en
        'full_spectrum' un plat reçoit juste son cosinus (bruit). Ici un point
        infaisable donne toujours la sentinelle -> nearest_high au fit du GP.
        """
        if np.std(spectrum.absorbance) < self.threshold:
            return float(self._flat_penalty_value()), 0
        return super().cost_function(spectrum)

    # --- les deux seuls points de contact avec le matériel dans optimise() ---
    def stop_flow_and_measure(self, flows, path=None, meas_type="measurement",
                              sample_id="", measure_ref=False):
        return self._measure(flows)

    def make_flow_and_measure(self, flows, path=None, meas_type="measurement"):
        return self._measure(flows)


def build_experiment(measure_fn, grid_wl, workdir, run_dir=C.RUN_S28,
                     variant="robust", budget=30, bounds=None,
                     prior_name=C.DEFAULT_PRIOR, n_restarts=None, target_csv=None):
    """Experiment simulé configuré comme la manip `run_dir` (+ variante kernel).

    budget   = nombre total d'évaluations (init LHS comprise)
    bounds   = (5,2) bornes par pompe ; None -> celles du _config.json
    n_restarts = override de gp_n_restarts (None -> valeur du _config.json, 100)
    target_csv = cible (format toofast) ; None -> <run_dir>/target_spectrum.csv
    """
    if not is_variant(variant):
        raise ValueError(f"variant {variant!r} ∉ {VARIANTS} ni frozen_ls<x>")
    cfg = load_run_config(run_dir)
    if cfg["target_style"] not in ("full_spectrum", "pca"):
        raise NotImplementedError(f"target_style={cfg['target_style']!r} non câblé")
    pca_kw = {}
    if cfg["target_style"] == "pca":
        from .cost import pca_model_for_run
        pca_kw = dict(pca_model_file=str(pca_model_for_run(run_dir)),
                      pca_variance_weighted=bool(cfg.get("pca_variance_weighted", False)),
                      pca_cost_metric=cfg.get("pca_cost_metric", "sqdist"),
                      pca_n_components=int(cfg.get("pca_n_components", 0) or 0))
    exp = SimulatedExperiment(measure_fn, grid_wl, workdir,
                              n_pumps=cfg["number_of_pumps"], total_flow=cfg["total_flow"])
    exp.update_optim_params(
        maxiter=budget - cfg["nb_init_samples"],
        reagents=np.array(cfg["reagents"]),
        nb_init_samples=cfg["nb_init_samples"], init_option=cfg["init_option"],
        target_style=cfg["target_style"], tolerance=cfg["tol"], tol_stop=False,
        threshold=cfg["threshold"], acq_func=cfg["acq_func"],
        bounds=np.array(cfg["bounds"], float),
        fixed_pumps=np.array(cfg["fixed_pumps"]), fixed_flows=np.array(cfg["fixed_flows"], float),
        target_spectrum_file=str(target_csv or os.path.join(run_dir, "target_spectrum.csv")),
        cost_method=cfg["cost_method"], **pca_kw,
    )
    # kernel / preset / ghosts / flat strategy / DE exactement comme la manip
    exp._apply_loaded_config(cfg)
    # ... sauf ce qui est propre au benchmark
    exp.maxiter = budget - cfg["nb_init_samples"]
    exp.tol_stop = False                       # on veut la courbe complète
    if bounds is not None:
        exp.bounds = np.array(bounds, float)
    if n_restarts is not None:
        exp.gp_n_restarts_override = int(n_restarts)

    m_ls = _FROZEN_LS.match(variant)
    if variant.startswith("prior"):
        exp.load_prior_kernel(str(C.PRIOR_KERNELS[prior_name]),
                              freeze=(variant == "prior_frozen"))
        exp._rebuild_kernel()
    elif m_ls:
        # Kernel du preset robuste (Matérn 2.5 iso + White, bornes de bruit du
        # _config.json), length scale figée : passe par le mécanisme prior_kernel
        # de toofast (freeze=False -> clone, les bornes 'fixed' sont respectées).
        exp.prior_kernel = (Matern(length_scale=float(m_ls.group(1)), length_scale_bounds="fixed", nu=2.5)
                            + WhiteKernel(noise_level=1e-3,
                                          noise_level_bounds=(exp.white_noise_floor, exp.white_noise_ceiling)))
        exp.prior_kernel_freeze = False
        exp._rebuild_kernel()
    else:
        exp.prior_kernel = None
        exp._rebuild_kernel()
    exp.sim_meta = dict(run_dir=str(run_dir), style=cfg["target_style"], target=str(target_csv or "real"),
                        variant=variant, budget=budget,
                        prior=prior_name if variant.startswith("prior") else None,
                        kernel=str(exp.kernel), bounds=exp.bounds.tolist(),
                        preset=exp.bo_preset_override, ghosts=exp.use_physics_ghosts,
                        acq=exp.acq_func, n_restarts=exp.gp_n_restarts_override)
    return exp


def run_experiment(exp, seed):
    """Lance optimise() de façon reproductible ; renvoie (flows, costs, gp)."""
    np.random.seed(seed)      # LHS init (pyDOE), DE, random-fallback : tous sur np.random
    flows, costs, gp = exp.optimise(plot_figs=False, save_figs=False)
    return np.asarray(flows, float), np.asarray(costs, float), gp
