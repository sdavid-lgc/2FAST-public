# [vendored] verbatim copy of paperAnalysis/bo_sim/cost.py (md5 8003b5ec30b97199c92cb19d54b6ce92), tools/vendor_bo_sim.py
"""
bo_sim/cost.py — coûts appliqués à l'oracle, miroirs vectorisés des coûts toofast.

  CosineCost     'full_spectrum' + cosine (S28-M01) : 1 - cos(spectre, cible) sur 315-750 nm,
                 = toofast.core.cost.CosineSpectrumCost (costs.csv reproduit à 4e-16).
  PCACosineCost  'pca' + cosine k (S24-M01) : projection sur une base PCA figée
                 (pca_model_norm6.npz : interpolation sur sa grille, normalisation 'max',
                 centrage, k=4 premières composantes), 1 - cos(scores, scores cible)
                 = toofast.core.cost.PCASpectrumCost(metric='cosine').

Le spectre de l'oracle est d'abord interpolé sur la grille de la cible (1219 pts),
comme le ferait le spectromètre. La BO simulée utilise les VRAIES classes toofast
via `Experiment.cost_function` ; ce module sert aux baselines et au coût "vrai"
(regret) — l'identité est vérifiée dans le notebook.

`make_cost_oracle(oracle, run_dir)` lit le `_config.json` de la manip pour choisir
le style et ses paramètres (métrique, k, base PCA).
"""
import csv
import json
import os
import re

import numpy as np

from . import config as C

WL_MIN, WL_MAX = 315.0, 750.0   # CosineSpectrumCost defaults

# "Vrai" coût d'un point infaisable (spectre plat, pas de particules) pour les
# courbes de convergence = coût d'un spectre NUL par la même fonction de coût
# (`cc.infeasible_cost`) : cosinus brut -> similarité 0 -> 1.0 ; PCA-cosinus ->
# 1 - cos(scores(-mean), cible) ≈ 1.40 = exactement la sentinelle `_pca_flat_sentinel`
# de toofast. Côté BO simulée, un plat donne la sentinelle, remappée par nearest_high.
INFEASIBLE_COST = 1.0   # valeur du style cosinus brut (rétro-compat) ; préférer cc.infeasible_cost


def load_target_csv(path):
    """target_spectrum.csv toofast (';', header timestamp+wl, dernière ligne)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    wl = np.array([float(x) for x in rows[0][1:]])
    ab = np.array([float(x) for x in rows[-1][1:]])
    return wl, ab


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "_config.json"), encoding="utf-8") as f:
        return json.load(f)


def pca_model_for_run(run_dir):
    """Chemin local de la base PCA d'une manip 'pca' : basename de la ligne
    'PCA decomposition file' de <run>_parameters.txt, cherché dans C.PCA_DIR."""
    run_dir = os.fspath(run_dir)
    txt = [f for f in os.listdir(run_dir) if f.endswith("_parameters.txt")]
    if not txt:
        raise FileNotFoundError(f"pas de *_parameters.txt dans {run_dir}")
    with open(os.path.join(run_dir, txt[0]), encoding="utf-8", errors="replace") as f:
        m = re.search(r"PCA decomposition file:\s*(.+)", f.read())
    if not m:
        raise ValueError("ligne 'PCA decomposition file' absente de " + txt[0])
    name = os.path.basename(m.group(1).strip().replace("\\", "/"))
    p = C.PCA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{name} introuvable dans {C.PCA_DIR}")
    return p


class _TargetGrid:
    def __init__(self, target_csv, oracle_wl):
        self.target_wl, self.target_abs = load_target_csv(target_csv)
        self.oracle_wl = np.asarray(oracle_wl, float)

    def to_target_grid(self, S):
        S = np.atleast_2d(S)
        return np.vstack([np.interp(self.target_wl, self.oracle_wl, s) for s in S])

    @property
    def infeasible_cost(self):
        """Coût d'un spectre nul (plat) par cette fonction de coût."""
        return float(self(np.zeros((1, len(self.oracle_wl))))[0])


class CosineCost(_TargetGrid):
    """cost(spectres sur la grille de l'oracle) -> 1 - cosinus sur [315,750]."""
    style = "cosine"

    def __init__(self, target_csv, oracle_wl):
        super().__init__(target_csv, oracle_wl)
        self.mask = (self.target_wl >= WL_MIN) & (self.target_wl <= WL_MAX)
        t = self.target_abs[self.mask]
        self._t_unit = t / np.linalg.norm(t)

    def __call__(self, S_oracle):
        M = self.to_target_grid(S_oracle)[:, self.mask]
        nrm = np.linalg.norm(M, axis=1)
        nrm[nrm == 0] = 1.0
        return 1.0 - (M @ self._t_unit) / nrm


class PCACosineCost(_TargetGrid):
    """cost(spectres sur la grille de l'oracle) -> 1 - cosinus des k premiers scores PCA."""
    style = "pca"

    def __init__(self, target_csv, oracle_wl, pca_npz, n_components=4, metric="cosine"):
        super().__init__(target_csv, oracle_wl)
        if metric != "cosine":
            raise NotImplementedError("seul pca_cost_metric='cosine' est miroité")
        from .toofast_sim import SpectralPCA   # stub avaspec + sys.path gérés là-bas
        self.pca = SpectralPCA.load(os.fspath(pca_npz))
        self.k = max(1, min(int(n_components) or self.pca.n_components, self.pca.n_components))
        self.target_scores = self.pca.transform(self.target_wl, self.target_abs)
        t = self.target_scores[:self.k]
        self._t_unit = t / np.linalg.norm(t)

    def scores(self, S_oracle):
        """Projection vectorisée = SpectralPCA._prepare + components @ centered."""
        M = self.to_target_grid(S_oracle)
        Mg = np.vstack([np.interp(self.pca.wavelengths, self.target_wl, m) for m in M])
        mode = self.pca.normalize
        if mode == "max":
            p = np.max(np.abs(Mg), axis=1, keepdims=True); Mg = np.where(p > 1e-12, Mg / np.where(p > 1e-12, p, 1), Mg)
        elif mode == "area":
            a = np.trapz(np.abs(Mg), axis=1)[:, None]; Mg = np.where(a > 1e-12, Mg / np.where(a > 1e-12, a, 1), Mg)
        return (Mg - self.pca.mean) @ self.pca.components.T

    def __call__(self, S_oracle):
        s = self.scores(S_oracle)[:, :self.k]
        nrm = np.linalg.norm(s, axis=1)
        out = 1.0 - (s @ self._t_unit) / np.where(nrm > 1e-12, nrm, 1.0)
        out[nrm <= 1e-12] = 1.0
        return out


def build_cost(oracle_wl, run_dir=C.RUN_S28, target_csv=None):
    """Objet coût miroir de la manip `run_dir` (style + paramètres du _config.json)."""
    cfg = load_run_config(run_dir)
    target_csv = target_csv or os.path.join(run_dir, "target_spectrum.csv")
    if cfg["target_style"] == "full_spectrum" and cfg.get("cost_method") == "cosine":
        return CosineCost(target_csv, oracle_wl)
    if cfg["target_style"] == "pca":
        return PCACosineCost(target_csv, oracle_wl, pca_model_for_run(run_dir),
                             n_components=cfg.get("pca_n_components", 0),
                             metric=cfg.get("pca_cost_metric", "sqdist"))
    raise NotImplementedError(f"style {cfg['target_style']}/{cfg.get('cost_method')} non miroité")


def make_cost_oracle(oracle, run_dir=C.RUN_S28, target_csv=None):
    """Retourne f(Q5) -> coût vrai (vectorisé, gate incluse), et l'objet coût."""
    cc = build_cost(oracle.wl, run_dir, target_csv)
    c_flat = cc.infeasible_cost

    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        c = cc(oracle.predict_spectra(Q))
        c[~oracle.is_feasible(Q)] = c_flat
        return c
    return f, cc
