# [vendored] verbatim copy of paperAnalysis/bo_sim/targets.py (md5 9ae74797bdeb7de254dd51aa605850d8), tools/vendor_bo_sim.py
"""
bo_sim/targets.py — panel de cibles pour agréger le benchmark.

  'real'      : la cible de la manip (`<run>/target_spectrum.csv`, S24 et S28 partagent
                la même, pic 0.587 à 607 nm).
  't<k>'      : cibles ATTEIGNABLES = spectres de l'oracle (moyenne) en des points
                faisables Q*_k du domaine, choisis pour étaler λmax. Par construction
                l'optimum vaut 0 en Q*_k (regret exact), et la cible est réaliste
                (c'est ce que le système sait produire selon les données).

Les cibles sont écrites au format toofast (`target_spectrum.csv`, ';', header
timestamp + wl, une ligne) sur la grille de la cible réelle (1219 pts), dans
RESULTS_DIR/targets/, avec un `targets.json` (Q*, λmax, seed).
"""
import csv
import json
import os

import numpy as np

from . import config as C
from .cost import load_target_csv

TARGET_DIR = C.RESULTS_DIR / "targets"


def _write_target_csv(path, wl, ab):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["timestamp"] + [repr(float(x)) for x in wl])
        w.writerow(["oracle"] + [repr(float(x)) for x in ab])


def make_target_panel(orc, sampler, n_targets=8, seed=0, lam_range=(430.0, 720.0),
                      n_candidates=20_000, min_peak_ratio=1.5, overwrite=False):
    """Choisit n_targets points faisables dont les spectres oracle étalent λmax
    uniformément dans lam_range ; écrit les csv. Renvoie la liste des entrées
    [{'id', 'csv', 'Q', 'lambda_max'}] (+ 'real' en tête)."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = TARGET_DIR / "targets.json"
    if meta_path.exists() and not overwrite:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    grid_wl, _ = load_target_csv(C.RUN_S28 / "target_spectrum.csv")
    Q = sampler.random_dirichlet(n_candidates, seed)
    Q = Q[orc.is_feasible(Q)]
    S = orc.predict_spectra(Q)
    vis = (orc.wl >= 350) & (orc.wl <= 900)
    lam = orc.wl[vis][np.argmax(S[:, vis], axis=1)]
    # pic net : max / médiane du spectre normalisé (évite les spectres "bosse molle")
    ratio = S[:, vis].max(1) / np.maximum(np.median(S[:, vis], 1), 1e-6)
    ok = (lam >= lam_range[0]) & (lam <= lam_range[1]) & (ratio >= min_peak_ratio)
    Q, S, lam = Q[ok], S[ok], lam[ok]

    entries = [dict(id="real", csv=str(C.RUN_S28 / "target_spectrum.csv"), Q=None,
                    lambda_max=float(grid_wl[np.argmax(load_target_csv(C.RUN_S28 / 'target_spectrum.csv')[1])]))]
    rng = np.random.default_rng(seed)
    for k, lam_t in enumerate(np.linspace(lam_range[0], lam_range[1], n_targets)):
        # parmi les 2 % de candidats les plus proches de lam_t, un au hasard
        near = np.argsort(np.abs(lam - lam_t))[:max(5, len(lam) // 50)]
        i = int(rng.choice(near))
        ab = np.interp(grid_wl, orc.wl, S[i])
        p = TARGET_DIR / f"target_t{k}.csv"
        _write_target_csv(p, grid_wl, ab)
        entries.append(dict(id=f"t{k}", csv=str(p), Q=Q[i].tolist(), lambda_max=float(lam[i])))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1)
    return entries


def load_target_panel():
    with open(TARGET_DIR / "targets.json", encoding="utf-8") as f:
        return json.load(f)
