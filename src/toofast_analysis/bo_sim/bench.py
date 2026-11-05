# [vendored] verbatim copy of paperAnalysis/bo_sim/bench.py (md5 4d8842d6b08bf7ee933865c245aa8d54), tools/vendor_bo_sim.py
"""
bo_sim/bench.py — orchestration : variantes × seeds, en parallèle, avec cache disque.

Chaque run BO est écrit dans  RESULTS_DIR/<tag>/<variant>_seed<k>.npz
(flows, costs, kernel_final, meta). Relancer avec les mêmes arguments ne
recalcule que ce qui manque.
"""
import json
import os
import shutil
import tempfile
import time
import warnings
import zlib
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed

from . import config as C


# ----------------------------------------------------------------- workers
@lru_cache(maxsize=None)
def _oracle_and_cost(run_dir, gate="classifier+rule", target_csv=None, world=None):
    """Chargé une fois par processus worker (et par cible / monde). Le coût
    (cosinus brut / PCA-cosinus) est celui du _config.json de `run_dir` ; la cible
    est `target_csv` (défaut : <run_dir>/target_spectrum.csv) ; `world` = n° du
    tirage a posteriori (None = moyenne)."""
    from .oracle import get_oracle
    from .cost import make_cost_oracle
    orc = get_oracle(gate=gate, world=world)
    f_cost, cc = make_cost_oracle(orc, run_dir, target_csv)
    return orc, f_cost, cc, make_measure(orc, cc)


def make_measure(orc, cc, noise=0.0, rng=None):
    """Q5 -> absorbance sur la grille de la cible, ou None si infaisable (plat).

    noise > 0 : mesure BRUITÉE — bruit gaussien sur les scores PCA, d'écart-type
    `noise` × bruit blanc estimé par le GP de l'oracle sur chaque PC
    (`oracle.score_noise_sd`). noise = 0.6 reproduit la répétabilité mesurée de
    λmax entre synthèses d'une même campagne (σ ≈ 15.5 nm, analyse_reproducibility)."""
    sd = None
    if noise:
        from .oracle import score_noise_sd
        sd = noise * score_noise_sd(orc)
        rng = rng if rng is not None else np.random.default_rng()

    def measure(q):
        if not orc.is_feasible(q)[0]:
            return None
        if sd is None:
            return cc.to_target_grid(orc.predict_spectra(q))[0]
        s = orc.predict_scores(q) + rng.normal(size=sd.shape) * sd
        return cc.to_target_grid(orc.mu + s @ orc.comps)[0]
    return measure


def _run_one(out_path, variant, seed, budget, bounds, run_dir, prior_name, n_restarts,
             gate="classifier+rule", target_csv=None, world=None, noise=0.0):
    from . import toofast_sim as ts
    warnings.filterwarnings("ignore")
    orc, f_cost, cc, measure = _oracle_and_cost(str(run_dir), gate, target_csv, world)
    if noise:
        # bruit propre au run, reproductible (seed du run + cible)
        rng = np.random.default_rng([int(seed), 7_919, zlib.crc32(str(target_csv).encode())])
        measure = make_measure(orc, cc, noise=noise, rng=rng)
    workdir = tempfile.mkdtemp(prefix=f"bo_sim_{variant}_{seed}_")
    try:
        exp = ts.build_experiment(measure, cc.target_wl, workdir, run_dir=run_dir,
                                  variant=variant, budget=budget, bounds=bounds,
                                  prior_name=prior_name, n_restarts=n_restarts,
                                  target_csv=target_csv)
        t0 = time.time()
        flows, costs, gp = ts.run_experiment(exp, seed)
        meta = dict(exp.sim_meta, seed=seed, elapsed_s=time.time() - t0,
                    kernel_final=str(gp.kernel_), gate=gate, world=world, noise=noise,
                    n_infeasible=int(sum(exp.sim_infeasible)))
        # costs        = ce que la BO a "mesuré" (sentinelle pour un plat ; bruité si noise > 0)
        # costs_mirror = coût vrai (oracle + gate : plat -> INFEASIBLE_COST), pour les courbes
        np.savez(out_path, flows=flows, costs=costs, costs_mirror=f_cost(flows),
                 infeasible=np.array(exp.sim_infeasible, bool), meta=json.dumps(meta))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out_path


# ------------------------------------------------------------------ driver
def _grid_dir(tag):
    d = C.RESULTS_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_name(variant, prior_name):
    """Nom de fichier : le prior fait partie de l'identité d'un run 'prior_*'."""
    return f"{variant}[{prior_name}]" if variant.startswith("prior") else variant


def run_bo_grid(variants, seeds, budget=30, bounds=None, run_dir=C.RUN_S28,
                prior_name=C.DEFAULT_PRIOR, n_restarts=None, tag="rg043_gate2",
                gate="classifier+rule", n_jobs=4, verbose=5, target_csv=None, world=None,
                noise=0.0):
    """Lance (ou complète) la grille variantes × seeds. Renvoie les chemins.

    Le `tag` identifie la grille (dossier de cache) : bornes / gate / budget doivent
    être constants dans un tag — ils sont relus depuis les métadonnées des runs."""
    d = _grid_dir(tag)
    bounds = None if bounds is None else np.asarray(bounds, float).tolist()
    jobs = []
    for v in variants:
        for s in seeds:
            p = d / f"{run_name(v, prior_name)}_seed{s}.npz"
            if not p.exists():
                jobs.append(delayed(_run_one)(str(p), v, int(s), int(budget), bounds,
                                              str(run_dir), prior_name, n_restarts, gate,
                                              target_csv, world, noise))
    print(f"[bench] {tag}: {len(jobs)} runs à faire "
          f"({len(variants)*len(seeds) - len(jobs)} déjà en cache) — n_jobs={n_jobs}")
    if jobs:
        Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    return sorted(d.glob("*.npz"))


def load_bo_grid(variants, seeds, tag="rg043_gate2", budget=None,
                 prior_name=C.DEFAULT_PRIOR):
    """dict variant -> dict(costs_true (n_seeds, budget) [oracle+gate, pour les courbes],
    costs_measured (ce que la BO a vu, sentinelle pour un plat), flows (n_seeds, budget, 5),
    infeasible (bool), bsf = best-so-far du coût vrai, meta [list])."""
    d = _grid_dir(tag)
    out = {}
    for v in variants:
        ct, cm, flows, infeas, metas = [], [], [], [], []
        for s in seeds:
            p = d / f"{run_name(v, prior_name)}_seed{s}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            n = len(z["costs"]) if budget is None else min(budget, len(z["costs"]))
            cm.append(z["costs"][:n]); flows.append(z["flows"][:n])
            ct.append(z["costs_mirror"][:n] if "costs_mirror" in z.files else z["costs"][:n])
            infeas.append(z["infeasible"][:n] if "infeasible" in z.files else np.zeros(n, bool))
            metas.append(json.loads(str(z["meta"])))
        if not ct:
            continue
        n = min(len(c) for c in ct)
        cut = lambda L: np.array([x[:n] for x in L])
        ct, cm, flows, infeas = cut(ct), cut(cm), cut(flows), cut(infeas)
        out[v] = dict(costs_true=ct, costs_measured=cm, costs=ct, flows=flows,
                      infeasible=infeas, bsf=np.minimum.accumulate(ct, axis=1), meta=metas)
    return out


# ------------------------------------------------------------------ résumé
def summarize(curves, budgets_report=(6, 10, 15, 20, 30), c_min=0.0):
    """Tableau : médiane / P10 / P90 du best-so-far à quelques budgets.
    curves : dict nom -> (n_seeds, budget) best-so-far."""
    import pandas as pd
    rows = []
    for name, arr in curves.items():
        for N in budgets_report:
            if N > arr.shape[1]:
                continue
            col = arr[:, N - 1]
            rows.append(dict(strategy=name, N=N, n_runs=len(col),
                             median=np.median(col), p10=np.percentile(col, 10),
                             p90=np.percentile(col, 90),
                             regret_med=np.median(col) - c_min))
    return pd.DataFrame(rows)


def edge_stats(results, bounds, n_init=6, eps=1e-3):
    """Fraction de points BO posés sur une borne du domaine, par variante."""
    import pandas as pd
    B = np.asarray(bounds, float)
    rows = []
    for v, r in results.items():
        F = r["flows"]
        edge = (np.isclose(F, B[:, 0], atol=eps) | np.isclose(F, B[:, 1], atol=eps)).any(-1)
        c = r["costs_true"]
        rows.append(dict(variant=v,
                         edge_init=edge[:, :n_init].mean(),
                         edge_bo=edge[:, n_init:].mean(),
                         edge_late=edge[:, (F.shape[1] + n_init) // 2:].mean(),
                         infeasible_bo=r["infeasible"][:, n_init:].mean(),
                         best_on_edge=np.mean([edge[s, c[s].argmin()] for s in range(len(c))])))
    return pd.DataFrame(rows).set_index("variant").round(3)


def iters_to_threshold(bsf, thr):
    """Pour chaque run, première itération (1-based) où best-so-far <= thr (nan sinon)."""
    hit = bsf <= thr
    out = np.full(len(bsf), np.nan)
    for i, h in enumerate(hit):
        idx = np.flatnonzero(h)
        if len(idx):
            out[i] = idx[0] + 1
    return out


# ------------------------------------------------------------ panel cibles × mondes
def panel_tag(base_tag, target_id, world):
    return f"{base_tag}/{target_id}/{'mean' if world is None else f'w{world}'}"


def run_panel(base_tag, targets, worlds, variants, seeds, budget=30, bounds=None,
              run_dir=C.RUN_S28, prior_name=C.DEFAULT_PRIOR, n_restarts=None,
              gate="classifier+rule", n_jobs=8, verbose=0, noise=0.0):
    """Grille complète cibles × mondes × variantes × seeds, un sous-dossier par
    (cible, monde). `targets` = liste d'entrées de targets.py ; `worlds` = [None, 0, 1, ...].
    noise > 0 : mesures bruitées (cf. make_measure) — utiliser un base_tag distinct."""
    for t in targets:
        for w in worlds:
            run_bo_grid(variants, seeds, budget=budget, bounds=bounds, run_dir=run_dir,
                        prior_name=prior_name, n_restarts=n_restarts,
                        tag=panel_tag(base_tag, t["id"], w), gate=gate, n_jobs=n_jobs,
                        verbose=verbose,
                        target_csv=None if t["id"] == "real" else t["csv"], world=w,
                        noise=noise)


def load_panel(base_tag, targets, worlds, variants, seeds, prior_name=C.DEFAULT_PRIOR):
    """dict (target_id, world) -> load_bo_grid(...)"""
    return {(t["id"], w): load_bo_grid(variants, seeds, tag=panel_tag(base_tag, t["id"], w),
                                       prior_name=prior_name)
            for t in targets for w in worlds}


def baseline_panel(orc_by_world, cost_by_key, sampler, targets, worlds, budget, seeds,
                   lhs_budgets, n_ref=100_000):
    """Pour chaque (cible, monde) : courbes random/Dirichlet, LHS(N), et référence
    (min + percentiles du paysage). cost_by_key[(tid, w)] = f_cost vectorisé."""
    from . import baselines as bl
    out = {}
    for t in targets:
        for w in worlds:
            f = cost_by_key[(t["id"], w)]
            ref = bl.oracle_reference(f, sampler, n=n_ref, seed=12345)
            feas = orc_by_world[w].is_feasible(ref["Q"])
            out[(t["id"], w)] = dict(
                random_toofast=bl.random_curves(f, sampler, budget, seeds, "random_toofast"),
                random_dirichlet=bl.random_curves(f, sampler, budget, seeds, "random_dirichlet"),
                lhs=bl.lhs_curve(f, sampler, lhs_budgets, seeds, "lhs_toofast", n_init=lhs_budgets[0]),
                lhs_simplex=bl.lhs_curve(f, sampler, lhs_budgets, seeds, "lhs_simplex", n_init=lhs_budgets[0]),
                c_min=ref["c_min"], q_min=ref["q_min"],
                pct=np.percentile(ref["costs"][feas], [0.1, 1, 5, 25, 50]),
                feasible_frac=float(feas.mean()))
    return out


def add_prefix_baselines(base, cost_by_key, sampler, seeds, budget=30, force=False):
    """Ajoute aux cellules de `base` les courbes LHS *en préfixe* (un seul plan de
    `budget` points par seed, best-so-far monotone) : 'lhs_prefix' (projected LHS
    toofast) et 'lhs_simplex_prefix' (vrai LHS simplexe), plus 'random_dirichlet'
    recalculé avec ces mêmes seeds. Même distribution à N fixé que "un plan par N",
    mais courbes lisses et statistiques séquentielles valides."""
    from . import baselines as bl
    for key, b in base.items():
        if not force and "lhs_prefix" in b and len(b["lhs_prefix"]) == len(seeds):
            continue
        f = cost_by_key[key]
        b["lhs_prefix"] = bl.random_curves(f, sampler, budget, seeds, "lhs_toofast")
        b["lhs_simplex_prefix"] = bl.random_curves(f, sampler, budget, seeds, "lhs_simplex")
        b["random_dirichlet"] = bl.random_curves(f, sampler, budget, seeds, "random_dirichlet")
    return base


def aggregate_panel(results, base, variants, budget):
    """Agrège sur (cible, monde) : regret simple (bsf − c_min) et indicatrices
    'top 1 % / 0.1 % du paysage' → dict strategy -> {regret, top1, top01} de forme
    (n_runs_total, budget) (concaténation des runs de toutes les cellules ;
    'lhs' est de forme (n, len(lhs_budgets)))."""
    agg = {}
    for key, r in results.items():
        b = base[key]
        thr1, thr01 = b["pct"][1], b["pct"][0]
        cells = {v: r[v]["bsf"][:, :budget] for v in variants if v in r}
        cells["random_dirichlet"] = b["random_dirichlet"][:, :budget]
        cells["random_toofast"] = b["random_toofast"][:, :budget]
        cells["lhs"] = b["lhs"]
        for extra in ("lhs_simplex", "lhs_prefix", "lhs_simplex_prefix"):
            if extra in b:
                cells[extra] = b[extra][:, :budget] if extra.endswith("prefix") else b[extra]
        for name, bsf in cells.items():
            d = agg.setdefault(name, dict(regret=[], top1=[], top01=[]))
            d["regret"].append(bsf - b["c_min"]); d["top1"].append(bsf <= thr1); d["top01"].append(bsf <= thr01)
    return {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in agg.items()}


# ------------------------------------------------------------------ statistiques
def _boot_median_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def stats_table(agg, N, lhs_budgets, pairs, metric="regret"):
    """Médiane [IC 95 % bootstrap] du regret à N pour chaque stratégie, et tests de
    Mann-Whitney (unilatéral : A < B) pour les paires demandées.
    pairs = [("robust", "lhs"), ("prior_frozen", "robust"), ...]"""
    import pandas as pd
    from scipy.stats import mannwhitneyu

    def col(name):
        a = agg[name][metric]
        per_n = name.startswith("lhs") and not name.endswith("prefix")
        return a[:, list(lhs_budgets).index(N)] if per_n else a[:, N - 1]
    rows = []
    for name in agg:
        m, lo, hi = _boot_median_ci(col(name))
        rows.append(dict(strategy=name, N=N, n=len(col(name)), median=m, ci_lo=lo, ci_hi=hi))
    summary = pd.DataFrame(rows).set_index("strategy")
    tests = []
    for a, b in pairs:
        if a in agg and b in agg:
            u = mannwhitneyu(col(a), col(b), alternative="less")
            tests.append(dict(A=a, B=b, N=N, p_value=u.pvalue,
                              median_ratio=float(np.median(col(a)) / max(np.median(col(b)), 1e-12))))
    return summary, pd.DataFrame(tests)
