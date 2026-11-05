# [vendored] verbatim copy of paperAnalysis/bo_sim/paper_bench.py (md5 8bc63c4c1ab33f004ebf25ee10f35390), tools/vendor_bo_sim.py
"""
bo_sim/paper_bench.py — protocole de comparaison « article » BO vs échantillonnage.

Remplace, pour l'article, la comparaison de `paper_figure.py` (oracle moyen seul,
déterministe, tests non appariés sur des runs supposés indépendants). Le protocole :

QUESTION. À budget égal, l'acquisition de la BO de l'instrument trouve-t-elle une
meilleure recette que l'échantillonneur de l'instrument ? Et de combien
d'expériences a-t-elle besoin pour atteindre la même qualité ?

BRAS (même budget 30, même domaine, même fonction de coût que la manip réelle)
  bo     la BO réellement utilisée (S28-M01 -> robust, S24-M01 -> prior_warm),
         code `toofast` d'origine, init = 6 points du sampler de l'instrument.
  instr  CONTRASTE PRINCIPAL : mêmes 6 points d'init (même seed) puis 24 points du
         même sampler (LHS cube puis reprojection à Qtot). Comparaison APPARIÉE seed
         par seed : la seule différence est la règle de choix des points 7…30.
  unif   référence naïve : uniforme sur le simplexe (Dirichlet), indépendant dès N=1.
         (Le LHS « vrai » sur le simplexe a la même loi à N fixé — non tracé.)
  L'écart instr / unif mesure l'effet du SAMPLER (la reprojection concentre les
  recettes au centre, plus souvent faisables) — à ne pas attribuer à la BO.

PAYSAGES. 9 cibles (la cible réelle + 8 cibles atteignables étalées en λmax) ×
  3 versions de l'oracle : la moyenne a posteriori et 2 tirages a posteriori
  cohérents (« mondes », `oracle.WorldOracle`) = 27 paysages par fonction de coût.
  Les mondes testent la robustesse à la forme du paysage, que l'oracle moyen lisse.

MESURE. Déterministe : l'oracle renvoie le spectre exact. Une BO tourne dans une
  seule session expérimentale, où la répétabilité est très bonne ; le bruit de mesure
  est donc négligé (la variabilité entre sessions est absorbée par les « mondes »).
  Recommandation = meilleure recette évaluée.

MÉTRIQUE. Regret simple normalisé de la recommandation après N expériences :
    r_N = (c_vrai(x̂_N) − c*) / (c_med − c*)
  c*    = optimum du paysage : min(échantillon uniforme de 60 000 points, polissage
          Nelder-Mead des 10 meilleurs, tout point évalué par un bras) -> r ≥ 0 ;
  c_med = coût médian des recettes FAISABLES (r = 1 : recette faisable « typique »).
  Une recette infaisable (plat) coûte le coût d'un spectre nul : r peut dépasser 1.

STATISTIQUE. L'unité indépendante est la CIBLE (les mondes d'une cible et les seeds
  d'un paysage ne sont pas indépendants).
  - IC 95 % par bootstrap hiérarchique : cibles tirées avec remise (avec tous leurs
    mondes), puis seeds tirés avec remise dans chaque paysage — appariés entre bo et
    instr (même indice de seed).
  - Effet apparié : log10 du rapport r_bo / r_instr par seed (plancher 1e-3),
    médiane par cible -> test de Wilcoxon signé unilatéral exact sur les 9 cibles
    (p minimal possible 1/512 ≈ 0.002) + nombre de cibles gagnées.
  - « Budget équivalent » : plus petit N où la médiane de bo passe sous la médiane
    de instr à N = 30 (IC bootstrap).
  - Temps d'atteinte : P(r_N ≤ seuil top-1 % des recettes faisables du paysage).

    py -3.11 -m bo_sim.paper_bench                 # tout (cache bo_sim_results/paper_bench/)
    py -3.11 -m bo_sim.paper_bench --recompute

Sorties : figures_publication_BO/bench_{curves,time_to_target,forest,infeasible}.{png,pdf,svg},
bench_metrics.csv,
bench_per_target.csv, bench_oracle_fidelity.csv.
"""
import argparse
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from . import config as C, baselines, bench, targets as T
from .oracle import get_oracle
from .cost import make_cost_oracle
from .paper_figure import FIG, _apply_rc, _save, COST_NAME, BO_OF_RUN

RUNS = ("S28-M01", "S24-M01")
BUDGET, N_INIT = 30, 6
SEEDS_BO = range(10)
SEEDS_UNIF = range(100)
WORLDS = (None, 0, 1)
N_REF = 60_000
LOG_FLOOR = 1e-3                 # plancher du regret normalisé (log, rapports)
NS_REPORT = (10, 15, 20, 30)
N_BOOT = 2000
CACHE_DIR = C.RESULTS_DIR / "paper_bench"
OUT_DIR = C.FIG_OUT
ARMS = ("unif", "instr", "bo")
ARM_LABEL = dict(unif="Uniform random", instr="Projected LHS", bo="BO")
COST_YLABEL = {"S28-M01": "raw-spectrum cosine cost", "S24-M01": "normalised PCA-cosine cost"}
ARM_COLOR = dict(unif="0.35", instr="#1f77b4", bo="#d62728")
ARM_LS = dict(unif=":", instr="--", bo="-")


# =============================================================================
#  évaluation
# =============================================================================
def _polish_min(f, smp, Q_start, n_start=10):
    """Polissage Nelder-Mead (4 débits libres, le 5e = Qtot − somme) des meilleurs points."""
    B, Qt = smp.bounds, smp.total_flow

    def g(x):
        q = np.append(x, Qt - x.sum())
        if np.any(q < B[:, 0] - 1e-9) or np.any(q > B[:, 1] + 1e-9):
            return 10.0
        return float(f(q[None])[0])
    best = np.inf
    for q0 in Q_start[:n_start]:
        r = minimize(g, q0[:4], method="Nelder-Mead",
                     options=dict(maxiter=400, xatol=1e-4, fatol=1e-7))
        best = min(best, r.fun)
    return best


def _designs(smp):
    """Séquences de recettes des bras sans modèle (indépendantes du paysage)."""
    instr = np.stack([np.vstack([smp.lhs_toofast(N_INIT, s),
                                 smp.lhs_toofast(BUDGET - N_INIT, 10_000 * s + BUDGET)])
                      for s in SEEDS_BO])
    unif = np.stack([smp.random_dirichlet(BUDGET, s) for s in SEEDS_UNIF])
    return dict(instr=instr, unif=unif)


def _bo_runs(run, tag, tid, world):
    var = BO_OF_RUN[run]
    r = bench.load_panel(tag, [dict(id=tid)], [world], [var], SEEDS_BO)[(tid, world)][var]
    return dict(true=r["costs_true"], flows=r["flows"], infeasible=r["infeasible"])


def build_case(run, recompute=False):
    """Toutes les courbes de regret d'une manip, par paysage."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{run}_noiseless.pkl"
    if cache.exists() and not recompute:
        return pickle.load(open(cache, "rb"))
    warnings.filterwarnings("ignore")
    run_dir = C.BO_RUNS_DIR / run
    tag = f"panel_{run.lower().split('_')[0]}"
    smp = baselines.Sampler(baselines.run_bounds(run_dir, 0.0))
    D = _designs(smp)
    base_orc = get_oracle()
    cells, init_check = {}, []
    for t in T.load_target_panel():
        for w in WORLDS:
            orc = base_orc if w is None else get_oracle(world=w)
            f, _ = make_cost_oracle(orc, run_dir, None if t["id"] == "real" else t["csv"])
            ref = baselines.oracle_reference(f, smp, n=N_REF, seed=12345)
            feas = orc.is_feasible(ref["Q"])
            order = np.argsort(ref["costs"])
            c_pol = _polish_min(f, smp, ref["Q"][order])
            bo = _bo_runs(run, tag, t["id"], w)
            init_check.append(float(np.abs(bo["flows"][:, :N_INIT] - D["instr"][:, :N_INIT]).max()))
            arms = {"bo": bo["true"]}
            for k in ("instr", "unif"):
                Q = D[k]
                arms[k] = f(Q.reshape(-1, 5)).reshape(Q.shape[:2])
            c_star = min(ref["c_min"], c_pol, *(a.min() for a in arms.values()))
            c_med = float(np.median(ref["costs"][feas]))
            scale = c_med - c_star
            thr1 = (np.percentile(ref["costs"][feas], 1) - c_star) / scale
            cell = dict(c_star=c_star, c_med=c_med, scale=scale, thr1=thr1,
                        feasible_frac=float(feas.mean()), lambda_max=t["lambda_max"],
                        c_star_sampled=ref["c_min"], c_star_polished=c_pol,
                        bo_infeasible=bo["infeasible"])
            for k, a in arms.items():
                cell[k] = (np.minimum.accumulate(a, axis=1) - c_star) / scale
            cells[(t["id"], w)] = cell
            print(f"[{run}] {t['id']:5s} w={w}  c*={c_star:.5f} "
                  f"(sample {ref['c_min']:.5f}, polish {c_pol:.5f})")
    case = dict(run=run, variant=BO_OF_RUN[run], cells=cells,
                init_max_abs_diff=max(init_check))
    pickle.dump(case, open(cache, "wb"))
    return case


# =============================================================================
#  statistiques
# =============================================================================
def _targets(case):
    return sorted({k[0] for k in case["cells"]}, key=lambda s: (s != "real", s))


def _pool(case, arm, keys=None):
    keys = keys or list(case["cells"])
    return np.concatenate([case["cells"][k][arm] for k in keys])


def _boot_samples(case, rng):
    """Un tirage bootstrap hiérarchique -> dict arm -> (n, N) poolé. Seeds appariés
    pour bo / instr (même indice), indépendants pour unif."""
    tids = _targets(case)
    pick = rng.choice(tids, len(tids))
    out = {a: [] for a in ARMS}
    for tid in pick:
        for key in [k for k in case["cells"] if k[0] == tid]:
            c = case["cells"][key]
            s = rng.integers(0, len(c["bo"]), len(c["bo"]))
            out["bo"].append(c["bo"][s]); out["instr"].append(c["instr"][s])
            out["unif"].append(c["unif"][rng.integers(0, len(c["unif"]), len(c["unif"]))])
    return {a: np.concatenate(v) for a, v in out.items()}


def bootstrap(case, n_boot=N_BOOT, seed=0):
    """Distributions bootstrap des quantités rapportées."""
    rng = np.random.default_rng(seed)
    med = {a: [] for a in ARMS}
    ratio = {(a, N): [] for a in ("instr", "unif") for N in NS_REPORT}
    beq = []
    for _ in range(n_boot):
        b = _boot_samples(case, rng)
        m = {a: np.median(b[a], 0) for a in ARMS}
        for a in ARMS:
            med[a].append(m[a])
        for a in ("instr", "unif"):
            for N in NS_REPORT:
                ratio[(a, N)].append(max(m["bo"][N - 1], LOG_FLOOR) / max(m[a][N - 1], LOG_FLOOR))
        hit = np.flatnonzero(m["bo"] <= m["instr"][BUDGET - 1])
        beq.append(hit[0] + 1 if len(hit) else np.nan)
    return dict(med={a: np.array(v) for a, v in med.items()},
                ratio={k: np.array(v) for k, v in ratio.items()}, beq=np.array(beq))


def paired_effect(case, N):
    """log10(r_bo / r_instr) par seed (apparié), médiane par paysage puis par cible."""
    per_cell = {k: np.median(np.log10(np.maximum(c["bo"][:, N - 1], LOG_FLOOR)
                                      / np.maximum(c["instr"][:, N - 1], LOG_FLOOR)))
                for k, c in case["cells"].items()}
    per_target = {tid: np.median([v for k, v in per_cell.items() if k[0] == tid])
                  for tid in _targets(case)}
    return per_cell, per_target


def metrics(case, boot=None):
    boot = boot or bootstrap(case)
    rows = []
    P = {a: _pool(case, a) for a in ARMS}
    for N in NS_REPORT:
        for a in ARMS:
            lo, hi = np.percentile(boot["med"][a][:, N - 1], [2.5, 97.5])
            rows.append(dict(metric="regret_median", N=N, arm=a, value=np.median(P[a][:, N - 1]), ci_lo=lo, ci_hi=hi))
            rows.append(dict(metric="P_regret_below_top1pct", N=N, arm=a,
                             value=np.mean(np.concatenate([c[a][:, N - 1] <= c["thr1"] for c in case["cells"].values()]))))
        for a in ("instr", "unif"):
            r = boot["ratio"][(a, N)]
            point = max(np.median(P["bo"][:, N - 1]), LOG_FLOOR) / max(np.median(P[a][:, N - 1]), LOG_FLOOR)
            rows.append(dict(metric="ratio_median_regret_bo_over", N=N, arm=a, value=point,
                             ci_lo=np.percentile(r, 2.5), ci_hi=np.percentile(r, 97.5)))
        _, pt = paired_effect(case, N)
        v = np.array(list(pt.values()))
        rows.append(dict(metric="paired_log10_ratio_bo_instr_median_over_targets", N=N, arm="instr", value=np.median(v)))
        rows.append(dict(metric="targets_won_by_bo_vs_instr", N=N, arm="instr", value=f"{int((v < 0).sum())}/{len(v)}"))
        rows.append(dict(metric="wilcoxon_targets_p_one_sided", N=N, arm="instr",
                         value=wilcoxon(v, alternative="less").pvalue))
        rows.append(dict(metric="P_bo_beats_paired_instr_run", N=N, arm="instr",
                         value=np.mean(P["bo"][:, N - 1] < P["instr"][:, N - 1])))
    hit = np.flatnonzero(np.median(P["bo"], 0) <= np.median(P["instr"][:, BUDGET - 1]))
    rows.append(dict(metric="bo_budget_to_match_instr_at_30", N=BUDGET, arm="bo",
                     value=hit[0] + 1 if len(hit) else np.nan,
                     ci_lo=np.nanpercentile(boot["beq"], 2.5), ci_hi=np.nanpercentile(boot["beq"], 97.5)))
    for a in ARMS:
        rows.append(dict(metric="N_for_50pct_runs_below_top1pct", arm=a, value=_n_for(case, a, "thr1")))
    inf = np.concatenate([c["bo_infeasible"] for c in case["cells"].values()])
    rows.append(dict(metric="bo_infeasible_fraction_iter7_30", arm="bo", value=inf[:, N_INIT:].mean()))
    rows.append(dict(metric="bo_infeasible_fraction_iter25_30", arm="bo", value=inf[:, 24:].mean()))
    rows.append(dict(metric="n_landscapes", arm="-", value=len(case["cells"])))
    rows.append(dict(metric="n_runs_bo", arm="bo", value=len(P["bo"])))
    rows.append(dict(metric="init_shared_max_abs_diff_flows", arm="-", value=case["init_max_abs_diff"]))
    df = pd.DataFrame(rows)
    df.insert(0, "run", case["run"])
    return df


def _n_for(case, arm, thr_key, frac=0.5):
    hits = np.concatenate([c[arm] <= c[thr_key] for c in case["cells"].values()]).mean(0)
    idx = np.flatnonzero(hits >= frac)
    return int(idx[0] + 1) if len(idx) else f"> {BUDGET}"


def per_target_table(case):
    rows = []
    for N in (15, 30):
        per_cell, _ = paired_effect(case, N)
        for (tid, w), v in per_cell.items():
            c = case["cells"][(tid, w)]
            rows.append(dict(run=case["run"], target=tid, lambda_max=round(c["lambda_max"], 1),
                             world="mean" if w is None else f"w{w}", N=N,
                             median_regret_bo=np.median(c["bo"][:, N - 1]),
                             median_regret_instr=np.median(c["instr"][:, N - 1]),
                             median_regret_unif=np.median(c["unif"][:, N - 1]),
                             paired_ratio_bo_instr=10 ** v,
                             c_star=c["c_star"], c_star_gain_from_polish=c["c_star_sampled"] - c["c_star"]))
    return pd.DataFrame(rows)


def oracle_fidelity():
    """Erreur de l'oracle (moyenne) contre les coûts MESURÉS des deux manips
    (sessions incluses dans l'entraînement de l'oracle -> borne optimiste)."""
    orc = get_oracle(); rows = []
    for run in RUNS:
        d = C.BO_RUNS_DIR / run
        f, _ = make_cost_oracle(orc, d, None)
        z = np.load(sorted(glob.glob(str(d / "data" / "bo_data_*.npz")))[-1])
        c, p = z["costs"], f(z["flows"])
        e = np.abs(p - c)
        rows.append(dict(run=run, n=len(c), median_abs_error=np.median(e), mae=e.mean(),
                         rmse=np.sqrt(((p - c) ** 2).mean()),
                         spearman=pd.Series(p).corr(pd.Series(c), method="spearman"),
                         best_measured=c.min(), oracle_at_best_measured=p[np.argmin(c)]))
    return pd.DataFrame(rows)


# =============================================================================
#  figures
# =============================================================================
def _style(F):
    F = {**FIG, **(F or {})}
    _apply_rc(F)
    return F


PANEL_SIZE = (5.4, 4.6)          # un panneau, presque carré (largeur, hauteur en pouces)


def _init_marks(ax, F):
    """Zone grise = plan initial (expériences 1…N_INIT), trait = début de la BO."""
    ax.axvspan(0, N_INIT, color=F["init_shade_color"], zorder=0, lw=0)
    ax.axvline(N_INIT, color="0.35", ls="--", lw=1, zorder=1)
    tr = ax.get_xaxis_transform()
    ax.text(N_INIT / 2, 0.985, "initial\ndesign", transform=tr, ha="center", va="top",
            fontsize=F["legend_size"] - 1, color="0.35")
    # flèche de l'étiquette vers le trait vertical (début de la BO)
    ax.annotate("BO starts", xy=(N_INIT, 0.95), xycoords=tr, xytext=(N_INIT + 5, 0.95), textcoords=tr,
                ha="left", va="center", fontsize=F["legend_size"] - 1, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.25", lw=1))


def _curve_panel(ax, case, boot, F, letter=None):
    x = np.arange(1, BUDGET + 1)
    for a in ARMS:
        arr = np.maximum(_pool(case, a), F["floor_norm"])
        ax.plot(x, np.median(arr, 0), color=ARM_COLOR[a], ls=ARM_LS[a], lw=F["lw"], label=ARM_LABEL[a])
        lo, hi = np.percentile(np.maximum(boot["med"][a], F["floor_norm"]), [2.5, 97.5], axis=0)
        ax.fill_between(x, lo, hi, color=ARM_COLOR[a], alpha=0.25, lw=0)
    ax.axhline(1.0, color="0.5", ls="-.", lw=1, zorder=0)
    ax.text(0.98, 1.0, "median feasible recipe", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=F["legend_size"] - 1, color="0.4")
    _init_marks(ax, F)
    ax.set_yscale("log"); ax.set_xlim(0, BUDGET + 0.5); ax.set_ylim(*(F["ylim"] or (4e-3, 6)))
    ax.set_xlabel(F["xlabel"])
    ax.set_ylabel(f"Normalised simple regret\n({COST_YLABEL[case['run']]})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.77), frameon=F["legend_frame"], fontsize=F["legend_size"])
    ax.set_box_aspect(1)
    if letter:
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=F["panel_letter_size"],
                fontweight="bold", ha="right", va="bottom")


def fig_curves(cases, boots, stem, F=None):
    F = _style(F)
    w, h = PANEL_SIZE
    f, axes = plt.subplots(1, len(cases), figsize=(w * len(cases), h))
    for i, (ax, c, b) in enumerate(zip(np.atleast_1d(axes), cases, boots)):
        _curve_panel(ax, c, b, F, letter=f"({chr(97 + i)})")
    f.tight_layout(); _save(f, stem, F)
    return f


def fig_time_to_target(cases, stem, F=None):
    """P(meilleure recette dans le top 1 % des recettes faisables) en fonction de N."""
    F = _style(F)
    w, h = PANEL_SIZE
    f, axes = plt.subplots(1, len(cases), figsize=(w * len(cases), h), sharey=True)
    x = np.arange(1, BUDGET + 1)
    for j, (ax, case) in enumerate(zip(np.atleast_1d(axes), cases)):
        for a in ARMS:
            p = np.concatenate([c[a] <= c["thr1"] for c in case["cells"].values()]).mean(0)
            ax.plot(x, p, color=ARM_COLOR[a], ls=ARM_LS[a], lw=F["lw"], label=ARM_LABEL[a])
        _init_marks(ax, F)
        ax.set_ylim(0, 1); ax.set_xlim(0, BUDGET + 0.5); ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"P(best recipe in top 1 %)\n({COST_YLABEL[case['run']]})")
        ax.set_xlabel(F["xlabel"])
        ax.legend(loc="upper right", fontsize=F["legend_size"])
        ax.tick_params(labelleft=True)
        ax.set_box_aspect(1)
        ax.text(-0.02, 1.02, f"({chr(97 + j)})", transform=ax.transAxes, fontsize=F["panel_letter_size"],
                fontweight="bold", ha="right", va="bottom")
    f.tight_layout(); _save(f, stem, F)
    return f


def fig_forest(cases, stem, F=None, Ns=(15, 30)):
    """Effet apparié BO / instr par paysage (point) et par cible (losange)."""
    F = _style(F)
    f, axes = plt.subplots(1, len(Ns), figsize=(F["figsize"][0] * 1.6, 0.55 * F["figsize"][1] + 0.22 * 20),
                           sharey=True)
    rows, ylabels = [], []
    for case in cases:
        for tid in _targets(case):
            lam = case["cells"][next(k for k in case["cells"] if k[0] == tid)]["lambda_max"]
            rows.append((case, tid)); ylabels.append(f"{case['run']:9s} {tid:4s} ({lam:.0f} nm)")
    y = np.arange(len(rows))[::-1]
    for ax, N in zip(np.atleast_1d(axes), Ns):
        for yi, (case, tid) in zip(y, rows):
            cell, tgt = paired_effect(case, N)
            for k, v in cell.items():
                if k[0] == tid:
                    ax.plot(10 ** v, yi, "o", ms=4, color="0.55", mfc="none")
            ax.plot(10 ** tgt[tid], yi, "D", ms=6, color="#d62728" if tgt[tid] < 0 else "0.2")
        ax.axvline(1, color="k", lw=1)
        ax.set_xscale("log"); ax.set_title(f"N = {N}")
        ax.set_xlabel("Paired regret ratio BO / instrument sampler\n(< 1: BO better)")
        ax.grid(True, alpha=0.3, axis="x")
    np.atleast_1d(axes)[0].set_yticks(y); np.atleast_1d(axes)[0].set_yticklabels(ylabels, fontsize=F["legend_size"] - 1,
                                                                                  family="monospace")
    from matplotlib.lines import Line2D
    np.atleast_1d(axes)[-1].legend(handles=[
        Line2D([], [], marker="o", ls="", color="0.55", mfc="none", label="one landscape (oracle mean / posterior world)"),
        Line2D([], [], marker="D", ls="", color="#d62728", label="target (median over landscapes)")],
        loc="lower right", fontsize=F["legend_size"] - 1)
    f.tight_layout(); _save(f, stem, F)
    return f


def fig_infeasible(cases, stem, F=None):
    F = _style(F)
    f, ax = plt.subplots(figsize=F["figsize"])
    x = np.arange(N_INIT + 1, BUDGET + 1)
    for case, ls in zip(cases, ("-", "--", ":", "-.")):
        inf = np.concatenate([c["bo_infeasible"] for c in case["cells"].values()])[:, N_INIT:]
        ax.plot(x, inf.mean(0), ls=ls, lw=F["lw"], label=f"{case['run']} ({case['variant'].replace('_', ' ')})")
    ax.set_xlabel("BO iteration (experiment number)"); ax.set_ylabel("Fraction of BO experiments\nwith no particles (flat spectrum)")
    ax.set_ylim(0, None); ax.grid(True, alpha=0.3); ax.legend(fontsize=F["legend_size"])
    f.tight_layout(); _save(f, stem, F)
    return f


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [build_case(r, a.recompute) for r in RUNS]
    boots = [bootstrap(c) for c in cases]
    pd.concat([metrics(c, b) for c, b in zip(cases, boots)]).to_csv(OUT_DIR / "bench_metrics.csv", index=False)
    pd.concat([per_target_table(c) for c in cases]).to_csv(OUT_DIR / "bench_per_target.csv", index=False)
    oracle_fidelity().to_csv(OUT_DIR / "bench_oracle_fidelity.csv", index=False)
    fig_curves(cases, boots, "bench_curves"); plt.close("all")
    fig_time_to_target(cases, "bench_time_to_target"); plt.close("all")
    fig_forest(cases, "bench_forest"); plt.close("all")
    fig_infeasible(cases, "bench_infeasible"); plt.close("all")
    m = pd.read_csv(OUT_DIR / "bench_metrics.csv")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(m[m.N.isin([15, 30]) | m.N.isna()].to_string(index=False))


# =============================================================================
#  prolongation des stratégies sans modèle (au-delà du budget de la BO)
# =============================================================================
N_EXT, SEEDS_EXT = 400, range(100)


def build_extended(run, recompute=False):
    """Projected LHS et uniforme prolongés jusqu'à N_EXT expériences (100 seeds par
    paysage) : regret normalisé (n, N_EXT) par paysage, mêmes c*, échelle et seuil
    top-1 % que build_case. La BO reste à 30 (cellules de build_case)."""
    cache = CACHE_DIR / f"{run}_extended{N_EXT}.pkl"
    if cache.exists() and not recompute:
        return pickle.load(open(cache, "rb"))
    warnings.filterwarnings("ignore")
    case = build_case(run)
    run_dir = C.BO_RUNS_DIR / run
    smp = baselines.Sampler(baselines.run_bounds(run_dir, 0.0))
    D = dict(instr=np.stack([np.vstack([smp.lhs_toofast(N_INIT, s),
                                        smp.lhs_toofast(N_EXT - N_INIT, 10_000 * s + N_EXT)]) for s in SEEDS_EXT]),
             unif=np.stack([smp.random_dirichlet(N_EXT, s) for s in SEEDS_EXT]))
    base_orc, cells = get_oracle(), {}
    for t in T.load_target_panel():
        for w in WORLDS:
            orc = base_orc if w is None else get_oracle(world=w)
            f, _ = make_cost_oracle(orc, run_dir, None if t["id"] == "real" else t["csv"])
            ref = case["cells"][(t["id"], w)]
            cells[(t["id"], w)] = {k: (np.minimum.accumulate(f(Q.reshape(-1, 5)).reshape(Q.shape[:2]), 1)
                                       - ref["c_star"]) / ref["scale"] for k, Q in D.items()}
            cells[(t["id"], w)]["thr1"] = ref["thr1"]
            print(f"[{run} extended] {t['id']} w={w}", flush=True)
    ext = dict(run=run, cells=cells)
    pickle.dump(ext, open(cache, "wb"))
    return ext


def _n50(p, frac=0.5):
    i = np.flatnonzero(p >= frac)
    return int(i[0] + 1) if len(i) else None


def fig_extended(cases, exts, stem="bench_extended", F=None, xmax=130):
    """Ligne 1 : regret médian ; ligne 2 : P(top 1 %). BO jusqu'à 30, les stratégies
    sans modèle jusqu'à xmax ; repère 50 % et N où chaque stratégie l'atteint."""
    F = _style(F)
    w, h = PANEL_SIZE
    f, axes = plt.subplots(2, len(cases), figsize=(w * len(cases), 2 * h), sharex=True)
    for j, (case, ext) in enumerate(zip(cases, exts)):
        series = {"bo": np.concatenate([c["bo"] for c in case["cells"].values()]),
                  **{k: np.concatenate([c[k] for c in ext["cells"].values()]) for k in ("instr", "unif")}}
        hits = {"bo": np.concatenate([c["bo"] <= c["thr1"] for c in case["cells"].values()]),
                **{k: np.concatenate([c[k] <= c["thr1"] for c in ext["cells"].values()]) for k in ("instr", "unif")}}
        ax_r, ax_p = axes[0, j], axes[1, j]
        for a in ARMS:
            n = min(series[a].shape[1], xmax)
            x = np.arange(1, n + 1)
            ax_r.plot(x, np.maximum(np.median(series[a][:, :n], 0), F["floor_norm"]),
                      color=ARM_COLOR[a], ls=ARM_LS[a], lw=F["lw"], label=ARM_LABEL[a])
            p = hits[a].mean(0)
            ax_p.plot(x, p[:n], color=ARM_COLOR[a], ls=ARM_LS[a], lw=F["lw"], label=ARM_LABEL[a])
            n50 = _n50(p)
            if n50 and n50 <= xmax:
                ax_p.plot([n50, n50], [0, 0.5], color=ARM_COLOR[a], lw=1, ls=":")
                ax_p.annotate(f"{n50}", (n50, 0.5), xytext=(0, 6), textcoords="offset points",
                              ha="center", fontsize=F["legend_size"], color=ARM_COLOR[a], fontweight="bold")
        ax_p.axhline(0.5, color="0.5", lw=1, ls="-.", zorder=0)
        for ax in (ax_r, ax_p):
            ax.axvspan(0, N_INIT, color=F["init_shade_color"], zorder=0, lw=0)
            ax.axvline(BUDGET, color="#d62728", lw=0.8, ls=":", zorder=0)
            ax.set_xlim(0, xmax); ax.grid(True, alpha=0.3, which="both"); ax.set_box_aspect(1)
        ax_r.text(BUDGET + 1, 0.97, "BO budget\n(30 exp.)", transform=ax_r.get_xaxis_transform(),
                  ha="left", va="top", fontsize=F["legend_size"] - 1, color="#d62728")
        ax_r.set_yscale("log"); ax_r.set_ylim(4e-3, 6)
        ax_r.axhline(1.0, color="0.5", ls="-.", lw=1, zorder=0)
        ax_r.set_ylabel(f"Normalised simple regret\n({COST_YLABEL[case['run']]})")
        ax_p.set_ylim(0, 1); ax_p.set_ylabel(f"P(best recipe in top 1 %)\n({COST_YLABEL[case['run']]})")
        ax_p.set_xlabel(F["xlabel"])
        ax_r.legend(loc="upper right", fontsize=F["legend_size"])
        for ax, k in ((ax_r, 0), (ax_p, 1)):
            ax.text(-0.02, 1.02, f"({chr(97 + 2 * k + j)})", transform=ax.transAxes,
                    fontsize=F["panel_letter_size"], fontweight="bold", ha="right", va="bottom")
    f.tight_layout(); _save(f, stem, F)
    return f


if __name__ == "__main__":
    main()
