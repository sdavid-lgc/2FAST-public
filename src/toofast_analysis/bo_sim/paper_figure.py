# [vendored] verbatim copy of paperAnalysis/bo_sim/paper_figure.py (md5 833baa5a300485850db7f7aa76f52835), tools/vendor_bo_sim.py
"""
bo_sim/paper_figure.py — figure et métriques « papier » : BO vs random / LHS,
oracle = moyenne a posteriori seulement, 9 cibles.

Stratégies tracées (une seule BO : celle de la manip réelle) :
  random        uniforme sur le simplexe (Dirichlet)
  lhs_simplex   vrai LHS sur le simplexe (stratifié + stick-breaking)
  lhs_projected générateur d'init de l'instrument (LHS cube -> reprojection)
  bo            S28-M01 -> 'robust' (cosinus brut, sans prior)
                S24-M01 -> 'prior_warm' (PCA-cosinus k=4, prior norm6 warm-start)

NORMALISE = True (défaut) : l'axe y est le REGRET NORMALISÉ, (coût − optimum du paysage)
/ (coût médian des points FAISABLES − optimum), calculé CIBLE PAR CIBLE avant agrégation.
1 = aussi bon qu'une recette médiane PARMI CELLES QUI FORMENT DES PARTICULES, 0 = optimum.
Attention : le random tire dans tout le domaine, dont 42 % est infaisable (coût plancher),
donc sa courbe démarre AU-DESSUS de 1 — ce n'est pas une incohérence, c'est la mesure de
ce que coûte l'absence de garde-fou sur la faisabilité. Indispensable ici : les
deux fonctions de coût vivent sur des échelles différentes (médiane du paysage 0.42 pour
le cosinus brut, 1.40 pour le PCA-cosinus) et l'oracle n'a pas la même résolution dans
chacune (plancher d'erreur 0.008 vs 0.046), donc les regrets bruts ne sont comparables ni
entre panneaux ni avec les coûts expérimentaux. NORMALISE = False rétablit le regret brut
(version SI) ; `regret_median_raw_cost_units` reste dans le CSV dans les deux cas.

PAIRED = True (défaut) : la BO et le `lhs_projected` partagent leurs 6 premières
expériences — c'est factuel, la BO s'initialise avec le sampler de l'instrument — puis
divergent ; leur comparaison est appariée et isole exactement ce qu'apporte l'acquisition.
`random` et `lhs_simplex` sont d'autres stratégies, indépendantes dès la 1re expérience,
donc leurs courbes divergent dès N = 1. Toutes les baselines sont séquentielles
(best-so-far d'un échantillon qui grandit) : courbes monotones, et « nombre
d'expériences pour atteindre un seuil » bien défini.
PAIRED = False : toutes indépendantes, LHS en « un plan par N » (non monotone).
Les deux définitions du LHS donnent la même loi à N fixé (KS p = 0.34–0.85).

Tout ce qui touche à l'apparence est dans le dict `FIG` (polices, légende, couleurs,
étiquettes, tailles) : modifier ici, ou passer `fig=dict(...)` à `paper_figure`.

    py -3.11 -m bo_sim.paper_figure                # les deux manips, figures + métriques
    py -3.11 -m bo_sim.paper_figure --run S28-M01  # une seule

Sorties : figures_publication_BO/<run>_bo_vs_sampling.{png,pdf,svg},
          figures_publication_BO/bo_vs_sampling_2panels.{png,pdf,svg},
          figures_publication_BO/<run>_metrics.csv (+ affichage).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from . import config as C, baselines, bench, targets as T
from .oracle import get_oracle
from .cost import make_cost_oracle

OUT_DIR = C.FIG_OUT
CACHE_DIR = C.RESULTS_DIR / "paper"

BO_OF_RUN = {"S28-M01": "robust", "S24-M01": "prior_warm"}     # la BO réellement utilisée
COST_NAME = {"S28-M01": "raw-spectrum cosine cost", "S24-M01": "normalised PCA-cosine cost"}

# =============================================================================
#  APPARENCE — tout se règle ici
# =============================================================================
FIG = dict(
    # --- polices ---
    font_family="sans-serif",        # ex. "Arial", "Helvetica", "DejaVu Sans"
    font_size=11,                    # taille de base (ticks, texte)
    label_size=12,                   # titres d'axes
    title_size=12,                   # titre du panneau
    legend_size=9,                   # légende
    panel_letter_size=13,            # (a), (b)
    panel_letter_xy=(0.02, 0.97),    # position de la lettre, en coordonnées d'axes (0-1)
    # --- légende ---
    legend_loc="upper right",        # 'upper right', 'lower left', 'best', ...
    legend_ncol=1,
    legend_frame=True,
    legend_show_n=False,             # ajouter "(n=…)" dans les étiquettes
    # --- figure ---
    figsize=(7.0, 4.6),              # un panneau ; le 2-panneaux utilise (2*w, h)
    dpi=300,
    formats=("png", "pdf", "svg"),
    band=(25, 75),                   # bande = percentiles (None -> pas de bande)
    band_alpha=0.15,
    lw=2.2,
    marker_lhs=None,                 # marqueur sur les courbes LHS ("o" ...) ; None -> ligne seule
    marker_size=4,
    log_y=True,
    floor=1e-4,                      # plancher du regret BRUT pour l'échelle log
    floor_norm=1e-3,                 # plancher du regret NORMALISÉ
    show_typical_line=True,          # ligne repère "regret = 1" (point typique du domaine)
    typical_label="median particle-forming recipe",
    typical_label_x=0.015,           # position horizontale de l'annotation (coord. d'axes)
    xlim=(0, 30.8),
    ylim=None,                       # ex. (5e-4, 1)
    show_init_shade=True,            # zone grise = phase d'init LHS(6) de la BO
    init_shade_color="0.92",
    grid=True,
    # --- textes ---
    xlabel="Number of experiments",
    ylabel="Simple regret (best cost so far − landscape minimum)",
    ylabel_norm="Normalised simple regret",   # (cost − optimum)/(median FEASIBLE cost − optimum)
    title=None,                      # None -> "S28-M01 — raw-spectrum cosine cost" ; "" -> pas de titre
    labels=dict(
        random="Random (uniform on the simplex)",
        lhs_simplex="LHS on the simplex (stratified uniform)",
        lhs_projected="Projected LHS (instrument sampler)",
        bo="Bayesian optimisation ({variant})",
    ),
    colors=dict(random="0.25", lhs_simplex="#2ca02c", lhs_projected="#1f77b4", bo="#d62728"),
    linestyles=dict(random="-", lhs_simplex=":", lhs_projected="--", bo="-"),
)

BUDGET, N_INIT = 30, 6
LHS_BUDGETS = list(range(N_INIT, BUDGET + 1))
SEEDS_BASE = range(100)
SEEDS_BO = range(10)
NORMALISE = True       # regret normalisé par cible (cf. docstring) ; False -> regret brut
PAIRED = True          # cf. docstring
# Stratégies qui partagent le plan d'init de la BO. UNIQUEMENT celles qui l'utilisent
# réellement : la BO s'initialise avec le sampler de l'instrument, donc `lhs_projected`
# partage ses 6 premiers points avec elle. `random` et `lhs_simplex` sont des stratégies
# différentes dès la 1re expérience et divergent donc dès N = 1.
SHARE_INIT_WITH_BO = ("lhs_projected",)


# =============================================================================
#  données
# =============================================================================
def _load_case(run, seeds_base=None, recompute=False):
    """Charge (ou calcule) tout ce qu'il faut pour une manip, oracle moyen, 9 cibles.
    Renvoie dict: curves (nom -> (n, N) regret), lhs_axis, thresholds, per-target data.

    En mode PAIRED les baselines utilisent LES MÊMES seeds que la BO : chaque run de
    baseline partage alors son plan d'init avec le run BO correspondant, et les courbes
    sont identiques point par point jusqu'à N = N_INIT."""
    if seeds_base is None:
        seeds_base = SEEDS_BASE
    run_dir = C.BO_RUNS_DIR / run
    bo_variant = BO_OF_RUN[run]
    tag = f"panel_{run.lower().split('_')[0]}"
    panel = T.load_target_panel()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{run}_baselines_mean_{len(seeds_base)}seeds{'_paired' if PAIRED else ''}.pkl"
    if cache.exists() and not recompute:
        base = pickle.load(open(cache, "rb"))
    else:
        orc = get_oracle()
        B = baselines.run_bounds(run_dir, 0.0); smp = baselines.Sampler(B)
        base = {}
        for t in panel:
            f, _ = make_cost_oracle(orc, run_dir, None if t["id"] == "real" else t["csv"])
            ref = baselines.oracle_reference(f, smp, n=60_000, seed=12345)
            feas = orc.is_feasible(ref["Q"])
            if PAIRED:
                # Appariée seulement pour les stratégies qui utilisent vraiment l'init
                # de la BO (SHARE_INIT_WITH_BO) ; les autres sont indépendantes dès N = 1,
                # mais restent séquentielles (best-so-far d'un échantillon qui grandit).
                # Seules les stratégies appariées sont limitées aux seeds de la BO ;
                # les indépendantes en utilisent beaucoup plus (courbes lisses).
                curves_b = {}
                for k, m in (("random", "random_dirichlet"), ("lhs_simplex", "lhs_simplex"),
                             ("lhs_projected", "lhs_toofast")):
                    curves_b[k] = (baselines.shared_init_curves(f, smp, BUDGET, SEEDS_BO, m, N_INIT)
                                   if k in SHARE_INIT_WITH_BO
                                   else baselines.random_curves(f, smp, BUDGET, SEEDS_BASE, m))
            else:
                curves_b = dict(
                    random=baselines.random_curves(f, smp, BUDGET, seeds_base, "random_dirichlet"),
                    lhs_simplex=baselines.lhs_curve(f, smp, LHS_BUDGETS, seeds_base, "lhs_simplex", n_init=N_INIT),
                    lhs_projected=baselines.lhs_curve(f, smp, LHS_BUDGETS, seeds_base, "lhs_toofast", n_init=N_INIT))
            base[t["id"]] = dict(
                **curves_b,
                c_min=ref["c_min"], pct=np.percentile(ref["costs"][feas], [0.1, 1, 5, 25, 50]),
                feasible_frac=float(feas.mean()))
        pickle.dump(base, open(cache, "wb"))
    # BO : cellules (cible, mean) du panel
    res = bench.load_panel(tag, panel, [None], [bo_variant], SEEDS_BO)
    per_target = {}
    for t in panel:
        b = base[t["id"]]; r = res[(t["id"], None)]
        bo = r[bo_variant]
        # échelle de normalisation propre à la cible : regret d'un point "typique"
        # du domaine faisable (médiane du paysage) -> regret normalisé 1 = aussi bon
        # qu'un tirage quelconque, 0 = optimum du paysage.
        scale = float(b["pct"][4] - b["c_min"])
        per_target[t["id"]] = dict(
            random=b["random"] - b["c_min"], lhs_simplex=b["lhs_simplex"] - b["c_min"],
            lhs_projected=b["lhs_projected"] - b["c_min"], bo=bo["bsf"] - b["c_min"],
            bo_infeasible=bo["infeasible"], thr1=b["pct"][1] - b["c_min"], thr01=b["pct"][0] - b["c_min"],
            c_min=b["c_min"], scale=scale, feasible_frac=b["feasible_frac"], lambda_max=t["lambda_max"])
    KEYS = ("random", "lhs_simplex", "lhs_projected", "bo")
    curves = {k: np.concatenate([d[k] for d in per_target.values()]) for k in KEYS}
    # version normalisée PAR CIBLE avant agrégation (sinon les cibles de grande
    # échelle dominent la médiane agrégée)
    curves_norm = {k: np.concatenate([d[k] / d["scale"] for d in per_target.values()]) for k in KEYS}
    # indicatrices "sous le seuil top 1 % / 0.1 %" (seuil propre à chaque cible ;
    # invariantes par normalisation)
    top = {}
    for key, thr_name in (("top1", "thr1"), ("top01", "thr01")):
        top[key] = {k: np.concatenate([d[k] <= d[thr_name] for d in per_target.values()]) for k in KEYS}
    return dict(run=run, bo_variant=bo_variant, curves=curves, curves_norm=curves_norm,
                top=top, per_target=per_target,
                lhs_axis=np.array(LHS_BUDGETS), x_axis=np.arange(1, BUDGET + 1))


def _curves(case):
    """Jeu de courbes utilisé pour la figure et les métriques (normalisé ou brut)."""
    return case["curves_norm"] if NORMALISE else case["curves"]


# =============================================================================
#  figure
# =============================================================================
def _apply_rc(F):
    matplotlib.rcParams.update({"font.family": F["font_family"], "font.size": F["font_size"],
                                "axes.labelsize": F["label_size"], "axes.titlesize": F["title_size"],
                                "legend.fontsize": F["legend_size"], "xtick.labelsize": F["font_size"],
                                "ytick.labelsize": F["font_size"], "savefig.dpi": F["dpi"],
                                "pdf.fonttype": 42, "ps.fonttype": 42})


def _label(F, key, case):
    lab = F["labels"][key].format(variant=case["bo_variant"].replace("_", " "))
    if F["legend_show_n"]:
        lab += f" (n={len(case['curves'][key])})"
    return lab


def draw_panel(ax, case, F, letter=None):
    """Trace un panneau (une manip) sur `ax`."""
    cv, x, xl = _curves(case), case["x_axis"], case["lhs_axis"]
    floor = F["floor_norm"] if NORMALISE else F["floor"]
    for key in ("random", "lhs_simplex", "lhs_projected", "bo"):
        arr = np.maximum(cv[key], floor) if F["log_y"] else cv[key]
        xx = x if PAIRED else (xl if key.startswith("lhs") else x)
        med = np.median(arr, 0)
        kw = dict(color=F["colors"][key], ls=F["linestyles"][key], lw=F["lw"], label=_label(F, key, case))
        if key.startswith("lhs") and F["marker_lhs"]:
            kw.update(marker=F["marker_lhs"], ms=F["marker_size"])
        ax.plot(xx, med, **kw)
        if F["band"]:
            ax.fill_between(xx, np.percentile(arr, F["band"][0], 0), np.percentile(arr, F["band"][1], 0),
                            color=F["colors"][key], alpha=F["band_alpha"], lw=0)
    if F["show_init_shade"]:
        ax.axvspan(0.5, N_INIT + 0.5, color=F["init_shade_color"], zorder=0, lw=0)
    if F["log_y"]:
        ax.set_yscale("log")
    ax.set_xlim(*F["xlim"])
    if F["ylim"]:
        ax.set_ylim(*F["ylim"])
    ax.set_xlabel(F["xlabel"])
    ax.set_ylabel(F["ylabel_norm"] if NORMALISE else F["ylabel"])
    if NORMALISE and F["show_typical_line"]:
        ax.axhline(1.0, color="0.5", ls="-.", lw=1, zorder=0)
        ax.text(F["typical_label_x"], 1.0, F["typical_label"], transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=F["legend_size"] - 1, color="0.4")
    title = F["title"] if F["title"] is not None else f"{case['run']} — {COST_NAME[case['run']]}"
    if title:
        ax.set_title(title)
    if F["grid"]:
        ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc=F["legend_loc"], ncol=F["legend_ncol"], frameon=F["legend_frame"])
    if letter:
        ax.text(*F["panel_letter_xy"], letter, transform=ax.transAxes, fontsize=F["panel_letter_size"],
                fontweight="bold", ha="left", va="top")
    return ax


def paper_figure(run, fig=None, save=True, name=None):
    F = {**FIG, **(fig or {})}
    if fig and "labels" in fig: F["labels"] = {**FIG["labels"], **fig["labels"]}
    if fig and "colors" in fig: F["colors"] = {**FIG["colors"], **fig["colors"]}
    _apply_rc(F)
    case = _load_case(run)
    f, ax = plt.subplots(figsize=F["figsize"])
    draw_panel(ax, case, F)
    f.tight_layout()
    if save:
        _save(f, name or f"{run}_bo_vs_sampling", F)
    return f, case


def paper_figure_2panels(fig=None, save=True, runs=("S28-M01", "S24-M01"), letters=("(a)", "(b)")):
    F = {**FIG, **(fig or {})}
    if fig and "labels" in fig: F["labels"] = {**FIG["labels"], **fig["labels"]}
    if fig and "colors" in fig: F["colors"] = {**FIG["colors"], **fig["colors"]}
    _apply_rc(F)
    w, h = F["figsize"]
    f, axes = plt.subplots(1, len(runs), figsize=(w * len(runs), h))
    cases = []
    for ax, run, letter in zip(np.atleast_1d(axes), runs, letters):
        case = _load_case(run); cases.append(case)
        draw_panel(ax, case, F, letter=letter)
    f.tight_layout()
    if save:
        _save(f, "bo_vs_sampling_2panels", F)
    return f, cases


def _save(f, stem, F):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in F["formats"]:
        f.savefig(OUT_DIR / f"{stem}.{ext}", dpi=F["dpi"], bbox_inches="tight", facecolor="white")
    print("saved", OUT_DIR / f"{stem}.[{','.join(F['formats'])}]")


# =============================================================================
#  métriques
# =============================================================================
def _boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def _at(case, key, N, raw=False):
    """valeurs (n,) à N expériences ; LHS = plan de taille N.
    raw=True force le regret brut même en mode NORMALISE."""
    arr = case["curves"][key] if raw else _curves(case)[key]
    per_n = key.startswith("lhs") and not PAIRED
    return arr[:, list(case["lhs_axis"]).index(N)] if per_n else arr[:, N - 1]


def _top_at(case, which, key, N):
    arr = case["top"][which][key]
    per_n = key.startswith("lhs") and not PAIRED
    return arr[:, list(case["lhs_axis"]).index(N)] if per_n else arr[:, N - 1]


def _n_for_frac(case, key, which="top1", frac=0.5):
    """plus petit N tel que P(best ≤ seuil) ≥ frac. Séquentiel (random, BO) ou taille de plan (LHS)."""
    axis = case["lhs_axis"] if (key.startswith("lhs") and not PAIRED) else case["x_axis"]
    p = case["top"][which][key].mean(0)
    idx = np.flatnonzero(p >= frac)
    return int(axis[idx[0]]) if len(idx) else "> 30"


def _budget_equivalent(case, ref_key, N_ref, key="bo"):
    """plus petit N où la médiane de `key` passe sous la médiane de `ref_key` à N_ref."""
    target = np.median(_at(case, ref_key, N_ref))
    med = np.median(_curves(case)[key], 0)
    idx = np.flatnonzero(med <= target)
    return int(case["x_axis"][idx[0]]) if len(idx) else "> 30"


def paper_metrics(run, Ns=(10, 15, 20, 30), save=True):
    """DataFrame long (metric, strategy, N, value) + tableau large ; imprime un résumé."""
    case = _load_case(run)
    keys = ("random", "lhs_simplex", "lhs_projected", "bo")
    rows = []
    for N in Ns:
        for k in keys:
            v = _at(case, k, N); lo, hi = _boot_ci(v)
            vr = _at(case, k, N, raw=True)
            rows += [dict(metric="regret_median", N=N, strategy=k, value=np.median(v)),
                     dict(metric="regret_ci_lo", N=N, strategy=k, value=lo),
                     dict(metric="regret_ci_hi", N=N, strategy=k, value=hi),
                     dict(metric="regret_p25", N=N, strategy=k, value=np.percentile(v, 25)),
                     dict(metric="regret_p75", N=N, strategy=k, value=np.percentile(v, 75)),
                     dict(metric="regret_median_raw_cost_units", N=N, strategy=k, value=np.median(vr)),
                     dict(metric="P_top1pct", N=N, strategy=k, value=_top_at(case, "top1", k, N).mean()),
                     dict(metric="P_top0.1pct", N=N, strategy=k, value=_top_at(case, "top01", k, N).mean())]
            if k != "bo":
                bo = _at(case, "bo", N)
                rows += [dict(metric="ratio_bo_over_strategy", N=N, strategy=k, value=np.median(bo) / np.median(v)),
                         dict(metric="mannwhitney_p_bo_less", N=N, strategy=k,
                              value=mannwhitneyu(bo, v, alternative="less").pvalue),
                         dict(metric="P_bo_run_beats_strategy_median", N=N, strategy=k, value=(bo < np.median(v)).mean())]
    for k in keys:
        rows += [dict(metric="N_for_50pct_runs_in_top1pct", N=np.nan, strategy=k, value=_n_for_frac(case, k, "top1", 0.5)),
                 dict(metric="N_for_80pct_runs_in_top1pct", N=np.nan, strategy=k, value=_n_for_frac(case, k, "top1", 0.8)),
                 dict(metric="N_for_50pct_runs_in_top0.1pct", N=np.nan, strategy=k, value=_n_for_frac(case, k, "top01", 0.5))]
    for k in ("random", "lhs_simplex", "lhs_projected"):
        rows.append(dict(metric="bo_budget_to_match_strategy_at_30", N=30, strategy=k, value=_budget_equivalent(case, k, 30)))
        rows.append(dict(metric="bo_budget_to_match_strategy_at_15", N=15, strategy=k, value=_budget_equivalent(case, k, 15)))
    # cohérence par cible : nb de cibles (sur 9) où la médiane BO < médiane stratégie à N=30
    for k in ("random", "lhs_simplex", "lhs_projected"):
        col = -1 if (k.startswith("lhs") and not PAIRED) else 29
        wins = sum(np.median(d["bo"][:, 29]) < np.median(d[k][:, col]) for d in case["per_target"].values())
        rows.append(dict(metric="targets_where_bo_median_better_at_30", N=30, strategy=k, value=f"{wins}/{len(case['per_target'])}"))
    # BO : expériences perdues en plats, part d'init
    inf = np.concatenate([d["bo_infeasible"] for d in case["per_target"].values()])
    rows.append(dict(metric="bo_fraction_infeasible_evals_after_init", N=np.nan, strategy="bo", value=inf[:, N_INIT:].mean()))
    rows.append(dict(metric="landscape_feasible_fraction", N=np.nan, strategy="-", value=np.mean([d["feasible_frac"] for d in case["per_target"].values()])))
    rows.append(dict(metric="normalised", N=np.nan, strategy="-", value=NORMALISE))
    rows.append(dict(metric="normalisation_scale_median_over_targets", N=np.nan, strategy="-",
                     value=float(np.median([d["scale"] for d in case["per_target"].values()]))))
    rows.append(dict(metric="n_runs_bo", N=np.nan, strategy="bo", value=len(case["curves"]["bo"])))
    rows.append(dict(metric="n_runs_baselines", N=np.nan, strategy="-", value=len(case["curves"]["random"])))
    df = pd.DataFrame(rows)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / f"{run}_metrics.csv", index=False)
    return df, case


def print_summary(run):
    df, case = paper_metrics(run, save=True)
    pd.set_option("display.width", 200)
    print(f"\n================ {run} — BO = {case['bo_variant']} — {COST_NAME[run]} ================")
    wide = df[df.metric.isin(["regret_median", "P_top1pct", "P_top0.1pct", "ratio_bo_over_strategy", "mannwhitney_p_bo_less"])]
    print(wide.pivot_table(index=["metric", "strategy"], columns="N", values="value").round(4).to_string())
    other = df[~df.metric.isin(wide.metric.unique()) & ~df.metric.str.startswith("regret_")]
    print(other[["metric", "strategy", "N", "value"]].to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=list(BO_OF_RUN))
    ap.add_argument("--recompute", action="store_true", help="recalcule les baselines (cache paper/)")
    a = ap.parse_args()
    if a.recompute:
        for run in a.run:
            _load_case(run, recompute=True)
    for run in a.run:
        paper_figure(run); plt.close("all")
        print_summary(run)
    if len(a.run) == 2:
        paper_figure_2panels(runs=tuple(a.run)); plt.close("all")


if __name__ == "__main__":
    main()
