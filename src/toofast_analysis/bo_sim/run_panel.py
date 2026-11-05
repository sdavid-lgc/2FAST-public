# [vendored] verbatim copy of paperAnalysis/bo_sim/run_panel.py (md5 7548a80fc28e2a3a2be0f0cf0dc662fe), tools/vendor_bo_sim.py
"""
Grille "papier" : cibles × mondes × variantes × seeds, hors notebook.

    py -3.11 -m bo_sim.run_panel --run S28-M01 --worlds mean 0 1 --seeds 10 --n-jobs 8
    py -3.11 -m bo_sim.run_panel --run S24-M01 --targets real t2 t4 --worlds mean
    py -3.11 -m bo_sim.run_panel --run S28-M01 --worlds mean --variants robust --noise 0.6

Cibles : `bo_sim_results/targets/targets.json` (targets.py ; 'real' + t0..t7).
Mondes : 'mean' (moyenne a posteriori) ou un entier (tirage a posteriori n° k).
Cache : bo_sim_results/<base_tag>/<cible>/<monde>/<variante>_seed<k>.npz
Coût par run ≈ 1 min seul, 4 min en parallèle ×8 (la DE domine, pas les restarts GP).
"""
import argparse
import warnings

from . import bench, baselines, config as C, targets as T
from .oracle import GATES
from .toofast_sim import VARIANTS


def parse_world(s):
    return None if s == "mean" else int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="S28-M01")
    ap.add_argument("--base-tag", default=None, help="défaut : panel_<run>")
    ap.add_argument("--targets", nargs="+", default=None, help="ids ; défaut : toutes")
    ap.add_argument("--worlds", nargs="+", default=["mean", "0", "1"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--gate", default="classifier+rule", choices=list(GATES))
    ap.add_argument("--prior", default=C.DEFAULT_PRIOR, choices=list(C.PRIOR_KERNELS))
    ap.add_argument("--n-restarts", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="mesures bruitées : fraction du bruit blanc GP de l'oracle (0.6 = répétabilité mesurée)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    run_dir = C.BO_RUNS_DIR / a.run
    base_tag = a.base_tag or f"panel_{a.run.lower().split('_')[0]}" + (f"_noise{a.noise:g}" if a.noise else "")
    panel = T.load_target_panel()
    if a.targets:
        panel = [t for t in panel if t["id"] in a.targets]
    worlds = [parse_world(w) for w in a.worlds]
    bounds = baselines.run_bounds(run_dir, 0.0)
    n = len(panel) * len(worlds) * len(a.variants) * a.seeds
    print(f"run={a.run} tag={base_tag} | {len(panel)} cibles × {len(worlds)} mondes × "
          f"{len(a.variants)} variantes × {a.seeds} seeds = {n} runs | gate={a.gate} | noise={a.noise:g}")
    bench.run_panel(base_tag, panel, worlds, a.variants, range(a.seeds), budget=a.budget,
                    bounds=bounds, run_dir=run_dir, prior_name=a.prior,
                    n_restarts=a.n_restarts, gate=a.gate, n_jobs=a.n_jobs, noise=a.noise)


if __name__ == "__main__":
    main()
