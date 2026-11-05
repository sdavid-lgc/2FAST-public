"""Command line: run a simulated stop-flow BO campaign (no hardware).

    python -m toofast.simulation --target-lambda 550 --iterations 6 --out sim_runs/demo
"""
import argparse

import numpy as np

from toofast.simulation.headless import run_campaign


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m toofast.simulation", description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="sim_runs/demo", help="run folder (created)")
    ap.add_argument("--target-lambda", type=float, default=550.0, help="target λmax (nm)")
    ap.add_argument("--iterations", type=int, default=6, help="BO iterations after the initial design")
    ap.add_argument("--n-init", type=int, default=6, help="experiments of the initial design")
    ap.add_argument("--init", default="latinhypercube", choices=["latinhypercube", "projected_lhs", "random"],
                    help="initial design: Latin hypercube on the simplex (default), projected LHS (as in the paper), random")
    ap.add_argument("--acq", default="EI", choices=["EI", "LCB", "TS"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.0, help="real-time factor of the virtual clock (0 = fastest)")
    a = ap.parse_args(argv)
    r = run_campaign(a.out, target_lambda_max=a.target_lambda, iterations=a.iterations, n_init=a.n_init,
                     acq_func=a.acq, seed=a.seed, speed=a.speed, init_option=a.init)
    b = r.best_index
    print(f"\n{len(r.costs)} experiments, {r.virtual_seconds / 60:.1f} virtual minutes -> {r.experiment_path}")
    print(f"best #{b}: cost {r.costs[b]:.4g}, lambda_max {r.lambda_max[b]:.1f} nm, flows {np.round(r.flows[b], 2)}")


if __name__ == "__main__":
    main()
