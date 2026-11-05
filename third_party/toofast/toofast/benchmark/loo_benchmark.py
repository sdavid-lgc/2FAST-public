"""
Leave-one-out (LOO) benchmark: does BO beat random sampling?

Fully offline and honest — it uses only *real* measured spectra (no surrogate
fantasy, no sim gap). For each held-out sample (the target), the candidate pool
is every other sample; the cost of a candidate is the distance between its
spectrum and the target's. Pool-based BO fits a GP on flows -> cost over the
points already tried and picks the next pool point by Expected Improvement;
the random baseline picks pool points uniformly. Repeating over many seeds and
many targets, with a simple-regret curve, gives a powered answer to
"is BO better than random?" without running a single new experiment.

Cost functions are pluggable so the *same* search can be benchmarked under the
degenerate lambda_max objective and under a richer full-spectrum / PCA cost,
which is exactly the comparison motivating the spectral-objective switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from toofast.core.pca_model import SpectralPCA, fit_spectral_pca

logger = logging.getLogger(__name__)

# A cost function: (target_spectrum, candidate_spectrum, wavelengths) -> float
CostFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def l2_cost(min_wl: float = 315.0, max_wl: float = 750.0) -> CostFn:
    """Full-spectrum L2 distance over a wavelength window."""
    def _cost(target: np.ndarray, cand: np.ndarray, wl: np.ndarray) -> float:
        mask = (wl >= min_wl) & (wl <= max_wl)
        return float(np.sum((target[mask] - cand[mask]) ** 2))
    return _cost


def lambda_max_cost(min_wl: float = 315.0, max_wl: float = 750.0, window: int = 11) -> CostFn:
    """Squared distance between target and candidate lambda_max (degenerate)."""
    def _peak(s: np.ndarray, wl: np.ndarray) -> float:
        mask = (wl >= min_wl) & (wl <= max_wl)
        sm = savgol_filter(s[mask], window, 3)
        return float(wl[mask][int(np.argmax(sm))])
    def _cost(target: np.ndarray, cand: np.ndarray, wl: np.ndarray) -> float:
        return float((_peak(target, wl) - _peak(cand, wl)) ** 2)
    return _cost


def pca_cost(pca: SpectralPCA, variance_weighted: bool = False) -> CostFn:
    """Distance in frozen-PCA score space (the objective we want to evaluate).

    Both spectra are projected onto the same frozen basis; the cost is the
    (optionally variance-weighted) squared distance between score vectors. This
    is an L2 restricted to the dominant spectral modes — it keeps peak
    position + width + amplitude jointly while discarding pixel noise, which is
    precisely the candidate objective for switching the BO away from lambda_max.

    The basis is fitted offline (see fit_spectral_pca / SpectralPCA) and frozen;
    pass the same ``pca`` used everywhere else so the cost is well-defined.
    """
    if variance_weighted:
        ev = np.asarray(pca.explained_variance, dtype=float)
        total = float(ev.sum())
        weights = ev / total if total > 1e-12 else np.ones_like(ev)
    else:
        weights = np.ones(pca.n_components, dtype=float)

    def _cost(target: np.ndarray, cand: np.ndarray, wl: np.ndarray) -> float:
        st = pca.transform(wl, target)
        sc = pca.transform(wl, cand)
        d = st - sc
        return float(np.sum(weights * d * d))

    return _cost


@dataclass
class BenchmarkResult:
    """Best-cost-so-far curves for BO and random, plus regret."""

    bo_best: np.ndarray      # (n_targets, n_seeds, n_evals)
    random_best: np.ndarray  # (n_targets, n_seeds, n_evals)
    pool_optimum: np.ndarray  # (n_targets,) best achievable cost in the pool

    def regret(self, which: str = "bo") -> np.ndarray:
        """Simple regret = best_so_far - pool_optimum, averaged over targets/seeds."""
        arr = self.bo_best if which == "bo" else self.random_best
        reg = arr - self.pool_optimum[:, None, None]
        return reg.mean(axis=(0, 1))

    def summary(self) -> str:
        bo = self.regret("bo")
        rnd = self.regret("random")
        lines = ["eval  BO_regret  RND_regret  improvement"]
        for i in range(len(bo)):
            imp = (rnd[i] - bo[i]) / rnd[i] * 100 if rnd[i] > 1e-12 else 0.0
            lines.append(f"{i+1:4d}  {bo[i]:9.4g}  {rnd[i]:10.4g}  {imp:9.1f}%")
        return "\n".join(lines)


def _build_gp(n_dim: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(n_dim), length_scale_bounds=(0.05, 50.0), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, alpha=1e-8
    )


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    """EI for minimization (improvement below the current best)."""
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def _pool_bo_run(
    flows_pool: np.ndarray,
    costs_pool: np.ndarray,
    n_init: int,
    n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One pool-based BO run -> best-cost-so-far after each evaluation."""
    n = len(costs_pool)
    fmean, fstd = flows_pool.mean(0), flows_pool.std(0)
    fstd = np.where(fstd > 1e-9, fstd, 1.0)
    Xn = (flows_pool - fmean) / fstd

    evaluated = list(rng.choice(n, size=n_init, replace=False))
    best_curve = [float(np.min(costs_pool[evaluated[: i + 1]])) for i in range(n_init)]

    for _ in range(n_iter):
        remaining = [i for i in range(n) if i not in evaluated]
        if not remaining:
            break
        gp = _build_gp(Xn.shape[1])
        gp.fit(Xn[evaluated], costs_pool[evaluated])
        mu, sigma = gp.predict(Xn[remaining], return_std=True)
        ei = _expected_improvement(mu, sigma, float(np.min(costs_pool[evaluated])))
        nxt = remaining[int(np.argmax(ei))]
        evaluated.append(nxt)
        best_curve.append(float(np.min(costs_pool[evaluated])))
    return np.array(best_curve)


def _pool_random_run(
    costs_pool: np.ndarray, n_init: int, n_iter: int, rng: np.random.Generator
) -> np.ndarray:
    """One random-sampling run -> best-cost-so-far after each evaluation."""
    n = len(costs_pool)
    order = list(rng.permutation(n))
    picks = order[: n_init + n_iter]
    return np.array([float(np.min(costs_pool[picks[: i + 1]])) for i in range(len(picks))])


def run_loo_benchmark(
    flows: np.ndarray,
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    cost_fn: Optional[CostFn] = None,
    n_targets: Optional[int] = None,
    n_seeds: int = 20,
    n_init: int = 4,
    n_iter: int = 14,
    seed: int = 0,
) -> BenchmarkResult:
    """Run the pool-based LOO benchmark of BO vs random.

    Parameters
    ----------
    flows, spectra, wavelengths : arrays
        Real dataset: flows (N, P), spectra (N, M), wavelengths (M,).
    cost_fn : callable, optional
        (target, candidate, wl) -> cost. Defaults to full-spectrum L2.
    n_targets : int, optional
        Number of held-out targets to average over (default: all samples).
    n_seeds : int
        Repetitions per target (different init/random draws).
    n_init, n_iter : int
        Initial random evaluations and subsequent BO/random iterations.
    """
    flows = np.atleast_2d(np.asarray(flows, dtype=float))
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    cost_fn = cost_fn or l2_cost()
    rng_master = np.random.default_rng(seed)

    n = flows.shape[0]
    target_idx = np.arange(n)
    if n_targets is not None and n_targets < n:
        target_idx = rng_master.choice(n, size=n_targets, replace=False)

    n_evals = n_init + n_iter
    bo_best = np.full((len(target_idx), n_seeds, n_evals), np.nan)
    rnd_best = np.full((len(target_idx), n_seeds, n_evals), np.nan)
    pool_opt = np.zeros(len(target_idx))

    for ti, t in enumerate(target_idx):
        pool = np.array([i for i in range(n) if i != t])
        flows_pool = flows[pool]
        costs_pool = np.array([cost_fn(spectra[t], spectra[i], wavelengths) for i in pool])
        pool_opt[ti] = float(costs_pool.min())

        for s in range(n_seeds):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            bo = _pool_bo_run(flows_pool, costs_pool, n_init, n_iter, rng)
            rng2 = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            rnd = _pool_random_run(costs_pool, n_init, n_iter, rng2)
            bo_best[ti, s, : len(bo)] = bo
            rnd_best[ti, s, : len(rnd)] = rnd

    return BenchmarkResult(bo_best=bo_best, random_best=rnd_best, pool_optimum=pool_opt)


def _main() -> None:  # pragma: no cover - CLI
    import argparse

    from toofast.data.h5_dataset import load_h5_dataset

    parser = argparse.ArgumentParser(description="LOO benchmark of BO vs random on an H5 spectra set.")
    parser.add_argument("directory", help="Directory of .h5 samples")
    parser.add_argument("--cost", choices=["lambda_max", "pca", "l2"], default="pca")
    parser.add_argument("--n-components", type=int, default=3, help="PCA components for --cost pca")
    parser.add_argument("--pca-normalize", choices=["none", "area", "max"], default="none")
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=14)
    parser.add_argument("--min-wl", type=float, default=315.0)
    parser.add_argument("--max-wl", type=float, default=750.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ds = load_h5_dataset(args.directory)
    if args.cost == "lambda_max":
        cost = lambda_max_cost(args.min_wl, args.max_wl)
    elif args.cost == "l2":
        cost = l2_cost(args.min_wl, args.max_wl)
    else:
        # Frozen PCA basis fitted offline on the whole set (representation only).
        pca = fit_spectral_pca(
            ds.wavelengths, ds.spectra, n_components=args.n_components,
            normalize=args.pca_normalize, min_wavelength=args.min_wl, max_wavelength=args.max_wl,
        )
        cost = pca_cost(pca)
    result = run_loo_benchmark(
        ds.flows, ds.spectra, ds.wavelengths, cost_fn=cost,
        n_targets=args.n_targets, n_seeds=args.n_seeds,
        n_init=args.n_init, n_iter=args.n_iter,
    )
    print(result.summary())


if __name__ == "__main__":
    _main()
