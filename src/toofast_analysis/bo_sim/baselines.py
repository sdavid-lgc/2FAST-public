# [vendored] verbatim copy of paperAnalysis/bo_sim/baselines.py (md5 ac34f80e80ae7de0261ef646040c2d8a), tools/vendor_bo_sim.py
"""
bo_sim/baselines.py — stratégies sans modèle, évaluées sur l'oracle.

  random_toofast  : `PumpSystem.generate_random_flows` (uniforme séquentiel, ordre
                    des pompes mélangé) — ce que fait `init_option='random'` de toofast
  lhs_toofast     : `PumpSystem.generate_lhs_flows_sets` (LHS pyDOE dans le cube
                    puis renormalisation à Qtot) — l'init de la BO réelle
  random_dirichlet: uniforme *sur le simplexe* (Dirichlet(1,…,1)·Qtot + rejet
                    sur les bornes) — la référence "vraiment uniforme"
  cube_projected  : uniforme dans le cube [lo,hi]^5 puis renormalisé à Qtot — le biais
                    du "LHS" toofast sans la stratification (contrôle : mêmes courbes)
  lhs_simplex     : VRAI LHS sur le simplexe — LHS 4D (scipy qmc) transformé par
                    stick-breaking (inverse-CDF séquentiel Beta(1,k)) = uniforme
                    stratifié sur le simplexe ; rejet hors bornes.

Constat (9 cibles, oracle moyen) : lhs_toofast ≡ cube_projected (l'avantage sur
l'uniforme vient de la reprojection radiale, qui concentre les recettes au centre
du simplexe — 74 % faisables vs 57 % — pas de la stratification) ; lhs_simplex ≡
random_dirichlet ; "un plan par N" ≡ préfixe d'un plan de 30.

Random se cumule (préfixe de k tirages = random(k)). LHS : on tire une LHS(N)
distincte pour chaque budget N (voir `lhs_curve`) ; à N = n_init le seed est
celui de la BO, donc LHS_toofast(6) = l'init de la BO, point par point.
"""
import numpy as np
from scipy.stats import qmc

from . import config as C
from .toofast_sim import make_pump_system, load_run_config

DEFAULT_BOUNDS = np.array([[0.5, 10], [0, 10], [0, 10], [0, 10], [0.5, 10]], float)


def run_bounds(run_dir=C.RUN_S28, ag_nabh4_min=0.5):
    """Bornes du _config.json, avec plancher sur AgNO3 (pompe 1) et NaBH4 (pompe 5)."""
    cfg = load_run_config(run_dir)
    b = np.array(cfg["bounds"], float)
    reagents = [str(r) for r in cfg["reagents"]]
    for key in ("AgNO3", "NaBH4"):
        i = reagents.index(key)
        b[i, 0] = max(b[i, 0], ag_nabh4_min)
    return b


class Sampler:
    def __init__(self, bounds=DEFAULT_BOUNDS, total_flow=14.0):
        self.bounds = np.asarray(bounds, float)
        self.total_flow = float(total_flow)
        self.n = len(self.bounds)
        self.pumps = make_pump_system(self.n, self.total_flow)
        self._fixed = np.zeros(self.n, bool)
        self._fixed_flows = np.zeros(self.n)

    def random_toofast(self, n, seed):
        np.random.seed(seed)
        return self.pumps.generate_random_flows_sets(n, self.bounds, self.total_flow,
                                                     self._fixed, self._fixed_flows)

    def lhs_toofast(self, n, seed):
        np.random.seed(seed)
        return self.pumps.generate_lhs_flows_sets(n, self.bounds, self.total_flow,
                                                  self._fixed, self._fixed_flows)

    def random_dirichlet(self, n, seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n:
            cand = rng.dirichlet(np.ones(self.n), size=4 * n) * self.total_flow
            ok = np.all((cand >= self.bounds[:, 0]) & (cand <= self.bounds[:, 1]), axis=1)
            out.extend(cand[ok])
        return np.array(out[:n])

    def cube_projected(self, n, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (n, self.n))
        return x / x.sum(1, keepdims=True) * self.total_flow

    @staticmethod
    def _stick(u):
        """u ∈ [0,1]^d -> point uniforme sur le d-simplexe (Dirichlet(1,…,1))."""
        n, d = u.shape; x = np.zeros((n, d + 1)); rem = np.ones(n)
        for j in range(d):
            xj = rem * (1 - (1 - u[:, j]) ** (1.0 / (d - j)))
            x[:, j] = xj; rem = rem - xj
        x[:, d] = rem
        return x

    def lhs_simplex(self, n, seed):
        for k in range(100):
            u = qmc.LatinHypercube(self.n - 1, seed=int(seed) + 100_003 * k).random(n)
            x = self._stick(u) * self.total_flow
            if np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])):
                return x
        return x   # (bornes 0-10 : jamais atteint en pratique)

    def get(self, name):
        return getattr(self, name)


def random_curves(cost_fn, sampler, budget, seeds, method="random_toofast"):
    """(n_seeds, budget) best-so-far d'un échantillonnage cumulatif."""
    gen = sampler.get(method)
    return np.array([np.minimum.accumulate(cost_fn(gen(budget, s))) for s in seeds])


def lhs_curve(cost_fn, sampler, budgets, seeds, method="lhs_toofast", n_init=6):
    """(n_seeds, len(budgets)) meilleur coût d'une LHS(N) indépendante par N.
    À N == n_init le seed est `s` lui-même : pour `lhs_toofast` c'est exactement
    l'init de la BO de seed s (np.random.seed(s) + generate_lhs_flows_sets)."""
    gen = sampler.get(method)
    out = np.empty((len(seeds), len(budgets)))
    for i, s in enumerate(seeds):
        for j, N in enumerate(budgets):
            out[i, j] = cost_fn(gen(N, s if N == n_init else 10_000 * s + N)).min()
    return out


def shared_init_curves(cost_fn, sampler, budget, seeds, method="random_dirichlet",
                       n_init=6, init_method="lhs_toofast"):
    """Courbes APPARIÉES : les `n_init` premières expériences sont le plan d'init
    de la BO de même seed (générateur de l'instrument, `np.random.seed(s)`), puis
    chaque stratégie continue avec son propre échantillonneur jusqu'à `budget`.

    (n_seeds, budget) best-so-far, monotone. Toutes les stratégies — BO comprise —
    partent donc EXACTEMENT du même point à N = n_init ; ce qui les sépare ensuite
    est la règle de choix des points suivants, et elle seule.
    """
    gen, gen_init = sampler.get(method), sampler.get(init_method)
    out = []
    for s in seeds:
        Q = np.vstack([gen_init(n_init, s), gen(budget - n_init, 10_000 * s + budget)])
        out.append(np.minimum.accumulate(cost_fn(Q)))
    return np.array(out)


def oracle_reference(cost_fn, sampler, n=200_000, seed=12345):
    """Minimum (et distribution) du coût sur un gros échantillon uniforme du domaine."""
    Q = sampler.random_dirichlet(n, seed)
    c = cost_fn(Q)
    i = np.argmin(c)
    return dict(c_min=float(c[i]), q_min=Q[i], costs=c, Q=Q)
