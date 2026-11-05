# [vendored] verbatim copy of paperAnalysis/bo_sim/oracle.py (md5 3e9b5df655bcc2b1ce0316c8289f3d7b), tools/vendor_bo_sim.py
"""
bo_sim/oracle.py — oracle spectral déterministe (modèle nb23 de 2FAST_ML).

    Q (5 débits, somme = 14) -> scores PCA(6) -> spectre max-normalisé 310-1000 nm

Reconstruit à l'identique le modèle de `23_spectral_model.ipynb`
(`spectral_pipeline.py` : normalise 'max', PCA(6), features 'raw',
un GP-Matérn52 isotrope par composante, tiers A/B/C, plats exclus).

La fonction de coût est appliquée *par-dessus* ce spectre — voir `cost.py`.
L'oracle ne prédit que la forme (spectre normalisé) : suffisant pour tout
coût invariant d'échelle (cosinus brut, cosinus PCA sur base normalisée).
"""
import os
import pickle
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import config as C


@contextmanager
def _in_ml_dir():
    """spectral_pipeline importe eda_rg024 qui crée `figures_eda/` dans le cwd."""
    old = os.getcwd()
    os.chdir(C.ML_DIR)
    if str(C.ML_DIR) not in sys.path:
        sys.path.insert(0, str(C.ML_DIR))
    try:
        yield
    finally:
        os.chdir(old)


def _import_sp():
    from toofast_analysis.ml import spectral_pipeline as sp   # [vendored] same code, no chdir into 2FAST_ML
    return sp


GATES = ("classifier+rule", "classifier", "rule", "none")


class SpectralOracle:
    """Modèle nb23 réentraîné sur toutes les données sélectionnées.

    predict_spectra(Q) : (n,5) débits -> (n, n_wl) spectres normalisés (pic=1)
    predict_scores(Q)  : (n,5) -> (n, n_pc) scores PCA
    predict_std(Q)     : (n,5) -> (n, n_pc) écart-type a posteriori par PC
    sample_spectra(Q, seed) : un tirage a posteriori (option "mondes" pour plus tard)
    is_feasible(Q)     : (n,) bool — gate de faisabilité (spectre plat = pas de particules)

    Gate (`gate`) :
      'classifier+rule' (défaut) : plat si le RF le dit OU si Ag/NaBH4 < rule_min —
                     le RF apprend les plats du catalogue (dont ceux à faible acide
                     ascorbique), la règle porte le prior physique des ghosts
                     (pas de particules sans Ag ni NaBH4), que le catalogue contredit
                     un peu (quelques non-plats à Q5 ≈ 0.1).
      'classifier' : RandomForest sur les 5 débits, entraîné sur `is_flat` des 274 pts
                     du catalogue (64 plats ; CV groupée par session : 45/64 plats
                     attrapés, 10/210 fausses alertes). p(plat) > 0.5 -> infaisable.
      'rule'       : Q_AgNO3 < rule_min ou Q_NaBH4 < rule_min (n'attrape que 28/64 plats).
      'none'       : tout est faisable.
    Le modèle spectral lui-même est toujours entraîné sans les plats (drop_flat).
    """

    def __init__(self, tiers=("A", "B", "C"), drop_flat=True, normalize="max",
                 n_pc=6, feature="raw", model_name="GP-Matern52",
                 exclude_sessions=(), gate="classifier+rule", rule_min=0.5):
        self.tiers, self.drop_flat = tuple(tiers), bool(drop_flat)
        self.normalize, self.n_pc = normalize, int(n_pc)
        self.feature, self.model_name = feature, model_name
        self.exclude_sessions = tuple(exclude_sessions)
        if gate not in GATES:
            raise ValueError(f"gate {gate!r} ∉ {GATES}")
        self.gate, self.rule_min = gate, float(rule_min)
        self.gate_clf = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, verbose=True):
        sp = _import_sp()
        sel = pd.read_csv(C.SPECTRAL_SELECTED)
        z = np.load(C.SPECTRA_MATRIX)
        S, wl = z["S"], z["wavelengths"]
        m = sel.tier.isin(self.tiers).values
        if self.exclude_sessions:
            m &= ~sel.session.isin(self.exclude_sessions).values
        # gate de faisabilité : tous les points (plats inclus) des tiers retenus
        self._fit_gate(sel[m][sp.FEATURES].values.astype(float), sel[m].is_flat.values,
                       verbose)
        if self.drop_flat:
            m &= (~sel.is_flat).values
        sel, S = sel[m].reset_index(drop=True), S[m]

        Q = sel[sp.FEATURES].values.astype(float)
        Sn, _ = sp.normalise(S, self.normalize)
        mu, comps, evr = sp.fit_pca(Sn, self.n_pc)
        scores = (Sn - mu) @ comps.T
        model = sp.SpectralModel(self.n_pc, sp.FEATURE_SETS[self.feature],
                                 sp.MODELS[self.model_name]).fit(Q, scores)

        self.wl, self.mu, self.comps, self.evr = wl, mu, comps, evr
        self.model = model
        self.train_Q, self.train_S, self.train_scores = Q, Sn, scores
        self.train_sessions = sel.session.values
        self.n_train = len(sel)
        self._fitted = True
        if verbose:
            print(f"[oracle] {self.n_train} pts / {sel.session.nunique()} sessions | "
                  f"normalise={self.normalize} PCA({self.n_pc})={100*evr.sum():.1f}% | "
                  f"{self.feature}+{self.model_name}")
            for k, gp in enumerate(model.models):
                print(f"   PC{k+1}: {gp[-1].kernel_}")
        return self

    def _fit_gate(self, Q, is_flat, verbose):
        if "classifier" not in self.gate:
            return
        from sklearn.ensemble import RandomForestClassifier
        self.gate_clf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0,
                                               n_jobs=1).fit(Q, is_flat.astype(int))
        if verbose:
            pf = self.gate_clf.predict_proba(Q)[:, 1] > 0.5
            print(f"[gate] RF sur {len(Q)} pts ({int(is_flat.sum())} plats) — resubstitution : "
                  f"{int((pf & is_flat).sum())}/{int(is_flat.sum())} plats, "
                  f"{int((pf & ~is_flat).sum())} fausses alertes")

    def is_feasible(self, Q):
        """(n,) bool. Ordre des colonnes : Q1 AgNO3, Q2 PVP, Q3 AcOH, Q4 AscA, Q5 NaBH4."""
        Q = self._as2d(Q)
        ok = np.ones(len(Q), bool)
        if "rule" in self.gate:
            ok &= (Q[:, 0] >= self.rule_min) & (Q[:, 4] >= self.rule_min)
        if "classifier" in self.gate:
            ok &= self.gate_clf.predict_proba(Q)[:, 1] <= 0.5
        return ok

    # -------------------------------------------------------------- predict
    def _check(self):
        if not self._fitted:
            raise RuntimeError("oracle non entraîné : appeler fit() ou load()")

    @staticmethod
    def _as2d(Q):
        Q = np.asarray(Q, float)
        return Q.reshape(1, -1) if Q.ndim == 1 else Q

    def predict_scores(self, Q):
        self._check()
        return self.model.predict(self._as2d(Q))

    def predict_std(self, Q):
        self._check()
        return self.model.predict_std(self._as2d(Q))

    def predict_spectra(self, Q):
        return self.mu + self.predict_scores(Q) @ self.comps

    def sample_spectra(self, Q, seed=0):
        """Un tirage a posteriori indépendant par composante (non corrélé entre
        points -> à n'utiliser que pour des évaluations ponctuelles)."""
        rng = np.random.default_rng(seed)
        mu, sd = self.predict_scores(Q), self.predict_std(Q)
        return self.mu + (mu + rng.normal(size=mu.shape) * sd) @ self.comps

    # ------------------------------------------------------------- persist
    def cache_path(self):
        return C.RESULTS_DIR / f"oracle_nb23_gate-{self.gate}.pkl"

    def save(self, path=None):
        self._check()
        path = os.fspath(path or self.cache_path())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path):
        _import_sp()   # les pipelines picklés référencent spectral_pipeline
        with open(os.fspath(path), "rb") as f:
            return pickle.load(f)


class WorldOracle:
    """Un "monde" = un tirage a posteriori COHÉRENT de la fonction spectrale.

    Pour chaque composante PC, le GP sklearn (C·Matérn52 + White, normalize_y, sur
    X standardisé) est échantillonné par conditionnement "pathwise" (Matheron) :
        f(x) = f_prior(x) + k(x,X) [K + σ²I]⁻¹ (y − f_prior(X) − ε)
    avec f_prior = random Fourier features de la densité spectrale de Matérn 5/2
    (ω ~ t₅(0, I/ℓ²)), D features. f est une fonction continue de x, évaluable
    partout (ce que la BO demande), dont la loi est celle de la postérieure du GP.
    Même interface que SpectralOracle (predict_scores / predict_spectra /
    is_feasible / wl / mu / comps). world=None ⇔ moyenne a posteriori = SpectralOracle.
    """

    def __init__(self, base, seed, n_features=2000):
        self.base, self.seed, self.D = base, int(seed), int(n_features)
        for a in ("wl", "mu", "comps", "evr", "n_pc", "gate", "n_train", "tiers"):
            setattr(self, a, getattr(base, a))
        rng = np.random.default_rng(10_000 + self.seed)
        self.parts = [self._draw(pipe, rng) for pipe in base.model.models]

    def _draw(self, pipe, rng):
        sc, gp = pipe[0], pipe[-1]
        X = gp.X_train_                                  # (n, d) standardisé
        n, d = X.shape
        k = gp.kernel_                                   # C**2 * Matern(ls) + White(noise)
        amp2 = float(k.k1.k1.constant_value); ls = float(k.k1.k2.length_scale)
        nu = float(k.k1.k2.nu)
        # RFF Matérn-ν : ω = z/ℓ · sqrt(2ν/u), z~N(0,I_d), u~χ²(2ν)
        z = rng.standard_normal((self.D, d)); u = rng.chisquare(2 * nu, self.D)
        omega = z / ls * np.sqrt(2 * nu / u)[:, None]
        b = rng.uniform(0, 2 * np.pi, self.D)
        w = rng.standard_normal(self.D)
        phi = lambda Xs: np.sqrt(2 * amp2 / self.D) * np.cos(Xs @ omega.T + b)
        f_prior_X = phi(X) @ w
        eps = rng.standard_normal(n) * np.sqrt(float(k.k2.noise_level))
        # v = (K + σ²I + αI)⁻¹ (y − f_prior(X) − ε), via le Cholesky de sklearn (L_ de K(X,X)+αI)
        resid = gp.y_train_ - f_prior_X - eps
        from scipy.linalg import cho_solve
        v = cho_solve((gp.L_, True), resid)
        return dict(scaler=sc, X=X, omega=omega, b=b, w=w, amp2=amp2, ls=ls, nu=nu, v=v,
                    y_mean=float(np.ravel(gp._y_train_mean)[0]), y_std=float(np.ravel(gp._y_train_std)[0]))

    @staticmethod
    def _matern52(r):
        s = np.sqrt(5.0) * r
        return (1 + s + s * s / 3.0) * np.exp(-s)

    def _eval(self, p, Q):
        Xs = p["scaler"].transform(Q)
        f_prior = np.sqrt(2 * p["amp2"] / self.D) * np.cos(Xs @ p["omega"].T + p["b"]) @ p["w"]
        r = np.sqrt(((Xs[:, None, :] - p["X"][None, :, :]) ** 2).sum(-1)) / p["ls"]
        kxX = p["amp2"] * self._matern52(r)
        return (f_prior + kxX @ p["v"]) * p["y_std"] + p["y_mean"]

    @staticmethod
    def _as2d(Q):
        Q = np.asarray(Q, float)
        return Q.reshape(1, -1) if Q.ndim == 1 else Q

    def predict_scores(self, Q, chunk=5000):
        Q = self._as2d(Q)
        out = np.empty((len(Q), self.n_pc))
        for i in range(0, len(Q), chunk):
            out[i:i + chunk] = np.column_stack([self._eval(p, Q[i:i + chunk]) for p in self.parts])
        return out

    def predict_spectra(self, Q):
        return self.mu + self.predict_scores(Q) @ self.comps

    def is_feasible(self, Q):
        return self.base.is_feasible(Q)

    def cache_path(self):
        return self.base.cache_path()


def score_noise_sd(orc):
    """(n_pc,) écart-type du bruit blanc estimé par le GP de chaque PC, en unités de
    score (WhiteKernel × std de normalisation de y). Il agrège répétabilité et
    écart au modèle ; `bench.make_measure(noise=a)` en utilise la fraction a."""
    base = getattr(orc, "base", orc)
    return np.array([np.sqrt(float(p[-1].kernel_.k2.noise_level)) * float(np.ravel(p[-1]._y_train_std)[0])
                     for p in base.model.models])


def get_oracle(gate="classifier+rule", refit=False, world=None, **kw):
    """Charge l'oracle en cache (un cache par gate), ou l'entraîne (≈ 1 min).
    world=None -> moyenne a posteriori ; world=k (int) -> tirage a posteriori n° k."""
    orc = SpectralOracle(gate=gate, **kw)
    if not refit and os.path.exists(orc.cache_path()):
        orc = SpectralOracle.load(orc.cache_path())
    else:
        orc.fit()
        orc.save()
    return orc if world is None else WorldOracle(orc, world)
