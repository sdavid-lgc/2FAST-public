"""
Spectral forward-model surrogate: flows -> full spectrum.

Instead of regressing a single scalar (lambda_max cost), this models the whole
spectral response of the platform. To keep it tractable the spectrum is first
compressed with a frozen PCA basis (a handful of scores), and one GP is fitted
per score:

    flows --GP_k--> score_k        (k = 1..K)
    scores --PCA^-1--> predicted spectrum

Each costly experiment then informs the model with K outputs derived from the
*entire* curve, not one degenerate number — far more signal per point. Any cost
(lambda_max, full-spectrum match, FWHM, ...) can be evaluated on the predicted
spectrum afterwards, without new experiments.

The surrogate also powers "option 3" target generation
(:meth:`make_feasible_target`): take a candidate target, project it onto the
achievable manifold (argmin over flows of the score distance) so the benchmark
target is guaranteed feasible with a known optimum and a ~0 cost floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from toofast.core.pca_model import SpectralPCA, fit_spectral_pca

logger = logging.getLogger(__name__)


@dataclass
class FeasibleTarget:
    """A projected, achievable benchmark target."""

    spectrum: np.ndarray       # reconstructed target absorbance on the PCA grid
    wavelengths: np.ndarray
    optimal_flows: np.ndarray  # flows that achieve it (the known optimum)
    target_scores: np.ndarray  # projected scores
    residual: float            # ||achieved_scores - requested_scores|| (feasibility)


class SpectralSurrogate:
    """PCA + per-score GP forward model mapping flows to spectra."""

    def __init__(
        self,
        pca: SpectralPCA,
        gps: List[GaussianProcessRegressor],
        flow_mean: np.ndarray,
        flow_std: np.ndarray,
        total_flow: float,
    ):
        self.pca = pca
        self.gps = gps
        self.flow_mean = flow_mean
        self.flow_std = flow_std
        self.total_flow = float(total_flow)

    # -- fitting ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        flows: np.ndarray,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        n_components: int = 3,
        normalize: str = "none",
        min_wavelength: float = 315.0,
        max_wavelength: float = 750.0,
        total_flow: Optional[float] = None,
        n_restarts: int = 3,
    ) -> "SpectralSurrogate":
        flows = np.atleast_2d(np.asarray(flows, dtype=float))
        spectra = np.atleast_2d(np.asarray(spectra, dtype=float))

        pca = fit_spectral_pca(
            wavelengths, spectra, n_components=n_components, normalize=normalize,
            min_wavelength=min_wavelength, max_wavelength=max_wavelength,
        )
        scores = np.vstack([
            pca.transform(wavelengths, row) for row in spectra
        ])  # (N, K)

        flow_mean = flows.mean(axis=0)
        flow_std = flows.std(axis=0)
        flow_std = np.where(flow_std > 1e-9, flow_std, 1.0)
        Xs = (flows - flow_mean) / flow_std

        gps: List[GaussianProcessRegressor] = []
        for k in range(scores.shape[1]):
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=np.ones(Xs.shape[1]), length_scale_bounds=(0.05, 50.0), nu=2.5
            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts, alpha=1e-8
            )
            gp.fit(Xs, scores[:, k])
            gps.append(gp)

        if total_flow is None:
            total_flow = float(np.median(flows.sum(axis=1)))

        logger.info(
            "Fitted SpectralSurrogate: %d components, %d GPs, %d training samples",
            scores.shape[1], len(gps), flows.shape[0],
        )
        return cls(pca, gps, flow_mean, flow_std, total_flow)

    # -- prediction ------------------------------------------------------

    def _standardize(self, flows: np.ndarray) -> np.ndarray:
        flows = np.atleast_2d(np.asarray(flows, dtype=float))
        return (flows - self.flow_mean) / self.flow_std

    def predict_scores(self, flows: np.ndarray) -> np.ndarray:
        """Predict PCA scores for one or many flow vectors -> (N, K)."""
        Xs = self._standardize(flows)
        cols = [gp.predict(Xs) for gp in self.gps]
        return np.vstack(cols).T

    def predict_spectrum(self, flows: np.ndarray) -> np.ndarray:
        """Reconstruct the predicted absorbance spectrum from predicted scores."""
        scores = self.predict_scores(flows)
        # Inverse PCA: spectrum = mean + scores @ components  (then de-normalize
        # is identity for normalize='none'; for 'max'/'area' the shape is what
        # matters for matching, so we return the centered reconstruction + mean).
        recon = scores @ self.pca.components + self.pca.mean
        return recon if recon.shape[0] > 1 else recon[0]

    # -- feasible-target projection (option 3) ---------------------------

    def _reduced_bounds(self) -> List[Tuple[float, float]]:
        d = len(self.flow_mean)
        return [(0.0, self.total_flow)] * (d - 1)

    def _reduced_to_full(self, reduced: np.ndarray) -> np.ndarray:
        full = np.empty(len(self.flow_mean))
        full[:-1] = reduced
        full[-1] = self.total_flow - np.sum(reduced)
        return full

    def project_to_manifold(
        self,
        target_scores: np.ndarray,
        seed: Optional[int] = None,
        maxiter: int = 200,
    ) -> Tuple[np.ndarray, float]:
        """Find feasible flows whose predicted scores are closest to a target.

        Returns ``(optimal_flows, residual)`` where residual is the Euclidean
        score distance at the optimum (small -> target is achievable).
        """
        target_scores = np.asarray(target_scores, dtype=float)

        def objective(reduced: np.ndarray) -> float:
            last = self.total_flow - np.sum(reduced)
            if last < -1e-9:
                return 1e6 + abs(last)  # infeasible budget
            full = self._reduced_to_full(reduced)
            pred = self.predict_scores(full)[0]
            return float(np.sum((pred - target_scores) ** 2))

        bounds = self._reduced_bounds()
        constraint = LinearConstraint(np.ones(len(bounds)), lb=0, ub=self.total_flow)
        result = differential_evolution(
            objective, bounds=bounds, constraints=[constraint],
            tol=1e-7, maxiter=maxiter, popsize=20,
            seed=seed if seed is not None else None,
        )
        opt_flows = np.round(self._reduced_to_full(result.x), 4)
        return opt_flows, float(np.sqrt(max(result.fun, 0.0)))

    def make_feasible_target(
        self,
        candidate_scores: Optional[np.ndarray] = None,
        candidate_flows: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> FeasibleTarget:
        """Build a feasible benchmark target (option 3).

        Provide either a candidate score vector (designed in PCA space) or a
        candidate flow vector (a held-out / interpolated setpoint). The target
        is projected onto the achievable manifold so the optimum is known and
        the cost floor is ~0.
        """
        if candidate_flows is not None:
            candidate_scores = self.predict_scores(candidate_flows)[0]
        if candidate_scores is None:
            raise ValueError("Provide candidate_scores or candidate_flows")

        opt_flows, residual = self.project_to_manifold(candidate_scores, seed=seed)
        projected_scores = self.predict_scores(opt_flows)[0]
        spectrum = projected_scores @ self.pca.components + self.pca.mean
        return FeasibleTarget(
            spectrum=spectrum,
            wavelengths=self.pca.wavelengths,
            optimal_flows=opt_flows,
            target_scores=projected_scores,
            residual=residual,
        )
