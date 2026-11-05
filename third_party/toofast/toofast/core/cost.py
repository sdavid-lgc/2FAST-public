"""
Cost function strategies for comparing measured spectra against targets.

Extracted from tooFAST.py:cost_function() which crammed 4 different
algorithms into one if/elif chain. Now each is a separate callable class
implementing the CostFunction protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity

from toofast.data.models import Spectrum

if TYPE_CHECKING:
    from toofast.core.pca_model import SpectralPCA

logger = logging.getLogger(__name__)


class CostFunction(ABC):
    """Base class for cost functions. Subclasses implement __call__."""

    @abstractmethod
    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        """
        Compute cost from a spectrum.

        Returns
        -------
        cost : float
            Distance to target (lower is better).
        measured_value : float
            The measured value used for comparison (e.g., absorbance).
        """

    def is_flat_spectrum(self, spectrum: Spectrum, threshold: float = 0.01) -> bool:
        """Check if spectrum is essentially flat (no nanoparticle signal)."""
        if spectrum.absorbance is None:
            return True
        return float(np.max(spectrum.absorbance) - np.min(spectrum.absorbance)) < threshold


class SinglePointCost(CostFunction):
    """
    Cost based on absorbance at a single target wavelength.

    Cost = (target_abs - measured_abs)^2
    """

    def __init__(self, target_wavelength: float, target_absorbance: float):
        self.target_wavelength = target_wavelength
        self.target_absorbance = target_absorbance

    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        measured_abs = spectrum.get_abs_at_wl(self.target_wavelength)
        cost = (self.target_absorbance - measured_abs) ** 2
        logger.debug(
            f"SinglePoint cost at {self.target_wavelength}nm: "
            f"target={self.target_absorbance:.4f}, measured={measured_abs:.4f}, cost={cost:.6f}"
        )
        return cost, measured_abs


class L2SpectrumCost(CostFunction):
    """
    L2 (Euclidean) distance between measured and target full spectrum.

    Cost = sum((target - measured)^2) over the wavelength range.
    """

    def __init__(
        self,
        target_spectrum: Spectrum,
        min_wavelength: float = 315.0,
        max_wavelength: float = 750.0,
    ):
        self.target_spectrum = target_spectrum
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        mask = (spectrum.wavelengths >= self.min_wavelength) & (
            spectrum.wavelengths <= self.max_wavelength
        )
        measured = spectrum.absorbance[mask]
        target = self.target_spectrum.absorbance[mask]
        cost = float(np.sum((target - measured) ** 2))
        measured_max = float(np.max(measured))
        logger.debug(f"L2 cost: {cost:.6f}, max_abs={measured_max:.4f}")
        return cost, measured_max


class CosineSpectrumCost(CostFunction):
    """
    Cosine similarity cost between measured and target full spectrum.

    Cost = 1 - cosine_similarity(target, measured)
    """

    def __init__(
        self,
        target_spectrum: Spectrum,
        min_wavelength: float = 315.0,
        max_wavelength: float = 750.0,
    ):
        self.target_spectrum = target_spectrum
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        mask = (spectrum.wavelengths >= self.min_wavelength) & (
            spectrum.wavelengths <= self.max_wavelength
        )
        measured = spectrum.absorbance[mask].reshape(1, -1)
        target = self.target_spectrum.absorbance[mask].reshape(1, -1)
        similarity = cosine_similarity(target, measured)[0][0]
        cost = 1.0 - similarity
        measured_max = float(np.max(measured))
        logger.debug(f"Cosine cost: {cost:.6f}, similarity={similarity:.4f}")
        return cost, measured_max


class LambdaMaxCost(CostFunction):
    """
    Cost based on distance between target and measured lambda_max.

    Cost = (target_wl - measured_lambda_max)^2
    """

    def __init__(
        self,
        target_wavelength: float,
        min_wavelength: float = 315.0,
        max_wavelength: float = 750.0,
    ):
        self.target_wavelength = target_wavelength
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        mask = (spectrum.wavelengths >= self.min_wavelength) & (
            spectrum.wavelengths <= self.max_wavelength
        )
        filtered = savgol_filter(spectrum.absorbance[mask], 11, 3)
        wl_subset = spectrum.wavelengths[mask]
        lambda_max = wl_subset[np.argmax(filtered)]
        cost = (self.target_wavelength - lambda_max) ** 2
        logger.debug(
            f"LambdaMax cost: target={self.target_wavelength}nm, "
            f"measured={lambda_max}nm, cost={cost:.2f}"
        )
        return cost, lambda_max


class PCASpectrumCost(CostFunction):
    """
    Cost in the score space of a *frozen* spectral PCA basis.

    Both the measured and the target spectrum are projected onto the same
    pre-fitted basis; the cost is the (optionally variance-weighted) squared
    Euclidean distance between the two score vectors:

        cost = sum_k w_k * (score_measured_k - score_target_k)^2

    This keeps far more information than lambda_max (it encodes peak position,
    width and amplitude jointly) while staying low-dimensional and giving the
    GP a smoother landscape than raw full-spectrum L2.

    Parameters
    ----------
    pca_model : SpectralPCA
        Frozen basis (fitted offline, never re-estimated during a BO run).
    target_spectrum : Spectrum
        Target whose projection defines the optimum (score distance 0).
    variance_weighted : bool
        If True, weight each component by its explained-variance ratio so the
        dominant modes dominate the cost. If False, plain Euclidean distance.
    lambda_weight : float
        If > 0, add a lambda_max term ``lambda_weight * ((lam - lam_target)
        / lambda_scale_nm)**2`` to the PCA score distance. Lets a NON-normalized
        basis (which keeps amplitude but is weakly sensitive to peak position)
        regain a sharp, interpretable sensitivity to the peak position without
        discarding amplitude (as max-normalization would). Default 0 (off).
    lambda_scale_nm : float
        Wavelength scale (nm) of the lambda_max term. Default 30 nm.
    amp_weight : float
        If > 0, add an amplitude term ``amp_weight * ((A - A_target) /
        A_target)**2`` (A = measured peak absorbance) to the score distance. On a
        max-normalized basis the score distance is amplitude-blind; this term
        discriminates good-shape compositions by yield/concentration. Keep it
        small so shape stays primary (the non-normalized basis over-weighted
        amplitude — this is the controllable inverse). Default 0 (off).
    metric : str
        Score-space distance: 'sqdist' (Σ w·d², default, current behaviour),
        'dist' (its sqrt), or 'cosine' (1 − cosine similarity, bounded [0,2],
        smooth → low GP uncertainty; needs n_components >= 2).
    n_components : int
        Use only the first k principal components (0 = all). Dropping noisy minor
        PCs lowers uncertainty; too few underfits. For 'cosine' the sweet spot is
        often k=2 (1D cosine is degenerate).
    """

    def __init__(
        self,
        pca_model: "SpectralPCA",
        target_spectrum: Spectrum,
        variance_weighted: bool = False,
        lambda_weight: float = 0.0,
        lambda_scale_nm: float = 30.0,
        amp_weight: float = 0.0,
        metric: str = "sqdist",
        n_components: int = 0,
    ):
        self.pca_model = pca_model
        self.variance_weighted = variance_weighted
        self.lambda_weight = float(lambda_weight)
        self.lambda_scale_nm = float(lambda_scale_nm)
        self.amp_weight = float(amp_weight)
        # Score-space distance metric and how many leading PCs to use (0 = all).
        # 'cosine' is bounded [0,2] and smooth (low GP uncertainty); needs k>=2
        # (1D cosine is degenerate: only the sign of PC1). See PCASpectrumCost doc.
        self.metric = str(metric)
        n = int(n_components) if n_components else pca_model.n_components
        self.k = max(1, min(n, pca_model.n_components))
        self.target_scores = pca_model.transform(
            target_spectrum.wavelengths, target_spectrum.absorbance
        )
        self.target_lambda_max = float(
            np.asarray(target_spectrum.wavelengths)[
                int(np.argmax(target_spectrum.absorbance))]
        )
        # Target peak amplitude; the amplitude term is RELATIVE to it so the
        # weight is dimensionless ((A - A_t)/A_t)^2. On a max-normalized basis
        # the PCA score distance is amplitude-blind, so this term lets the cost
        # discriminate good-shape compositions by yield/concentration WITHOUT the
        # amplitude dominating (keep amp_weight small).
        self.target_amplitude = float(np.max(target_spectrum.absorbance))
        if variance_weighted:
            ev = np.asarray(pca_model.explained_variance, dtype=float)
            total = float(ev.sum())
            self.weights = ev / total if total > 1e-12 else np.ones_like(ev)
        else:
            self.weights = np.ones(pca_model.n_components, dtype=float)

    def _base_cost(self, scores) -> float:
        """Score-space distance per self.metric, using the first self.k PCs."""
        k = self.k
        s = scores[:k]; t = self.target_scores[:k]; w = self.weights[:k]
        if self.metric == "cosine":
            denom = float(np.linalg.norm(s) * np.linalg.norm(t))
            return float(1.0 - (s @ t) / denom) if denom > 1e-12 else 1.0
        d = s - t
        sq = float(np.sum(w * d * d))
        if self.metric == "dist":
            return float(np.sqrt(sq))
        return sq  # 'sqdist' (default)

    def __call__(self, spectrum: Spectrum) -> Tuple[float, float]:
        scores = self.pca_model.transform(spectrum.wavelengths, spectrum.absorbance)
        cost = self._base_cost(scores)
        measured_max = float(np.max(spectrum.absorbance))
        if self.lambda_weight > 0.0:
            lam = float(np.asarray(spectrum.wavelengths)[
                int(np.argmax(spectrum.absorbance))])
            d = (lam - self.target_lambda_max) / self.lambda_scale_nm
            cost += self.lambda_weight * d * d
        if self.amp_weight > 0.0:
            scale = self.target_amplitude if self.target_amplitude > 1e-9 else 1.0
            da = (measured_max - self.target_amplitude) / scale
            cost += self.amp_weight * da * da
        logger.debug(
            f"PCA cost: {cost:.6f} (scores={np.round(scores, 4).tolist()}, "
            f"target={np.round(self.target_scores, 4).tolist()})"
        )
        return cost, measured_max


def huber_cost_nm(measured_nm: float, target_nm: float, delta_nm: float) -> float:
    """
    Huber loss between two wavelengths (in nm).

    Quadratic for |error| <= delta_nm, linear beyond. Delta defaults to 30 nm
    (validated by grid search across multiple BO runs as the most robust cost
    method when paired with a Matern 2.5 ARD kernel).
    """
    diff = float(measured_nm) - float(target_nm)
    abs_diff = abs(diff)
    if abs_diff <= delta_nm:
        return 0.5 * diff * diff
    return float(delta_nm) * (abs_diff - 0.5 * float(delta_nm))


def create_cost_function(
    target_style: str,
    target_wavelength: Optional[float] = None,
    target_absorbance: Optional[float] = None,
    target_spectrum: Optional[Spectrum] = None,
    cost_method: str = "L2",
    min_wavelength: float = 315.0,
    max_wavelength: float = 750.0,
    pca_model: Optional["SpectralPCA"] = None,
    pca_variance_weighted: bool = False,
    pca_lambda_weight: float = 0.0,
    pca_lambda_scale_nm: float = 30.0,
    pca_amp_weight: float = 0.0,
    pca_cost_metric: str = "sqdist",
    pca_n_components: int = 0,
) -> CostFunction:
    """
    Factory to create the appropriate cost function from config.

    Parameters
    ----------
    target_style : str
        One of 'single_point', 'full_spectrum', 'lambda_max', 'pca'.
    pca_model : SpectralPCA, optional
        Frozen PCA basis for target_style='pca'.
    pca_variance_weighted : bool
        Weight PCA components by explained variance ('pca' mode).
    target_wavelength : float, optional
        Target wavelength for single_point and lambda_max.
    target_absorbance : float, optional
        Target absorbance for single_point.
    target_spectrum : Spectrum, optional
        Target spectrum object for full_spectrum.
    cost_method : str
        'L2' or 'cosine' for full_spectrum mode.
    min_wavelength, max_wavelength : float
        Wavelength bounds for spectrum comparison.
    """
    if target_style == "single_point":
        if target_wavelength is None or target_absorbance is None:
            raise ValueError("single_point requires target_wavelength and target_absorbance")
        return SinglePointCost(target_wavelength, target_absorbance)

    elif target_style == "full_spectrum":
        if target_spectrum is None:
            raise ValueError("full_spectrum requires target_spectrum")
        if cost_method == "cosine":
            return CosineSpectrumCost(target_spectrum, min_wavelength, max_wavelength)
        return L2SpectrumCost(target_spectrum, min_wavelength, max_wavelength)

    elif target_style == "lambda_max":
        if target_wavelength is None:
            raise ValueError("lambda_max requires target_wavelength")
        return LambdaMaxCost(target_wavelength, min_wavelength, max_wavelength)

    elif target_style == "pca":
        if pca_model is None or target_spectrum is None:
            raise ValueError("pca requires pca_model and target_spectrum")
        return PCASpectrumCost(pca_model, target_spectrum, pca_variance_weighted,
                               lambda_weight=pca_lambda_weight,
                               lambda_scale_nm=pca_lambda_scale_nm,
                               amp_weight=pca_amp_weight,
                               metric=pca_cost_metric,
                               n_components=pca_n_components)

    else:
        raise ValueError(f"Unknown target_style: {target_style}")
