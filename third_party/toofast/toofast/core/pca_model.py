"""
Frozen spectral PCA model for low-dimensional cost functions.

Rationale
---------
``lambda_max`` keeps only the peak position; ``full_spectrum`` L2 keeps the
whole curve but conflates concentration with shape and gives the GP a rough,
multi-modal landscape. A PCA fitted *once* on a reference set of spectra and
then **frozen** is a good middle ground: 2-3 components typically capture
>95% of the variance of UV-Vis spectra, so the cost lives in a small,
smooth space that still encodes peak position, width and amplitude.

Crucial invariant: the PCA basis must be fitted offline and never re-estimated
during a BO run. If the basis moved between iterations the cost would change
definition and break the GP surrogate. ``SpectralPCA`` is therefore an
immutable container (load -> transform), and fitting is a separate offline step.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Normalisation modes applied to every spectrum before projection. 'area' and
# 'max' make the basis encode *shape* rather than concentration; 'none' keeps
# amplitude information.
NormalizeMode = str  # Literal["none", "area", "max"]


class SpectralPCA:
    """Immutable, frozen PCA projector for spectra.

    Parameters
    ----------
    wavelengths : np.ndarray
        Common wavelength grid (nm) the basis was fitted on.
    mean : np.ndarray
        Per-wavelength mean removed before projection (n_features,).
    components : np.ndarray
        PCA components (n_components, n_features).
    explained_variance : np.ndarray
        Variance explained by each component (n_components,).
    normalize : str
        Pre-projection normalisation: 'none', 'area' or 'max'.
    min_wavelength, max_wavelength : float
        Wavelength window the basis covers (used for masking on transform).
    """

    def __init__(
        self,
        wavelengths: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
        explained_variance: np.ndarray,
        normalize: NormalizeMode = "none",
        min_wavelength: float = 315.0,
        max_wavelength: float = 750.0,
    ):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.components = np.asarray(components, dtype=float)
        self.explained_variance = np.asarray(explained_variance, dtype=float)
        self.normalize = normalize
        self.min_wavelength = float(min_wavelength)
        self.max_wavelength = float(max_wavelength)

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    # -- preprocessing ---------------------------------------------------

    @staticmethod
    def _normalize_vector(y: np.ndarray, mode: NormalizeMode) -> np.ndarray:
        if mode == "none":
            return y
        if mode == "area":
            area = float(np.trapz(np.abs(y))) if hasattr(np, "trapz") else float(np.sum(np.abs(y)))
            return y / area if area > 1e-12 else y
        if mode == "max":
            peak = float(np.max(np.abs(y)))
            return y / peak if peak > 1e-12 else y
        raise ValueError(f"Unknown normalize mode: {mode}")

    def _prepare(self, wavelengths: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
        """Resample onto the frozen grid, normalise and mean-center."""
        wavelengths = np.asarray(wavelengths, dtype=float)
        absorbance = np.asarray(absorbance, dtype=float)
        # Resample onto the basis grid if the measured grid differs.
        if wavelengths.shape != self.wavelengths.shape or not np.allclose(
            wavelengths, self.wavelengths
        ):
            absorbance = np.interp(self.wavelengths, wavelengths, absorbance)
        absorbance = self._normalize_vector(absorbance, self.normalize)
        return absorbance - self.mean

    # -- projection ------------------------------------------------------

    def transform(self, wavelengths: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
        """Project a spectrum onto the frozen PCA basis -> score vector."""
        centered = self._prepare(wavelengths, absorbance)
        return self.components @ centered

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(
            path,
            wavelengths=self.wavelengths,
            mean=self.mean,
            components=self.components,
            explained_variance=self.explained_variance,
            normalize=np.array(self.normalize),
            min_wavelength=np.array(self.min_wavelength),
            max_wavelength=np.array(self.max_wavelength),
        )
        logger.info("Saved frozen SpectralPCA (%d components) to %s", self.n_components, path)

    @classmethod
    def load(cls, path: str) -> "SpectralPCA":
        if not path.endswith(".npz"):
            path = path + ".npz" if os.path.exists(path + ".npz") else path
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)

        def pick(*names):
            for n in names:
                if n in keys:
                    return data[n]
            raise KeyError(
                f"None of {names} found in {path} (available: {sorted(keys)})")

        # Accept legacy aliases (e.g. notebooks that saved wl/comps) and fill in
        # optional metadata from the data itself when absent.
        wavelengths = np.asarray(pick("wavelengths", "wl"), dtype=float)
        components = np.asarray(pick("components", "comps"), dtype=float)
        ev = (np.asarray(data["explained_variance"], dtype=float)
              if "explained_variance" in keys
              else np.ones(components.shape[0], dtype=float))
        normalize = str(data["normalize"]) if "normalize" in keys else "none"
        min_wl = (float(data["min_wavelength"]) if "min_wavelength" in keys
                  else float(wavelengths.min()))
        max_wl = (float(data["max_wavelength"]) if "max_wavelength" in keys
                  else float(wavelengths.max()))
        return cls(
            wavelengths=wavelengths,
            mean=np.asarray(pick("mean"), dtype=float),
            components=components,
            explained_variance=ev,
            normalize=normalize,
            min_wavelength=min_wl,
            max_wavelength=max_wl,
        )


def fit_spectral_pca(
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    n_components: int = 3,
    normalize: NormalizeMode = "max",
    min_wavelength: float = 315.0,
    max_wavelength: float = 750.0,
) -> SpectralPCA:
    """Fit a PCA basis offline from a matrix of reference spectra.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength grid (n_features,) shared by all rows of ``spectra``.
    spectra : np.ndarray
        Absorbance matrix (n_samples, n_features).
    n_components : int
        Number of components to keep (small: 2-3 recommended).
    normalize : str
        'none', 'area' or 'max'. 'max'/'area' make the basis shape-focused.
    min_wavelength, max_wavelength : float
        Window to restrict the fit to.
    """
    from sklearn.decomposition import PCA

    wavelengths = np.asarray(wavelengths, dtype=float)
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))

    mask = (wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)
    wl = wavelengths[mask]
    X = spectra[:, mask]

    # Normalise every spectrum before fitting (must match transform()).
    X = np.vstack([SpectralPCA._normalize_vector(row, normalize) for row in X])

    n_components = int(min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components)
    pca.fit(X)

    model = SpectralPCA(
        wavelengths=wl,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        normalize=normalize,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
    )
    logger.info(
        "Fitted SpectralPCA: %d components, explained variance ratio=%s",
        n_components,
        np.round(pca.explained_variance_ratio_, 4).tolist(),
    )
    return model


def load_spectra_from_dir(
    directory: str,
    pattern: str = "measurement_spectra_*.csv",
    use_last_only: bool = True,
    delimiter: str = ";",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load reference spectra from 2FAST measurement CSV files.

    Each file has a header row ``timestamp; wl1; wl2; ...`` followed by one
    measured spectrum per row. Returns ``(wavelengths, spectra_matrix)``.

    Parameters
    ----------
    use_last_only : bool
        If True, keep only the final (stabilised) spectrum of each series;
        otherwise keep every row. The final spectrum is usually the cleanest
        reference, but all rows give a richer basis.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    wavelengths: Optional[np.ndarray] = None
    rows: List[np.ndarray] = []
    for fpath in files:
        with open(fpath, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split(delimiter)
        wl = np.array([float(x) for x in header[1:]])
        if wavelengths is None:
            wavelengths = wl
        elif wl.shape != wavelengths.shape or not np.allclose(wl, wavelengths):
            logger.warning("Skipping %s: wavelength grid differs from reference", fpath)
            continue
        data_lines = [lines[-1]] if use_last_only else lines[1:]
        for ln in data_lines:
            parts = ln.split(delimiter)
            rows.append(np.array([float(x) for x in parts[1:]]))

    if wavelengths is None or not rows:
        raise ValueError(f"No usable spectra found in {directory}")

    return wavelengths, np.vstack(rows)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit and freeze a spectral PCA basis from 2FAST CSV spectra."
    )
    parser.add_argument("directory", help="Directory containing measurement_spectra_*.csv files")
    parser.add_argument("-o", "--output", default="pca_model.npz", help="Output .npz path")
    parser.add_argument("-n", "--n-components", type=int, default=3)
    parser.add_argument("--normalize", choices=["none", "area", "max"], default="max")
    parser.add_argument("--min-wl", type=float, default=315.0)
    parser.add_argument("--max-wl", type=float, default=750.0)
    parser.add_argument("--all-rows", action="store_true", help="Use every spectrum, not just the last of each series")
    parser.add_argument("--pattern", default="measurement_spectra_*.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    wl, spectra = load_spectra_from_dir(
        args.directory, pattern=args.pattern, use_last_only=not args.all_rows
    )
    logger.info("Loaded %d spectra (%d wavelengths) from %s", spectra.shape[0], spectra.shape[1], args.directory)
    model = fit_spectral_pca(
        wl, spectra, n_components=args.n_components, normalize=args.normalize,
        min_wavelength=args.min_wl, max_wavelength=args.max_wl,
    )
    model.save(args.output)
    print(f"Saved frozen PCA basis ({model.n_components} components) to {args.output}")


if __name__ == "__main__":
    _main()
