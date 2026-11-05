"""
Loader for 2FAST per-sample HDF5 files (Romain's LHS training set).

Each .h5 file is one sample: a fixed flow setpoint, a short growth time-series
of spectra, and a finalized absorbance curve. For surrogate modelling and
benchmarking we only need, per file, the flow vector and the final spectrum on
the common wavelength grid.

File layout (see S12_* samples)::

    experiment/flows_set        (5,)            set flows, uL/min
    experiment/reagents         (5,) bytes      reagent names
    experiment/varying_pumps    (5,) bool
    flows/measured_flows        (5,)            actually delivered flows
    spectra/wavelengths         (M,)            nm
    spectra/absorbances         (T, M)          growth time-series
    statistics/abs_final        (M,)            finalized absorbance (target)
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectraDataset:
    """A set of (flows -> final spectrum) samples on a common wavelength grid."""

    flows: np.ndarray          # (N, n_pumps) set flows
    measured_flows: np.ndarray  # (N, n_pumps) delivered flows
    spectra: np.ndarray        # (N, M) final absorbances
    wavelengths: np.ndarray    # (M,)
    reagents: List[str]
    sample_ids: List[str]

    @property
    def n_samples(self) -> int:
        return int(self.flows.shape[0])

    @property
    def n_pumps(self) -> int:
        return int(self.flows.shape[1])

    def restrict_wavelengths(self, min_wl: float, max_wl: float) -> "SpectraDataset":
        """Return a copy masked to [min_wl, max_wl]."""
        mask = (self.wavelengths >= min_wl) & (self.wavelengths <= max_wl)
        return SpectraDataset(
            flows=self.flows,
            measured_flows=self.measured_flows,
            spectra=self.spectra[:, mask],
            wavelengths=self.wavelengths[mask],
            reagents=self.reagents,
            sample_ids=self.sample_ids,
        )

    def _subset(self, keep: np.ndarray) -> "SpectraDataset":
        keep = np.asarray(keep, dtype=bool)
        return SpectraDataset(
            flows=self.flows[keep],
            measured_flows=self.measured_flows[keep],
            spectra=self.spectra[keep],
            wavelengths=self.wavelengths,
            reagents=self.reagents,
            sample_ids=[s for s, k in zip(self.sample_ids, keep) if k],
        )

    def drop_flat(
        self, threshold: float = 0.05, min_wl: float = 315.0, max_wl: float = 750.0
    ) -> Tuple["SpectraDataset", np.ndarray]:
        """Drop no-particle / flat spectra before fitting a PCA basis.

        A spectrum is "flat" when its peak-to-peak amplitude over the window is
        below ``threshold`` (no plasmon band, or pure scattering). These are
        synthesis failures, not points on the morphology manifold; keeping them
        wastes PCA components on a particle/no-particle contrast and produces
        red-edge lambda_max artefacts. Flat spectra are instead handled by the
        flat-penalty / ghost-point machinery during BO, not by the PCA cost.

        Returns ``(clean_dataset, flat_mask)`` where ``flat_mask`` is True for
        the dropped (flat) samples.
        """
        flat = flat_spectrum_mask(self.spectra, self.wavelengths, min_wl, max_wl, threshold)
        return self._subset(~flat), flat


def flat_spectrum_mask(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    min_wl: float = 315.0,
    max_wl: float = 750.0,
    threshold: float = 0.05,
) -> np.ndarray:
    """Boolean mask (True = flat) of no-particle/scattering spectra.

    Flatness is the peak-to-peak amplitude over [min_wl, max_wl] being below
    ``threshold``. Mirrors cost.CostFunction.is_flat_spectrum but vectorised
    over a whole matrix and restricted to the useful window.
    """
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    wavelengths = np.asarray(wavelengths, dtype=float)
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    window = spectra[:, mask]
    ptp = window.max(axis=1) - window.min(axis=1)
    return ptp < threshold


def _decode(values) -> List[str]:
    out = []
    for v in values:
        out.append(v.decode() if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def load_h5_sample(
    path: str, use_measured_flows: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load one .h5 sample -> (flows, measured_flows, final_absorbance, wavelengths... ).

    Returns ``(set_flows, measured_flows, abs_final, wavelengths, reagents)``.
    Falls back to the last row of the growth series if ``statistics/abs_final``
    is missing.
    """
    import h5py

    with h5py.File(path, "r") as f:
        set_flows = np.asarray(f["experiment/flows_set"][:], dtype=float)
        if "flows/measured_flows" in f:
            measured = np.asarray(f["flows/measured_flows"][:], dtype=float)
        else:
            measured = set_flows.copy()
        wavelengths = np.asarray(f["spectra/wavelengths"][:], dtype=float)
        if "statistics/abs_final" in f:
            abs_final = np.asarray(f["statistics/abs_final"][:], dtype=float)
        else:
            abs_final = np.asarray(f["spectra/absorbances"][-1], dtype=float)
        reagents = _decode(f["experiment/reagents"][:]) if "experiment/reagents" in f else []
    return set_flows, measured, abs_final, wavelengths, reagents


def load_h5_dataset(
    directory: str,
    pattern: str = "*.h5",
    use_measured_flows: bool = False,
) -> SpectraDataset:
    """Load every .h5 sample in a directory into a :class:`SpectraDataset`.

    Files whose wavelength grid differs from the first loaded file are skipped
    with a warning (the basis/surrogate require a common grid).
    """
    import h5py  # noqa: F401  (fail fast with a clear error if h5py missing)

    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    ref_wl: Optional[np.ndarray] = None
    flows: List[np.ndarray] = []
    measured: List[np.ndarray] = []
    spectra: List[np.ndarray] = []
    ids: List[str] = []
    reagents: List[str] = []

    for fpath in files:
        try:
            sflow, mflow, absf, wl, reag = load_h5_sample(fpath, use_measured_flows)
        except Exception as e:  # pragma: no cover - corrupt file guard
            logger.warning("Skipping %s: %s", fpath, e)
            continue
        if ref_wl is None:
            ref_wl = wl
            reagents = reag
        elif wl.shape != ref_wl.shape or not np.allclose(wl, ref_wl):
            logger.warning("Skipping %s: wavelength grid differs from reference", fpath)
            continue
        flows.append(sflow)
        measured.append(mflow)
        spectra.append(absf)
        ids.append(os.path.splitext(os.path.basename(fpath))[0])

    if ref_wl is None or not flows:
        raise ValueError(f"No usable .h5 samples found in {directory}")

    logger.info("Loaded %d samples (%d wavelengths) from %s", len(flows), len(ref_wl), directory)
    return SpectraDataset(
        flows=np.vstack(flows),
        measured_flows=np.vstack(measured),
        spectra=np.vstack(spectra),
        wavelengths=ref_wl,
        reagents=reagents,
        sample_ids=ids,
    )
