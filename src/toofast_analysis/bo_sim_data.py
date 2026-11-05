"""Training data of the in-silico benchmark oracle (S15, Table S3), rebuilt from raw/.

Original: 2FAST_ML/spectral_catalogue.py (nb20: flows, is_flat = std(last spectrum) < 0.03) and
2FAST_ML/spectral_selection.py (nb21: selection, tiers, last spectrum resampled on the 310-1000 nm grid of the first
selected file) -> spectral_selected.csv + spectra_matrix.npz. The selection itself (274 rows, with the recorded tiers)
is resources/bo_sim_oracle_files.csv; this module re-reads every H5 in raw/ and checks flows and is_flat against it.
"""
import csv

import h5py
import numpy as np
import pandas as pd

from .paths import RAW, RESOURCE_DIR

WL_RANGE = (310.0, 1000.0)
FLAT_STD_THR = 0.03
ABS_CANDS = ["spectra/extinction"]
WL_CANDS = ["spectra/wavelength"]


def selection():
    with open(RESOURCE_DIR / "bo_sim_oracle_files.csv", encoding="utf-8") as fh:
        return pd.read_csv(fh, float_precision="round_trip")


def read_spectrum(fp):
    """(wavelengths, LAST spectrum) — spectral_selection.read_spectrum."""
    with h5py.File(fp, "r") as f:
        loc = next((c for c in ABS_CANDS if c in f), None)
        wlk = next((w for w in WL_CANDS if w in f), None)
        if loc is None or wlk is None:
            return None, None
        return f[wlk][()], f[loc][-1]


def build(out_dir, raw=None):
    """Write spectral_selected.csv + spectra_matrix.npz (original names and columns) into out_dir."""
    from .ml import catalog_2fast
    raw = RAW if raw is None else raw
    sel = selection()
    paths = [raw / p for p in sel.raw_path]
    wl0, _ = read_spectrum(paths[0])
    grid = wl0[(wl0 >= WL_RANGE[0]) & (wl0 <= WL_RANGE[1])]
    S, bad = [], []
    for p, (_, r) in zip(paths, sel.iterrows()):
        wl, a = read_spectrum(p)
        S.append(np.interp(grid, wl, a))
        rec = catalog_2fast.parse_h5_file(p)
        q = np.array([rec.Q1, rec.Q2, rec.Q3, rec.Q4, rec.Q5])
        if not np.allclose(q, r[["Q1", "Q2", "Q3", "Q4", "Q5"]].values.astype(float), atol=1e-9):
            bad.append((r.filename, "flows"))
        if bool(np.std(a) < FLAT_STD_THR) != bool(r.is_flat):
            bad.append((r.filename, "is_flat"))
    if bad:
        raise ValueError(f"raw/ disagrees with the recorded selection: {bad[:10]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.rename(columns={"raw_path": "filepath"}).assign(filepath=[str(p) for p in paths]).to_csv(
        out_dir / "spectral_selected.csv", index=False)
    np.savez(out_dir / "spectra_matrix.npz", S=np.vstack(S), wavelengths=grid)
    return out_dir
