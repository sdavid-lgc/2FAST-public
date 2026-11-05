"""[vendored, replaced] bo_sim/config.py — paths of the benchmark, repository version (same names as the original
paperAnalysis/bo_sim/config.py)."""
from pathlib import Path

from toofast_analysis import paths as _p

SRC_DIR     = _p.ROOT / "src"                                     # importable by the joblib workers
TOOFAST_SRC = _p.ROOT / "third_party" / "toofast"                 # package `toofast` (snapshot)
PCA_DIR     = _p.INPUTS                                           # frozen PCA bases + prior kernels
BO_RUNS_DIR = _p.RAW                                              # was paperAnalysis/BO (md5-identical copies)

# --- oracle training data (were 2FAST_ML/spectral_selected.csv + spectra_matrix.npz), rebuilt from raw/
RESULTS_DIR = _p.PROCESSED / "bo_sim"                             # was paperAnalysis/bo_sim_results
SPECTRAL_SELECTED = RESULTS_DIR / "oracle_data" / "spectral_selected.csv"
SPECTRA_MATRIX    = RESULTS_DIR / "oracle_data" / "spectra_matrix.npz"

RUN_S28 = BO_RUNS_DIR / "S28-M01"      # raw spectrum, cosine, no prior
RUN_S24 = BO_RUNS_DIR / "S24-M01"    # PCA cosine k=4 + warm-start prior

PRIOR_KERNELS = {
    "pca_model_norm6": PCA_DIR / "prior_kernel_pca_model_norm6_cosine_k4.pkl",
    "pca_model_5":     PCA_DIR / "prior_kernel_pca_model_5_cosine_k4.pkl",   # not delivered (not used by the paper)
}
DEFAULT_PRIOR = "pca_model_norm6"

ORACLE_CACHE = RESULTS_DIR / "oracle_nb23.pkl"
FIG_OUT = _p.FIGURES / "bo_sim"                                   # was paperAnalysis/figures_publication_BO
