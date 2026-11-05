"""Repository paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# raw/ can live elsewhere (e.g. an extracted Zenodo archive): set TOOFAST_RAW
RAW = Path(os.environ.get("TOOFAST_RAW", ROOT / "raw"))
PROCESSED = ROOT / "data" / "processed"
INPUTS = ROOT / "data" / "inputs"      # frozen artefacts used by the platform during the runs (versioned, read-only)
RESOURCE_DIR = INPUTS
FIGURES = ROOT / "figures"
