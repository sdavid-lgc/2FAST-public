# [vendored] verbatim copy of paperAnalysis/bo_sim/__init__.py (md5 35577b042b0baa720558cc709f4ff0d2), tools/vendor_bo_sim.py
"""
bo_sim — benchmark de convergence BO (vrai code `toofast`) vs random / LHS
sur un oracle spectral déterministe (modèle nb23 de 2FAST_ML).

Modules :
  config       chemins
  oracle       SpectralOracle (Q5 -> spectre normalisé)
  cost         coût cosinus brut (miroir de toofast.core.cost.CosineSpectrumCost)
  toofast_sim  Experiment toofast en simulation, mesure remplacée par l'oracle
  baselines    random / LHS (générateurs toofast + Dirichlet)
  bench        orchestration (seeds, variantes, cache disque) + résumé

Env : Python 3.11 (Anaconda) — `py -3.11`.
"""
import os
import sys

from . import config as C

# Les workers joblib (processus séparés) doivent pouvoir importer bo_sim et toofast.
for p in (str(C.SRC_DIR), str(C.TOOFAST_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(C.SRC_DIR), str(C.TOOFAST_SRC)] +
    [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
