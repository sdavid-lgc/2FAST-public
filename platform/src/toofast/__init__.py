"""
2FAST - Fully Automated Synthesis Technology.

A Python platform for autonomous nanoparticle synthesis exploration
using Bayesian Optimization with Gaussian Process Regression.
"""

import sys as _sys

if _sys.version_info < (3, 10):
    raise ImportError(f"toofast needs Python >= 3.10 (this is {_sys.version.split()[0]}): "
                      "use the toofast-pub environment (conda env create -f environment.yml)")

__version__ ="1.2.0"   # 1.0.0 = acquisition of the paper's data (commit 2358b2e); 1.1.0 = public, vendor-neutral; 1.2.0 = sample files in format 2.0
