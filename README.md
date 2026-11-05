# 2FAST — simulated platform and paper figures

Companion repository for *Capturing and Predicting Reaction Kinetics Across Timescales: An Autonomous Switchable
Flow/Stop-Flow Platform for Nanoparticle Synthesis* (submitted).

It holds the platform software, which runs without any instrument, and the notebooks that regenerate the figures of the
paper from the raw data. The raw data are a separate Zenodo record: [10.5281/zenodo.23983430](https://doi.org/10.5281/zenodo.23983430).

## Layout
| Path | Content |
|---|---|
| `platform/` | Platform software `toofast` (MIT): simulated instruments, virtual reactor, headless campaigns (`toofast-sim`) and GUI; see [`platform/README.md`](platform/README.md). |
| `notebooks/` | One notebook per figure or group of figures, stored without outputs. Each holds all its code, except S15. |
| `src/toofast_analysis/` | Modules needed by the S15 notebook only: its parallel workers import them from files. |
| `data/inputs/` | Frozen artefacts the platform used during the runs (PCA bases, prior kernel), delivered as used, and the lists of the files used by each figure. |
| `third_party/toofast/` | Snapshot of the platform code driven in simulation by the in-silico benchmark (S15, Table S3); vendor files not included. |

## Run the platform in simulation
```bash
pip install -e platform
toofast-sim --target-lambda 550 --iterations 6 --out sim_runs/demo
```
The virtual reactor is a toy model: it exercises the platform logic and no result of the paper uses it.

## Regenerate the figures
```bash
conda env create -f environment.yml
conda activate toofast-pub
```
For S15 only, also run `pip install -e .`. Download the Zenodo record and extract it into `raw/`, or set `TOOFAST_RAW` to the extracted folder. `MANIFEST.csv` in the
record gives the md5 of every file. Then open a notebook from this repository, or run it from the repository root:
```bash
jupyter nbconvert --to notebook --execute notebooks/fig02_standard_duplicates.ipynb --output-dir data/processed/notebooks_executed --ExecutePreprocessor.timeout=-1
```
Figures are written to `figures/`, derived data to `data/processed/`.


## Environment
Python 3.11.5, scikit-learn 1.2.2 (the versions the original models were produced with); channel `defaults` with MKL
2023.1 and exact builds, see the header of `environment.yml`. These builds are pinned for Windows, the only system tested.

Two practical points on Windows:
- set `PYTHONNOUSERSITE=1` before creating or using the environment, otherwise packages from a user site-packages folder
  (`%APPDATA%\Python\Python311`) leak into it;
- the post-link script of scikit-learn calls `chcp`: `C:\Windows\System32` must be on `PATH` during `conda env create`.

## Licence
This repository is under the MIT licence, see [`LICENSE`](LICENSE). The data of the Zenodo record are under CC BY 4.0.
