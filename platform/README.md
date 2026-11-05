# toofast — 2FAST platform software

Control software of the 2FAST autonomous stop-flow microfluidic platform (silver nanoparticle synthesis, time-resolved
UV-Vis extinction, Bayesian optimisation of the reagent flows), as used in *Capturing and Predicting Reaction Kinetics Across Timescales: An Autonomous Switchable
Flow/Stop-Flow Platform for Nanoparticle Synthesis*.

It runs **with or without the instrument**: every instrument has a simulated counterpart, and a virtual reactor
produces plausible extinction spectra from what the experiment does (mixing, stop-flow growth, washing), so the GUI
and complete optimisation campaigns can be run on any computer.

## Install
```bash
pip install -e platform            # from the repository root
pip install -e "platform[serial]"  # + pyserial / propar, for the real serial instruments
```
Python ≥ 3.10. The analysis environment of the repository (`environment.yml`, Python 3.11, scikit-learn 1.2.2) works.

## Run without hardware
```bash
toofast-sim --target-lambda 550 --iterations 6 --out sim_runs/demo     # headless BO campaign (≈ 1 min)
toofast                                                                # GUI: tick "Simulate experiment"
```
The headless campaign runs the instrument's own loop (`Experiment.stop_flow()`: projected-LHS initial design,
stop-flow measurement with the platform's growth-termination rule, chip wash, GP fit, acquisition) on simulated
instruments, with a virtual clock (minutes of stop-flow waits take seconds; `--speed 1` for real time). The run folder
has the same layout as on the instrument: `costs.csv`, `pump_flows.csv`, `lambda_max.csv`, `samples/*.h5`,
`measurement_spectra_*.csv`, parameters and log. A seeded campaign is reproducible.

```python
from toofast.simulation.headless import run_campaign
r = run_campaign("sim_runs/demo", target_lambda_max=550, iterations=6, seed=0)
r.costs, r.flows, r.lambda_max
```

**The virtual reactor is a toy model** (`toofast/simulation/reactor.py`): a composition-dependent plasmon band with
a first-order growth after the flow stops, no particles without reducer or silver, measurement noise. It is meant
to exercise the platform logic, not to predict the chemistry; no result of the paper uses it. (The in-silico benchmark
of the paper, Figure S15, uses a data-driven oracle instead — see `src/toofast_analysis/bo_sim/` in the repository.)

## Initial designs
| GUI option | `init_option` | Design |
|---|---|---|
| Latin hypercube (default) | `latinhypercube` | Latin hypercube on the simplex of compositions: a hypercube LHS in d−1 dimensions mapped by the inverse Rosenblatt transform of the uniform composition (`toofast/core/sampling.py`) — uniform over the compositions, stratified, exact total flow and bounds |
| Projected LHS | `projected_lhs` | hypercube LHS rescaled to the total flow (toofast 1.0 "LatinHyperCube", **the design of the paper's runs**) |
| Random / User defined | `random` / `fixed` | unchanged |

The same two choices exist for the "LHS Flows" / "Projected LHS Flows" experiments and `toofast-sim --init`. A
`_config.json` written by toofast 1.0 (no `toofast_version` key) is reloaded with `projected_lhs`, so a reloaded run
keeps its design.

## Run the instrument
The syringe pumps (Cetoni Nemesys) and the spectrometer (Avantes AvaSpec) are driven through their vendors' SDKs,
which are proprietary and **not distributed here**. The package defines what it needs from them
(`toofast.hardware.base.PumpBackend`, `SpectrometerBackend`); a laboratory implements these two classes on top of
its SDK installation and names them:
```bash
set TOOFAST_PUMP_DRIVER=mylab_drivers.nemesys:NemesysBackend
set TOOFAST_SPECTROMETER_DRIVER=mylab_drivers.avantes:AvantesBackend
set TOOFAST_DEVICE_CONFIG_DIR=C:\path\to\vendor\device\configurations     :: passed to the pump driver
toofast
```
See [`docs/drivers.md`](docs/drivers.md) for the contract and a skeleton. The other instruments (Tricontinent C3000
rinse pumps, fraction collector, Bronkhorst mass-flow controller) use documented serial protocols and are driven by
the package itself (`pip install "platform[serial]"`); in simulation the same drivers talk to emulated firmwares.

## Layout
| Path | Content |
|---|---|
| `src/toofast/core/` | experiment orchestration (`experiment.py`, with mixins `flat_strategy`, `next_point`, `gp_model`, `warm_start`, `wash`, `run_config`), cost functions (`cost.py`), growth-termination rule (`growth.py`), PCA basis, GP diagnostics, multi-objective |
| `src/toofast/acquisition/` | acquisition functions (EI, LCB, Thompson sampling, exploration wrapper, feasibility) |
| `src/toofast/hardware/` | vendor-neutral contracts, pump system, spectrometer, serial drivers, simulated instruments, driver loading |
| `src/toofast/simulation/` | virtual reactor, headless runner (`python -m toofast.simulation`) |
| `src/toofast/clock.py` | time source (wall clock on the instrument, virtual clock in simulation) |
| `src/toofast/data/`, `gui/`, `visualization/` | HDF5 / CSV storage, Tk GUI, figures |

## Licence
MIT (see `LICENSE`).
