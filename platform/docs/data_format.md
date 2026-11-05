# Data written by the platform

One folder per run (`<data folder>/<experiment>/<experiment>_<k>/` from the GUI, the `--out` folder of `toofast-sim`).
The same layout is produced on the instrument and in simulation; `experiment@simulated` tells them apart.

## Run folder
| File | Content |
|---|---|
| `samples/<run>_<sample_id>.h5` | one HDF5 file per sample: the full time series of spectra and the settings (below) |
| `measurement_spectra_<sample_id>.csv` | same spectra as text: `;`-separated, header `timestamp;λ1;λ2;…`, one row per spectrum (extinction) |
| `reference_spectrum.csv`, `target_spectrum.csv` | reference (transmittance) and target spectra, same layout |
| `pump_flows.csv` | flows set for each evaluation (µL/s, one row per evaluation, `;`) |
| `costs.csv` | cost of each evaluation, in evaluation order (flat spectra: the flat sentinel cost) |
| `lambda_max.csv` / `absorbance_at_<λ>_nm.csv` | measured λmax (λmax target) or extinction at the target wavelength |
| `growth_times.csv` | `sample_id;seconds` growth (termination) time of each sample |
| `_config.json` | complete run configuration (reloaded by "Load previous") |
| `<run>_parameters.txt` | human-readable log of the BO settings, GP kernels and result |
| `gaussian_processes/gp_model_iter_NNN.pkl`, `data/bo_data_NN.npz`, `figures/` | GP snapshots (scikit-learn pickles), GP maps, figures |
| `experiment_log.log`, `metadata.txt` | log; set-up of the session (GUI) |

`sample_id` is `init_k` (initial design), `iter_k` (BO iteration k, 1-based) or `sample_k` (fixed / random / LHS / CSV
experiments).

## Sample HDF5 — format "2fast-sample" 2.0 (toofast ≥ 1.2)
Units: a `unit` attribute on each dataset, a suffix in the name of scalar attributes.
```
<run>_<sample_id>.h5
├── root attrs     format = "2fast-sample", format_version = "2.0", software ("toofast 1.2.0"), simulated (bool),
│                  finalized, finalized_at
├── sample/        attrs: sample_id, run (= the run folder), timestamp (creation, ISO), measurement_type
│                  (+ any free-form metadata given by the caller)
├── conditions/    reagents (one name per pump), flow_set [uL/s] (commanded), flow_measured [uL/s] (written at the
│                  end of the sample), stock_concentration [mM] (@source, PVP per 40 kDa chain), varying_pumps (bool)
│                  attrs: total_flow_uL_s (configured total flow of the run), residence_time_s
├── instrument/    attrs: spectrometer_model, spectrometer_serial (from the spectrometer driver), n_pixels,
│                  wavelength_min_nm, wavelength_max_nm, integration_time_ms, n_averages, sampling_interval_s,
│                  smoothing_px
├── spectra/       timestamp (N, ISO), elapsed (N) [s] since the first spectrum, wavelength (W) [nm],
│                  extinction (N × W) [a.u.] = −log10(I / I_ref), intensity (N × W) [counts] = detector counts I,
│                  lambda_max (N) [nm], extinction_max (N) [a.u.] (maximum of the Savitzky–Golay(50, 3)-smoothed spectrum)
├── result/        extinction_final (W) [a.u.] = last spectrum; attrs lambda_max_nm, extinction_max (argmax of the raw
│                  last spectrum), n_spectra, growth_time_s (only when the termination rule was met)
└── run_settings/  attrs: experiment_type, target_style, target_wavelength_nm, target_extinction, tolerance,
                   cost_method, acquisition_function, pca_model_file, termination_* (below), wash_acid_volume_uL,
                   wash_water_volume_uL, wash_backpressure_flow (+ any other run parameter)
    └── target_spectrum/{wavelength [nm], extinction [a.u.]}       full-spectrum / PCA targets
```
`toofast.data.sample_file` holds the names and units (`UNITS`), `read_sample` (any layout → the same record),
`write_sample` and `validate_sample`. The spectrometer driver reports its model and serial through
`SpectrometerBackend.identity()`; a driver that does not implement it leaves the two attributes out.

### Older layouts (read by `read_sample`, warm-start included)
| 2.0 | RG-flat (toofast 1.0 / 1.1: later sessions) | SD-hier (2025: earlier sessions) |
|---|---|---|
| `sample@run`, `@sample_id`, `@timestamp`, `@measurement_type` | `metadata@experiment_name`, … | same |
| `conditions/flow_set` | `flows/set_flows` (= `experiment/flows_set`) | `metadata/flows_set` |
| `conditions/flow_measured` | `flows/measured_flows` | `flows/flow_rates` |
| `conditions/reagents`, `varying_pumps` | `experiment/reagents`, `varying_pumps` (`fixed_pumps` = its complement) | — |
| `conditions@total_flow_uL_s`, `@residence_time_s` | `experiment@total_flow`, `@residence_time` | `experimental_parameters@…` |
| `instrument@integration_time_ms`, `n_averages`, `sampling_interval_s`, `smoothing_px` | `experiment@spectro_*` | — |
| `spectra/{timestamp, wavelength, extinction, intensity, lambda_max, extinction_max}` | `spectra/{timestamps, wavelengths, absorbances, transmittances, lambda_max, abs_max}` | `time_series/spectra/…` |
| `result/extinction_final`, `@lambda_max_nm`, `@extinction_max`, `@n_spectra` | `statistics/abs_final`, `@final_lambda_max`, `@final_abs_max`, `@total_spectra_count` | same (+ `abs_mean`, `abs_std` → `result/extinction_mean`, `_std`) |
| `run_settings@…`, `target_wavelength_nm`, `target_extinction` | `experiment@…`, `target_wavelength`, `target_absorbance` | `experimental_parameters@target_style` |
| root `@software`, `@simulated` | `experiment@software`, `@simulated` (1.1 only) | — |

Dropped: `metadata@chip_id` (always empty), the redundant `flows@total_flow`, `@measured_total_flow`,
`@number_of_pumps` (checked to agree before being dropped), and the `images/` group of SD-hier files (PNG renderings).
In the older layouts `transmittances` holds the detector counts (not a ratio): hence `intensity` [counts].

## Termination of a stop-flow measurement
Spectra are taken every `spectro_sampling_rate_s` after the flow stops. After each spectrum, the latest spectrum and
the one ~5 s earlier are smoothed (Savitzky–Golay `savgol_window`, `savgol_polyorder`) and compared: growth is over
when Pearson r > `termination_correlation_threshold` **and** the relative change of the maximum
< `termination_amplitude_threshold`, with more than `termination_min_spectra` spectra — or, if
`termination_allow_early_stop`, as soon as the criteria hold for `termination_early_stop_confirmations` consecutive
checks — or at `termination_timeout_s` (`toofast/core/growth.py`). A spectrum with std(extinction) < the "no AgNP"
threshold (0.01) is a flat spectrum (no particles).
