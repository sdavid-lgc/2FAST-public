"""
Per-sample HDF5 file, format "2fast-sample" 2.0: names, units, and one reader for every layout ever written.

Layouts:
- ``"2.0"``     toofast >= 1.2 (written by DataManager, or by `write_sample` from a converted older file);
- ``"RG-flat"`` toofast 1.0 / 1.1 (2026 runs: later sessions): ``spectra/…``, ``flows/set_flows``, ``experiment@…``;
- ``"SD-hier"`` 2025 runs (earlier sessions): ``time_series/spectra/…``, ``metadata/flows_set``, ``experimental_parameters@…``.

`read_sample` returns the same `SampleRecord` (2.0 names) for the three; `write_sample` writes a record as a 2.0 file.
The mapping of the older layouts is explicit (tables below): every array is copied unchanged, the attributes and
datasets that are redundant in the older layouts are checked to be redundant before being dropped, and anything not
in the tables is reported in `SampleRecord.unmapped` (``strict=True`` raises instead).

2.0 layout (units: ``unit`` attribute on each dataset, suffix of the attribute name for scalars)::

    @format="2fast-sample" @format_version="2.0" @software @simulated @finalized @finalized_at
    sample/        @sample_id @run @timestamp @measurement_type (+ free-form metadata)
    conditions/    reagents  flow_set [uL/s]  flow_measured [uL/s]  stock_concentration [mM]  varying_pumps
                   @total_flow_uL_s (configured total flow of the run)  @residence_time_s
    instrument/    @spectrometer_model @spectrometer_serial @n_pixels @wavelength_min_nm @wavelength_max_nm
                   @integration_time_ms @n_averages @sampling_interval_s @smoothing_px
    spectra/       timestamp (ISO)  elapsed [s]  wavelength [nm]  extinction (N x W) [a.u.]  intensity (N x W) [counts]
                   lambda_max [nm]  extinction_max [a.u.]
    result/        extinction_final [a.u.]  @lambda_max_nm @extinction_max @n_spectra @growth_time_s
    run_settings/  @experiment_type @target_style @target_wavelength_nm @target_extinction @tolerance @cost_method
                   @acquisition_function @pca_model_file @termination_* @wash_*
                   target_spectrum/{wavelength [nm], extinction [a.u.]}
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

FORMAT = "2fast-sample"
FORMAT_VERSION = "2.0"

GROUPS = ("sample", "conditions", "instrument", "spectra", "result", "run_settings")

UNITS = {
    "conditions/flow_set": "uL/s",
    "conditions/flow_measured": "uL/s",
    "conditions/stock_concentration": "mM",
    "spectra/elapsed": "s",
    "spectra/wavelength": "nm",
    "spectra/extinction": "a.u.",
    "spectra/intensity": "counts",
    "spectra/lambda_max": "nm",
    "spectra/extinction_max": "a.u.",
    "result/extinction_final": "a.u.",
    "result/extinction_mean": "a.u.",
    "result/extinction_std": "a.u.",
    "run_settings/target_spectrum/wavelength": "nm",
    "run_settings/target_spectrum/extinction": "a.u.",
}

# --- older layouts -> 2.0 ------------------------------------------------------------------------------------------
# datasets: old path -> new path
_RG_DATASETS = {
    "experiment/reagents": "conditions/reagents",
    "experiment/varying_pumps": "conditions/varying_pumps",
    "flows/set_flows": "conditions/flow_set",
    "flows/measured_flows": "conditions/flow_measured",
    "experiment/target_spectrum/wavelengths": "run_settings/target_spectrum/wavelength",
    "experiment/target_spectrum/absorbance": "run_settings/target_spectrum/extinction",
    "spectra/timestamps": "spectra/timestamp",
    "spectra/wavelengths": "spectra/wavelength",
    "spectra/absorbances": "spectra/extinction",
    "spectra/transmittances": "spectra/intensity",
    "spectra/lambda_max": "spectra/lambda_max",
    "spectra/abs_max": "spectra/extinction_max",
    "statistics/abs_final": "result/extinction_final",
}
_SD_DATASETS = {
    "metadata/flows_set": "conditions/flow_set",
    "flows/flow_rates": "conditions/flow_measured",
    **{"time_series/" + k: v for k, v in _RG_DATASETS.items() if k.startswith("spectra/")},
    "statistics/abs_final": "result/extinction_final",
    "statistics/abs_mean": "result/extinction_mean",
    "statistics/abs_std": "result/extinction_std",
}
# attributes: (old group, name) -> (new group, name); group "/" = root
_COMMON_ATTRS = {
    ("/", "finalized"): ("/", "finalized"),
    ("/", "finalized_at"): ("/", "finalized_at"),
    ("metadata", "sample_id"): ("sample", "sample_id"),
    ("metadata", "timestamp"): ("sample", "timestamp"),
    ("metadata", "experiment_name"): ("sample", "run"),
    ("metadata", "measurement_type"): ("sample", "measurement_type"),
    ("statistics", "final_lambda_max"): ("result", "lambda_max_nm"),
    ("statistics", "final_abs_max"): ("result", "extinction_max"),
    ("statistics", "total_spectra_count"): ("result", "n_spectra"),
}
_RG_ATTRS = {
    **_COMMON_ATTRS,
    ("experiment", "residence_time"): ("conditions", "residence_time_s"),
    ("experiment", "total_flow"): ("conditions", "total_flow_uL_s"),
    ("experiment", "spectro_integration_time_ms"): ("instrument", "integration_time_ms"),
    ("experiment", "spectro_number_averages"): ("instrument", "n_averages"),
    ("experiment", "spectro_sampling_rate_s"): ("instrument", "sampling_interval_s"),
    ("experiment", "spectro_smoothing_px"): ("instrument", "smoothing_px"),
    ("experiment", "target_wavelength"): ("run_settings", "target_wavelength_nm"),
    ("experiment", "target_absorbance"): ("run_settings", "target_extinction"),
    ("experiment", "software"): ("/", "software"),
    ("experiment", "simulated"): ("/", "simulated"),
    **{("experiment", k): ("run_settings", k) for k in (
        "experiment_type", "target_style", "tolerance", "cost_method", "acquisition_function", "pca_model_file",
        "wash_acid_volume_uL", "wash_water_volume_uL", "wash_backpressure_flow",
        "termination_correlation_threshold", "termination_amplitude_threshold", "termination_min_spectra",
        "termination_timeout_s", "termination_allow_early_stop", "termination_early_stop_confirmations",
        "termination_savgol_window", "termination_savgol_polyorder")},
}
_SD_ATTRS = {
    **_COMMON_ATTRS,
    ("experimental_parameters", "residence_time"): ("conditions", "residence_time_s"),
    ("experimental_parameters", "total_flow"): ("conditions", "total_flow_uL_s"),
    ("experimental_parameters", "target_style"): ("run_settings", "target_style"),
}
# Redundant in the older layouts: dropped after checking (see _check_redundant).
_RG_REDUNDANT_DATASETS = ("experiment/flows_set", "experiment/fixed_pumps")
_RG_REDUNDANT_ATTRS = (("flows", "total_flow"), ("flows", "measured_total_flow"), ("flows", "number_of_pumps"))
_SD_REDUNDANT_ATTRS = (("flows", "total_flow"),)
# Not data: PNG renderings of the spectra stored inside the 2025 files (the figures/ folder has the same plots).
_SD_IGNORED_PREFIX = "images"


@dataclass
class SampleRecord:
    """One sample in 2.0 names: ``datasets["spectra/extinction"]``, ``attrs["run_settings"]["cost_method"]``."""

    layout: str
    datasets: Dict[str, np.ndarray] = field(default_factory=dict)
    attrs: Dict[str, dict] = field(default_factory=lambda: {g: {} for g in ("/",) + GROUPS})
    dataset_attrs: Dict[str, dict] = field(default_factory=dict)
    unmapped: List[str] = field(default_factory=list)

    @property
    def flow_set(self) -> Optional[np.ndarray]:
        return self.datasets.get("conditions/flow_set")

    @property
    def flow_measured(self) -> Optional[np.ndarray]:
        return self.datasets.get("conditions/flow_measured")

    @property
    def wavelength(self) -> Optional[np.ndarray]:
        return self.datasets.get("spectra/wavelength")

    @property
    def extinction_final(self) -> Optional[np.ndarray]:
        """result/extinction_final, or the last spectrum of the series if it is missing."""
        if "result/extinction_final" in self.datasets:
            return self.datasets["result/extinction_final"]
        ext = self.datasets.get("spectra/extinction")
        return None if ext is None or len(ext) == 0 else ext[-1]

    @property
    def reagents(self) -> List[str]:
        return decode_strings(self.datasets.get("conditions/reagents", []))


def decode_strings(values) -> List[str]:
    return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in values]


def elapsed_seconds(timestamps) -> np.ndarray:
    """Seconds since the first spectrum, from ISO timestamps (NaN where a timestamp cannot be parsed)."""
    def parse(v):
        try:
            return datetime.datetime.fromisoformat(v.decode() if isinstance(v, bytes) else str(v))
        except ValueError:
            return None
    ts = [parse(v) for v in timestamps]
    if not ts or ts[0] is None:
        return np.full(len(ts), np.nan)
    return np.array([(t - ts[0]).total_seconds() if t is not None else np.nan for t in ts])


def detect_layout(h5) -> Optional[str]:
    if h5.attrs.get("format") == FORMAT:
        return str(h5.attrs.get("format_version"))
    if "time_series" in h5:
        return "SD-hier"
    if "spectra" in h5 and ("experiment" in h5 or "flows" in h5):
        return "RG-flat"
    return None


def read_sample(path: str, strict: bool = False) -> SampleRecord:
    """Read a sample file of any layout into a 2.0 `SampleRecord`.

    For older layouts, the derived fields of 2.0 that follow exactly from the stored data are added
    (`spectra/elapsed`, `instrument@n_pixels`, `@wavelength_min_nm`, `@wavelength_max_nm`); nothing else is added.
    """
    import h5py

    with h5py.File(path, "r") as h5:
        layout = detect_layout(h5)
        if layout is None:
            raise ValueError(f"{path}: not a 2FAST sample file")
        if layout == FORMAT_VERSION:
            return _read_v2(h5)
        if layout == "RG-flat":
            rec = _read_legacy(h5, layout, _RG_DATASETS, _RG_ATTRS, _RG_REDUNDANT_DATASETS, _RG_REDUNDANT_ATTRS)
        else:
            rec = _read_legacy(h5, layout, _SD_DATASETS, _SD_ATTRS, (), _SD_REDUNDANT_ATTRS)
    _add_derived(rec)
    if strict and rec.unmapped:
        raise ValueError(f"{path}: not mapped to format 2.0: {rec.unmapped}")
    return rec


def _read_v2(h5) -> SampleRecord:
    import h5py

    rec = SampleRecord(layout=FORMAT_VERSION)
    rec.attrs["/"] = {k: v for k, v in h5.attrs.items() if k not in ("format", "format_version")}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            rec.datasets[name] = obj[()]
            extra = {k: v for k, v in obj.attrs.items() if not (k == "unit" and UNITS.get(name) == v)}
            if extra:
                rec.dataset_attrs[name] = extra
        elif name in rec.attrs or "/" not in name:
            rec.attrs[name] = dict(obj.attrs)
        elif obj.attrs:
            rec.unmapped.append(name + "@")
    h5.visititems(visit)
    return rec


def _read_legacy(h5, layout, ds_map, attr_map, redundant_ds, redundant_attrs) -> SampleRecord:
    import h5py

    rec = SampleRecord(layout=layout)
    seen = set()
    for old, new in ds_map.items():
        if old in h5:
            rec.datasets[new] = h5[old][()]
            seen.add(old)
    for (grp, name), (ngrp, nname) in attr_map.items():
        g = h5 if grp == "/" else h5.get(grp)
        if g is not None and name in g.attrs:
            rec.attrs[ngrp][nname] = g.attrs[name]
    chip = h5["metadata"].attrs.get("chip_id", "") if "metadata" in h5 else ""
    if chip not in ("", b""):
        rec.attrs["sample"]["chip_id"] = chip          # always empty in the data of the paper
    _check_redundant(h5, rec, layout)
    seen.update(redundant_ds)
    known_attrs = set(attr_map) | set(redundant_attrs) | {("metadata", "chip_id")}

    def visit(name, obj):
        if name.split("/")[0] == _SD_IGNORED_PREFIX and layout == "SD-hier":
            return
        if isinstance(obj, h5py.Dataset):
            if name not in seen:
                rec.unmapped.append(name)
        for a in obj.attrs:
            if (name, a) not in known_attrs:
                rec.unmapped.append(f"{name}@{a}")
    h5.visititems(visit)
    rec.unmapped += [f"/@{a}" for a in h5.attrs if ("/", a) not in known_attrs]
    return rec


def _check_redundant(h5, rec, layout):
    """The older layouts store some values twice; check they agree before keeping one copy."""
    problems = []
    fs = rec.flow_set
    if layout == "RG-flat":
        if "experiment/flows_set" in h5 and not np.array_equal(h5["experiment/flows_set"][()], fs):
            problems.append("experiment/flows_set != flows/set_flows")
        if "experiment/fixed_pumps" in h5 and "experiment/varying_pumps" in h5 and not np.array_equal(
                h5["experiment/fixed_pumps"][()], ~h5["experiment/varying_pumps"][()]):
            problems.append("fixed_pumps != ~varying_pumps")
    fl = h5["flows"].attrs if "flows" in h5 else {}
    if "total_flow" in fl:
        if layout == "RG-flat":
            if not float(fl["total_flow"]) == float(rec.attrs["conditions"].get("total_flow_uL_s", np.nan)):
                problems.append("flows@total_flow != experiment@total_flow")
        elif rec.flow_measured is not None and not np.isclose(fl["total_flow"], np.sum(rec.flow_measured)):
            problems.append("flows@total_flow != sum(flows/flow_rates)")
    if "measured_total_flow" in fl and rec.flow_measured is not None and not np.isclose(
            fl["measured_total_flow"], np.sum(rec.flow_measured)):
        problems.append("flows@measured_total_flow != sum(measured_flows)")
    if "number_of_pumps" in fl and fs is not None and int(fl["number_of_pumps"]) != len(fs):
        problems.append("flows@number_of_pumps != len(set_flows)")
    if problems:
        raise ValueError(f"{h5.filename}: redundant values disagree: {problems}")


def _add_derived(rec: SampleRecord):
    if "spectra/timestamp" in rec.datasets:
        rec.datasets["spectra/elapsed"] = elapsed_seconds(rec.datasets["spectra/timestamp"])
    wl = rec.wavelength
    if wl is not None and len(wl):
        rec.attrs["instrument"].update(n_pixels=int(len(wl)), wavelength_min_nm=float(wl[0]),
                                       wavelength_max_nm=float(wl[-1]))


def write_sample(path: str, rec: SampleRecord, root_attrs: Optional[dict] = None) -> None:
    """Write a record as a 2.0 file (all groups present, units on the datasets)."""
    import h5py

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = FORMAT
        h5.attrs["format_version"] = FORMAT_VERSION
        for k, v in {**rec.attrs.get("/", {}), **(root_attrs or {})}.items():
            h5.attrs[k] = v
        for g in GROUPS:
            h5.require_group(g)
        for name, data in rec.datasets.items():
            arr = np.asarray(data)
            if arr.dtype.kind == "U":
                arr = arr.astype("S")
            ds = h5.create_dataset(name, data=arr)
            if name in UNITS:
                ds.attrs["unit"] = UNITS[name]
            for k, v in rec.dataset_attrs.get(name, {}).items():
                ds.attrs[k] = v
        for g, attrs in rec.attrs.items():
            if g == "/":
                continue
            grp = h5.require_group(g)
            for k, v in attrs.items():
                grp.attrs[k] = v


def validate_sample(path: str) -> List[str]:
    """Problems of a 2.0 file against the format (empty list = conforms)."""
    import h5py

    problems = []
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != FORMAT or h5.attrs.get("format_version") != FORMAT_VERSION:
            return [f"not a {FORMAT} {FORMAT_VERSION} file"]
        problems += [f"missing group {g}" for g in GROUPS if g not in h5]
        for name, unit in UNITS.items():
            if name in h5 and h5[name].attrs.get("unit") != unit:
                problems.append(f"{name}: unit {h5[name].attrs.get('unit')!r}, expected {unit!r}")
        for name in ("sample_id", "run", "timestamp"):
            if name not in h5.get("sample", {}).attrs:
                problems.append(f"missing sample@{name}")
        sp = h5.get("spectra")
        if sp is not None and "extinction" in sp:
            n = sp["extinction"].shape[0]
            for name in ("timestamp", "elapsed", "lambda_max", "extinction_max", "intensity"):
                if name in sp and sp[name].shape[0] != n:
                    problems.append(f"spectra/{name}: {sp[name].shape[0]} rows, extinction has {n}")
    return problems
