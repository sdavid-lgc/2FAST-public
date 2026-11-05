"""
Data Manager for 2FAST experiment data.

Writes one HDF5 file per sample, format "2fast-sample" 2.0 (layout, names and units: `toofast.data.sample_file`,
`docs/data_format.md`). The file is written while the sample runs: header groups at creation, one row of the
`spectra/` time series per spectrum, `conditions/flow_measured` and `result/` at finalization.
"""

import os
import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime
from io import BytesIO
import logging
from typing import Optional, Dict, Any, List
import json
from toofast import clock
from toofast.data.sample_file import FORMAT, FORMAT_VERSION, GROUPS, UNITS, elapsed_seconds

# Experiment-level keyword arguments of create_sample_file -> (group, attribute) of the 2.0 file.
# Scalars not listed here go to run_settings/ under their own name (termination_*, wash_*, ...).
_SCALAR_TARGETS = {
    'residence_time': ('conditions', 'residence_time_s'),
    'total_flow': ('conditions', 'total_flow_uL_s'),
    'measurement_type': ('sample', 'measurement_type'),
    'spectrometer_model': ('instrument', 'spectrometer_model'),
    'spectrometer_serial': ('instrument', 'spectrometer_serial'),
    'spectro_integration_time_ms': ('instrument', 'integration_time_ms'),
    'spectro_number_averages': ('instrument', 'n_averages'),
    'spectro_sampling_rate_s': ('instrument', 'sampling_interval_s'),
    'spectro_smoothing_px': ('instrument', 'smoothing_px'),
    'target_wavelength': ('run_settings', 'target_wavelength_nm'),
    'target_absorbance': ('run_settings', 'target_extinction'),
    'software': ('/', 'software'),
    'simulated': ('/', 'simulated'),
}
# Array keyword arguments -> dataset. fixed_pumps is not stored: it is ~varying_pumps.
_ARRAY_TARGETS = {
    'reagents': 'conditions/reagents',
    'varying_pumps': 'conditions/varying_pumps',
    'flows_set': 'conditions/flow_set',
    'stock_concentration': 'conditions/stock_concentration',
}
_NOT_STORED = {'fixed_pumps', 'target_spectrum_wavelengths', 'target_spectrum_absorbance'}


class DataManager:
    """Centralized manager for saving and loading experimental data (HDF5 + CSV fallback)."""

    def __init__(self, experiment_path: str, experiment_name: str = "",
                 logger: Optional[logging.Logger] = None):
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_path
        self.logger = logger or logging.getLogger(__name__)

        self._create_directory_structure()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _create_directory_structure(self):
        """Create directory structure for the experiment."""
        for directory in ('samples', 'figures', 'raw_data'):
            os.makedirs(os.path.join(self.experiment_dir, directory), exist_ok=True)

    def _sample_filepath(self, sample_id: str) -> str:
        filename = f"{self.experiment_name}_{sample_id}.h5"
        return os.path.join(self.experiment_dir, 'samples', filename)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_sample_file(self, sample_id: str, metadata: Optional[Dict] = None,
                           **kwargs) -> str:
        """
        Create an HDF5 file for a sample.

        Parameters
        ----------
        sample_id : str
            Unique identifier for the sample (e.g. "init_0", "sample_3").
        metadata : dict, optional
            Free-form metadata of the sample (measurement_type, operator, ...) -> /sample attributes.
        **kwargs
            Experiment-level parameters forwarded from the Experiment class (names: _SCALAR_TARGETS,
            _ARRAY_TARGETS; other scalars and arrays go to /run_settings), e.g. experiment_type,
            residence_time, total_flow, reagents, stock_concentration (mM), flows_set, varying_pumps,
            spectrometer_model, spectro_*, target_*, termination_*, wash_*, software, simulated.

        Returns
        -------
        str
            Absolute path to the created HDF5 file.
        """
        filepath = self._sample_filepath(sample_id)

        try:
            with h5py.File(filepath, 'w') as h5:
                h5.attrs['format'] = FORMAT
                h5.attrs['format_version'] = FORMAT_VERSION
                for g in GROUPS:
                    h5.create_group(g)
                self._write_sample(h5, sample_id, metadata)
                self._write_experiment_params(h5, **kwargs)
                # --- extensible time series ---
                h5['spectra'].create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype='S32')

            self.logger.info(f"Sample file created: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error creating file for {sample_id}: {e}")
            raise

    def add_spectrum_to_sample(self, sample_id: str, spectrum,
                               timestamp: Optional[str] = None) -> bool:
        """
        Append a spectrum to the sample's time-series.

        Parameters
        ----------
        sample_id : str
            Sample identifier matching a previously created file.
        spectrum : Spectrum
            Spectrum object (must expose .wavelengths, .absorbance / .absorbances,
            optionally .transmittance, .lamb_max, .abs_max).
        timestamp : str, optional
            ISO-format timestamp; auto-generated if omitted.

        Returns
        -------
        bool
            True if the spectrum was successfully appended.
        """
        filepath = self._sample_filepath(sample_id)
        if not os.path.exists(filepath):
            self.logger.error(f"File not found for {sample_id}. Call create_sample_file() first.")
            return False

        if timestamp is None:
            timestamp = clock.now().isoformat()

        try:
            with h5py.File(filepath, 'a') as h5:
                grp = h5['spectra']
                ts_ds = grp['timestamp']
                idx = ts_ds.shape[0]

                # Timestamp, and seconds since the first spectrum
                ts_ds.resize((idx + 1,))
                ts_ds[idx] = timestamp.encode('utf-8')
                elapsed = float(elapsed_seconds([ts_ds[0], ts_ds[idx]])[-1])
                self._append_scalar(grp, 'elapsed', elapsed, idx)

                # Extract arrays from the Spectrum object
                wavelengths = getattr(spectrum, 'wavelengths', None)
                extinction = getattr(spectrum, 'absorbances', None)
                if extinction is None:
                    extinction = getattr(spectrum, 'absorbance', None)
                intensity = getattr(spectrum, 'transmittance', None)    # detector counts of the sample

                # First spectrum: create wavelength + data datasets
                if idx == 0 and wavelengths is not None:
                    wl = np.asarray(wavelengths, dtype=float)
                    self._dataset(grp, 'wavelength', data=wl)
                    h5['instrument'].attrs.update(n_pixels=int(len(wl)), wavelength_min_nm=float(wl[0]),
                                                  wavelength_max_nm=float(wl[-1]))
                    for name, data in (('extinction', extinction), ('intensity', intensity)):
                        if data is not None:
                            self._dataset(grp, name, shape=(1, len(data)), maxshape=(None, len(data)),
                                          dtype=np.float64, data=np.asarray(data).reshape(1, -1))
                else:
                    self._append_2d(grp, 'extinction', extinction, idx)
                    self._append_2d(grp, 'intensity', intensity, idx)

                # Scalar per-spectrum values (argmax of the Savitzky-Golay(50, 3)-smoothed spectrum, see Spectrum)
                self._append_scalar(grp, 'lambda_max', getattr(spectrum, 'lamb_max', None), idx)
                self._append_scalar(grp, 'extinction_max', getattr(spectrum, 'abs_max', None), idx)

            self.logger.debug(f"Spectrum #{idx + 1} added to {sample_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding spectrum to {sample_id}: {e}")
            return False

    def finalize_sample(self, sample_id: str, flows: np.ndarray = None,
                        growth_time_s: Optional[float] = None) -> bool:
        """
        Finalize a sample: store measured flows, compute the result, save figures.

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        flows : np.ndarray, optional
            Actual measured flow rates for this sample.
        growth_time_s : float, optional
            Growth (termination) time of the sample; None if growth never met the termination rule.

        Returns
        -------
        bool
            True if finalization succeeded.
        """
        filepath = self._sample_filepath(sample_id)

        try:
            with h5py.File(filepath, 'a') as h5:
                # Measured (actual) flows
                if flows is not None:
                    cond = h5.require_group('conditions')
                    if 'flow_measured' in cond:
                        del cond['flow_measured']
                    self._dataset(cond, 'flow_measured', data=np.asarray(flows, dtype=float))

                # Result
                self._compute_statistics(h5)
                if growth_time_s is not None:
                    h5['result'].attrs['growth_time_s'] = float(growth_time_s)

                # Save figures to disk (not inside H5)
                self._save_figures_to_disk(h5, sample_id)

                # Mark finalized
                h5.attrs['finalized'] = True
                h5.attrs['finalized_at'] = clock.now().isoformat()

            self.logger.info(f"Sample {sample_id} finalized")
            return True

        except Exception as e:
            self.logger.error(f"Error finalizing {sample_id}: {e}")
            return False

    def load_sample_h5(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Load all data from a sample HDF5 file into a nested dict.

        Returns
        -------
        dict or None
            ``{group_name: {dataset_name: ndarray, 'attrs': dict}, ...}``
        """
        filepath = self._sample_filepath(sample_id)
        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            return None

        try:
            data = {}
            with h5py.File(filepath, 'r') as h5:
                data['attrs'] = dict(h5.attrs)
                for group_name in h5.keys():
                    item = h5[group_name]
                    if isinstance(item, h5py.Group):
                        data[group_name] = {'attrs': dict(item.attrs)}
                        for ds_name in item.keys():
                            ds = item[ds_name]
                            if isinstance(ds, h5py.Dataset):
                                data[group_name][ds_name] = ds[:]
                            elif isinstance(ds, h5py.Group):
                                # Nested group (shouldn't happen often)
                                data[group_name][ds_name] = {'attrs': dict(ds.attrs)}
                    elif isinstance(item, h5py.Dataset):
                        data[group_name] = item[:]
            return data

        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None

    def list_samples(self) -> List[str]:
        """Return sorted list of sample IDs found in the samples directory."""
        samples_dir = os.path.join(self.experiment_dir, 'samples')
        if not os.path.exists(samples_dir):
            return []
        samples = []
        for fname in os.listdir(samples_dir):
            if fname.endswith('.h5') and fname.startswith(self.experiment_name):
                # strip prefix and suffix
                sample_id = fname[len(self.experiment_name) + 1:].replace('.h5', '')
                samples.append(sample_id)
        return sorted(samples)

    def save_experiment_summary(self, summary_data: Dict[str, Any]):
        """Save an experiment-level summary as JSON."""
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(self._prepare_for_json(summary_data), f, indent=2, default=str)
            self.logger.info(f"Summary saved: {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")

    def save_csv_fallback(self, spectrum, flows: np.ndarray, sample_id: str) -> bool:
        """Fallback: save spectrum + flows as CSV if HDF5 fails."""
        try:
            if spectrum and hasattr(spectrum, 'save_data'):
                spectrum.save_data(os.path.join(self.experiment_dir, 'raw_data'), sample_id)
            if flows is not None:
                path = os.path.join(self.experiment_dir, 'raw_data', f"flows_{sample_id}.csv")
                np.savetxt(path, flows, delimiter=';', header='Flow_rates_uL_per_s', comments='')
            self.logger.info(f"CSV fallback saved for {sample_id}")
            return True
        except Exception as e:
            self.logger.error(f"CSV fallback error: {e}")
            return False

    # ------------------------------------------------------------------
    # Private: H5 write helpers
    # ------------------------------------------------------------------
    def _write_sample(self, h5, sample_id: str, metadata: Optional[Dict]):
        """Write the /sample attributes (identity of the sample + free-form metadata)."""
        grp = h5['sample']
        grp.attrs['sample_id'] = sample_id
        grp.attrs['run'] = self.experiment_name
        grp.attrs['timestamp'] = clock.now().isoformat()
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    grp.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    self._dataset(grp, key, data=value)

    def _write_experiment_params(self, h5, **kwargs):
        """
        Write the experiment-level parameters into /conditions, /instrument, /run_settings and the root.

        Known keys are mapped by _SCALAR_TARGETS / _ARRAY_TARGETS; any other scalar or array goes to
        /run_settings under its own name, so no information is lost.
        """
        for key, val in kwargs.items():
            if val is None or key in _NOT_STORED:
                continue
            try:
                if key in _ARRAY_TARGETS:
                    group, name = _ARRAY_TARGETS[key].split('/')
                    self._dataset(h5[group], name, data=val)
                elif isinstance(val, (str, int, float, bool, np.generic)):
                    group, name = _SCALAR_TARGETS.get(key, ('run_settings', key))
                    (h5 if group == '/' else h5[group]).attrs[name] = val
                elif isinstance(val, (list, tuple, np.ndarray)):
                    self._dataset(h5['run_settings'], key, data=val)
                else:
                    self.logger.debug(f"experiment parameter {key!r} not stored (unsupported type {type(val).__name__})")
            except Exception as e:
                self.logger.debug(f"experiment parameter {key!r} not stored ({e})")

        conc = h5['conditions'].get('stock_concentration')
        if conc is not None:
            conc.attrs['source'] = 'GUI (pump cards)'
            if 'PVP' in [r.decode() if isinstance(r, bytes) else str(r)
                         for r in h5['conditions'].get('reagents', [])]:
                conc.attrs['note'] = 'PVP: per polymer chain (40 kDa)'

        # Target spectrum curve -> dedicated subgroup (full_spectrum / pca).
        tw = kwargs.get('target_spectrum_wavelengths')
        ta = kwargs.get('target_spectrum_absorbance')
        if tw is not None and ta is not None:
            tgrp = h5['run_settings'].create_group('target_spectrum')
            self._dataset(tgrp, 'wavelength', data=np.asarray(tw, dtype=float))
            self._dataset(tgrp, 'extinction', data=np.asarray(ta, dtype=float))

    @staticmethod
    def _dataset(grp, name, data=None, **kwargs):
        """create_dataset + the unit of the 2.0 format (byte strings for text arrays)."""
        if data is not None:
            arr = np.asarray(data)
            if arr.dtype.kind == 'U':
                arr = arr.astype('S')      # h5py needs byte-strings for variable-length string arrays
            data = arr
        ds = grp.create_dataset(name, data=data, **kwargs)
        unit = UNITS.get(ds.name.lstrip('/'))
        if unit:
            ds.attrs['unit'] = unit
        return ds

    @staticmethod
    def _append_2d(grp, name, data, idx):
        """Append a 1-D array as a new row to an existing 2-D dataset."""
        if data is not None and name in grp:
            ds = grp[name]
            ds.resize((idx + 1, ds.shape[1]))
            ds[idx, :] = data

    @staticmethod
    def _append_scalar(grp, name, value, idx):
        """Append a scalar value to an extensible 1-D dataset (create if needed)."""
        if value is None:
            return
        if name not in grp:
            DataManager._dataset(grp, name, shape=(1,), maxshape=(None,), dtype=np.float64, data=[value])
        else:
            ds = grp[name]
            ds.resize((idx + 1,))
            ds[idx] = value

    # ------------------------------------------------------------------
    # Private: statistics + figures
    # ------------------------------------------------------------------
    def _compute_statistics(self, h5):
        """Fill /result from the time series: final spectrum, its maximum, number of spectra."""
        if 'spectra' not in h5:
            return
        spectra = h5['spectra']
        res = h5.require_group('result')
        if 'extinction_final' in res:
            del res['extinction_final']

        if 'extinction' in spectra:
            final = spectra['extinction'][-1, :]
            self._dataset(res, 'extinction_final', data=final)
            if 'wavelength' in spectra:
                idx = np.argmax(final)
                res.attrs['lambda_max_nm'] = float(spectra['wavelength'][idx])
                res.attrs['extinction_max'] = float(final[idx])

        if 'timestamp' in spectra:
            res.attrs['n_spectra'] = int(spectra['timestamp'].shape[0])

    def _save_figures_to_disk(self, h5, sample_id: str):
        """
        Generate summary figures and save them as PNG files in the figures/ directory.

        This avoids bloating the H5 file with binary image data.
        """
        try:
            if 'spectra' not in h5:
                return
            spectra = h5['spectra']
            if 'extinction' not in spectra or 'wavelength' not in spectra:
                return

            wl = spectra['wavelength'][:]
            abs_data = spectra['extinction'][:]
            n = abs_data.shape[0]
            fig_dir = os.path.join(self.experiment_dir, 'figures')

            # 1. Time evolution
            fig1 = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig1)
            ax1 = fig1.add_subplot(111)
            if n <= 10:
                for i in range(n):
                    alpha = 0.3 + 0.7 * (i / max(1, n - 1))
                    ax1.plot(wl, abs_data[i, :], alpha=alpha,
                             label=f't={i + 1}' if n <= 5 else None)
            else:
                indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
                for idx in indices:
                    ax1.plot(wl, abs_data[idx, :], label=f'#{idx + 1}', linewidth=1.5)
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Extinction (a.u.)')
            ax1.set_title(f'Time Evolution - {sample_id}')
            ax1.grid(True, alpha=0.3)
            if ax1.get_legend_handles_labels()[0]:
                ax1.legend()
            fig1.savefig(os.path.join(fig_dir, f'{sample_id}_time_evolution.png'),
                         dpi=150, bbox_inches='tight')
            fig1.clear()

            # 2. Final spectrum
            fig2 = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig2)
            ax2 = fig2.add_subplot(111)
            final = abs_data[-1, :]
            ax2.plot(wl, final, 'b-', linewidth=2, label='Final spectrum')
            lm_idx = np.argmax(final)
            ax2.plot(wl[lm_idx], final[lm_idx], 'ro', markersize=8,
                     label=f'$\\lambda_{{max}}$ = {wl[lm_idx]:.1f} nm')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Extinction (a.u.)')
            ax2.set_title(f'Final Spectrum - {sample_id}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            fig2.savefig(os.path.join(fig_dir, f'{sample_id}_final_spectrum.png'),
                         dpi=150, bbox_inches='tight')
            fig2.clear()

        except Exception as e:
            self.logger.warning(f"Could not save figures for {sample_id}: {e}")

    # ------------------------------------------------------------------
    # Private: JSON helper
    # ------------------------------------------------------------------
    def _prepare_for_json(self, data):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.generic):
            return data.item()
        return data
