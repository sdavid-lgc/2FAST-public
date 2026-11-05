"""
Data Manager for 2FAST experiment data.

Manages HDF5 sample files with the following structure:

    sample_file.h5
    ├── metadata/                     (attrs: sample_id, timestamp, experiment_name)
    │   └── chip_id, measurement_type, operator, ...   (from caller metadata dict)
    │
    ├── experiment/                    (attrs: all experiment-level parameters)
    │   ├── experiment_type            "Fixed", "Random", "LHS Flows", "CSV Flows",
    │   │                              "Bayesian Optimization"
    │   ├── residence_time             float  (seconds)
    │   ├── total_flow                 float  (uL/s)
    │   ├── reagents                   dataset  (string array, one per pump)
    │   ├── target_style               "single_point", "full_spectrum", "lambda_max", "pca"
    │   ├── target_wavelength          float  (nm)
    │   ├── target_absorbance          float
    │   ├── pca_model_file             str   (PCA decomposition basis used, pca style)
    │   ├── target_spectrum/           group (full_spectrum / pca targets)
    │   │   ├── wavelengths            dataset  (nm)
    │   │   └── absorbance             dataset  (target curve)
    │   ├── tolerance                  float
    │   ├── cost_method                "L2", "cosine", ...
    │   ├── acquisition_function       "EI", "LCB", "TS"
    │   ├── fixed_pumps                dataset  (bool array)
    │   ├── varying_pumps              dataset  (bool array)
    │   ├── spectro_integration_time_ms  float
    │   ├── spectro_number_averages      int
    │   ├── spectro_smoothing_px         int
    │   ├── spectro_sampling_rate_s      float
    │   ├── wash_acid_volume_uL          float
    │   ├── wash_water_volume_uL         float
    │   └── wash_backpressure_flow       float
    │
    ├── flows/
    │   ├── set_flows                  dataset  (flows requested for this sample)
    │   ├── measured_flows             dataset  (actual flows, written at finalization)
    │   └── (attrs: total_flow, number_of_pumps)
    │
    ├── spectra/                       (extensible time-series)
    │   ├── timestamps                 dataset  (S32 strings, ISO datetime)
    │   ├── wavelengths                dataset  (1-D, written on first spectrum)
    │   ├── absorbances                dataset  (N x W, extensible along axis 0)
    │   ├── transmittances             dataset  (N x W, extensible along axis 0)
    │   ├── lambda_max                 dataset  (N,)
    │   └── abs_max                    dataset  (N,)
    │
    ├── statistics/                    (computed at finalization)
    │   ├── abs_final                  dataset  (last absorbance spectrum)
    │   └── (attrs: final_lambda_max, final_abs_max, total_spectra_count)
    │
    └── (root attrs: finalized, finalized_at)
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
            Free-form metadata (chip_id, measurement_type, operator, ...).
        **kwargs
            Experiment-level parameters forwarded from the Experiment class.
            Recognised keys: experiment_type, residence_time, total_flow,
            reagents, target_style, target_wavelength, target_absorbance,
            tolerance, cost_method, acquisition_function, fixed_pumps,
            varying_pumps, flows_set, and any extra key.

        Returns
        -------
        str
            Absolute path to the created HDF5 file.
        """
        filepath = self._sample_filepath(sample_id)

        try:
            with h5py.File(filepath, 'w') as h5:
                # --- metadata ---
                self._write_metadata(h5, sample_id, metadata)
                # --- experiment parameters ---
                self._write_experiment_params(h5, **kwargs)
                # --- initial (set) flows ---
                self._write_set_flows(h5, **kwargs)
                # --- extensible spectra group ---
                spectra = h5.create_group('spectra')
                spectra.create_dataset('timestamps', shape=(0,), maxshape=(None,),
                                       dtype='S32')

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
            timestamp = datetime.now().isoformat()

        try:
            with h5py.File(filepath, 'a') as h5:
                grp = h5['spectra']
                ts_ds = grp['timestamps']
                idx = ts_ds.shape[0]

                # Timestamp
                ts_ds.resize((idx + 1,))
                ts_ds[idx] = timestamp.encode('utf-8')

                # Extract arrays from the Spectrum object
                wavelengths = getattr(spectrum, 'wavelengths', None)
                absorbances = getattr(spectrum, 'absorbances', None) or getattr(spectrum, 'absorbance', None)
                transmittances = getattr(spectrum, 'transmittance', None)

                # First spectrum: create wavelength + data datasets
                if idx == 0 and wavelengths is not None:
                    grp.create_dataset('wavelengths', data=wavelengths)

                    if absorbances is not None:
                        grp.create_dataset('absorbances',
                                           shape=(1, len(absorbances)),
                                           maxshape=(None, len(absorbances)),
                                           dtype=np.float64,
                                           data=absorbances.reshape(1, -1))

                    if transmittances is not None:
                        grp.create_dataset('transmittances',
                                           shape=(1, len(transmittances)),
                                           maxshape=(None, len(transmittances)),
                                           dtype=np.float64,
                                           data=transmittances.reshape(1, -1))
                else:
                    self._append_2d(grp, 'absorbances', absorbances, idx)
                    self._append_2d(grp, 'transmittances', transmittances, idx)

                # Scalar per-spectrum values
                self._append_scalar(grp, 'lambda_max', getattr(spectrum, 'lamb_max', None), idx)
                self._append_scalar(grp, 'abs_max', getattr(spectrum, 'abs_max', None), idx)

            self.logger.debug(f"Spectrum #{idx + 1} added to {sample_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding spectrum to {sample_id}: {e}")
            return False

    def finalize_sample(self, sample_id: str, flows: np.ndarray = None) -> bool:
        """
        Finalize a sample: store measured flows, compute statistics, save figures.

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        flows : np.ndarray, optional
            Actual measured flow rates for this sample.

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
                    flow_grp = h5.require_group('flows')
                    if 'measured_flows' in flow_grp:
                        del flow_grp['measured_flows']
                    flow_grp.create_dataset('measured_flows', data=flows)
                    flow_grp.attrs['measured_total_flow'] = float(np.sum(flows))

                # Statistics
                self._compute_statistics(h5)

                # Save figures to disk (not inside H5)
                self._save_figures_to_disk(h5, sample_id)

                # Mark finalized
                h5.attrs['finalized'] = True
                h5.attrs['finalized_at'] = datetime.now().isoformat()

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
    def _write_metadata(self, h5, sample_id: str, metadata: Optional[Dict]):
        """Write the /metadata group."""
        grp = h5.create_group('metadata')
        grp.attrs['sample_id'] = sample_id
        grp.attrs['timestamp'] = datetime.now().isoformat()
        grp.attrs['experiment_name'] = self.experiment_name
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    grp.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    grp.create_dataset(key, data=value)

    def _write_experiment_params(self, h5, **kwargs):
        """
        Write the /experiment group with all experiment-level parameters.

        Stores every recognised key as an attribute or dataset (for arrays).
        Unknown extra kwargs are stored as well so no information is lost.
        """
        grp = h5.create_group('experiment')

        # Keys that should be stored as scalar attributes
        scalar_keys = [
            'experiment_type', 'residence_time', 'total_flow',
            'target_style', 'target_wavelength', 'target_absorbance',
            'tolerance', 'cost_method', 'acquisition_function',
            'pca_model_file',
            'chip_id', 'measurement_type',
            # Spectrometer parameters
            'spectro_integration_time_ms', 'spectro_number_averages',
            'spectro_smoothing_px', 'spectro_sampling_rate_s',
            # Wash / flush parameters
            'wash_acid_volume_uL', 'wash_water_volume_uL',
            'wash_backpressure_flow',
        ]
        # Keys that should be stored as datasets (arrays)
        array_keys = [
            'reagents', 'fixed_pumps', 'varying_pumps', 'flows_set',
        ]

        for key in scalar_keys:
            val = kwargs.get(key)
            if val is not None:
                grp.attrs[key] = str(val) if isinstance(val, str) else val

        for key in array_keys:
            val = kwargs.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.dtype.kind == 'U':
                    # h5py needs byte-strings for variable-length string arrays
                    grp.create_dataset(key, data=arr.astype('S'))
                else:
                    grp.create_dataset(key, data=arr)

        # Target spectrum curve -> dedicated subgroup (full_spectrum / pca).
        tw = kwargs.get('target_spectrum_wavelengths')
        ta = kwargs.get('target_spectrum_absorbance')
        if tw is not None and ta is not None:
            tgrp = grp.create_group('target_spectrum')
            tgrp.create_dataset('wavelengths', data=np.asarray(tw, dtype=float))
            tgrp.create_dataset('absorbance', data=np.asarray(ta, dtype=float))

        # Store any remaining kwargs that we didn't explicitly handle
        handled = (set(scalar_keys) | set(array_keys)
                   | {'target_spectrum_wavelengths', 'target_spectrum_absorbance'})
        for key, val in kwargs.items():
            if key in handled or val is None:
                continue
            try:
                if isinstance(val, (str, int, float, bool)):
                    grp.attrs[key] = val
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.dtype.kind == 'U':
                        grp.create_dataset(key, data=arr.astype('S'))
                    else:
                        grp.create_dataset(key, data=arr)
            except Exception:
                pass  # skip unsupported types silently

    def _write_set_flows(self, h5, **kwargs):
        """Write the /flows group with the requested (set) flows."""
        flows_set = kwargs.get('flows_set')
        grp = h5.create_group('flows')
        if flows_set is not None:
            grp.create_dataset('set_flows', data=np.asarray(flows_set))
            grp.attrs['number_of_pumps'] = len(flows_set)
        total = kwargs.get('total_flow')
        if total is not None:
            grp.attrs['total_flow'] = total

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
            grp.create_dataset(name, shape=(1,), maxshape=(None,),
                               dtype=np.float64, data=[value])
        else:
            ds = grp[name]
            ds.resize((idx + 1,))
            ds[idx] = value

    # ------------------------------------------------------------------
    # Private: statistics + figures
    # ------------------------------------------------------------------
    def _compute_statistics(self, h5):
        """Compute summary statistics on the time-series spectra."""
        if 'spectra' not in h5:
            return
        spectra = h5['spectra']

        if 'statistics' in h5:
            del h5['statistics']
        stats = h5.create_group('statistics')

        if 'absorbances' in spectra:
            abs_data = spectra['absorbances'][:]
            stats.create_dataset('abs_final', data=abs_data[-1, :])

            if 'wavelengths' in spectra:
                wl = spectra['wavelengths'][:]
                final_abs = abs_data[-1, :]
                idx = np.argmax(final_abs)
                stats.attrs['final_lambda_max'] = float(wl[idx])
                stats.attrs['final_abs_max'] = float(final_abs[idx])

        if 'timestamps' in spectra:
            stats.attrs['total_spectra_count'] = int(spectra['timestamps'].shape[0])

    def _save_figures_to_disk(self, h5, sample_id: str):
        """
        Generate summary figures and save them as PNG files in the figures/ directory.

        This avoids bloating the H5 file with binary image data.
        """
        try:
            if 'spectra' not in h5:
                return
            spectra = h5['spectra']
            if 'absorbances' not in spectra or 'wavelengths' not in spectra:
                return

            wl = spectra['wavelengths'][:]
            abs_data = spectra['absorbances'][:]
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
            ax1.set_ylabel('Absorbance')
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
            ax2.set_ylabel('Absorbance')
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
