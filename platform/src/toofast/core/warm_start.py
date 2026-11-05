"""
Warm start of a BO run from a previous run folder or from a library of H5 samples.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
import os
import time
import csv
import shutil
import numpy as np

from toofast.hardware.spectrometer import Spectrum


class WarmStartMixin:
    """Warm start of a BO run from a previous run folder or from a library of H5 samples (part of `Experiment`)."""

    def _validate_previous_data(self) -> bool:
        """Check that previous_flows/previous_costs are usable with the current
        experiment configuration. If invalid, clear them and warn.

        Returns True if previous data is set AND valid, False otherwise.
        """
        if self.previous_flows is None or self.previous_costs is None:
            return False
        n_prev = len(self.previous_costs)
        n_pumps_prev = self.previous_flows.shape[1] if self.previous_flows.ndim > 1 else 1
        reasons = []
        if n_pumps_prev != self.number_of_pumps:
            reasons.append(
                f"pump count mismatch (previous={n_pumps_prev}, current={self.number_of_pumps})")
        if n_prev < self.nb_init_samples:
            reasons.append(
                f"too few previous samples ({n_prev}) for current "
                f"nb_init_samples={self.nb_init_samples}")
        if reasons:
            self.logger.warning(
                "Discarding previous experiment data — " + "; ".join(reasons))
            self.previous_flows = None
            self.previous_costs = None
            self._n_warmstart = 0
            return False
        return True

    def warm_start_from_h5(self, h5_dir, drop_flat=False, flat_threshold=0.05,
                           min_wl=315.0, max_wl=750.0, warmstart_noise_level="auto"):
        """Warm-start the BO from a directory of historical .h5 samples.

        For the *current* PCA target, project every historical spectrum onto the
        frozen basis and compute its PCA cost; the (flows, cost) pairs seed the
        GP via previous_flows / previous_costs, so the BO starts from a model
        already informed by all those experiments instead of a cold GP.

        Because the PCA cost is target-dependent, this recomputes the costs for
        the target currently configured (call after update_optim_params). No new
        experiment is run — costs come from the historical spectra.

        Parameters
        ----------
        h5_dir : str
            Directory of per-sample .h5 files, any layout (set flows + final spectrum, see toofast.data.sample_file).
        drop_flat : bool
            Drop no-particle / flat spectra (handled by the flat machinery at
            run time, not by the PCA cost) before seeding.
        flat_threshold, min_wl, max_wl : float
            Flat-detection parameters (peak-to-peak amplitude over the window).

        Returns
        -------
        (flows, costs) : (np.ndarray, np.ndarray)
            The warm-start data that was loaded (also stored on the experiment).
        """
        if self.target_style != 'pca' or getattr(self, 'pca_cost_fn', None) is None:
            raise ValueError(
                "warm_start_from_h5 requires a configured PCA target "
                "(call update_optim_params with target_style='pca' first)."
            )

        from toofast.data.h5_dataset import load_h5_dataset

        ds = load_h5_dataset(h5_dir)
        if drop_flat:
            ds, flat = ds.drop_flat(threshold=flat_threshold, min_wl=min_wl, max_wl=max_wl)
            self.logger.info(f"Warm-start: dropped {int(flat.sum())} flat spectra")

        # Flat (no-particle) spectra are KEPT by default and assigned the flat
        # sentinel — the same value cost_function stores for a live flat — so the
        # flat machinery (nearest_high remap + soft noise) handles them exactly
        # like a measured flat, instead of feeding their (meaningless) raw PCA
        # projection to the GP. Detection mirrors the live criterion in
        # cost_function (std of absorbance below self.threshold).
        sentinel = float(self._flat_penalty_value())
        flat_thr = float(getattr(self, "threshold", flat_threshold))
        costs = []
        n_flat = 0
        for i in range(ds.n_samples):
            absb = ds.spectra[i]
            spec = Spectrum(ds.wavelengths, 10 ** (-absb), 'measurement', absorbance=absb)
            if float(np.std(spec.absorbance)) < flat_thr:
                costs.append(sentinel)
                n_flat += 1
            else:
                cost, _ = self.pca_cost_fn(spec)
                costs.append(float(cost))
        if not drop_flat and n_flat:
            self.logger.info(
                f"Warm-start: {n_flat} flat spectra kept with the flat sentinel "
                f"(handled by flat_strategy, not dropped)")

        self.previous_flows = np.asarray(ds.flows, dtype=float)
        self.previous_costs = np.asarray(costs, dtype=float)
        # Soft warm-start: tag these historical points so the GP fit gives them
        # an inflated per-point noise (prior, not gospel) — robust to drift in
        # old experiments. Reset to 0 wherever previous_* are cleared.
        self._n_warmstart = int(len(self.previous_costs))
        self.warmstart_noise_level = warmstart_noise_level
        self.logger.info(
            f"Warm-start: seeded BO with {ds.n_samples} historical points "
            f"(best PCA cost {self.previous_costs.min():.5f}); "
            f"soft prior noise={warmstart_noise_level}"
        )
        return self.previous_flows, self.previous_costs

    def load_previous_experiment(self, previous_exp_path):
        """
        Load costs, pump flows, and other data from a previous experiment.
        Also copies all the previous experiment files to the current experiment directory.
        Validates that the target parameters match the current experiment.

        Parameters
        ----------
        previous_exp_path : str
            Path to the directory containing the previous experiment data.

        Returns
        -------
        flows : np.ndarray or None
            Array of shape (n_samples, n_pumps) containing the pump flows.
        costs : np.ndarray or None
            Array of shape (n_samples,) containing the costs.
        """
        costs_file = os.path.join(previous_exp_path, "costs.csv")
        flows_file = os.path.join(previous_exp_path, "pump_flows.csv")

        # Check if both files exist
        if not os.path.exists(costs_file):
            self.logger.error(f"Costs file not found: {costs_file}")
            return None, None
        if not os.path.exists(flows_file):
            self.logger.error(f"Pump flows file not found: {flows_file}")
            return None, None

        # Load and validate target parameters from previous experiment
        params_file = None
        for filename in os.listdir(previous_exp_path):
            if filename.endswith("_parameters.txt"):
                params_file = os.path.join(previous_exp_path, filename)
                break

        if params_file and os.path.exists(params_file):
            try:
                prev_target_style = None
                prev_tg_wl = None
                prev_tg_abs = None

                with open(params_file, 'r') as f:
                    for line in f:
                        if 'Optimization strategy:' in line:
                            prev_target_style = line.split(':')[1].strip().rstrip('.')
                        elif 'Lambda_max target at' in line:
                            prev_tg_wl = float(line.split('at')[1].strip().split()[0])
                        elif 'Target absorbance of' in line:
                            parts = line.split()
                            prev_tg_abs = float(parts[3])
                            prev_tg_wl = float(parts[5])

                # Check if target parameters match
                target_mismatch = False
                mismatch_msg = []

                if prev_target_style and prev_target_style != self.target_style:
                    target_mismatch = True
                    mismatch_msg.append(f"Target style: previous={prev_target_style}, current={self.target_style}")

                if prev_tg_wl is not None and abs(prev_tg_wl - self.tg_wl) > 0.1:
                    target_mismatch = True
                    mismatch_msg.append(f"Target wavelength: previous={prev_tg_wl} nm, current={self.tg_wl} nm")

                if prev_tg_abs is not None and abs(prev_tg_abs - self.tg_abs) > 0.01:
                    target_mismatch = True
                    mismatch_msg.append(f"Target absorbance: previous={prev_tg_abs}, current={self.tg_abs}")

                if target_mismatch:
                    self.logger.warning("Target parameters mismatch detected!")
                    for msg in mismatch_msg:
                        self.logger.warning(f"  - {msg}")
                    self.logger.warning("The loaded costs were calculated for a different target.")
                    self.logger.warning("You should use the same target parameters as the previous experiment.")

                    # Store the previous target info for the user
                    self.prev_target_info = {
                        'style': prev_target_style,
                        'wavelength': prev_tg_wl,
                        'absorbance': prev_tg_abs
                    }
                    return None, None
                else:
                    self.logger.info("Target parameters match - safe to load previous data")

            except Exception as e:
                self.logger.warning(f"Could not validate target parameters: {e}")
                self.logger.warning("Proceeding with caution - target may not match!")

        # Restore experiment parameters from machine-readable config
        cfg = self._load_config_json(previous_exp_path)
        if cfg is not None:
            self._apply_loaded_config(cfg)
            self.loaded_config = cfg
        else:
            self.loaded_config = None
            self.logger.warning("No _config.json found — parameters not restored")

        try:
            # Load costs
            costs = []
            with open(costs_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row:  # Skip empty rows
                        costs.append(float(row[0]))
            costs = np.array(costs)

            # Load pump flows
            flows = []
            with open(flows_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row:  # Skip empty rows
                        flows.append([float(x) for x in row])
            flows = np.array(flows)

            # Validate data consistency
            if len(costs) != len(flows):
                self.logger.error(f"Data mismatch: {len(costs)} costs but {len(flows)} flow sets")
                return None, None

            if len(flows) > 0 and flows.shape[1] != self.number_of_pumps:
                self.logger.error(f"Flow data has {flows.shape[1]} pumps but experiment expects {self.number_of_pumps}")
                return None, None

            self.logger.info(f"Successfully loaded {len(costs)} data points from previous experiment")
            self.logger.info(f"Previous costs range: [{costs.min():.4f}, {costs.max():.4f}]")

            # Store source path — files will be copied when the BO sub-folder
            # is created (in _copy_previous_files, called from optimise/stop_flow)
            self._previous_exp_source = previous_exp_path

            return flows, costs

        except Exception as e:
            self.logger.error(f"Error loading previous experiment data: {e}")
            return None, None

    def _copy_previous_files(self):
        """Copy previous experiment files into the current BO sub-folder.

        Called from optimise/stop_flow once experiment_path points to the
        sub-folder (SDXXX/SDXXX_XX).
        """
        src = getattr(self, '_previous_exp_source', None)
        if src is None:
            return

        try:
            copied_files = []

            costs_file = os.path.join(src, "costs.csv")
            if os.path.exists(costs_file):
                shutil.copy2(costs_file, os.path.join(self.experiment_path, "previous_costs.csv"))
                copied_files.append("costs.csv")

            flows_file = os.path.join(src, "pump_flows.csv")
            if os.path.exists(flows_file):
                shutil.copy2(flows_file, os.path.join(self.experiment_path, "previous_pump_flows.csv"))
                copied_files.append("pump_flows.csv")

            lambda_max_file = os.path.join(src, "lambda_max.csv")
            if os.path.exists(lambda_max_file):
                shutil.copy2(lambda_max_file, os.path.join(self.experiment_path, "previous_lambda_max.csv"))
                copied_files.append("lambda_max.csv")

            for filename in os.listdir(src):
                if filename.startswith("absorbance_at_") and filename.endswith("_nm.csv"):
                    shutil.copy2(
                        os.path.join(src, filename),
                        os.path.join(self.experiment_path, f"previous_{filename}"),
                    )
                    copied_files.append(filename)

            for filename in os.listdir(src):
                if filename.endswith("_parameters.txt"):
                    shutil.copy2(
                        os.path.join(src, filename),
                        os.path.join(self.experiment_path, "previous_parameters.txt"),
                    )
                    copied_files.append(filename)
                    break

            samples_dir = os.path.join(src, "samples")
            if os.path.exists(samples_dir):
                dest_samples = os.path.join(self.experiment_path, "previous_samples")
                shutil.copytree(samples_dir, dest_samples, dirs_exist_ok=True)
                copied_files.append("samples/")

            metadata_file = os.path.join(self.experiment_path, "previous_experiment_info.txt")
            with open(metadata_file, 'w') as f:
                f.write(f"Source path: {src}\n")
                f.write(f"Loaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data points: {len(self.previous_costs)}\n")
                f.write(f"Best cost: {self.previous_costs.min():.6f}\n")
                f.write(f"Copied files: {', '.join(copied_files)}\n")

            self.logger.info(f"Copied {len(copied_files)} previous experiment files to {self.experiment_path}")

        except Exception as e:
            self.logger.warning(f"Could not copy previous experiment files: {e}")
