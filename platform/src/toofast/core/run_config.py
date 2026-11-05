"""
Run configuration files: _config.json (saved / reloaded) and the human-readable <run>_parameters.txt.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
import json
import os
import numpy as np

from toofast import clock



def _toofast_version():
    from toofast import __version__
    return __version__


def init_option_from_config(cfg, default="latinhypercube"):
    """Initial-design option of a saved _config.json. Files without 'toofast_version' were written by toofast 1.0,
    whose "latinhypercube" was the projected LHS (now "projected_lhs"): mapped so that a reloaded run keeps its
    design."""
    opt = cfg.get('init_option', default)
    if opt == 'latinhypercube' and 'toofast_version' not in cfg:
        return 'projected_lhs'
    return opt


class RunConfigMixin:
    """Run configuration files: _config.json (saved / reloaded) and the human-readable <run>_parameters.txt (part of `Experiment`)."""

    # ------------------------------------------------------------------
    # Machine-readable config save / load for load_previous
    # ------------------------------------------------------------------
    def _save_config_json(self):
        """Save all experiment parameters as JSON for reliable reload."""
        config = {
            # BO parameters
            'maxiter': self.maxiter,
            'acq_func': self.acq_func,
            'nb_init_samples': self.nb_init_samples,
            'init_option': self.init_option,
            'toofast_version': _toofast_version(),
            'de_tol': self.de_tol,
            'de_maxiter': self.de_maxiter,
            # Target
            'target_style': self.target_style,
            'tg_wl': self.tg_wl,
            'tg_abs': self.tg_abs,
            'tol': self.tol,
            'rel_tol': self.rel_tol,
            'tol_stop': self.tol_stop,
            'cost_method': getattr(self, 'cost_method', 'L2'),
            'threshold': self.threshold,
            # Timing
            'residence_time': self.residence_time,
            # Pump constraints
            'bounds': self.bounds.tolist() if self.bounds is not None else None,
            'fixed_pumps': self.fixed_pumps.tolist() if self.fixed_pumps is not None else None,
            'fixed_flows': [float(f) if f is not None else None for f in self.fixed_flows],
            'reagents': list(self.reagents) if self.reagents is not None else None,
            'total_flow': self.total_flow,
            'number_of_pumps': self.number_of_pumps,
            # GP kernel overrides
            'kernel_type_override': self.kernel_type_override,
            'rbf_length_scale_override': self.rbf_length_scale_override,
            'white_noise_override': self.white_noise_override,
            'white_noise_floor': self.white_noise_floor,
            'white_noise_ceiling': self.white_noise_ceiling,
            'length_scale_min': self.length_scale_min,
            'cost_transform': self.cost_transform,
            'pca_cost_metric': self.pca_cost_metric,
            'pca_n_components': self.pca_n_components,
            'gp_n_restarts_override': self.gp_n_restarts_override,
            'bo_preset_override': getattr(self, 'bo_preset_override', 'default'),
            'huber_delta_nm_override': getattr(self, 'huber_delta_nm_override', 30.0),
            'flat_strategy_override': getattr(self, 'flat_strategy_override', 'nearest_high'),
            'flat_penalty_factor_override': getattr(self, 'flat_penalty_factor_override', 2.0),
            'flat_nearest_margin_override': getattr(self, 'flat_nearest_margin_override', 1.25),
            'flat_nearest_cap_override': getattr(self, 'flat_nearest_cap_override', 3.0),
            'feasibility_aware_override': getattr(self, 'feasibility_aware_override', False),
            'use_physics_ghosts': getattr(self, 'use_physics_ghosts', True),
            'ghost_noise_level': getattr(self, 'ghost_noise_level', 'auto'),
            'ghost_penalty_lift': getattr(self, 'ghost_penalty_lift', 1.5),
            # Spectrometer
            'spectro_integration_time': getattr(self.spectro, 'integration_time', None),
            'spectro_number_average': getattr(self.spectro, 'number_average', None),
            'spectro_smoothing': getattr(self.spectro, 'smoothing', None),
            # Wash / flush (read from GUI if available)
            'wash_acid_volume': self._gui_get_safe('nitric_acid_volume_entry'),
            'wash_water_volume': self._gui_get_safe('water_volume_entry'),
            'wash_backpressure_flow': self._gui_get_safe('back_pressure_entry'),
            'spectro_sampling_rate': self._gui_get_safe('spectro_sampling_rate_entry'),
            # Termination
            'termination_correlation': self.termination.correlation_threshold,
            'termination_amplitude': self.termination.amplitude_threshold,
            'termination_min_spectra': self.termination.min_spectra,
            'termination_timeout': self.termination.monitoring_timeout_s,
        }
        path = os.path.join(self.experiment_path, '_config.json')
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"Config JSON saved: {path}")
        except Exception as e:
            self.logger.warning(f"Could not save config JSON: {e}")

    def _gui_get_safe(self, attr_name):
        """Read a GUI entry value without crashing if the GUI is absent."""
        try:
            widget = getattr(self.app, attr_name, None)
            if widget is not None:
                return float(self._gui_get(widget))
        except Exception:
            self.logger.debug(f"GUI value {attr_name!r} unavailable", exc_info=True)
        return None

    @staticmethod
    def _load_config_json(previous_exp_path):
        """Load config JSON from a previous experiment directory."""
        path = os.path.join(previous_exp_path, '_config.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _apply_loaded_config(self, cfg):
        """Apply parameters from a loaded config JSON to this experiment."""
        self.maxiter = cfg.get('maxiter', self.maxiter)
        self.acq_func = cfg.get('acq_func', self.acq_func)
        self.nb_init_samples = cfg.get('nb_init_samples', self.nb_init_samples)
        self.init_option = init_option_from_config(cfg, self.init_option)
        self.de_tol = cfg.get('de_tol', self.de_tol)
        self.de_maxiter = cfg.get('de_maxiter', self.de_maxiter)

        self.target_style = cfg.get('target_style', self.target_style)
        self.tg_wl = cfg.get('tg_wl', self.tg_wl)
        self.tg_abs = cfg.get('tg_abs', self.tg_abs)
        self.tol = cfg.get('tol', self.tol)
        self.rel_tol = cfg.get('rel_tol', self.rel_tol)
        self.tol_stop = cfg.get('tol_stop', self.tol_stop)
        self.cost_method = cfg.get('cost_method', getattr(self, 'cost_method', 'L2'))
        self.threshold = cfg.get('threshold', self.threshold)

        self.residence_time = cfg.get('residence_time', self.residence_time)

        if cfg.get('total_flow') is not None:
            self.total_flow = cfg['total_flow']
        if cfg.get('bounds') is not None:
            self.bounds = np.array(cfg['bounds'])
        if cfg.get('fixed_pumps') is not None:
            self.fixed_pumps = np.array(cfg['fixed_pumps'])
            self.varying_pumps = ~self.fixed_pumps
            self.nb_of_var_pumps = self.varying_pumps.sum()
        if cfg.get('fixed_flows') is not None:
            self.fixed_flows = np.array([
                f if f is not None else 0.0 for f in cfg['fixed_flows']
            ])
        self.total_var_flow = self.total_flow - sum(
            self.fixed_flows[self.fixed_pumps]
        )
        if cfg.get('reagents') is not None:
            self.reagents = np.array(cfg['reagents'])

        self.kernel_type_override = cfg.get('kernel_type_override', self.kernel_type_override)
        self.rbf_length_scale_override = cfg.get('rbf_length_scale_override', self.rbf_length_scale_override)
        self.white_noise_override = cfg.get('white_noise_override', self.white_noise_override)
        self.white_noise_floor = cfg.get('white_noise_floor', self.white_noise_floor)
        self.white_noise_ceiling = cfg.get('white_noise_ceiling', self.white_noise_ceiling)
        self.length_scale_min = cfg.get('length_scale_min', self.length_scale_min)
        self.cost_transform = cfg.get('cost_transform', self.cost_transform)
        self.pca_cost_metric = cfg.get('pca_cost_metric', self.pca_cost_metric)
        self.pca_n_components = cfg.get('pca_n_components', self.pca_n_components)
        self.gp_n_restarts_override = cfg.get('gp_n_restarts_override', self.gp_n_restarts_override)
        self.bo_preset_override = cfg.get('bo_preset_override', getattr(self, 'bo_preset_override', 'default'))
        self.huber_delta_nm_override = cfg.get('huber_delta_nm_override', getattr(self, 'huber_delta_nm_override', 30.0))
        self.flat_strategy_override = cfg.get('flat_strategy_override', getattr(self, 'flat_strategy_override', 'nearest_high'))
        self.flat_penalty_factor_override = cfg.get('flat_penalty_factor_override', getattr(self, 'flat_penalty_factor_override', 2.0))
        self.flat_nearest_margin_override = cfg.get('flat_nearest_margin_override', getattr(self, 'flat_nearest_margin_override', 1.25))
        self.flat_nearest_cap_override = cfg.get('flat_nearest_cap_override', getattr(self, 'flat_nearest_cap_override', 3.0))
        self.feasibility_aware_override = cfg.get('feasibility_aware_override', getattr(self, 'feasibility_aware_override', False))
        self.use_physics_ghosts = bool(cfg.get('use_physics_ghosts', getattr(self, 'use_physics_ghosts', True)))
        self.ghost_noise_level = cfg.get('ghost_noise_level', getattr(self, 'ghost_noise_level', 'auto'))
        self.ghost_penalty_lift = cfg.get('ghost_penalty_lift', getattr(self, 'ghost_penalty_lift', 1.5))
        self._rebuild_kernel()

        tc = self.termination
        tc.correlation_threshold = cfg.get('termination_correlation', tc.correlation_threshold)
        tc.amplitude_threshold = cfg.get('termination_amplitude', tc.amplitude_threshold)
        tc.min_spectra = cfg.get('termination_min_spectra', tc.min_spectra)
        tc.monitoring_timeout_s = cfg.get('termination_timeout', tc.monitoring_timeout_s)

        self.logger.info("Experiment parameters restored from previous config")

    def write_params(self, gaussian_process, end=False):
        """
        Writes a .txt file containing the parameters of the experiment.

        Parameters
        ----------
        gaussian_process : sk-learn GaussianProcessRegressor
            The GPR.
        end : bool, optional
            If set to True, writes the optimal measured points' characteristics.
            The default is False.

        """
        parameters_file_path = os.path.join(self.experiment_path,
                                            f"{self.experiment_name}_parameters.txt")
        with open(parameters_file_path, 'w') as file:
            file.write(f'{self.experiment_name}:   {clock.now().isoformat()}\n\n')

            # Add information about loaded previous experiment if applicable
            if self.previous_flows is not None and self.previous_costs is not None:
                file.write('=' * 60 + '\n')
                file.write('WARM START FROM PREVIOUS EXPERIMENT\n')
                file.write('=' * 60 + '\n')
                file.write(f'Loaded {len(self.previous_costs)} data points from previous experiment\n')
                file.write(f'Previous best cost: {self.previous_costs.min():.6f}\n')

                # Check if metadata file exists
                metadata_file = os.path.join(self.experiment_path, "previous_experiment_info.txt")
                if os.path.exists(metadata_file):
                    file.write(f'See "previous_experiment_info.txt" for complete details\n')
                file.write('\n')

            file.write(f'Features: {[pump.dye for pump in self.pump_syst.pumps]}\n')
            file.write(f"Total Flow: {self.total_flow}\n")
            file.write(f'Bounds to the pumps: {self.bounds}\n\n')

            file.write(f"Number of maximum iterations: {self.maxiter} iterations.\n")
            file.write(f'Optimization strategy: {self.target_style}.\n')
            if self.target_style == 'lambda_max':
                file.write(f'Lambda_max target at {self.tg_wl} nm.\n')
            elif self.target_style == 'single_point':
                file.write(f'Target absorbance of {self.tg_abs} at {self.tg_wl} nm.\n')
            elif self.target_style == 'full_spectrum':
                file.write(f'Full spectrum target measured at {self.tg_init_flows}.\n')
            elif self.target_style == 'pca':
                file.write('PCA target (score-space distance).\n')
                if getattr(self, 'pca_model_file', None):
                    file.write(f'PCA decomposition file: {self.pca_model_file}\n')
                file.write(f'PCA cost metric: {self.pca_cost_metric} | '
                           f'n_components: {self.pca_n_components or "all"}\n')
                file.write(f'PCA variance-weighted: {self.pca_variance_weighted}\n')
                if getattr(self, 'pca_lambda_weight', 0.0):
                    file.write(f'Hybrid lambda_max term: weight={self.pca_lambda_weight}, '
                               f'scale={self.pca_lambda_scale_nm} nm\n')
                if getattr(self, 'pca_amp_weight', 0.0):
                    file.write(f'Hybrid amplitude term: weight={self.pca_amp_weight} '
                               f'(relative to target peak absorbance)\n')
            # Target spectrum (full_spectrum / pca): saved alongside as a CSV.
            if getattr(self, 'tg_spectrum', None) is not None:
                wl = np.asarray(self.tg_spectrum.wavelengths, dtype=float)
                ab = np.asarray(self.tg_spectrum.absorbance, dtype=float)
                src = (os.path.basename(self.target_spectrum_file)
                       if getattr(self, 'target_spectrum_file', None) else 'in-memory')
                file.write(
                    f'Target spectrum: {len(wl)} points, {wl[0]:.1f}-{wl[-1]:.1f} nm, '
                    f'peak abs {np.max(ab):.4f} at {wl[int(np.argmax(ab))]:.1f} nm '
                    f'(source: {src}; saved as target_spectrum.csv).\n')
            file.write(f'Tolerence: {self.tol}\n\n')

            file.write(f'Initiation option: {self.init_option} ({self.nb_init_samples} init samples).\n')
            file.write('Initialisation points:\n')
            file.write(f'{self.initial_flows}')

            file.write('\n\nGPR parameters:\n')
            file.write(f'Acquisition function: {self.acq_func}.\n')
            file.write(str(gaussian_process.get_params()) + '\n\n')
            file.write('Differential evolution parameters:\nMaxiter: '
                       f'{self.de_maxiter}\nTolerance: {self.de_tol}\n')
            file.write('\nSpectrometer parameters:\n')
            file.write(f'Integration time = {self.spectro.integration_time} ms\n')
            file.write(f'Average = {self.spectro.number_average} measurements\n')

            if end:
                file.write('\n----------------\n')
                file.write('Results: ' + ('Success' if self.success else 'Failure') + '\n\n')
                file.write(f"Optimal flows: {self.opt_flows}\n")
                file.write(f"Minimal cost: {self.min_cost}\n")
                file.write(f'Obtained after {self.best_iter} iterations.\n')
                file.write(f'\nLast kernel parameters : {gaussian_process.kernel_}\n')
                file.write("\nErrors during the GP:\n")
                file.write(f"Predicted negative variance: {self.neg_var}.")

        self._save_config_json()
