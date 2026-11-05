"""
GP kernel construction (presets, overrides, offline-tuned prior) and fit-quality diagnostics.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
import pickle
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


def _freeze_kernel(kernel):
    """Return a copy of ``kernel`` with all hyperparameters frozen.

    Sets every ``*_bounds`` param to 'fixed' so a GaussianProcessRegressor keeps
    the (offline-tuned) hyperparameters instead of re-optimising them.
    """
    k = clone(kernel)
    k.set_params(**{n: "fixed" for n in k.get_params() if n.endswith("_bounds")})
    return k


def _prior_kernel_matches(kernel, n_dims):
    """True if the kernel's ARD lengthscales are compatible with ``n_dims``.

    A per-dimension (ARD) lengthscale vector must have length n_dims; scalar
    (isotropic) lengthscales are compatible with any dimensionality.
    """
    for name, v in kernel.get_params().items():
        if name.endswith("length_scale") and not name.endswith("_bounds"):
            v = np.atleast_1d(v)
            if v.size > 1 and v.size != n_dims:
                return False
    return True


class GPModelMixin:
    """GP kernel construction (presets, overrides, offline-tuned prior) and fit-quality diagnostics (part of `Experiment`)."""

    def load_prior_kernel(self, path, freeze=False):
        """Load an offline-tuned kernel to use as a hyperparameter prior.

        The kernel (e.g. produced by notebooks/kernel_benchmark_pca.ipynb by
        fitting a GP on historical data) carries learned lengthscales + noise.
        Used by _rebuild_kernel as the live kernel: the historical data informs
        the function's SHAPE (smoothness/noise), not where the optimum is.

        Parameters
        ----------
        path : str
            Pickle file containing a fitted sklearn kernel.
        freeze : bool
            If False (default), the offline hyperparameters are only a WARM
            START: the live GP keeps re-optimising them as points arrive (i.e.
            just a hyperparameter optimisation seeded from history). If True,
            freeze them (bounds -> 'fixed') so the prior regime is kept fixed.
        """
        with open(path, "rb") as f:
            self.prior_kernel = pickle.load(f)
        self.prior_kernel_freeze = bool(freeze)
        self.logger.info("Loaded prior kernel from %s (freeze=%s): %s",
                         path, freeze, self.prior_kernel)
        return self.prior_kernel

    def _rebuild_kernel(self):
        """Rebuild the GP kernel from current override settings."""
        preset = getattr(self, "bo_preset_override", "default") or "default"
        n_dims = max(1, int(np.sum(self.varying_pumps)) - 1) if hasattr(self, 'varying_pumps') else 1

        # Hyperparameter prior takes precedence: use the offline-tuned kernel.
        if getattr(self, "prior_kernel", None) is not None:
            if _prior_kernel_matches(self.prior_kernel, n_dims):
                self.kernel = (_freeze_kernel(self.prior_kernel)
                               if getattr(self, "prior_kernel_freeze", False)
                               else clone(self.prior_kernel))
                self.logger.info(
                    "Using PRIOR kernel (hyperparameter prior, freeze=%s): %s",
                    getattr(self, "prior_kernel_freeze", False), self.kernel)
                return
            self.logger.warning(
                "Prior kernel dim mismatch (live n_dims=%d) — ignoring it and "
                "rebuilding the default kernel.", n_dims)

        if preset == "robust":
            # Robust preset is atomic: ignore UI length_scale / noise overrides.
            # Matern 2.5 ISOTROPIC: ARD's per-dimension length scales came out
            # ~equal on well-fit data (mechanically redundant) and fitted noise
            # on poor data — isotropic is the defensible choice.
            self.kernel = (
                Matern(length_scale=max(1.0, self.length_scale_min),
                       length_scale_bounds=(self.length_scale_min, 10.0),
                       nu=2.5)
                + WhiteKernel(noise_level=1e-3,
                              noise_level_bounds=(self.white_noise_floor,
                                                   self.white_noise_ceiling))
            )
            self.logger.info(
                "Robust BO preset: Matern 2.5 isotropic kernel "
                "(length_scale_bounds=(%.3g, 10.0), white noise initial 1e-3). "
                "UI Length scale / Noise level entries are ignored.",
                self.length_scale_min,
            )
            return

        # Default preset: honour UI overrides
        ls = self.rbf_length_scale_override or 1.0
        noise = self.white_noise_override or 1e-3
        kernel_type = self.kernel_type_override or "RBF"

        if kernel_type == "ARD":
            ls = np.ones(n_dims) * ls
            self.logger.warning(
                "ARD kernel: per-dimension length scales (n_dims=%d). "
                "Requires sufficient data to avoid overfitting.", n_dims,
            )
        # Keep the initial length scale within the (length_scale_min, 1e5) bounds.
        ls = np.maximum(ls, self.length_scale_min)

        self.kernel = (
            RBF(length_scale=ls, length_scale_bounds=(self.length_scale_min, 1e5))
            + WhiteKernel(noise_level=noise,
                          noise_level_bounds=(self.white_noise_floor,
                                              self.white_noise_ceiling))
        )

    def _is_gp_fit_reliable(self, gaussian_process):
        """Quick check: is the GP fit reliable enough for acquisition optimization?

        Returns False when there are too few samples or when the RBF length
        scale has hit its upper bound (sign of underfitting with sparse data).
        """
        try:
            n_samples = len(gaussian_process.y_train_)
            n_features = gaussian_process.X_train_.shape[1]
            if n_samples < n_features + 2:
                self.logger.info(
                    f"Only {n_samples} samples for {n_features}D — "
                    "insufficient for reliable GP"
                )
                return False

            kernel = gaussian_process.kernel_
            params = kernel.get_params()
            for key, value in params.items():
                if "length_scale" in key and "bounds" not in key:
                    ls = np.atleast_1d(value)
                    bounds_key = key + "_bounds"
                    if bounds_key in params and params[bounds_key] != "fixed":
                        lo, hi = params[bounds_key]
                        if np.any(ls >= hi * 0.9):
                            self.logger.info(
                                f"Length scale {ls} near upper bound {hi:.0f} "
                                "— GP likely underfitting"
                            )
                            return False
            return True
        except Exception:
            self.logger.debug("GP reliability check failed; GP treated as reliable", exc_info=True)
            return True

    def diagnose_gp_convergence(self, gaussian_process, flows, costs):
        """
        Diagnose GP convergence and parameter quality after fitting.

        Delegates to toofast.core.diagnostics.GPDiagnostics module.

        Parameters
        ----------
        gaussian_process : GaussianProcessRegressor
            Fitted GP model
        flows : array-like
            Training flows
        costs : array-like
            Training costs

        Returns
        -------
        DiagnosticReport : Diagnostic information with converged, warnings, suggestions
        """
        # The GP is fit in GP-target space (warped if cost_transform == 'log'),
        # so residuals must be computed against the warped targets; the 4th arg
        # (raw cost history) stays in raw units for range/scale reporting.
        report = self._gp_diagnostics.diagnose(
            gaussian_process, flows, self._cost_to_gp(np.asarray(costs)), costs
        )

        # Log report using the experiment logger
        lml = gaussian_process.log_marginal_likelihood_value_
        self.logger.info("=" * 60)
        self.logger.info("GP CONVERGENCE DIAGNOSTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Log Marginal Likelihood: {lml:.2f}")
        self.logger.info(f"Converged: {report.is_converged}")

        if report.rbf_length_scale is not None:
            self.logger.info(f"Length scale: {report.rbf_length_scale}")
        if report.noise_level is not None:
            self.logger.info(f"Noise level: {report.noise_level:.4f}")

        if report.warnings:
            for warn in report.warnings:
                self.logger.warning(f"  - {warn}")
        if report.suggestions:
            self.logger.info("Suggestions:")
            for sug in report.suggestions:
                self.logger.info(f"  - {sug}")

        if report.is_converged and not report.warnings:
            self.logger.info("GP converged successfully with no issues detected")
        self.logger.info("=" * 60)

        # Return legacy-compatible dict
        kernel_params = {}
        if report.rbf_length_scale is not None:
            kernel_params['length_scale'] = report.rbf_length_scale
        if report.noise_level is not None:
            kernel_params['noise_level'] = report.noise_level

        return {
            'converged': report.is_converged,
            'warnings': report.warnings,
            'suggestions': report.suggestions,
            'kernel_params': kernel_params,
        }
