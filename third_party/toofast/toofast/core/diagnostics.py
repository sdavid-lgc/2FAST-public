"""
Gaussian Process convergence diagnostics.

Extracted from tooFAST.py:diagnose_gp_convergence() (155 lines) which was
embedded in the Experiment class. Now standalone for testability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Results of GP convergence analysis."""

    rbf_length_scale: Optional[np.ndarray] = None
    noise_level: Optional[float] = None
    signal_to_noise: Optional[float] = None
    mean_residual: Optional[float] = None
    max_residual: Optional[float] = None
    cost_trend_slope: Optional[float] = None
    cost_trend_r: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_converged: bool = False


class GPDiagnostics:
    """
    Analyzes GP model health and convergence.

    Performs checks on:
    - Kernel hyperparameters (length scale, noise)
    - Signal-to-noise ratio
    - Residual analysis
    - Cost evolution trend
    """

    def __init__(self, min_observations: int = 4):
        self.min_observations = min_observations

    def diagnose(
        self,
        gp_model: object,
        X: np.ndarray,
        y: np.ndarray,
        costs: np.ndarray,
    ) -> DiagnosticReport:
        """
        Run full diagnostics on a fitted GP model.

        Parameters
        ----------
        gp_model : GaussianProcessRegressor
            Fitted GP model.
        X : np.ndarray
            Training input data (n_samples, n_features).
        y : np.ndarray
            Training target data (n_samples,).
        costs : np.ndarray
            Full cost history (may include initial experiments).
        """
        report = DiagnosticReport()

        if len(y) < self.min_observations:
            report.warnings.append(
                f"Too few observations ({len(y)}) for reliable diagnostics"
            )
            return report

        self._analyze_kernel(gp_model, X, report)
        self._analyze_residuals(gp_model, X, y, report)
        self._analyze_cost_trend(costs, report)
        self._check_convergence(report)

        self._log_report(report)
        return report

    def _analyze_kernel(
        self, gp_model: object, X: np.ndarray, report: DiagnosticReport
    ) -> None:
        """Analyze kernel hyperparameters."""
        try:
            kernel = gp_model.kernel_
            params = kernel.get_params()

            # Extract length scale from kernel parameters
            for key, value in params.items():
                if "length_scale" in key and "bounds" not in key:
                    report.rbf_length_scale = np.atleast_1d(value)
                if "noise_level" in key and "bounds" not in key:
                    report.noise_level = float(value)

            if report.rbf_length_scale is not None:
                data_range = np.ptp(X, axis=0)
                for i, (ls, dr) in enumerate(
                    zip(report.rbf_length_scale, data_range)
                ):
                    if dr > 0:
                        ratio = ls / dr
                        if ratio > 10:
                            report.warnings.append(
                                f"Length scale dim {i} ({ls:.2f}) >> data range ({dr:.2f}): "
                                f"GP may be underfitting"
                            )
                        elif ratio < 0.01:
                            report.warnings.append(
                                f"Length scale dim {i} ({ls:.4f}) << data range ({dr:.2f}): "
                                f"GP may be overfitting"
                            )

            if report.noise_level is not None:
                y_var = float(np.var(gp_model.y_train_))
                if y_var > 0:
                    report.signal_to_noise = y_var / report.noise_level
                    if report.signal_to_noise < 1.0:
                        report.warnings.append(
                            f"Low signal-to-noise ratio ({report.signal_to_noise:.2f}): "
                            f"noise dominates the signal"
                        )

        except Exception as e:
            report.warnings.append(f"Kernel analysis failed: {e}")

    def _analyze_residuals(
        self,
        gp_model: object,
        X: np.ndarray,
        y: np.ndarray,
        report: DiagnosticReport,
    ) -> None:
        """Analyze prediction residuals on training data."""
        try:
            y_pred = gp_model.predict(X)
            residuals = y - y_pred
            report.mean_residual = float(np.mean(np.abs(residuals)))
            report.max_residual = float(np.max(np.abs(residuals)))

            y_range = float(np.ptp(y))
            if y_range > 0 and report.max_residual / y_range > 0.3:
                report.warnings.append(
                    f"Large residuals detected (max={report.max_residual:.4f}, "
                    f"data range={y_range:.4f}): GP fit may be poor"
                )

        except Exception as e:
            report.warnings.append(f"Residual analysis failed: {e}")

    def _analyze_cost_trend(
        self, costs: np.ndarray, report: DiagnosticReport
    ) -> None:
        """Analyze the trend in cost values over iterations."""
        if len(costs) < 3:
            return

        try:
            recent = costs[-min(5, len(costs)) :]
            x = np.arange(len(recent))
            r, _ = pearsonr(x, recent)
            slope = np.polyfit(x, recent, 1)[0]

            report.cost_trend_slope = float(slope)
            report.cost_trend_r = float(r)

            if slope > 0 and abs(r) > 0.7:
                report.suggestions.append(
                    "Costs are increasing: consider adjusting acquisition function "
                    "or expanding search bounds"
                )
            elif abs(slope) < 1e-6 and len(costs) > 5:
                report.suggestions.append(
                    "Costs have plateaued: optimizer may have converged or be stuck"
                )

        except Exception as e:
            report.warnings.append(f"Trend analysis failed: {e}")

    def _check_convergence(self, report: DiagnosticReport) -> None:
        """Determine if the GP appears converged."""
        has_critical_warnings = any(
            "underfitting" in w or "noise dominates" in w
            for w in report.warnings
        )
        report.is_converged = (
            len(report.warnings) == 0
            and report.cost_trend_slope is not None
            and abs(report.cost_trend_slope) < 1e-4
            and not has_critical_warnings
        )

    def _log_report(self, report: DiagnosticReport) -> None:
        """Log the diagnostic report."""
        logger.info("=== GP Diagnostics ===")
        if report.rbf_length_scale is not None:
            logger.info(f"  Length scale: {report.rbf_length_scale}")
        if report.noise_level is not None:
            logger.info(f"  Noise level: {report.noise_level:.6f}")
        if report.signal_to_noise is not None:
            logger.info(f"  Signal/Noise: {report.signal_to_noise:.2f}")
        if report.mean_residual is not None:
            logger.info(
                f"  Residuals: mean={report.mean_residual:.6f}, max={report.max_residual:.6f}"
            )
        for w in report.warnings:
            logger.warning(f"  WARNING: {w}")
        for s in report.suggestions:
            logger.info(f"  SUGGESTION: {s}")
        logger.info(f"  Converged: {report.is_converged}")
