"""
Visualization module for Bayesian Optimization results.

Provides a clean interface to the existing Figure_plotting functionality.
The original module is preserved as-is since it handles complex visualization
logic for 2-5+ pump dimensions with ternary diagrams, 3D surfaces, etc.

This wrapper decouples the plotting from the Experiment object by accepting
explicit parameters instead of reaching into exp attributes.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_iteration(
    iteration: int,
    flows: np.ndarray,
    costs: np.ndarray,
    gp_model: object,
    best_min: List[float],
    times: List[float],
    spectrum: object,
    predicted_flows: Optional[np.ndarray] = None,
    final_fig: bool = False,
    save_figs: bool = True,
    # Experiment config needed for plotting
    nb_init_samples: int = 3,
    nb_of_var_pumps: int = 2,
    varying_pumps: Optional[np.ndarray] = None,
    total_var_flow: float = 8.0,
    total_flow: float = 8.0,
    tolerance: float = 1e-7,
    target_style: str = "single_point",
    target_wavelength: Optional[float] = None,
    target_absorbance: Optional[float] = None,
    target_spectrum: object = None,
    reagents: Optional[np.ndarray] = None,
    wavelengths: Optional[np.ndarray] = None,
    experiment_path: str = "",
    experiment_name: str = "",
    opt_flows: Optional[np.ndarray] = None,
    best_iter: int = 0,
    pump_colors: Optional[List] = None,
) -> Optional[plt.Figure]:
    """
    Plot a full iteration figure for the BO experiment.

    This is a clean-interface wrapper. For the original complex plotting
    logic (ternary diagrams, 4D visualization, etc.), the existing
    Figure_plotting.py module is imported and used.
    """
    try:
        # Try to use the original plotting module
        from toofast.visualization.figure_plotting import plot_fig as _original_plot_fig

        # The original function expects an experiment-like object
        # Create a lightweight proxy
        class _ExpProxy:
            pass

        proxy = _ExpProxy()
        proxy.nb_init_samples = nb_init_samples
        proxy.nb_of_var_pumps = nb_of_var_pumps
        proxy.varying_pumps = varying_pumps if varying_pumps is not None else np.ones(flows.shape[1], dtype=bool)
        proxy.total_var_flow = total_var_flow
        proxy.total_flow = total_flow
        proxy.tol = tolerance
        proxy.target_style = target_style
        proxy.tg_wl = target_wavelength
        proxy.tg_abs = target_absorbance
        proxy.tg_spectrum = target_spectrum
        proxy.reagents = reagents if reagents is not None else np.array([""] * flows.shape[1])
        proxy.experiment_path = experiment_path
        proxy.experiment_name = experiment_name
        proxy.opt_flows = opt_flows
        proxy.best_iter = best_iter
        proxy.maxiter = iteration

        # Create a mock spectro for wavelength access
        class _SpectroProxy:
            pass
        spectro_proxy = _SpectroProxy()
        spectro_proxy.wavelengths = wavelengths if wavelengths is not None else np.linspace(300, 800, 4094)
        spectro_proxy.find_index_for_wavelength = lambda wl, domain=None: int(
            np.argmin(np.abs((domain if domain is not None else spectro_proxy.wavelengths) - wl))
        )
        proxy.spectro = spectro_proxy

        # Create mock pump_syst for colors
        class _PumpProxy:
            pass
        class _PumpSystProxy:
            pass
        pump_syst_proxy = _PumpSystProxy()
        n_pumps = flows.shape[1] if flows is not None and flows.ndim > 1 else 1
        pump_syst_proxy.pumps = []
        colors = pump_colors or [None] * n_pumps
        for c in colors:
            p = _PumpProxy()
            p.color = c
            pump_syst_proxy.pumps.append(p)
        proxy.pump_syst = pump_syst_proxy

        fig = _original_plot_fig(
            proxy, iteration, flows, costs, gp_model, best_min, times,
            spectrum, predicted_flows, final_fig, save_figs,
        )
        return fig

    except ImportError:
        logger.warning("Figure_plotting module not available, using basic plot")
        return _basic_plot(
            iteration, flows, costs, best_min, spectrum,
            target_style, target_wavelength, target_absorbance,
            experiment_path, experiment_name, final_fig, save_figs,
        )


def _basic_plot(
    iteration, flows, costs, best_min, spectrum,
    target_style, target_wavelength, target_absorbance,
    experiment_path, experiment_name, final_fig, save_figs,
) -> plt.Figure:
    """Fallback simple plot when full visualization isn't available."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cost evolution
    axes[0].plot(range(len(best_min)), best_min, "b-o", label="Best cost")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Iteration {iteration}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spectrum
    if spectrum is not None and hasattr(spectrum, "absorbance") and spectrum.absorbance is not None:
        axes[1].plot(spectrum.wavelengths, spectrum.absorbance, label="Measured")
        if target_style == "single_point" and target_wavelength and target_absorbance:
            axes[1].scatter(target_wavelength, target_absorbance, c="red", s=50, label="Target")
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Absorbance")
        axes[1].set_title("Spectrum")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_figs and experiment_path:
        fig_dir = os.path.join(experiment_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        suffix = "final" if final_fig else f"iter_{iteration}"
        fig.savefig(os.path.join(fig_dir, f"{experiment_name}_{suffix}.png"), dpi=150)

    return fig
