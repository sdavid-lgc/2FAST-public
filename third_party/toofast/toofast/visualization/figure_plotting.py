# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:54:56 2024

@author: HL279987
"""
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.ticker
import imageio.v2 as imageio
import ternary
from ternary.helpers import simplex_iterator
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from ternary.colormapping import colorbar_hack
import glob
import mpld3
import plotly.io as pio

logger = logging.getLogger(__name__)

# Sigma multiplier for the plotted GP uncertainty band. Was 2.807 (~99.5% CI),
# which is visually huge — especially with the log cost warp, where between-point
# sigma is large. 1.0 (~68% CI / +/-1 sigma) keeps the band informative without
# swamping the mean surface. Raise it for wider coverage.
BAND_SIGMA = 1.0


def _cap_uncertainty(upper, *refs, factor=5.0):
    """Cap an upper-uncertainty surface for display.

    With the log cost warp, the displayed band is exp(mu + k*sigma): a normal
    log-space CI becomes an exponentially large, spiky multiplicative band in raw
    cost (tens of thousands), squashing the mean surface flat. This caps the band
    at ``factor`` x the largest reference (mean surface / measured costs) so the
    plot stays readable. The mean surface itself is never capped.
    """
    ref = 0.0
    for r in refs:
        try:
            r = float(np.nanmax(np.asarray(r, dtype=float)))
            if np.isfinite(r):
                ref = max(ref, r)
        except (ValueError, TypeError):
            continue
    if ref <= 0:
        return upper
    return np.minimum(upper, ref * factor)

def init_fig(exp, i):
    """
    Generate the figure layout and target (depending on the number of features)

    Parameters
    ----------
    exp : Experiment object
        The current experiment.
    i : int
        The current iteration.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plotted figure.
    ax : list of matplotlib.axes._axes.Axes
        The figure's axes.

    """
    logger.debug("init fig")
    fig = plt.figure(dpi=90)
    fig.suptitle(f'Iteration {i}', fontsize=10)
    fig.set_size_inches(9, 6)
    ax = []

    gs = GridSpec(3, 4, figure=fig, hspace=0.6, wspace=0.4,
                  height_ratios=[1, 1, 2], width_ratios=[9, 0, 6, 1])

    # Plot GPR posterior
    if exp.nb_of_var_pumps == 2:
        gs2 = GridSpec(3, 4, figure=fig, hspace=0, wspace=0.3,
                       height_ratios=[0.5, 3, 3.5], width_ratios=[9, 0.5, 6, 1])
        ax.append(fig.add_subplot(gs2[1:2, 0]))
        ax[-1].set_title('GP Regression Posterior', fontsize=10)
        ax[-1].set_ylabel('cost' + exp._cost_axis_suffix())
        ax[-1].set_xlim(0, exp.total_var_flow)
        varying_indices = np.where(exp.varying_pumps)[0]
        reagents_varying = exp.reagents[varying_indices]

        logger.debug("Reagents varying: %s", reagents_varying)
        logger.debug("exp reagent varying pumps: %s", reagents_varying[0] if len(reagents_varying) > 0 else "No reagent")

        ax[-1].set_xlabel("Q_" + (reagents_varying[0] if len(reagents_varying) > 0 else "pump1"))
        ax2 = ax[-1].secondary_xaxis("top", functions=(lambda x: exp.total_var_flow - x,
                                                       lambda x: exp.total_var_flow - x))
        ax2.set_xlabel("Q_" + (reagents_varying[1] if len(reagents_varying) > 1 else "pump2"))
#ba
    elif exp.nb_of_var_pumps == 3:
        ax.append(fig.add_subplot(gs[0:2, 0]))
        tax = ternary.TernaryAxesSubplot(ax=ax[-1], scale=int(np.ceil(exp.total_var_flow)))

        tax.boundary(linewidth=1.5)
        fontsize = 12
        offset = 0.12
        tax.bottom_axis_label("Q" + exp.reagents[exp.varying_pumps][0], fontsize=fontsize, offset=offset)
        tax.set_axis_limits(int(np.ceil(exp.total_var_flow)))
        tax.right_axis_label("Q" + exp.reagents[exp.varying_pumps][1], fontsize=fontsize, offset=offset)
        tax.left_axis_label("Q" + exp.reagents[exp.varying_pumps][2], fontsize=fontsize, offset=offset)
        tax.ticks(axis='lbr', linewidth=1, multiple=1, offset=0.02)
        tax.set_background_color(color="whitesmoke", alpha=0.7)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        ax[-1] = tax
        ax[-1].set_title('GP Regression Posterior')
    elif exp.nb_of_var_pumps == 4:
        ax.append(fig.add_subplot(gs[0:2, 0], projection='3d'))
        init_5d_visualisation(exp, ax[-1])
    else:
        ax.append(fig.add_subplot(gs[0:3, 0]))
        ax[-1].text(0.4, 0.5, 'No GPR visual')
        ax[-1].set_xticks([], minor=False)
        ax[-1].set_yticks([], minor=False)
        ax[-1].axis('off')

    # Plot Sampled costs
    ax.append(fig.add_subplot(gs[0:2, 2:]))
    ax[-1].set_title('Cost evolution', fontsize=10)
    ax[-1].set_ylabel('Cost')
    ax[-1].set_xlabel('Iteration')
    ax[-1].set_yscale('log')
    ax[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[-1].set_xlim(0, i if i > 0 else 1)
    ax[-1].grid(False, axis='y')
    ax[-1].plot(np.arange(0, i + 1), np.ones(i + 1) * exp.tol, 'r--', linewidth=0.5,
                label='Tolerance')

    # Plot the Minimal sampled cost
    ax.append(None)
    # ax[-1].set_title('Minimal sampled cost')
    # ax[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # ax[-1].set_xlim(0, i if i > 0 else 1)
    # ax[-1].set_yscale('log')
    # ax[-1].set_ylabel('cost')
    # ax[-1].grid(True, axis='y')
    # ax[-1].plot(np.arange(0, exp.maxiter + 1), np.ones(exp.maxiter + 1) * exp.tol,
    #             'r--', linewidth=0.5)

    # Plot the Computation time
    ax.append(None)
    # ax.append(fig.add_subplot(gs[2, 1:]))
    # ax[-1].set_title('Computation time')
    # ax[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # ax[-1].set_xlim(0, i if i > 0 else 1)
    # ax[-1].set_xlabel('nb of iteration')
    # ax[-1].set_ylabel('time (s)')
    # ax[-1].grid(True, axis='y')

    # Plot the current absorbance spectrum
    ax.append(fig.add_subplot(gs[2, :3]))
    ax[-1].set_title('Spectrum', fontsize=10)
    ax[-1].set_xlim(exp.spectro.wavelengths[0], exp.spectro.wavelengths[-1])
    ax[-1].set_ylim(0, 1)
    ax[-1].set_xlabel('Wavelength (nm)')
    ax[-1].set_ylabel('Extinction')
    ax[-1].grid(False)
    # Plot the target
    tg_style = exp.target_style
    if tg_style == 'full_spectrum':
        ax[-1].plot(exp.tg_spectrum.wavelengths, exp.tg_spectrum.absorbance,
                    '--', c='grey', label="target spectrum")
    elif tg_style == 'single_point':
        ax[-1].scatter(exp.tg_wl, exp.tg_abs, c='red', marker='o',
                       label='target absorbance')
    elif tg_style == 'lambda_max':
        ax[-1].plot([exp.tg_wl, exp.tg_wl], [0, 1], c='grey', linewidth=1, label='target lambda max')
    elif tg_style == 'pca' and getattr(exp, 'tg_spectrum', None) is not None:
        ax[-1].plot(exp.tg_spectrum.wavelengths, exp.tg_spectrum.absorbance,
                    '--', c='grey', label="target spectrum")

    # Plot the flows
    ax.append(fig.add_subplot(gs[2, -1]))
    ax[-1].set_ylim(0, exp.total_flow)
    ax[-1].set_ylabel('Flow (µL/s)')
    ax[-1].set_xlim(0, 1)
    ax[-1].set_xticks([], minor=False)

    return fig, ax


def _overlay_pca_decomposition(spec_ax, exp, spectrum):
    """Overlay the PCA view on the spectrum panel for a PCA target.

    On the spectrum axis: the basis reconstruction of the measured spectrum and
    of the target (what the frozen basis actually captures). As an inset: the
    score decomposition of both (PC1..PCn), the quantity the PCA cost compares.

    Fully defensive: returns silently if the PCA model / target is unavailable.
    """
    model = getattr(exp, 'pca_model', None)
    cost_fn = getattr(exp, 'pca_cost_fn', None)
    target = getattr(exp, 'tg_spectrum', None)
    if model is None or cost_fn is None or spectrum is None:
        return

    wl = model.wavelengths
    # Measured scores + reconstruction (in the basis' normalized space).
    meas_scores = model.transform(spectrum.wavelengths, spectrum.absorbance)
    meas_norm = model._normalize_vector(
        np.interp(wl, spectrum.wavelengths, spectrum.absorbance), model.normalize)
    meas_recon = meas_scores @ model.components + model.mean
    spec_ax.plot(wl, meas_recon, ':', color='#16a085', linewidth=1.1,
                 label='Measured (PCA reconstr.)')

    target_scores = getattr(cost_fn, 'target_scores', None)
    if target is not None and target_scores is not None:
        tgt_recon = target_scores @ model.components + model.mean
        spec_ax.plot(wl, tgt_recon, ':', color='grey', linewidth=1.0,
                     label='Target (PCA reconstr.)')

    # Inset: score bars (target vs measured) = the decomposition the cost uses.
    try:
        inset = spec_ax.inset_axes([0.62, 0.55, 0.36, 0.42])
        n = len(meas_scores)
        x = np.arange(n)
        if target_scores is not None:
            inset.bar(x - 0.2, target_scores, width=0.4, color='grey', label='target')
            inset.bar(x + 0.2, meas_scores, width=0.4, color='#16a085', label='measured')
        else:
            inset.bar(x, meas_scores, width=0.6, color='#16a085', label='measured')
        inset.axhline(0, color='k', linewidth=0.5)
        inset.set_title('PCA scores', fontsize=7)
        inset.set_xticks(x)
        inset.set_xticklabels([f'PC{k+1}' for k in range(n)], fontsize=6)
        inset.tick_params(axis='y', labelsize=6)
        inset.legend(fontsize=5, loc='upper right')
    except Exception as e:
        logger.debug("PCA score inset skipped: %s", e)


def plot_fig(exp, i, flows, costs, gaussian_process, best_min, times, spectrum,
             predicted_flows=None, final_fig=False, save_figs=True):
    """
    Plots the characteristics of the experiment at iteration i (or at the best
    iteration for the final figure). Include the visualisation of the GP
    regression if applicable (2 to 4 pumps), the measured costs, the best cost so
    far, the measured spectra (including a decomposition of the dyes if simulated)
    and the current flows.

    Parameters
    ----------
    exp : Experiment object
        The current experiment.
    i : int
        The current iteration.
    flows : array-like of shape (n_samples, n_features)
        List of the sampled flows.
    costs : array-like of shape (n_samples)
        List of the corresponding costs.
    gaussian_process : sk-learn GaussianProcessRegressor
        The experiment GPR.
    best_min : list of float
        List of the best cost so far.
    times : list of float
        List of the computation time at each iteration.
    spectrum : Spectrum object
        Measured spectrum.
    predicted_flows : array-like of shape n_features, optional
        The predicted flows to be evaluated next. The default is None.
    final_fig : bool, optional
        If True, plots the final figure, consisting of the best observed spectrum.
        The default is False.
    save_figs : bool, optional
        If True, saves the intermediate figures. The default is True.

    """
    fig, ax= init_fig(exp, i)

    if exp.nb_of_var_pumps == 2:
        # Plot the GPR with higher resolution for smoother visualization
        domain = np.linspace(0, exp.total_var_flow, 2000)
        mu_pred, sigma_pred = gaussian_process.predict(domain.reshape(-1, 1),
                                                       return_std=True)
        logger.debug("gaussian process predicted")
        # Inverse-warp to raw cost units (identity unless cost_transform=='log')
        # so the surface matches the measured points plotted in raw cost.
        std_surface_min = exp._gp_disp(mu_pred - BAND_SIGMA * sigma_pred)  # 99% CI
        std_surface_max = exp._gp_disp(mu_pred + BAND_SIGMA * sigma_pred)
        mu_pred = exp._gp_disp(mu_pred)
        std_surface_max = _cap_uncertainty(std_surface_max, mu_pred, costs)

        ax[0].plot(domain, mu_pred, "b")
        ax[0].fill_between(domain, std_surface_min, std_surface_max,
                           color='blue', alpha=0.2)

        # Plot the samples (and predicted sample)

        ax[0].scatter(flows[:, exp.varying_pumps][:, 0], exp._cost_to_gp(costs), c='red', marker='o', label='measured')
        if final_fig:
            ax[0].scatter(exp.opt_flows[exp.varying_pumps][0], exp._cost_to_gp(costs).min(), c='lime', marker='o',
                          label='optimal measurement', zorder=2)
        else:
            pred_flow = predicted_flows[exp.varying_pumps][0]
            idx = np.argmin(np.abs(domain - pred_flow))
            ax[0].plot([pred_flow] * 2,
                       [mu_pred[idx] - max(mu_pred) / 10, mu_pred[idx] + max(mu_pred) / 10],
                       c='green', linestyle='--', label='next measurement')

            ax[0].scatter([flows[-1][exp.varying_pumps][0]], [exp._cost_to_gp(costs)[-1]], c='blue',
                          marker='o', label='last measurement')

        # save BO subplot
        save_bo_plot(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs, predicted_flows,
                     final_fig)

    elif exp.nb_of_var_pumps == 3:
        # Generate display grid
        data_mu = dict()
        data_std_min = dict()
        data_std_max = dict()
        q1_list, q2_list = [], []
        scale = int(np.ceil(exp.total_var_flow))


        for j, k, l in simplex_iterator(scale=scale):
            # Convert to calculation space
            calc_j = j * exp.total_var_flow / scale
            calc_k = k * exp.total_var_flow / scale
            q1_list.append(calc_j)
            q2_list.append(calc_k)
        X_ternary = np.column_stack([q1_list, q2_list])
        mu_pred, sigma_pred = gaussian_process.predict(X_ternary, return_std=True)
        mu_pred = exp._gp_disp(mu_pred)  # raw cost units for the colormap
        mu_min, mu_max = mu_pred.min(), mu_pred.max()
        for idx, (j, k, l) in enumerate(simplex_iterator(scale=scale)):
            mu_val = mu_pred[idx]
            sigma_val = sigma_pred[idx]
            data_mu[(j, k)] = mu_val
            data_std_min[(j, k)] = mu_val - BAND_SIGMA * sigma_val
            data_std_max[(j, k)] = mu_val + BAND_SIGMA * sigma_val
        # Plot with display scale
        tax = ax[0]
        ternary.heatmap(data_mu, scale=scale, ax=tax.get_axes(),
                        cmap='viridis', style="triangular", permutation=None,
                        cbarlabel='Predicted cost (GP)' + exp._cost_axis_suffix(), vmin=mu_min, vmax=mu_max)

        # Plot measurement points
        ax[0].scatter(flows[:, exp.varying_pumps], c='red', marker='o', s=100,
                      label='measured', zorder=10, edgecolors='black', linewidth=2)
        if final_fig:
            ax[0].scatter([exp.opt_flows[exp.varying_pumps]], c='lime', marker='*', s=200,
                          label='optimal measurement', zorder=15, edgecolors='darkgreen', linewidth=2)
        else:
            ax[0].scatter([flows[-1][exp.varying_pumps]], c='blue', marker='o', s=120,
                          label='last measurement', zorder=12, edgecolors='darkblue', linewidth=2)
            ax[0].scatter([predicted_flows[exp.varying_pumps]], c='green', marker='*', s=150,
                          label='next measurement', zorder=13, edgecolors='darkgreen', linewidth=2)

        # Ternary diagram configuration
        ax[0].set_title(f'GP Ternary Diagram - Iteration {i}', fontsize=14)

        # Convert dicts to numpy arrays for saving
        domain = np.array(list(data_mu.keys()))  # shape (N, 2)
        mu_pred = np.array(list(data_mu.values()))
        std_surface_min = np.array(list(data_std_min.values()))
        std_surface_max = np.array(list(data_std_max.values()))

        if save_figs:
            save_bo_raw_data(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs)

        # === CORRECTED 3D FIGURE ===
        fig_3d = plt.figure(figsize=(12, 9))
        ax3d = fig_3d.add_subplot(111, projection='3d')

        # High resolution 3D grid for smooth surface
        resolution_3d = 100  # Resolution for 3D grid
        q1_range = np.linspace(0, exp.total_var_flow, resolution_3d)
        q2_range = np.linspace(0, exp.total_var_flow, resolution_3d)
        Q1_grid, Q2_grid = np.meshgrid(q1_range, q2_range)

        # Mask to respect ternary constraint
        Q3_grid = exp.total_var_flow - Q1_grid - Q2_grid
        mask_3d = Q3_grid >= 0

        # Valid points for GP prediction
        Q1_valid = Q1_grid[mask_3d]
        Q2_valid = Q2_grid[mask_3d]
        Q3_valid = Q3_grid[mask_3d]

        # GP predictions on 3D grid
        X_3d_gp = np.column_stack([Q1_valid, Q2_valid])
        mu_3d_pred, sigma_3d_pred = gaussian_process.predict(X_3d_gp, return_std=True)
        # Inverse-warp the mean to raw cost (identity unless cost_transform=='log')
        # so the surface and the red measured points share the same axis.
        mu_3d_disp = exp._gp_disp(mu_3d_pred)

        logger.debug("Generating 3D surface with %d valid points", len(Q1_valid))

        # === MAIN SURFACE with Z variation ===
        # Create complete grids with NaN for invalid zones
        Z_surface = np.full_like(Q1_grid, np.nan)
        Z_surface[mask_3d] = mu_3d_disp

        # Main surface with colors based on Z height (cost)
        surf = ax3d.plot_surface(Q1_grid, Q2_grid, Z_surface,
                                 alpha=0.8, cmap='viridis',
                                 linewidth=0, antialiased=True,
                                 vmin=mu_3d_disp.min(), vmax=mu_3d_disp.max())

        # === UNCERTAINTY SURFACES ===
        # Bounds computed in GP space then inverse-warped (asymmetric in raw cost).
        Z_lower = np.full_like(Q1_grid, np.nan)
        Z_lower[mask_3d] = exp._gp_disp(mu_3d_pred - BAND_SIGMA * sigma_3d_pred)

        ax3d.plot_surface(Q1_grid, Q2_grid, Z_lower,
                          alpha=0.3, color='lightblue',
                          linewidth=0, antialiased=True)

        # Upper surface (upper uncertainty bound), capped for display so an
        # exp-amplified log-space band cannot blow up the z-axis.
        Z_upper = np.full_like(Q1_grid, np.nan)
        Z_upper[mask_3d] = _cap_uncertainty(
            exp._gp_disp(mu_3d_pred + BAND_SIGMA * sigma_3d_pred), mu_3d_disp, costs)

        ax3d.plot_surface(Q1_grid, Q2_grid, Z_upper,
                          alpha=0.3, color='lightcoral',
                          linewidth=0, antialiased=True)

        # === EXPERIMENTAL POINTS IN 3D ===
        if len(flows) > 0:
            flows_3d = flows[:, exp.varying_pumps]  # Take only variable pumps
            ax3d.scatter(flows_3d[:, 0], flows_3d[:, 1], exp._cost_to_gp(costs),
                         c='red', s=80, marker='o',
                         label='Measured points', zorder=10)

            # Cost value annotations
            costs_disp = exp._cost_to_gp(np.asarray(costs, dtype=float))
            for idx, (flow, cost) in enumerate(zip(flows_3d, costs)):
                ax3d.text(flow[0], flow[1], costs_disp[idx] + (mu_3d_disp.max() - mu_3d_disp.min()) * 0.05,
                          f'{cost:.2f}', fontsize=9, color='black', weight='bold')

        # === PREDICTED NEXT POINT ===
        if not final_fig and predicted_flows is not None:
            pred_3d = predicted_flows[exp.varying_pumps]
            logger.debug("pred_3d: %s", pred_3d)

            pred_cost = exp._gp_disp(gaussian_process.predict([[pred_3d[0], pred_3d[1]]]))
            logger.debug("pred_cost: %s", pred_cost)
            ax3d.scatter([pred_3d[0]], [pred_3d[1]], [pred_cost],
                         c='lime', s=150, marker='o',
                         label='Next measurement', zorder=15)

        # === OPTIMAL POINT ===
        if final_fig and hasattr(exp, 'opt_flows') and exp.opt_flows is not None:
            opt_3d = exp.opt_flows[exp.varying_pumps]
            opt_cost = exp._cost_to_gp(costs).min()
            ax3d.scatter([opt_3d[0]], [opt_3d[1]], [opt_cost],
                         c='gold', s=200, marker='*',
                         label='Optimal point', zorder=20,
                         edgecolors='orange', linewidth=3)

        # === 3D CONFIGURATION ===
        ax3d.set_xlabel(f'Q1 - {exp.reagents[0]} (µL/min)', fontsize=12)
        ax3d.set_ylabel(f'Q2 - {exp.reagents[1]} (µL/min)', fontsize=12)
        ax3d.set_zlabel('Predicted cost' + exp._cost_axis_suffix(), fontsize=12)
        ax3d.set_title(f'3D GP Surface - Iteration {i}', fontsize=14)

        # Color bar
        fig_3d.colorbar(surf, ax=ax3d, shrink=0.6, aspect=30,
                        label='Predicted cost (GP)' + exp._cost_axis_suffix())

        # Legend
        ax3d.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

        # Optimized view
        ax3d.view_init(elev=25, azim=45)

        # Axis limits
        ax3d.set_xlim(0, exp.total_var_flow)
        ax3d.set_ylim(0, exp.total_var_flow)

        plt.tight_layout()

        # === SAVE ===
        if save_figs:
            fig_dir = os.path.join(exp.experiment_path, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            # Save PNG
            fig_3d_path = os.path.join(fig_dir, f"surface3D_iter_{i:03d}.png")
            fig_3d.savefig(fig_3d_path, dpi=300, bbox_inches='tight')
            logger.info("3D surface saved: %s", fig_3d_path)

            # Save pickle for reuse
            pickle_path = os.path.join(fig_dir, f"surface3D_iter_{i:03d}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(fig_3d, f)


    elif exp.nb_of_var_pumps == 4:
        # Create 4D visualization and get raw data
        domain, mu_pred, std_surface_min, std_surface_max = create_4d_plotly_volume_visualization(
            exp, i, flows, costs, gaussian_process,
            predicted_flows=predicted_flows,
            final_fig=final_fig, save_figs=save_figs)

        if save_figs:
            save_bo_raw_data(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs)

        # Matplotlib 3D voxel cloud: GP-predicted cost over the feasible
        # simplex (Q1+Q2+Q3 <= total), coloured by cost.
        # Deviation-weighted opacity: points near the GP mean are nearly
        # transparent so ridges/valleys of the cost landscape stand out.
        res = 18
        gaxis = np.linspace(0, exp.total_var_flow, res)
        Xg, Yg, Zg = np.meshgrid(gaxis, gaxis, gaxis, indexing='ij')
        grid = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        feas = grid.sum(axis=1) <= exp.total_var_flow + 1e-9
        mu_grid = exp._gp_disp(gaussian_process.predict(grid[feas]))
        feas_pts = grid[feas]

        meas_flows = flows[:, exp.varying_pumps][:, :3]
        meas_costs = exp._cost_to_gp(np.asarray(costs))[:len(meas_flows)]
        vmin = float(min(mu_grid.min(), meas_costs.min())) if len(meas_costs) else float(mu_grid.min())
        vmax = float(max(mu_grid.max(), meas_costs.max())) if len(meas_costs) else float(mu_grid.max())

        # Per-bin opacity: (dev_lo_σ, dev_hi_σ, alpha)
        mu_std = max(float(mu_grid.std()), 1e-9)
        deviation = np.abs(mu_grid - mu_grid.mean()) / mu_std
        dev_bins = [(0.0, 0.5, 0.06), (0.5, 1.0, 0.16),
                    (1.0, 1.5, 0.32), (1.5, 2.0, 0.55), (2.0, np.inf, 0.85)]
        sc = None
        for lo, hi, op in dev_bins:
            mask = (deviation >= lo) & (deviation < hi)
            if mask.sum() == 0:
                continue
            sc = ax[0].scatter(feas_pts[mask, 0], feas_pts[mask, 1], feas_pts[mask, 2],
                               c=mu_grid[mask], cmap='viridis', vmin=vmin, vmax=vmax,
                               alpha=op, s=10, depthshade=False)
        if sc is not None:
            fig.colorbar(sc, ax=ax[0], shrink=0.6, pad=0.1).set_label('cost')

        # Measured points on the same colour scale
        ax[0].scatter(meas_flows[:, 0], meas_flows[:, 1], meas_flows[:, 2],
                      c=meas_costs, cmap='viridis', vmin=vmin, vmax=vmax,
                      s=45, edgecolors='black', linewidths=0.6,
                      depthshade=False, label='measured')

        # Highlight the optimal (final) or next predicted point
        if final_fig and getattr(exp, 'opt_flows', None) is not None:
            o = np.asarray(exp.opt_flows)[exp.varying_pumps][:3]
            ax[0].scatter([o[0]], [o[1]], [o[2]], c='red', marker='*', s=220,
                          edgecolors='black', linewidths=0.6, depthshade=False,
                          label='optimal')
        elif predicted_flows is not None:
            p = np.asarray(predicted_flows)[exp.varying_pumps][:3]
            ax[0].scatter([p[0]], [p[1]], [p[2]], c='lime', marker='^', s=90,
                          edgecolors='black', linewidths=0.6, depthshade=False,
                          label='next predicted')

    else:  # exp.nb_of_var_pumps >= 5
        # For 5+ pumps, no visualization but save raw data
        logger.info("Saving raw GP data for %d variable pumps...", exp.nb_of_var_pumps)

        # Generate random sample points in the constrained space
        # Note: with constraint Q1+Q2+...+Qn = total_flow, we only need n-1 dimensions
        n_samples = 1000  # Number of points to sample
        sampled_points = []

        for _ in range(n_samples):
            # Generate random point that satisfies constraint using Dirichlet distribution
            point = np.random.dirichlet(np.ones(exp.nb_of_var_pumps)) * exp.total_var_flow
            # Keep only the first n-1 dimensions (the last one is determined by the constraint)
            sampled_points.append(point[:-1])

        domain = np.array(sampled_points)

        # Get GP predictions for these points (GP expects n-1 dimensions)
        mu_pred, sigma_pred = gaussian_process.predict(domain, return_std=True)
        # Inverse-warp to raw cost units (identity unless cost_transform=='log').
        std_surface_min = exp._gp_disp(mu_pred - BAND_SIGMA * sigma_pred)  # 99% CI
        std_surface_max = exp._gp_disp(mu_pred + BAND_SIGMA * sigma_pred)
        mu_pred = exp._gp_disp(mu_pred)
        std_surface_max = _cap_uncertainty(std_surface_max, mu_pred, costs)

        logger.info("GP predictions computed for %d variable pumps (iteration %d)", exp.nb_of_var_pumps, i)

        if save_figs:
            save_bo_raw_data(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs)

    # Plot the sampled cost with color gradient
    n_total = exp.nb_init_samples + i
    all_iterations = np.arange(n_total)
    all_costs = costs[:n_total]
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, max(n_total, 1)))

    # Init samples at x=0 with distinct marker
    ax[1].scatter([0] * exp.nb_init_samples, costs[:exp.nb_init_samples],
                  c=[colors[j] for j in range(exp.nb_init_samples)],
                  marker='s', s=40, edgecolors='black', linewidths=0.5,
                  zorder=3, label='Initial samples')

    # BO iterations with gradient
    if i > 0:
        bo_iters = np.arange(1, i + 1)
        bo_costs = costs[exp.nb_init_samples:n_total]
        bo_colors = [colors[exp.nb_init_samples + j] for j in range(len(bo_costs))]
        ax[1].scatter(bo_iters, bo_costs, c=bo_colors, marker='o', s=30,
                      edgecolors='black', linewidths=0.5, zorder=3, label='BO iterations')

    # Best cost so far line — start at first BO iteration (x=1) with running
    # minimum that includes the init batch + iter_1..iter_i. Skipping x=0
    # avoids the misleading "init-only" point.
    if i >= 1 and len(best_min) >= 2:
        current_best_min = best_min[1:i + 1]
        ax[1].plot(np.arange(1, i + 1), current_best_min, color='#e74c3c',
                   linewidth=1.5, label='Best cost')

    if not final_fig:
        ax[1].set_title(f'Cost evolution — current: {costs[-1]:.2e}', fontsize=10)

    # Plot the last/best spectrum
    if spectrum.absorbance is not None:
        cur_flows = flows[-1]
        ax[4].plot(spectrum.wavelengths, spectrum.absorbance, color='#2c3e50',
                   linewidth=1.2, label='Measured spectrum')
        # PCA target: overlay the basis reconstructions + an inset of the score
        # decompositions (target vs measured). Never let a plotting hiccup abort
        # the BO — the figure is non-critical.
        if getattr(exp, 'target_style', None) == 'pca':
            try:
                _overlay_pca_decomposition(ax[4], exp, spectrum)
            except Exception as e:
                logger.warning("PCA decomposition overlay skipped: %s", e)
        ax[4].set_title(f'Spectrum — flows: {[round(flw, 2) for flw in flows[-1].tolist()]} µL/s',
                        fontsize=10)
        if final_fig:
            fig.suptitle(f'Final figure (after {i} iterations)', fontsize=20)
            ax[1].scatter(exp.best_iter, best_min[-1], c='#e74c3c', s=80,
                          marker='*', zorder=5, label='Optimum')
            ax[4].set_title(f'Best spectrum (iter {exp.best_iter})\n'
                            f'Optimal flows: {[round(flw, 2) for flw in exp.opt_flows.tolist()]} µL/s')
            ax[1].legend(fontsize='small')
            cur_flows = exp.opt_flows

        # Plot the flow bar with gradient colors
        btm = 0
        n_pumps = len(cur_flows)
        bar_cmap = plt.cm.Set2
        for j, flw in enumerate(cur_flows):
            color = exp.pump_syst.pumps[j].color if hasattr(exp.pump_syst.pumps[j], 'color') and exp.pump_syst.pumps[j].color else bar_cmap(j / max(n_pumps - 1, 1))
            ax[5].bar(0.5, height=flw, bottom=btm, width=1, edgecolor='black',
                      color=color, alpha=0.7)
            btm += flw

    if exp.nb_of_var_pumps < 5:
        ax[0].legend(loc='upper left', fontsize='small', bbox_to_anchor=(-0.4, 1))
    ax[1].legend(fontsize='small')
    ax[4].legend(fontsize='small')

    # Save the Gaussian Process model for future use
    if save_figs:
        save_gaussian_process(exp, i, gaussian_process, flows, costs)

    if save_figs:
        fig_dir = os.path.join(exp.experiment_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f"{exp.experiment_name}_final_figure" if final_fig else f"{exp.experiment_name}_iteration_{i}"
        fig_file_path = os.path.join(fig_dir, f"{fig_name}.png")
        fig.savefig(fig_file_path, dpi=250)
        # Make de GIF
        if final_fig:
            figs = []
            img_files = sorted(glob.glob(os.path.join(fig_dir, f"{exp.experiment_name}_iteration_*.png")))

            # Add all existing iteration figures
            for img_file in img_files:
                figs.append(imageio.imread(img_file))

            # Add final figure
            final_img_path = os.path.join(fig_dir, f"{exp.experiment_name}_final_figure.png")
            if os.path.exists(final_img_path):
                figs.append(imageio.imread(final_img_path))

            # Save GIF only if images were found
            if figs:
                gif_path = os.path.join(exp.experiment_path, f"{exp.experiment_name}_gif.gif")
                imageio.mimsave(gif_path, figs, format='GIF', fps=2)
    return fig


def save_bo_plot(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs, predicted_flows, final_fig):
    """
    Save the Bayesian Optimization (BO) plot as an image and pickle file.
    """
    fig_bo, ax_bo = plt.subplots()

    # Plot Gaussian Process curve
    ax_bo.plot(domain, mu_pred, "b", label="GP mean")
    ax_bo.fill_between(domain, std_surface_min, std_surface_max, color='blue', alpha=0.2, label="99% CI")

    # Plot measured points
    ax_bo.scatter(flows[:, exp.varying_pumps][:, 0], exp._cost_to_gp(costs), c='red', marker='o', label='Measured')

    if final_fig:
        ax_bo.scatter(exp.opt_flows[exp.varying_pumps][0], exp._cost_to_gp(costs).min(), c='lime', marker='o',
                      label='Optimal measurement', zorder=2)
    else:
        idx = np.argmin(np.abs(domain - predicted_flows[exp.varying_pumps][0]))
        ax_bo.plot([predicted_flows[exp.varying_pumps][0]] * 2,
                   [mu_pred[idx] - max(mu_pred) / 10, mu_pred[idx] + max(mu_pred) / 10],
                   c='green', linestyle='--', label='Next measurement')

        ax_bo.scatter([flows[-1][exp.varying_pumps][0]], [exp._cost_to_gp(costs)[-1]], c='blue',
                      marker='o', label='Last measurement')

    # Save files
    figure_dir = os.path.join(exp.experiment_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    fig_bo.savefig(os.path.join(figure_dir, f"bo_{i}.png"), dpi=250)
    plot_data = {
        'domain': domain,
        'mu_pred': mu_pred,
        'std_surface_min': std_surface_min,
        'std_surface_max': std_surface_max,
        'flows': flows,
        'costs': costs,
        'predicted_flows': predicted_flows,
        'final_fig': final_fig,
        'iteration': i,
        'varying_pumps': exp.varying_pumps,
        'reagents': exp.reagents
    }

    with open(os.path.join(figure_dir, f"bo_data_{i}.pkl"), 'wb') as f:
        pickle.dump(plot_data, f)

    plt.close(fig_bo)

    save_bo_raw_data(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs)


def save_bo_raw_data(exp, i, domain, mu_pred, std_surface_min, std_surface_max, flows, costs):
    """Save raw BO GP data (domain, mu, sigma) to a .npz file for post-analysis."""
    data_dir = os.path.join(exp.experiment_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    np.savez(
        os.path.join(data_dir, f"bo_data_{i:02d}.npz"),
        domain=domain,
        mu_pred=mu_pred,
        std_surface_min=std_surface_min,
        std_surface_max=std_surface_max,
        flows=flows,
        costs=costs,
    )
    logger.debug("Raw BO data saved: bo_data_%02d.npz", i)


def save_gaussian_process(exp, i, gaussian_process, flows, costs):
    """
    Save the Gaussian Process object for future predictions.

    Parameters
    ----------
    exp : Experiment object
        The current experiment.
    i : int
        The current iteration.
    gaussian_process : sk-learn GaussianProcessRegressor
        The trained Gaussian Process model.
    flows : array-like of shape (n_samples, n_features)
        Training flows used for this GP.
    costs : array-like of shape (n_samples)
        Training costs used for this GP.
    """
    gp_dir = os.path.join(exp.experiment_path, "gaussian_processes")
    os.makedirs(gp_dir, exist_ok=True)

    # Save the GP model object
    gp_data = {
        'gaussian_process': gaussian_process,
        'iteration': i,
        'training_flows': flows,
        'training_costs': costs,
        'nb_of_var_pumps': exp.nb_of_var_pumps,
        'total_var_flow': exp.total_var_flow,
        'varying_pumps': exp.varying_pumps,
        'reagents': exp.reagents
    }

    gp_path = os.path.join(gp_dir, f"gp_model_iter_{i:03d}.pkl")
    with open(gp_path, 'wb') as f:
        pickle.dump(gp_data, f)

    logger.info("Gaussian Process saved: %s", gp_path)

def _varying_reagent_label(exp, k):
    """Axis label 'Q<reagent>' for the k-th varying pump (fallback Q<pump>)."""
    varying_reagents = list(np.asarray(exp.reagents)[exp.varying_pumps])
    name = varying_reagents[k] if k < len(varying_reagents) else f'pump{k + 1}'
    return 'Q' + str(name)


def init_5d_visualisation(exp, ax):
    """
    Clean 3D axes for the 4-variable GP posterior.

    For 4 varying pumps the GP lives in a 3D simplex (the 4th flow is fixed by
    the sum constraint Q1+Q2+Q3+Q4 = total). We therefore show the 3 free flows
    on the x/y/z axes and encode the predicted cost as colour (filled by
    plot_fig as a voxel cloud). The previous version hand-drew a fake axis
    system and stacked cost slices at z*1000 offsets, which was unreadable.

    Parameters
    ----------
    exp : Experiment object
        The current experiment.
    ax : matplotlib 3D axes
        The axis on which to plot the GPR.
    """
    ax.set_title('GP Regression Posterior (4 var)', fontsize=10)
    ax.set_xlabel(_varying_reagent_label(exp, 0))
    ax.set_ylabel(_varying_reagent_label(exp, 1))
    ax.set_zlabel(_varying_reagent_label(exp, 2))
    ax.set_xlim(0, exp.total_var_flow)
    ax.set_ylim(0, exp.total_var_flow)
    ax.set_zlim(0, exp.total_var_flow)
    ax.view_init(elev=20., azim=300)


def create_4d_plotly_volume_visualization(exp, i, flows, costs, gaussian_process, predicted_flows=None, final_fig=False,
                                          save_figs=True):
    """
    Create a 4D visualization using Plotly for 4 variable pumps.

    Returns
    -------
    tuple
        (points, mu, sigma, std_surface_min, std_surface_max) for data saving
    """

    # Dense regular grid of feasible points in the 3 free flow dimensions.
    # Using go.Scatter3d instead of go.Volume: Scatter3d renders reliably in
    # all browsers/Plotly versions. go.Volume's ray-marching WebGL pipeline
    # silently produces nothing on many setups.
    resolution = 25
    max_flow = exp.total_var_flow
    axis = np.linspace(0, max_flow, resolution)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Feasible cells: Q1 + Q2 + Q3 <= total (4th flow = remainder >= 0)
    feasible = grid.sum(axis=1) <= max_flow + 1e-9
    points = grid[feasible]
    mu_feas, sigma_feas = gaussian_process.predict(points, return_std=True)
    # Inverse-warp to raw cost units (identity unless cost_transform=='log').
    std_surface_min = exp._gp_disp(mu_feas - BAND_SIGMA * sigma_feas)
    mu = exp._gp_disp(mu_feas)
    std_surface_max = _cap_uncertainty(
        exp._gp_disp(mu_feas + BAND_SIGMA * sigma_feas), mu, costs)
    sigma = sigma_feas
    isomin = float(mu.min()) if mu.size else 0.0
    isomax = float(mu.max()) if mu.size else 1.0

    # Axis labels: the 3 free varying reagents (4th is implied by the sum).
    varying_reagents = list(np.asarray(exp.reagents)[exp.varying_pumps])
    lbl = [f'Q{varying_reagents[k]} (uL/min)' if k < len(varying_reagents)
           else f'Q{k+1} (uL/min)' for k in range(3)]

    # Deviation-weighted opacity: points near the GP mean are nearly invisible
    # so ridges/valleys of the cost landscape stand out clearly.
    mu_std = max(float(mu.std()), 1e-9)
    deviation = np.abs(mu - mu.mean()) / mu_std
    dev_bins = [(0.0, 0.5, 0.06), (0.5, 1.0, 0.16),
                (1.0, 1.5, 0.32), (1.5, 2.0, 0.55), (2.0, np.inf, 0.85)]

    fig = go.Figure()
    first = True
    for lo, hi, op in dev_bins:
        mask = (deviation >= lo) & (deviation < hi)
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=mu[mask],
                colorscale='Viridis',
                cmin=isomin, cmax=isomax,
                opacity=op,
                colorbar=dict(title='Predicted cost' + exp._cost_axis_suffix()) if first else None,
                showscale=first,
            ),
            showlegend=False,
            hovertemplate=(
                f'{lbl[0]}=%{{x:.1f}}<br>{lbl[1]}=%{{y:.1f}}<br>'
                f'{lbl[2]}=%{{z:.1f}}<br>cost=%{{marker.color:.3g}}<extra></extra>'
            ),
        ))
        first = False

    fig.update_layout(
        title=f'4D GP cost volume — Iteration {i}'
              + ('  (4th flow = total − Q1 − Q2 − Q3)'),
        scene=dict(
            xaxis_title=lbl[0],
            yaxis_title=lbl[1],
            zaxis_title=lbl[2],
            xaxis=dict(range=[0, max_flow]),
            yaxis=dict(range=[0, max_flow]),
            zaxis=dict(range=[0, max_flow]),
        )
    )

    # Add measured points, coloured by their actual measured cost
    if len(flows) > 0:
        measured_flows = flows[:, exp.varying_pumps][:, :3]
        measured_costs = exp._cost_to_gp(np.asarray(costs))[:len(measured_flows)]
        fig.add_trace(go.Scatter3d(
            x=measured_flows[:, 0],
            y=measured_flows[:, 1],
            z=measured_flows[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=measured_costs,
                colorscale='Viridis',
                cmin=isomin, cmax=isomax,
                symbol='circle',
                line=dict(width=1, color='DarkSlateGrey'),
                showscale=False,
            ),
            text=[f'cost={c:.3g}' for c in measured_costs],
            name='Measured points'
        ))

    # Add next predicted point
    if not final_fig and predicted_flows is not None:
        pred_3d = predicted_flows[exp.varying_pumps][:3]
        fig.add_trace(go.Scatter3d(
            x=[pred_3d[0]],
            y=[pred_3d[1]],
            z=[pred_3d[2]],
            mode='markers',
            marker=dict(
                size=8,
                color='lime',
                symbol='diamond',
                line=dict(width=2, color='DarkGreen')
            ),
            name='Next Predicted Point'
        ))

    # Add optimal point
    if final_fig and hasattr(exp, 'opt_flows') and exp.opt_flows is not None:
        opt_3d = exp.opt_flows[exp.varying_pumps][:3]
        fig.add_trace(go.Scatter3d(
            x=[opt_3d[0]],
            y=[opt_3d[1]],
            z=[opt_3d[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='gold',
                symbol='diamond',
                line=dict(width=2, color='DarkOrange')
            ),
            name='Optimal Point'
        ))

    # Open in a browser only for the final figure (avoid one tab per
    # iteration), and never let a display failure crash the BO thread.
    if final_fig:
        try:
            fig.show(renderer="browser")
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("Could not open 4D volume in browser: %s", exc)

    # Save the figure only if save_figs is True
    if save_figs:
        fig_dir = os.path.join(exp.experiment_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"4d_volume_iter_{i:03d}.html")
        fig.write_html(fig_path)
        logger.info("4D volume visualization saved: %s", fig_path)

    # Return raw data for saving
    return points, mu, std_surface_min, std_surface_max


def save_time_evolution_figure(experiment_path, sample_id, wavelengths,
                                spectra_data, flows=None, save_dir=None):
    """
    Save a publication-style time evolution figure showing all spectra
    with a smooth color gradient from early (light) to late (dark).

    Parameters
    ----------
    experiment_path : str
        Root experiment path.
    sample_id : str
        Sample identifier.
    wavelengths : np.ndarray
        Wavelength array.
    spectra_data : list of np.ndarray
        List of absorbance arrays, one per time step.
    flows : array-like, optional
        Flow rates for the title.
    save_dir : str, optional
        Override directory for saving. Defaults to experiment_path/figures.
    """
    if not spectra_data or len(spectra_data) < 2:
        return

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib import cm as _mpl_cm

    n = len(spectra_data)
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    cmap = _mpl_cm.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.9, n))

    for i, spectrum in enumerate(spectra_data):
        alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
        lw = 0.5 + 0.8 * (i / max(n - 1, 1))
        ax.plot(wavelengths, spectrum, color=colors[i], alpha=alpha,
                linewidth=lw)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Spectrum index (time →)', fontsize=10)

    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Extinction', fontsize=11)
    title = f'Time evolution — {sample_id}'
    if flows is not None:
        title += f'\nflows: {[round(f, 2) for f in flows]} µL/s'
    ax.set_title(title, fontsize=11)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)

    fig.tight_layout()

    out_dir = save_dir or os.path.join(experiment_path, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"time_evolution_{sample_id}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info("Time evolution figure saved: %s", fig_path)
