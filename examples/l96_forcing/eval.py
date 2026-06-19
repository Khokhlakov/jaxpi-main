import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax
from jax.tree_util import tree_map
from flax.jax_utils import replicate
from typing import Callable, Dict, Optional

from jaxpi.utils import restore_checkpoint
import examples.l96_forcing.models as models
from examples.l96_forcing.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps

import numpy as np
from scipy.integrate import solve_ivp
import h5py
from functools import partial


def _plot_trajectory_summary(
    t_ax:       np.ndarray,
    x_true:     np.ndarray,
    x_est:      np.ndarray,
    ic_idx:     int,
    F_val:      float,
    save_path:  str,
    x_std:      Optional[np.ndarray] = None,
    est_label:  str = "Prediction",
    N:          int = 40,
    dt_window:  Optional[float] = None,
    obs_coords: Optional[list[tuple[int, float, float]]] = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single IC.
    Includes uncertainty bands and observation markers.
    """
    x_true = np.asarray(x_true)
    x_est  = np.asarray(x_est)
    x_std  = np.asarray(x_std) if x_std is not None else None
 
    abs_error    = np.abs(x_true - x_est)
    mean_abs_err = abs_error.mean(axis=1)
 
    n_var_rows = N // 2
 
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window, t_max + 1e-12 * dt_window, dt_window)
    else:
        window_boundaries = np.array([])
        
    if obs_coords is not None:
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        obs_by_var = {k: sorted(v, key=lambda x: x[0]) for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}
 
    top_height   = 3.2
    var_row_h    = 1.9
    total_height = top_height + n_var_rows * var_row_h
 
    fig = plt.figure(figsize=(14, total_height))
    gs  = gridspec.GridSpec(
        nrows        = 1 + n_var_rows,
        ncols        = 2,
        figure       = fig,
        height_ratios= [top_height] + [var_row_h] * n_var_rows,
        hspace       = 0.55,
        wspace       = 0.32,
    )
 
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err, color="#E53935", linewidth=1.6, label="Mean |error| over variables")
 
    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.8, alpha=0.55, label="Window boundary" if wb == window_boundaries[0] else None)
 
    ax_top.set_xlabel("Time (t)", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} (F = {F_val:.2f}) — Mean absolute error across all {N} variables ({est_label})",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    TRUTH_COLOR = "#37474F"
    EST_COLOR   = "#1E88E5"
    BAND_COLOR  = "#90CAF9"
    OBS_COLOR   = "#E53935"
 
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.6, alpha=0.45)
 
        ax.plot(t_ax, x_true[:, i], color=TRUTH_COLOR, linewidth=1.0, label="Truth")
        ax.plot(t_ax, x_est[:, i], color=EST_COLOR, linewidth=1.0, linestyle="--", label=est_label)
        
        if x_std is not None:
            ax.fill_between(
                t_ax, x_est[:, i] - x_std[:, i], x_est[:, i] + x_std[:, i],
                color=BAND_COLOR, alpha=0.40, linewidth=0, label="±1σ",
            )
            
        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])
            ax.scatter(obs_times_i, obs_vals_i, marker="x", s=20, linewidths=0.7,
                       color=OBS_COLOR, zorder=5, label="Observation" if i == min(obs_by_var) else None)
 
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
 
        if row == 1 + n_var_rows - 1:
            ax.set_xlabel("Time (t)", fontsize=8)
        if col == 0:
            ax.set_ylabel("State", fontsize=8)
 
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", handlelength=1.2, framealpha=0.7)
 
    fig.suptitle(
        f"Trajectory summary — IC {ic_idx}  |  Forcing F = {F_val:.3f} | Estimator: {est_label}",
        fontsize=13, fontweight="bold", y=1.002,
    )
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Trajectory summary for IC {ic_idx} saved to: {save_path}")


def _plot_batch_l2_over_time(
    t_ax: np.ndarray,
    overall_mean_l2: np.ndarray,
    grouped_l2: Dict[str, np.ndarray],
    save_path: str
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t_ax, overall_mean_l2, color="#1E88E5", linewidth=2.5, label="Overall Mean (All Trajectories)")
    axes[0].set_ylabel("Mean Relative L2 Error", fontsize=11)
    axes[0].set_title("Overall Mean L2 Error Over Time", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=11)

    colors = ["#43A047", "#FB8C00", "#8E24AA", "#E53935", "#3949AB"]
    for i, (label, data) in enumerate(grouped_l2.items()):
        axes[1].plot(t_ax, data, linewidth=2.0, label=label, color=colors[i % len(colors)])
    
    axes[1].set_xlabel("Time (t)", fontsize=11)
    axes[1].set_ylabel("Mean Relative L2 Error", fontsize=11)
    axes[1].set_title("Mean L2 Error Grouped by Forcing Parameter (F)", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=11, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_erf(
    obs_times:  np.ndarray,
    erf_mean:   np.ndarray,
    erf_std:    np.ndarray,
    n_traj:     int,
    title:      str,
    save_path:  str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(obs_times, erf_mean, color="#FF5722", linewidth=2.0, marker="o", markersize=4, label=f"Mean ERF  (n = {n_traj} trajectories)")
    ax.fill_between(obs_times, erf_mean - erf_std, erf_mean + erf_std, color="#FF5722", alpha=0.20, linewidth=0, label="±1 std across trajectories")
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4, label="ERF = 1  (no reduction)")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_rmse_comparison(
    obs_times:       np.ndarray,
    prior_rmse_mean: np.ndarray,
    prior_rmse_std:  np.ndarray,
    post_rmse_mean:  np.ndarray,
    post_rmse_std:   np.ndarray,
    sigma_obs:       float,
    n_traj:          int,
    title:           str,
    save_path:       str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(obs_times, prior_rmse_mean, color="#2196F3", linewidth=2.0, marker="o", markersize=4, label=f"Prior RMSE  (n = {n_traj})")
    ax.fill_between(obs_times, prior_rmse_mean - prior_rmse_std, prior_rmse_mean + prior_rmse_std, color="#2196F3", alpha=0.18, linewidth=0, label="Prior ±1 std")
    ax.plot(obs_times, post_rmse_mean, color="#FF5722", linewidth=2.0, marker="s", markersize=4, label=f"Posterior RMSE  (n = {n_traj})")
    ax.fill_between(obs_times, post_rmse_mean - post_rmse_std, post_rmse_mean + post_rmse_std, color="#FF5722", alpha=0.18, linewidth=0, label="Posterior ±1 std")
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", linewidth=1.6, label=f"Measurement noise  σ_obs = {sigma_obs}")
    ax.set_yscale("log")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("RMSE  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_erf_split(
    obs_times:       np.ndarray,
    erf_mean_obs:    np.ndarray,
    erf_std_obs:     np.ndarray,
    erf_mean_unobs:  np.ndarray,
    erf_std_unobs:   np.ndarray,
    n_traj:          int,
    title:           str,
    save_path:       str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(obs_times, erf_mean_obs, color="#FF5722", linewidth=2.0, marker="o", markersize=4, label=f"Observed ERF (n={n_traj})")
    ax.fill_between(obs_times, erf_mean_obs - erf_std_obs, erf_mean_obs + erf_std_obs, color="#FF5722", alpha=0.20, linewidth=0)
    ax.plot(obs_times, erf_mean_unobs, color="#2196F3", linewidth=2.0, marker="^", markersize=4, linestyle="--", label=f"Unobserved ERF (n={n_traj})")
    ax.fill_between(obs_times, erf_mean_unobs - erf_std_unobs, erf_mean_unobs + erf_std_unobs, color="#2196F3", alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4, label="ERF = 1 (no reduction)")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_rmse_comparison_split(
    obs_times:            np.ndarray,
    prior_rmse_mean_obs:  np.ndarray,
    prior_rmse_std_obs:   np.ndarray,
    post_rmse_mean_obs:   np.ndarray,
    post_rmse_std_obs:    np.ndarray,
    prior_rmse_mean_unobs:np.ndarray,
    prior_rmse_std_unobs: np.ndarray,
    post_rmse_mean_unobs: np.ndarray,
    post_rmse_std_unobs:  np.ndarray,
    sigma_obs:            float,
    n_traj:               int,
    title:                str,
    save_path:            str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(obs_times, prior_rmse_mean_obs, color="#2196F3", linewidth=2.0, marker="o", markersize=4, label="Obs Prior")
    ax.fill_between(obs_times, prior_rmse_mean_obs - prior_rmse_std_obs, prior_rmse_mean_obs + prior_rmse_std_obs, color="#2196F3", alpha=0.15, linewidth=0)
    ax.plot(obs_times, post_rmse_mean_obs, color="#0D47A1", linewidth=2.0, marker="s", markersize=4, label="Obs Posterior")
    ax.fill_between(obs_times, post_rmse_mean_obs - post_rmse_std_obs, post_rmse_mean_obs + post_rmse_std_obs, color="#0D47A1", alpha=0.15, linewidth=0)
    ax.plot(obs_times, prior_rmse_mean_unobs, color="#FF9800", linewidth=2.0, marker="^", markersize=4, linestyle="--", label="Unobs Prior")
    ax.fill_between(obs_times, prior_rmse_mean_unobs - prior_rmse_std_unobs, prior_rmse_mean_unobs + prior_rmse_std_unobs, color="#FF9800", alpha=0.15, linewidth=0)
    ax.plot(obs_times, post_rmse_mean_unobs, color="#D84315", linewidth=2.0, marker="v", markersize=4, linestyle="--", label="Unobs Posterior")
    ax.fill_between(obs_times, post_rmse_mean_unobs - post_rmse_std_unobs, post_rmse_mean_unobs + post_rmse_std_unobs, color="#D84315", alpha=0.15, linewidth=0)
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", linewidth=1.6, label=f"Meas noise σ_obs = {sigma_obs}")
    ax.set_yscale("log")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("RMSE  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # ── 1. Load Dense Test Dataset ──────────────────────────────────────────
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "l96_forcing_test.h5")

    logging.info(f"Loading test dataset from {test_file}...")
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:])     # Shape: (num_ics, num_test_pts, 40)
        F_test = jnp.array(f['F'][:])     # Shape: (num_ics,)
        t_test = jnp.array(f['t'][:])     # Shape: (num_test_pts,)
        L_windows = f.attrs['L']
        window_size = f.attrs['window_size']
        
    num_ics, num_test_pts, N = u_test.shape
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))

    # Single-window relative time grid required by the surrogate model
    t_star_window = t_test[:pts_pw + 1]

    # ── 2. Setup Model & Load Checkpoint ────────────────────────────────────
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    
    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor
    predict_batch = jax.jit(jax.vmap(lambda u_aug: model.x_pred_fn(params, u_aug, t_star_window), in_axes=0))

    # ── 3. Batched Autoregressive Rollout ───────────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")
    
    u_current_batch = u_test[:, 0, :]               
    F_batch = F_test[:, None]                       

    x_pred_list = []
    
    for w in range(L_windows):
        u_aug_batch = jnp.concatenate([u_current_batch, F_batch], axis=-1)
        pred_window = predict_batch(u_aug_batch)    

        if w == 0:
            x_pred_list.append(pred_window)
        else:
            x_pred_list.append(pred_window[:, 1:, :])

        u_current_batch = pred_window[:, -1, :]

    x_pred_full = jnp.concatenate(x_pred_list, axis=1)  

    # ── 4. Generate Individual Trajectory Plots ─────────────────────────────
    total_plots = config.saving.get("total_plots", 2)
    for ic_idx in range(min(total_plots, num_ics)):
        logging.info(f"--- Generating detailed summary for IC {ic_idx} (F={F_test[ic_idx]:.2f}) ---")
        
        save_path = os.path.join(
            workdir, "figures", config.wandb.name, f"trajectory_summary_ic_{ic_idx}.pdf"
        )
        
        _plot_trajectory_summary(
            t_ax=np.array(t_test),
            x_true=np.array(u_test[ic_idx]),
            x_est=np.array(x_pred_full[ic_idx]),
            ic_idx=ic_idx,
            F_val=float(F_test[ic_idx]),
            est_label="DeepONet",
            save_path=save_path,
            N=model.N,
            dt_window=window_size
        )

    # ── 5. Generate Batch Error & Binned F Analysis ─────────────────────────
    logging.info("--- Computing Batch L2 Error Statistics ---")
    
    err = x_pred_full - u_test
    norm_err = jnp.linalg.norm(err, axis=-1)
    norm_ref = jnp.linalg.norm(u_test, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    l2_rel_np = np.array(l2_rel_per_traj_time)
    F_np = np.array(F_test)
    t_test_np = np.array(t_test)

    overall_mean_l2 = np.mean(l2_rel_np, axis=0)

    grouped_l2 = {}
    bins = [(5, 6), (6, 7), (7, 8), (8, 9.01)] 
    
    for lower, upper in bins:
        mask = (F_np >= lower) & (F_np < upper)
        if np.any(mask):
            group_mean = np.mean(l2_rel_np[mask], axis=0)
            upper_label = int(upper) if upper > 9 else upper
            label = f"F ∈ [{lower}, {upper_label}) (n={np.sum(mask)})"
            grouped_l2[label] = group_mean

    batch_save_path = os.path.join(
        workdir, "figures", config.wandb.name, "batch_l2_error_analysis.pdf"
    )
    
    _plot_batch_l2_over_time(
        t_ax=t_test_np, 
        overall_mean_l2=overall_mean_l2, 
        grouped_l2=grouped_l2, 
        save_path=batch_save_path
    )
    
    logging.info(f"Batch L2 error breakdown plot saved to: {batch_save_path}")


def evaluate_with_enkf(config: ml_collections.ConfigDict, workdir: str):
    from examples.l96_forcing.kf import run_enkf_smoother, init_ensemble, make_enkf

    obs_every_n  = config.ekf.get("obs_every_n",   4)
    sigma_obs    = config.ekf.get("sigma_obs",     0.5)
    P0_sigma     = config.ekf.get("P0_sigma",      1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",  False)
    N_ens        = config.enkf.get("N_ens",        50)
    sigma_model  = config.enkf.get("sigma_model",  0.1)

    specify_obs_idx   = config.kf.get("specify_obs_idx", False)
    obs_idx_list      = config.kf.get("obs_idx_list", None)

    # ── Load Dataset ────────────────────────────────────────────────────────
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "l96_forcing_test.h5")
    
    logging.info(f"Loading test dataset from {test_file} for EnKF...")
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:])     
        F_test = jnp.array(f['F'][:])     
        t_test = jnp.array(f['t'][:])     
        L_windows = f.attrs['L']
        window_size = f.attrs['window_size']

    num_ics, num_test_pts, N = u_test.shape
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))
    t_star_window = t_test[:pts_pw + 1]

    DT_WINDOW = float(config.get("dt_window", window_size))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))

    # ── Setup Model ─────────────────────────────────────────────────────────
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m_vars = len(obs_indices)
    R  = jnp.eye(m_vars) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    num_windows = 100#L_windows
    total_time  = num_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time = total_time, dt_fine = DT_FINE, dt_obs = DT_OBS,
    )

    # ── Binned Trajectory Selection ─────────────────────────────────────────
    try:
        # Attempt to parse m from ckpt_name as requested
        m_samples = int(config.wandb.ckpt_name)
    except (ValueError, TypeError, AttributeError):
        # Fallback if ckpt_name is a standard string
        m_samples = config.saving.get("total_plots", 2)

    F_np = np.array(F_test)
    bins = [(5, 6), (6, 7), (7, 8), (8, 9.01)]
    selected_ic_indices = []

    for lower, upper in bins:
        # Find all IC indices falling within the current F bin
        indices_in_bin = np.where((F_np >= lower) & (F_np < upper))[0]
        # Pick the first m_samples (or as many as available)
        selected_for_bin = indices_in_bin[:m_samples]
        selected_ic_indices.extend(selected_for_bin.tolist())
        
    logging.info(f"Selected {len(selected_ic_indices)} total trajectories across {len(bins)} F-parameter bins (target m={m_samples} per bin).")

    # ── Individual Trajectory Evaluation ────────────────────────────────────
    for ic_idx in selected_ic_indices:
        logging.info(f"--- EnKF Evaluation for IC {ic_idx} (N_ens={N_ens}, F={F_test[ic_idx]:.2f}) ---")
        u_current_true = u_test[ic_idx, 0, :]
        F_val = float(F_test[ic_idx])

        def lorenz_96(t, state, F=F_val):
            xp1 = np.roll(state, -1)
            xm1 = np.roll(state, 1)
            xm2 = np.roll(state, 2)
            return (xp1 - xm2) * xm1 - state + F

        t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time],
            y0=np.array(u_current_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = jnp.array(sol.y.T)                      
        x_true_at_obs = x_true_fine[obs_step_indices + 1]  

        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []

        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]

            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m_vars,), replace=False)
            else:
                obs_idx_vars = obs_indices

            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise

            H_list.append(H_t)
            y_obs_list.append(y_t)
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))

        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)

        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        # Build custom DeepONet propagator matching current trajectory's F
        def propagator(u, t):
            u_aug = jnp.concatenate([u, jnp.array([F_val])], axis=-1)
            preds = model.x_pred_fn(params, u_aug, t_star_window)  # (pts_pw+1, N)
            idx   = round(t / DT_FINE)
            return preds[idx]
            
        predict_fn, update_fn = make_enkf(propagator, N, N_ens)

        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0, y_obs_seq, obs_step_indices,
            H_seq, Q_fine, R, key,
            total_fine_steps, dt_fine=DT_FINE, dt_window=DT_WINDOW,
        )

        t_fine_axis = t_eval_fine[1:]
        
        # Added F_val directly into the output filename for easy visual sorting
        save_path = os.path.join(
            workdir, "figures", config.wandb.name, f"trajectory_summary_F_{F_val:.2f}_ic_{ic_idx}.pdf",
        )

        _plot_trajectory_summary(
            t_ax      = t_fine_axis,
            x_true    = np.array(x_true_fine[1:]),
            x_est     = np.array(x_means),
            x_std     = np.array(x_spreads),
            ic_idx    = ic_idx,
            F_val     = F_val,
            est_label = "EnKF mean",
            save_path = save_path,
            N         = model.N,
            dt_window = DT_WINDOW,
            obs_coords= obs_coords,
        )

        window_step_indices = np.array([round((w + 1) * DT_WINDOW / DT_FINE) - 1 for w in range(num_windows)])
        x_means_at_windows = x_means[window_step_indices]
        x_true_at_windows  = x_true_fine[window_step_indices + 1]

        l2_enkf     = jnp.linalg.norm(x_means_at_windows - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
        mean_spread = float(jnp.mean(x_spreads))
        logging.info(f"IC {ic_idx} | EnKF L2: {l2_enkf:.3e} | Mean σ: {mean_spread:.3e}")

    # ── Batch Evaluation ────────────────────────────────────────────────────
    _evaluate_batch_l2_enkf(
        model, params, t_star_window,
        Q_fine, P0, N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars, DT_FINE, DT_OBS, config, workdir,
        u_test, F_test, L_windows, DT_WINDOW
    )


def _evaluate_batch_l2_enkf(
    model, params, t_star_window,
    Q_fine, P0, N_ens, obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars, dt_fine: float, dt_obs: float,
    config, workdir, u_test, F_test, max_additions, dt_window
):
    from examples.l96_forcing.kf import run_enkf_smoother, init_ensemble, make_enkf

    specify_obs_idx   = config.kf.get("specify_obs_idx", False)
    obs_idx_list      = config.kf.get("obs_idx_list", None)
    
    N = model.N
    enkf_batch_size = config.ekf.get("batch_l2_size", 200)

    logging.info("Computing batch L2 per window (open-loop vs EnKF) …")
    B = min(u_test.shape[0], enkf_batch_size)
    logging.info(f"Using {B} ICs from pool for batch L2 / ERF evaluation (N_ens={N_ens}).")

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    np_obs_idx = np.array(obs_indices)
    np_unobs_idx = np.setdiff1d(np.arange(N), np_obs_idx)
    m = len(obs_indices)
    R_fixed = jnp.eye(m) * sigma_obs ** 2

    # Open-loop vmapped single-window predictor
    predict_batch = jax.jit(jax.vmap(lambda u_aug: model.x_pred_fn(params, u_aug, t_star_window)[-1], in_axes=0))

    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=dt_fine, dt_obs=dt_obs,
    )
    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    window_step_indices = np.array([round((k + 1) * dt_window / dt_fine) - 1 for k in range(max_additions)])

    # Accumulators (Original)
    enkf_l2_sum     = np.zeros(max_additions)
    enkf_spread_sum = np.zeros(max_additions)
    enkf_rmse_sum   = np.zeros(max_additions)
    erf_sum         = np.zeros(T_obs)
    erf_sq_sum      = np.zeros(T_obs)
    prior_rmse_sum    = np.zeros(T_obs)
    prior_rmse_sq_sum = np.zeros(T_obs)
    post_rmse_sum     = np.zeros(T_obs)
    post_rmse_sq_sum  = np.zeros(T_obs)

    # Accumulators (Split)
    enkf_l2_sum_obs       = np.zeros(max_additions)
    enkf_spread_sum_obs   = np.zeros(max_additions)
    enkf_rmse_sum_obs     = np.zeros(max_additions)
    enkf_l2_sum_unobs     = np.zeros(max_additions)
    enkf_spread_sum_unobs = np.zeros(max_additions)
    enkf_rmse_sum_unobs   = np.zeros(max_additions)

    erf_sum_obs      = np.zeros(T_obs)
    erf_sq_sum_obs   = np.zeros(T_obs)
    erf_sum_unobs    = np.zeros(T_obs)
    erf_sq_sum_unobs = np.zeros(T_obs)

    prior_rmse_sum_obs      = np.zeros(T_obs)
    prior_rmse_sq_sum_obs   = np.zeros(T_obs)
    post_rmse_sum_obs       = np.zeros(T_obs)
    post_rmse_sq_sum_obs    = np.zeros(T_obs)
    prior_rmse_sum_unobs    = np.zeros(T_obs)
    prior_rmse_sq_sum_unobs = np.zeros(T_obs)
    post_rmse_sum_unobs     = np.zeros(T_obs)
    post_rmse_sq_sum_unobs  = np.zeros(T_obs)

    @partial(jax.jit, static_argnums=(3,))   # t is still static; F is dynamic
    def _propagator_kernel(u, F_jax, t_star, t):
        u_aug = jnp.concatenate([u, F_jax], axis=-1)
        preds = model.x_pred_fn(params, u_aug, t_star)
        return preds[round(t / dt_fine)]

    # Build predict_fn/update_fn ONCE with a placeholder F (any concrete array works
    # to set the shape; JAX will compile a general version)
    _F_placeholder = jnp.array([0.0])
    def _base_propagator(u, t):
        return _propagator_kernel(u, _F_placeholder, t_star_window, t)
    
    predict_fn, update_fn = make_enkf(_base_propagator, N, N_ens)

    for ic in range(B):
        key = jax.random.PRNGKey(ic + 77777)
        u_true = u_test[ic, 0, :]
        F_val = float(F_test[ic])
        F_val_jax = jnp.array([float(F_test[ic])])

        def lorenz_96(t, state, F=F_val):
            xp1 = np.roll(state, -1)
            xm1 = np.roll(state,  1)
            xm2 = np.roll(state,  2)
            return (xp1 - xm2) * xm1 - state + F

        t_eval_fine = np.linspace(0.0, total_time_batch, total_fine_steps_batch + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time_batch],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = sol.y.T                
        x_true_at_obs = x_true_fine[obs_step_indices_batch + 1]

        H_list, y_obs_list = [], []

        for obs_idx in range(T_obs):
            x_true_t = x_true_at_obs[obs_idx]

            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m,), replace=False)
            else:
                obs_idx_vars = obs_indices

            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise

            H_list.append(H_t)
            y_obs_list.append(y_t)

        H_seq     = jnp.stack(H_list)      
        y_obs_seq = jnp.stack(y_obs_list)  

        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        def propagator(u, t, _F=F_val_jax):       # default-arg trick captures current value
            return _propagator_kernel(u, _F, t_star_window, t)
            
        predict_fn_ic, update_fn_ic = make_enkf(propagator, N, N_ens)
        x_means, x_spreads, prior_means_at_obs = run_enkf_smoother(
            predict_fn_ic, update_fn_ic,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, Q_fine, R_fixed, key, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )

        post_means_at_obs = x_means[obs_step_indices_batch]   
        
        # Tracking RMSE and ERF
        err_prior = np.array(prior_means_at_obs) - x_true_at_obs
        err_post  = np.array(post_means_at_obs) - x_true_at_obs

        # Global ERF/RMSE
        prior_rmse = np.sqrt(np.mean(err_prior ** 2, axis=1))
        post_rmse  = np.sqrt(np.mean(err_post ** 2, axis=1))
        erf_ic = prior_rmse / (post_rmse + 1e-12)

        erf_sum += erf_ic
        erf_sq_sum += erf_ic ** 2
        prior_rmse_sum += prior_rmse
        prior_rmse_sq_sum += prior_rmse ** 2
        post_rmse_sum += post_rmse
        post_rmse_sq_sum += post_rmse ** 2

        # Split: Observed
        prior_rmse_obs = np.sqrt(np.mean(err_prior[:, np_obs_idx] ** 2, axis=1))
        post_rmse_obs  = np.sqrt(np.mean(err_post[:, np_obs_idx] ** 2, axis=1))
        erf_ic_obs = prior_rmse_obs / (post_rmse_obs + 1e-12)

        erf_sum_obs += erf_ic_obs
        erf_sq_sum_obs += erf_ic_obs ** 2
        prior_rmse_sum_obs += prior_rmse_obs
        prior_rmse_sq_sum_obs += prior_rmse_obs ** 2
        post_rmse_sum_obs += post_rmse_obs
        post_rmse_sq_sum_obs += post_rmse_obs ** 2

        # Split: Unobserved
        prior_rmse_unobs = np.sqrt(np.mean(err_prior[:, np_unobs_idx] ** 2, axis=1))
        post_rmse_unobs  = np.sqrt(np.mean(err_post[:, np_unobs_idx] ** 2, axis=1))
        erf_ic_unobs = prior_rmse_unobs / (post_rmse_unobs + 1e-12)

        erf_sum_unobs += erf_ic_unobs
        erf_sq_sum_unobs += erf_ic_unobs ** 2
        prior_rmse_sum_unobs += prior_rmse_unobs
        prior_rmse_sq_sum_unobs += prior_rmse_unobs ** 2
        post_rmse_sum_unobs += post_rmse_unobs
        post_rmse_sq_sum_unobs += post_rmse_unobs ** 2

        # Accumulate L2 at boundaries
        for k in range(max_additions):
            # Evaluate using actual dataset truths
            ref_k   = u_test[ic, window_step_indices[k] + 1, :]
            step_k  = window_step_indices[k]
            x_hat_k = x_means[step_k]
            spread_k = x_spreads[step_k]
            err_k   = x_hat_k - ref_k

            # Global
            enkf_l2_sum[k] += float(jnp.linalg.norm(err_k) / (jnp.linalg.norm(ref_k) + 1e-12))
            enkf_rmse_sum[k] += float(jnp.sqrt(jnp.mean(err_k ** 2)))
            enkf_spread_sum[k] += float(jnp.sqrt(jnp.mean(spread_k ** 2)))

            # Split: Observed
            enkf_l2_sum_obs[k] += float(np.linalg.norm(err_k[np_obs_idx]) / (np.linalg.norm(ref_k[np_obs_idx]) + 1e-12))
            enkf_rmse_sum_obs[k] += float(np.sqrt(np.mean(err_k[np_obs_idx] ** 2)))
            enkf_spread_sum_obs[k] += float(np.sqrt(np.mean(spread_k[np_obs_idx] ** 2)))

            # Split: Unobserved
            enkf_l2_sum_unobs[k] += float(np.linalg.norm(err_k[np_unobs_idx]) / (np.linalg.norm(ref_k[np_unobs_idx]) + 1e-12))
            enkf_rmse_sum_unobs[k] += float(np.sqrt(np.mean(err_k[np_unobs_idx] ** 2)))
            enkf_spread_sum_unobs[k] += float(np.sqrt(np.mean(spread_k[np_unobs_idx] ** 2)))

    # Open-loop batched rollout over B
    ol_l2       = np.zeros(max_additions)
    ol_l2_obs   = np.zeros(max_additions)
    ol_l2_unobs = np.zeros(max_additions)
  
    u_current = u_test[:B, 0, :]
    F_batch = F_test[:B, None]
    
    for k in range(max_additions):
        u_aug = jnp.concatenate([u_current, F_batch], axis=-1)
        u_current = predict_batch(u_aug)
        ref_k = u_test[:B, window_step_indices[k] + 1, :]
        
        # Global
        numer = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom = jnp.linalg.norm(ref_k, axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))

        # Split
        err_k_np = np.array(u_current - ref_k)
        ref_k_np = np.array(ref_k)

        num_obs = np.linalg.norm(err_k_np[:, np_obs_idx], axis=1)
        den_obs = np.linalg.norm(ref_k_np[:, np_obs_idx], axis=1)
        ol_l2_obs[k] = float(np.mean(num_obs / (den_obs + 1e-12)))

        num_unobs = np.linalg.norm(err_k_np[:, np_unobs_idx], axis=1)
        den_unobs = np.linalg.norm(ref_k_np[:, np_unobs_idx], axis=1)
        ol_l2_unobs[k] = float(np.mean(num_unobs / (den_unobs + 1e-12)))

    # Averages
    l2_enkf     = enkf_l2_sum / B
    rmse_enkf   = enkf_rmse_sum / B
    spread_mean = enkf_spread_sum / B
    erf_mean = erf_sum / B
    erf_std  = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))
    prior_rmse_mean = prior_rmse_sum / B
    prior_rmse_std  = np.sqrt(np.maximum(prior_rmse_sq_sum / B - prior_rmse_mean ** 2, 0.0))
    post_rmse_mean  = post_rmse_sum / B
    post_rmse_std   = np.sqrt(np.maximum(post_rmse_sq_sum / B - post_rmse_mean ** 2, 0.0))

    l2_enkf_obs       = enkf_l2_sum_obs / B
    rmse_enkf_obs     = enkf_rmse_sum_obs / B
    spread_mean_obs   = enkf_spread_sum_obs / B
    l2_enkf_unobs     = enkf_l2_sum_unobs / B
    rmse_enkf_unobs   = enkf_rmse_sum_unobs / B
    spread_mean_unobs = enkf_spread_sum_unobs / B

    erf_mean_obs = erf_sum_obs / B
    erf_std_obs  = np.sqrt(np.maximum(erf_sq_sum_obs / B - erf_mean_obs ** 2, 0.0))
    erf_mean_unobs = erf_sum_unobs / B
    erf_std_unobs  = np.sqrt(np.maximum(erf_sq_sum_unobs / B - erf_mean_unobs ** 2, 0.0))

    prior_rmse_mean_obs = prior_rmse_sum_obs / B
    prior_rmse_std_obs  = np.sqrt(np.maximum(prior_rmse_sq_sum_obs / B - prior_rmse_mean_obs ** 2, 0.0))
    post_rmse_mean_obs  = post_rmse_sum_obs / B
    post_rmse_std_obs   = np.sqrt(np.maximum(post_rmse_sq_sum_obs / B - post_rmse_mean_obs ** 2, 0.0))

    prior_rmse_mean_unobs = prior_rmse_sum_unobs / B
    prior_rmse_std_unobs  = np.sqrt(np.maximum(prior_rmse_sq_sum_unobs / B - prior_rmse_mean_unobs ** 2, 0.0))
    post_rmse_mean_unobs  = post_rmse_sum_unobs / B
    post_rmse_std_unobs   = np.sqrt(np.maximum(post_rmse_sq_sum_unobs / B - post_rmse_mean_unobs ** 2, 0.0))

    # ── Plots ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    window_idx = np.arange(1, max_additions + 1)
    
    # 1A. Calibration Plot (Global)
    save_path = os.path.join(save_dir, "batch_l2_per_window_enkf.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(window_idx, ol_l2,   marker="o", markersize=4, linewidth=1.8, label="Open-loop (DeepONet)", color="#2196F3")
    ax.plot(window_idx, l2_enkf, marker="s", markersize=4, linewidth=1.8, label=f"EnKF mean (N_ens={N_ens})", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel(f"Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop: L2 per window", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx], fontsize=8, rotation=45, ha="left")
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)

    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean, marker="^", markersize=4, linewidth=1.8, label="RMS ensemble σ", color="#4CAF50")
    ax2.plot(window_idx, rmse_enkf,   marker="s", markersize=4, linewidth=1.8, linestyle="--", label="EnKF RMSE", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Window index", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs RMSE", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx], fontsize=8, rotation=45, ha="left")
    ax2_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)

    fig.suptitle(f"EnKF batch evaluation (B={B}, N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # 1B. Calibration Plot (Split)
    save_path_split = os.path.join(save_dir, "batch_l2_per_window_enkf_split.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(window_idx, ol_l2_obs,   marker="o", markersize=4, linewidth=1.8, label="Open-loop Obs", color="#2196F3")
    ax.plot(window_idx, l2_enkf_obs, marker="s", markersize=4, linewidth=1.8, label="EnKF Obs", color="#FF5722")
    ax.plot(window_idx, ol_l2_unobs, marker="^", markersize=4, linewidth=1.8, linestyle="--", label="Open-loop Unobs", color="#2196F3")
    ax.plot(window_idx, l2_enkf_unobs, marker="v", markersize=4, linewidth=1.8, linestyle="--", label="EnKF Unobs", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel(f"Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop (Split)", fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx], fontsize=8, rotation=45, ha="left")

    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean_obs, marker="o", markersize=4, linewidth=1.8, label="RMS σ Obs", color="#4CAF50")
    ax2.plot(window_idx, rmse_enkf_obs,   marker="s", markersize=4, linewidth=1.8, linestyle="--", label="RMSE Obs", color="#FF5722")
    ax2.plot(window_idx, spread_mean_unobs, marker="^", markersize=4, linewidth=1.8, linestyle=":", label="RMS σ Unobs", color="#4CAF50")
    ax2.plot(window_idx, rmse_enkf_unobs, marker="v", markersize=4, linewidth=1.8, linestyle="-.", label="RMSE Unobs", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Window index", fontsize=12)
    ax2.set_title("Calibration (Split)", fontsize=13)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx], fontsize=8, rotation=45, ha="left")

    fig.suptitle(f"EnKF Split L2/Calibration (Obs vs Unobs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path_split, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # 2A. Existing ERF Plot
    erf_save_path = os.path.join(save_dir, "batch_erf_enkf.pdf")
    _plot_erf(
        obs_times=obs_times_batch, erf_mean=erf_mean, erf_std=erf_std, n_traj=B,
        title=f"EnKF ERF\n(B={B}, N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        save_path=erf_save_path,
    )

    # 2B. Split ERF Plot
    erf_split_save_path = os.path.join(save_dir, "batch_erf_enkf_split.pdf")
    _plot_erf_split(
        obs_times=obs_times_batch, erf_mean_obs=erf_mean_obs, erf_std_obs=erf_std_obs,
        erf_mean_unobs=erf_mean_unobs, erf_std_unobs=erf_std_unobs, n_traj=B,
        title=f"Split ERF (Observed vs Unobserved)\n(B={B}, N_ens={N_ens}, σ_obs={sigma_obs})",
        save_path=erf_split_save_path,
    )

    # 3A. Prior/Posterior RMSE
    rmse_save_path = os.path.join(save_dir, "batch_rmse_enkf.pdf")
    _plot_rmse_comparison(
        obs_times=obs_times_batch, prior_rmse_mean=prior_rmse_mean, prior_rmse_std=prior_rmse_std,
        post_rmse_mean=post_rmse_mean, post_rmse_std=post_rmse_std, sigma_obs=sigma_obs, n_traj=B,
        title=f"EnKF prior vs posterior RMSE\n(B={B}, N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        save_path=rmse_save_path,
    )

    # 3B. Split Prior/Posterior RMSE
    rmse_split_save_path = os.path.join(save_dir, "batch_rmse_enkf_split.pdf")
    _plot_rmse_comparison_split(
        obs_times=obs_times_batch, prior_rmse_mean_obs=prior_rmse_mean_obs, prior_rmse_std_obs=prior_rmse_std_obs,
        post_rmse_mean_obs=post_rmse_mean_obs, post_rmse_std_obs=post_rmse_std_obs,
        prior_rmse_mean_unobs=prior_rmse_mean_unobs, prior_rmse_std_unobs=prior_rmse_std_unobs,
        post_rmse_mean_unobs=post_rmse_mean_unobs, post_rmse_std_unobs=post_rmse_std_unobs,
        sigma_obs=sigma_obs, n_traj=B,
        title=f"Split RMSE (Obs vs Unobs)\n(B={B}, N_ens={N_ens}, σ_obs={sigma_obs})",
        save_path=rmse_split_save_path,
    )