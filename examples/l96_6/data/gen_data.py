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
        t_test = jnp.array(f['t'][:])     # Shape: (num_test_pts,)
        L_windows = f.attrs['L']
        window_size = f.attrs['window_size']
        
    num_ics, num_test_pts, N = u_test.shape
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))

    # F is fixed to 6
    F_test = jnp.full((num_ics,), 6.0)

    # Single-window relative time grid required by the surrogate model
    t_star_window = t_test[:pts_pw + 1]

    # ── 2. Setup Model & Load Checkpoint ────────────────────────────────────
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    
    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor
    predict_batch = jax.jit(jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=0))

    # ── 3. Batched Autoregressive Rollout ───────────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")
    
    u_current_batch = u_test[:, 0, :]               

    x_pred_list = []
    
    for w in range(L_windows):
        pred_window = predict_batch(u_current_batch)    

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
    bins = [(5.9, 6.1)] 
    
    for lower, upper in bins:
        mask = (F_np >= lower) & (F_np < upper)
        if np.any(mask):
            group_mean = np.mean(l2_rel_np[mask], axis=0)
            label = f"F ∈ [{lower}, {upper}) (n={np.sum(mask)})"
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
        t_test = jnp.array(f['t'][:])     
        L_windows = f.attrs['L']
        window_size = f.attrs['window_size']

    num_ics, num_test_pts, N = u_test.shape
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))
    t_star_window = t_test[:pts_pw + 1]

    # F is fixed to 6
    F_test = jnp.full((num_ics,), 6.0)

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

    # ── Standard EnKF Setup (No F Augmentation) ──────────────────────────────
    def base_propagator(u, t):
        # u is strictly N variables
        preds = model.x_pred_fn(params, u, t_star_window)  # (pts_pw+1, 40)
        idx = round(t / DT_FINE)
        return preds[idx]
        
    predict_fn, update_fn = make_enkf(base_propagator, N, N_ens)
    # ────────────────────────────────────────────────────────────────────────

    # ── Binned Trajectory Selection ─────────────────────────────────────────
    try:
        m_samples = int(config.wandb.ckpt_name)
    except (ValueError, TypeError, AttributeError):
        m_samples = config.saving.get("total_plots", 2)

    F_np = np.array(F_test)
    bins = [(5.9, 6.1)]
    selected_ic_indices = []

    for lower, upper in bins:
        indices_in_bin = np.where((F_np >= lower) & (F_np < upper))[0]
        selected_for_bin = indices_in_bin[:m_samples]
        selected_ic_indices.extend(selected_for_bin.tolist())
        
    logging.info(f"Selected {len(selected_ic_indices)} total trajectories for F=6.0.")

    # ── Individual Trajectory Evaluation ────────────────────────────────────
    for ic_idx in selected_ic_indices:
        logging.info(f"--- EnKF Evaluation for IC {ic_idx} (N_ens={N_ens}, F={F_test[ic_idx]:.2f}) ---")
        u_current_true = u_test[ic_idx, 0, :]
        F_val = float(F_test[ic_idx])

        def lorenz_96(t, state, F=6.0):
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
        x0_hat = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        # Execute EnKF (Using pre-compiled predict_fn and update_fn)
        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0, y_obs_seq, obs_step_indices,
            H_seq, Q_fine, R, key,
            total_fine_steps, dt_fine=DT_FINE, dt_window=DT_WINDOW,
        )

        t_fine_axis = t_eval_fine[1:]
        
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
    predict_batch = jax.jit(jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window)[-1], in_axes=0))

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

    # ── EnKF Setup (No F Augmentation) ──────────────────────────────────────
    def base_propagator(u, t):
        preds = model.x_pred_fn(params, u, t_star_window)
        idx = round(t / dt_fine)
        return preds[idx]
        
    predict_fn, update_fn = make_enkf(base_propagator, N, N_ens)
    # ────────────────────────────────────────────────────────────────────────

    for ic in range(B):
        key = jax.random.PRNGKey(ic + 77777)
        u_true = u_test[ic, 0, :]
        F_val = float(F_test[ic])

        def lorenz_96(t, state, F=6.0):
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
        x0_hat = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        x_means, x_spreads, prior_means_at_obs = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, Q_fine, R_fixed, key, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )

        post_means_at_obs  = x_means[obs_step_indices_batch]   
        
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
    
    for k in range(max_additions):
        u_current = predict_batch(u_current)
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

def _plot_trajectory_summary_compare(
    t_ax: np.ndarray, x_true: np.ndarray, x_est_pi: np.ndarray, x_est_dd: np.ndarray,
    ic_idx: int, F_val: float, save_path: str, N: int = 40, dt_window: float = None,
    obs_coords: list = None
):
    """Generates a trajectory summary comparing Truth, PI, and DD models."""
    n_var_rows = N // 2
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    window_boundaries = np.arange(0, t_max + 1e-12, dt_window) if dt_window else []

    top_height, var_row_h = 3.2, 1.9
    fig = plt.figure(figsize=(14, top_height + n_var_rows * var_row_h))
    gs = gridspec.GridSpec(nrows=1 + n_var_rows, ncols=2, height_ratios=[top_height] + [var_row_h] * n_var_rows, hspace=0.55, wspace=0.32)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, np.abs(x_true - x_est_pi).mean(axis=1), color="#2196F3", label="Mean |error| PI")
    ax_top.plot(t_ax, np.abs(x_true - x_est_dd).mean(axis=1), color="#FF5722", label="Mean |error| DD")
    ax_top.set_yscale("log")
    ax_top.set_title(f"IC {ic_idx} (F = {F_val:.2f}) — Mean absolute error (PI vs DD)", fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", alpha=0.6)

    for i in range(N):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        for wb in window_boundaries: ax.axvline(x=wb, color="#78909C", linestyle="--", alpha=0.45, linewidth=0.6)
        
        ax.plot(t_ax, x_true[:, i], color="#37474F", linewidth=1.2, label="Truth")
        ax.plot(t_ax, x_est_pi[:, i], color="#2196F3", linewidth=1.0, linestyle="--", label="PI")
        ax.plot(t_ax, x_est_dd[:, i], color="#FF5722", linewidth=1.0, linestyle=":", label="DD")
        
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0: ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def _plot_grouped_metrics(
    x_ax: np.ndarray, data_pi: np.ndarray, data_dd: np.ndarray, F_vals: np.ndarray, 
    bins: list, y_label: str, title_base: str, save_path: str, log_scale: bool = True
):
    """Plots a 2x2 grid of metrics (e.g., L2, ERF, RMSE) grouped by F parameter bins."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (lower, upper) in enumerate(bins):
        ax = axes[i]
        mask = (F_vals >= lower) & (F_vals < upper)
        if np.any(mask):
            mean_pi = np.mean(data_pi[mask], axis=0)
            mean_dd = np.mean(data_dd[mask], axis=0)
            std_pi = np.std(data_pi[mask], axis=0)
            std_dd = np.std(data_dd[mask], axis=0)

            ax.plot(x_ax, mean_pi, color="#2196F3", marker="o", label="PI Mean")
            ax.fill_between(x_ax, mean_pi - std_pi, mean_pi + std_pi, color="#2196F3", alpha=0.15)
            ax.plot(x_ax, mean_dd, color="#FF5722", marker="s", label="DD Mean")
            ax.fill_between(x_ax, mean_dd - std_dd, mean_dd + std_dd, color="#FF5722", alpha=0.15)
        
        ax.set_title(f"F ∈ [{lower}, {upper}) (n={np.sum(mask)})", fontsize=11)
        if log_scale: ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
        
        if i >= 2: ax.set_xlabel("Time Step / Window", fontsize=10)
        if i % 2 == 0: ax.set_ylabel(y_label, fontsize=10)

    fig.suptitle(title_base, fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def evaluate_pi_vs_dd(config: ml_collections.ConfigDict, workdir: str):
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "l96_forcing_test.h5")

    logging.info(f"Loading dense test dataset from {test_file}...")
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:])     
        t_test = jnp.array(f['t'][:])     
        L_windows = f.attrs['L']
        window_size = f.attrs['window_size']
        
    num_ics, num_test_pts, N = u_test.shape
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))
    t_star_window = t_test[:pts_pw + 1]

    # F is fixed to 6
    F_test = jnp.full((num_ics,), 6.0)

    # Load PI
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_pi = os.path.join(os.getcwd(), config.wandb.ckpt_name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_pi)
    predict_pi = jax.jit(jax.vmap(lambda u: model_pi.x_pred_fn(model_pi.state.params, u, t_star_window), in_axes=0))

    # Load DD
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_dd = os.path.join(os.getcwd(), config.wandb.ckpt_name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_dd)
    predict_dd = jax.jit(jax.vmap(lambda u: model_dd.x_pred_fn(model_dd.state.params, u, t_star_window), in_axes=0))

    logging.info("Initiating batched rollout...")
    u_curr_pi, u_curr_dd = u_test[:, 0, :], u_test[:, 0, :]

    x_pred_pi, x_pred_dd = [], []
    for w in range(L_windows):
        pred_pi = predict_pi(u_curr_pi)
        pred_dd = predict_dd(u_curr_dd)

        x_pred_pi.append(pred_pi if w == 0 else pred_pi[:, 1:, :])
        x_pred_dd.append(pred_dd if w == 0 else pred_dd[:, 1:, :])
        
        u_curr_pi, u_curr_dd = pred_pi[:, -1, :], pred_dd[:, -1, :]

    x_pred_full_pi = jnp.concatenate(x_pred_pi, axis=1)
    x_pred_full_dd = jnp.concatenate(x_pred_dd, axis=1)

    # 1. Trajectory Summaries
    save_dir = os.path.join(workdir, "figures", "pi_vs_dd_openloop")
    for ic_idx in range(min(config.saving.get("total_plots", 2), num_ics)):
        _plot_trajectory_summary_compare(
            np.array(t_test), np.array(u_test[ic_idx]), np.array(x_pred_full_pi[ic_idx]), 
            np.array(x_pred_full_dd[ic_idx]), ic_idx, float(F_test[ic_idx]),
            os.path.join(save_dir, f"trajectory_ic_{ic_idx}.pdf"), N, window_size
        )

    # 2. Batch L2 Calculation & Grouped Plots
    norm_ref = jnp.linalg.norm(u_test, axis=-1) + 1e-12
    l2_pi = np.array(jnp.linalg.norm(x_pred_full_pi - u_test, axis=-1) / norm_ref)
    l2_dd = np.array(jnp.linalg.norm(x_pred_full_dd - u_test, axis=-1) / norm_ref)

    bins = [(5.9, 6.1)]
    _plot_grouped_metrics(
        np.array(t_test), l2_pi, l2_dd, np.array(F_test), bins,
        "Mean Relative L2 Error", "Open-Loop L2 Error by Forcing Parameter (PI vs DD)",
        os.path.join(save_dir, "batch_l2_grouped.pdf")
    )

def evaluate_with_enkf_pi_vs_dd(config: ml_collections.ConfigDict, workdir: str):
    from examples.l96_forcing.kf import run_enkf_smoother, init_ensemble, make_enkf
    obs_every_n = config.ekf.get("obs_every_n", 4)
    sigma_obs = config.ekf.get("sigma_obs", 0.5)
    N_ens = config.enkf.get("N_ens", 50)
    dynamic_vars = config.ekf.get("dynamic_vars", False)
    
    data_dir = config.training.get("data_dir", "data")
    with h5py.File(os.path.join(data_dir, "l96_forcing_test.h5"), 'r') as f:
        u_test = jnp.array(f['u'][:])     
        t_test = jnp.array(f['t'][:])     
        window_size = f.attrs['window_size']
        L_windows = f.attrs['L']

    N = u_test.shape[2]
    dt = float(t_test[1] - t_test[0])
    pts_pw = int(round(window_size / dt))
    t_star_window = t_test[:pts_pw + 1]
    DT_WINDOW, DT_FINE, DT_OBS = window_size, config.ekf.get("dt_fine", window_size), config.ekf.get("dt_obs", window_size)

    # F is fixed to 6
    F_test = jnp.full(u_test.shape[0], 6.0)

    # Reconstruct PI & DD (Standard State Propagators)
    def build_enkf(model_cls, ckpt_name):
        model = model_cls(config, t_star_window)
        ckpt = os.path.join(os.getcwd(), ckpt_name, "ckpt", model.state.__class__.__name__.lower() + "_model")
        ckpt = ckpt.replace("trainstate_model", "udon_dd_model").replace("forwardivp_model", "udon_model")
        
        model.state = restore_checkpoint(model.state, ckpt)
        def base_propagator(u, t):
            preds = model.x_pred_fn(model.state.params, u, t_star_window)
            return preds[round(t / DT_FINE)]
        return make_enkf(base_propagator, N, N_ens), model

    (pred_pi, upd_pi), model_pi = build_enkf(models.L96UDON, config.wandb.ckpt_name_pi)
    (pred_dd, upd_dd), model_dd = build_enkf(models.L96UDON_DD, config.wandb.ckpt_name_dd)

    Q_fine = scale_Q_for_fine_steps(jnp.eye(N) * config.enkf.get("sigma_model", 0.1)**2, round(DT_WINDOW / DT_FINE))
    P0 = jnp.eye(N) * config.ekf.get("P0_sigma", 1.0)**2
    
    m_vars = len(jnp.arange(0, N, obs_every_n))
    R = jnp.eye(m_vars) * sigma_obs**2
    
    # Ensure num_windows doesn't exceed available data
    num_windows = min(config.training.get("max_additions", 10), L_windows)
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(num_windows * DT_WINDOW, DT_FINE, DT_OBS)
    
    B = min(u_test.shape[0], config.ekf.get("batch_l2_size", 200))
    window_step_indices = np.array([round((k + 1) * DT_WINDOW / DT_FINE) - 1 for k in range(num_windows)])
    
    metrics = {
        'ol_l2_pi': np.zeros((B, num_windows)), 'ol_l2_dd': np.zeros((B, num_windows)),
        'enkf_l2_pi': np.zeros((B, num_windows)), 'enkf_l2_dd': np.zeros((B, num_windows)),
        'enkf_rmse_pi': np.zeros((B, num_windows)), 'enkf_rmse_dd': np.zeros((B, num_windows)),
        'spread_pi': np.zeros((B, num_windows)), 'spread_dd': np.zeros((B, num_windows)),
        'erf_pi': np.zeros((B, len(obs_times))), 'erf_dd': np.zeros((B, len(obs_times))),
        'rmse_prior_pi': np.zeros((B, len(obs_times))), 'rmse_prior_dd': np.zeros((B, len(obs_times))),
        'rmse_post_pi': np.zeros((B, len(obs_times))), 'rmse_post_dd': np.zeros((B, len(obs_times)))
    }

    logging.info(f"Running EnKF Batch Evaluation on {B} ICs for PI and DD...")
    
    # Open-Loop Rollout 
    step_ol_pi = jax.jit(jax.vmap(lambda u: model_pi.x_pred_fn(model_pi.state.params, u, t_star_window)[-1], in_axes=0))
    step_ol_dd = jax.jit(jax.vmap(lambda u: model_dd.x_pred_fn(model_dd.state.params, u, t_star_window)[-1], in_axes=0))
    
    u_curr_pi, u_curr_dd = u_test[:B, 0, :], u_test[:B, 0, :]
    for k in range(num_windows):
        u_curr_pi = step_ol_pi(u_curr_pi)
        u_curr_dd = step_ol_dd(u_curr_dd)
        
        # CORRECTED INDEXING: Use dataset temporal grid (dt) not EnKF fine steps (DT_FINE)
        ref_k = u_test[:B, (k + 1) * pts_pw, :] 
        metrics['ol_l2_pi'][:, k] = jnp.linalg.norm(u_curr_pi - ref_k, axis=1) / (jnp.linalg.norm(ref_k, axis=1) + 1e-12)
        metrics['ol_l2_dd'][:, k] = jnp.linalg.norm(u_curr_dd - ref_k, axis=1) / (jnp.linalg.norm(ref_k, axis=1) + 1e-12)

    # Solve trajectory & Run EnKF
    for ic in range(B):
        key = jax.random.PRNGKey(ic + 77777)
        u_true, F_val = u_test[ic, 0, :], float(F_test[ic])
        
        sol = solve_ivp(
            lambda t, state: (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F_val,
            [0.0, num_windows * DT_WINDOW], np.array(u_true), 
            t_eval=np.linspace(0.0, num_windows * DT_WINDOW, total_fine_steps + 1), rtol=1e-9, atol=1e-11
        )
        x_true_at_obs = sol.y.T[obs_step_indices + 1]

        H_list, y_obs_list = [], []
        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]
            
            # Reinstated Dynamic Vars Logic
            if dynamic_vars:
                key, sub = jax.random.split(key)
                obs_idx_vars = jax.random.choice(sub, N, shape=(m_vars,), replace=False)
            else:
                obs_idx_vars = jnp.arange(0, N, obs_every_n)
                
            H_t = jnp.zeros((m_vars, N)).at[jnp.arange(m_vars), obs_idx_vars].set(1.0)
            key, sub = jax.random.split(key)
            y_t = x_true_t[obs_idx_vars] + sigma_obs * jax.random.normal(sub, shape=(m_vars,))
            
            H_list.append(H_t)
            y_obs_list.append(y_t)
        
        H_seq = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)

        x0_hat = u_true + config.ekf.get("P0_sigma", 1.0) * jax.random.normal(jax.random.split(key)[0], shape=(N,))
        ens0 = init_ensemble(x0_hat, P0, N_ens, jax.random.split(key)[1])

        # Run Smoothers
        mean_pi, std_pi, prior_pi = run_enkf_smoother(pred_pi, upd_pi, ens0, y_obs_seq, obs_step_indices, H_seq, Q_fine, R, key, total_fine_steps, DT_FINE, DT_WINDOW)
        mean_dd, std_dd, prior_dd = run_enkf_smoother(pred_dd, upd_dd, ens0, y_obs_seq, obs_step_indices, H_seq, Q_fine, R, key, total_fine_steps, DT_FINE, DT_WINDOW)

        for suffix, x_m, x_s, p_m in [('pi', mean_pi, std_pi, prior_pi), ('dd', mean_dd, std_dd, prior_dd)]:
            
            err_prior = np.array(p_m) - x_true_at_obs
            err_post = np.array(x_m[obs_step_indices]) - x_true_at_obs
            
            metrics[f'rmse_prior_{suffix}'][ic, :] = np.sqrt(np.mean(err_prior**2, axis=1))
            metrics[f'rmse_post_{suffix}'][ic, :] = np.sqrt(np.mean(err_post**2, axis=1))
            metrics[f'erf_{suffix}'][ic, :] = metrics[f'rmse_prior_{suffix}'][ic, :] / (metrics[f'rmse_post_{suffix}'][ic, :] + 1e-12)

            for k, w_idx in enumerate(window_step_indices):
                ref_k = sol.y.T[w_idx + 1]
                metrics[f'enkf_l2_{suffix}'][ic, k] = jnp.linalg.norm(x_m[w_idx] - ref_k) / (jnp.linalg.norm(ref_k) + 1e-12)
                metrics[f'enkf_rmse_{suffix}'][ic, k] = jnp.sqrt(jnp.mean((x_m[w_idx] - ref_k)**2))
                metrics[f'spread_{suffix}'][ic, k] = jnp.sqrt(jnp.mean(x_s[w_idx]**2))

    # --- PLOTTING ---
    save_dir = os.path.join(workdir, "figures", "pi_vs_dd_enkf")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. OVERALL COMPARISON PLOTS (Averaged across all trajectories)
    window_idx = np.arange(1, num_windows + 1)
    
    # L2 & Calibration
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(window_idx, metrics['ol_l2_pi'].mean(axis=0),   marker="o", markersize=5, label="OL (PI)", color="#2196F3")
    axes[0].plot(window_idx, metrics['enkf_l2_pi'].mean(axis=0), marker="s", markersize=5, label=f"EnKF PI", color="#1976D2")
    axes[0].plot(window_idx, metrics['ol_l2_dd'].mean(axis=0),   marker="^", markersize=5, label="OL (DD)", color="#FF5722")
    axes[0].plot(window_idx, metrics['enkf_l2_dd'].mean(axis=0), marker="v", markersize=5, label=f"EnKF DD", color="#E64A19")
    axes[0].set(yscale="log", xlabel="Window index", ylabel="Mean relative L2 error", title="EnKF vs Open-loop L2")
    axes[0].grid(True, linestyle="--", alpha=0.6); axes[0].legend(fontsize=10)

    axes[1].plot(window_idx, metrics['spread_pi'].mean(axis=0),    marker="^", markersize=5, label="PI RMS σ", color="#4CAF50")
    axes[1].plot(window_idx, metrics['enkf_rmse_pi'].mean(axis=0), marker="s", markersize=5, linestyle="--", label="PI RMSE", color="#FF5722")
    axes[1].plot(window_idx, metrics['spread_dd'].mean(axis=0),    marker="^", markersize=5, label="DD RMS σ", color="#2196F3")
    axes[1].plot(window_idx, metrics['enkf_rmse_dd'].mean(axis=0), marker="s", markersize=5, linestyle="--", label="DD RMSE", color="#9C27B0")
    axes[1].set(yscale="log", xlabel="Window index", ylabel="Log scale", title="Calibration: Ensemble Spread vs RMSE")
    axes[1].grid(True, linestyle="--", alpha=0.6); axes[1].legend(fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, "batch_l2_per_window_enkf.pdf"), bbox_inches="tight", dpi=300); plt.close(fig)

    # ERF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(obs_times, metrics['erf_pi'].mean(axis=0), color="#2196F3", marker="o", label="PI EnKF ERF")
    ax.fill_between(obs_times, metrics['erf_pi'].mean(axis=0) - metrics['erf_pi'].std(axis=0), metrics['erf_pi'].mean(axis=0) + metrics['erf_pi'].std(axis=0), color="#2196F3", alpha=0.15)
    ax.plot(obs_times, metrics['erf_dd'].mean(axis=0), color="#FF5722", marker="s", label="DD EnKF ERF")
    ax.fill_between(obs_times, metrics['erf_dd'].mean(axis=0) - metrics['erf_dd'].std(axis=0), metrics['erf_dd'].mean(axis=0) + metrics['erf_dd'].std(axis=0), color="#FF5722", alpha=0.15)
    ax.axhline(y=1.0, color="#37474F", linestyle="--"); ax.set(yscale="log", xlabel="Observation time t", ylabel="ERF", title="Error Reduction Factor: PI vs DD")
    ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, "batch_erf_enkf.pdf"), bbox_inches="tight", dpi=300); plt.close(fig)

    # RMSE Comparison 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(obs_times, metrics['rmse_prior_pi'].mean(axis=0), color="#90CAF9", marker="o", linestyle=":", label="PI Prior")
    ax.plot(obs_times, metrics['rmse_post_pi'].mean(axis=0), color="#1E88E5", marker="o", label="PI Post")
    ax.plot(obs_times, metrics['rmse_prior_dd'].mean(axis=0), color="#FFAB91", marker="s", linestyle=":", label="DD Prior")
    ax.plot(obs_times, metrics['rmse_post_dd'].mean(axis=0), color="#F4511E", marker="s", label="DD Post")
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", label=f"σ_obs = {sigma_obs}")
    ax.set(yscale="log", xlabel="Observation time t", ylabel="RMSE", title="Prior vs Posterior RMSE: PI vs DD")
    ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, "batch_rmse_enkf.pdf"), bbox_inches="tight", dpi=300); plt.close(fig)

    # 2. GROUPED PLOTS (F Bins)
    bins = [(5.9, 6.1)]
    F_vals_B = np.array(F_test[:B])
    
    _plot_grouped_metrics(
        np.arange(1, num_windows + 1), metrics['enkf_l2_pi'], metrics['enkf_l2_dd'], F_vals_B, bins,
        "Relative L2 Error", "EnKF L2 Error Grouped by Forcing (PI vs DD)",
        os.path.join(save_dir, "grouped_enkf_l2.pdf")
    )
    
    _plot_grouped_metrics(
        obs_times, metrics['erf_pi'], metrics['erf_dd'], F_vals_B, bins,
        "ERF (Prior/Post RMSE)", "Error Reduction Factor Grouped by Forcing (PI vs DD)",
        os.path.join(save_dir, "grouped_erf.pdf")
    )
    
    _plot_grouped_metrics(
        np.arange(1, num_windows + 1), metrics['spread_pi'], metrics['spread_dd'], F_vals_B, bins,
        "RMS Ensemble Spread", "Ensemble Spread Grouped by Forcing (PI vs DD)",
        os.path.join(save_dir, "grouped_calibration.pdf")
    )