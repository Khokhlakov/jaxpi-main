"""
Data-driven evaluation functions for L96 UDON.
 
Mirrors the physics-informed eval functions but uses pre-computed test data
instead of reference ODE solutions.
"""
 
import os
import logging
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import gridspec
 
from utils import dd_get_test_data_rollout, build_obs_schedule, scale_Q_for_fine_steps
from jaxpi.utils import restore_checkpoint
import examples.l96_n40_f6_ics.models as models
from examples.l96_n40_f6_ics.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps

import ml_collections
 
 
# ── Plotting functions (reused from eval.py) ──────────────────────────────────
 
def _plot_l2_per_window(
    curves:    dict[str, np.ndarray],
    dt:        float,
    title:     str,
    save_path: str,
    colors:    dict[str, str] | None = None,
) -> None:
    """
    Plot one or more average-L2-per-window curves on a log-scale y-axis.
    (Reused from eval.py)
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for i, (label, l2_arr) in enumerate(curves.items()):
        num_windows = len(l2_arr)
        window_idx  = np.arange(1, num_windows + 1)
        color       = (colors or {}).get(label, default_colors[i % len(default_colors)])
        ax.plot(window_idx, l2_arr, marker="o", markersize=4,
                linewidth=1.8, label=label, color=color)
 
    num_windows = len(next(iter(curves.values())))
    window_idx  = np.arange(1, num_windows + 1)
 
    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # Secondary x-axis showing simulation time
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Batch L2-per-window plot saved to: {save_path}")
 
 
def _plot_trajectory_summary(
    t_ax:       np.ndarray,
    x_true:     np.ndarray,
    x_est:      np.ndarray,
    x_std:      np.ndarray | None,
    ic_idx:     int,
    est_label:  str,
    save_path:  str,
    N:          int = 40,
    dt_window:  float | None = None,
    obs_coords: list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single IC.
    (Reused from eval.py)
    """
    x_true = np.asarray(x_true)
    x_est  = np.asarray(x_est)
    x_std  = np.asarray(x_std) if x_std is not None else None
 
    abs_error    = np.abs(x_true - x_est)
    mean_abs_err = abs_error.mean(axis=1)
 
    n_var_rows = N // 2
 
    # Pre-compute window-boundary times
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])
 
    # Pre-compute per-variable observation times
    if obs_coords is not None:
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        obs_by_var = {k: sorted(v, key=lambda x: x[0])
                      for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}
 
    # Figure & GridSpec
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
 
    # Top panel: mean absolute error vs time
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err, color="#E53935", linewidth=1.6,
                label="Mean |error| over variables")
 
    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)
 
    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables  ({est_label})",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    # Colour palette
    TRUTH_COLOR = "#37474F"
    EST_COLOR   = "#1E88E5"
    BAND_COLOR  = "#90CAF9"
    OBS_COLOR   = "#E53935"
 
    # Per-variable panels
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.6, alpha=0.45)
 
        ax.plot(t_ax, x_true[:, i],
                color=TRUTH_COLOR, linewidth=1.0, label="Truth")
 
        ax.plot(t_ax, x_est[:, i],
                color=EST_COLOR, linewidth=1.0, linestyle="--",
                label=est_label)
 
        if x_std is not None:
            ax.fill_between(
                t_ax,
                x_est[:, i] - x_std[:, i],
                x_est[:, i] + x_std[:, i],
                color=BAND_COLOR, alpha=0.40, linewidth=0,
                label="±1σ",
            )
 
        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])
            ax.scatter(obs_times_i, obs_vals_i,
                       marker="x", s=25, linewidths=0.9,
                       color=OBS_COLOR, zorder=5,
                       label="Observation" if i == min(obs_by_var) else None)
 
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
 
        if row == 1 + n_var_rows - 1:
            ax.set_xlabel("t", fontsize=8)
        if col == 0:
            ax.set_ylabel("state", fontsize=8)
 
        if i == 0:
            ax.legend(fontsize=7, loc="upper right",
                      handlelength=1.2, framealpha=0.7)
 
    fig.suptitle(
        f"Trajectory summary — IC {ic_idx}  |  estimator: {est_label}",
        fontsize=13, fontweight="bold", y=1.002,
    )
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Trajectory summary for IC {ic_idx} saved to: {save_path}")
 
 
# ── Data-driven evaluation functions ───────────────────────────────────────────
 
def evaluate_dd(config: ml_collections.ConfigDict, workdir: str):
    """
    Data-driven evaluation: compare DeepONet predictions against test dataset.
    
    The test data has shape (num_ics=200, num_windows=31, num_t=51, N=40).
    For each IC, we:
      1. Extract the initial condition from the first time step of window 0
      2. Autoregressively roll out the network for num_windows steps
      3. Compare against the ground truth test data
      4. Generate plots
    """
    time_steps = 51
    num_windows_test = 31
    
    # Load test data: shape (200, 31, 51, 40)
    test_data_rollout = dd_get_test_data_rollout(
        data_dir=config.training.get("data_dir", "data/"),
        windows_per_traj=num_windows_test,
    )
    logging.info(f"Loaded test data: {test_data_rollout.shape}")
    
    # Setup model & load checkpoint
    model = models.L96UDON_DD(config, jnp.linspace(0.0, 0.25, time_steps))
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    logging.info(f"Restored data-driven DeepONet model for evaluation.")
    
    # Assimilation window settings
    dt_window = float(config.get("dt_window", 0.25))
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    
    for ic_idx in range(config.saving.total_plots):
        if ic_idx >= test_data_rollout.shape[0]:
            logging.warning(f"IC index {ic_idx} exceeds test data size; stopping.")
            break
            
        logging.info(f"--- Evaluating Data-Driven Trajectory for IC {ic_idx} ---")
        
        # Extract IC from first time step of first window
        u_current = test_data_rollout[ic_idx, 0, 0, :]  # shape (40,)
        logging.info(f"Initial condition shape: {u_current.shape}")
        
        # Autoregressive rollout
        x_pred_list = []
        t_full_list = []
        
        for win_idx in range(num_windows_test):
            # Predict over this window
            preds = model.x_net(params, u_current, t_star_window[:, None])  # (51, 40)
            x_pred_window = jnp.squeeze(preds)
            
            # Handle overlapping boundary (skip first point after window 0)
            if win_idx == 0:
                x_pred_list.append(x_pred_window)
                t_full_list.append(t_star_window)
            else:
                x_pred_list.append(x_pred_window[1:])  # skip first point
                t_offset = win_idx * dt_window
                t_full_list.append(t_star_window[1:] + t_offset)
            
            # Use last point as next IC
            u_current = x_pred_window[-1, :]
        
        x_pred_full = jnp.concatenate(x_pred_list, axis=0)
        t_star_full = jnp.concatenate(t_full_list, axis=0)
        
        # Ground truth: concatenate all windows from test data
        x_ref_windows = []
        for win_idx in range(num_windows_test):
            if win_idx == 0:
                x_ref_windows.append(test_data_rollout[ic_idx, win_idx, :, :])
            else:
                # Skip first point to match prediction overlap handling
                x_ref_windows.append(test_data_rollout[ic_idx, win_idx, 1:, :])
        
        x_ref_matched = jnp.concatenate(x_ref_windows, axis=0)
        
        # Compute L2 error
        total_l2_error = jnp.linalg.norm(x_pred_full - x_ref_matched) / jnp.linalg.norm(x_ref_matched)
        print(f"IC {ic_idx} | Full Rollout Trajectory L2 error: {total_l2_error:.3e}")
        
        # Plot trajectory summary
        _plot_trajectory_summary(
            t_ax       = np.array(t_star_full),
            x_true     = np.array(x_ref_matched),
            x_est      = np.array(x_pred_full),
            x_std      = None,
            ic_idx     = ic_idx,
            est_label  = "DeepONet (DD)",
            save_path  = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_dd_ic_{ic_idx}.pdf",
            ),
            N          = model.N,
            dt_window  = float(dt_window),
            obs_coords = None,
        )
        
        # Heatmap plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        im0 = axes[0].pcolormesh(np.arange(model.N), t_star_full, x_ref_matched, 
                                  cmap='viridis', shading='auto')
        axes[0].set_title(f"Ground Truth (IC {ic_idx})", fontsize=14)
        axes[0].set_ylabel("Time (t)", fontsize=14)
        axes[0].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].pcolormesh(np.arange(model.N), t_star_full, x_pred_full, 
                                  cmap='viridis', shading='auto')
        axes[1].set_title(f"DeepONet (DD) Prediction (IC {ic_idx})", fontsize=14)
        axes[1].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im1, ax=axes[1])
        
        abs_error = jnp.abs(x_ref_matched - x_pred_full)
        im2 = axes[2].pcolormesh(np.arange(model.N), t_star_full, abs_error, 
                                  cmap='magma', shading='auto')
        axes[2].set_title(f"Absolute Error (IC {ic_idx})", fontsize=14)
        axes[2].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im2, ax=axes[2])
        
        # Mark window boundaries
        for ax in axes:
            for w in range(1, num_windows_test):
                boundary_time = w * dt_window
                ax.axhline(y=boundary_time, color='white', linestyle=':', alpha=0.5)
        
        fig.tight_layout()
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"udon_dd_rollout_analysis_ic_{ic_idx}.pdf")
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        
        logging.info(f"Evaluation plot for IC {ic_idx} saved to: {fig_path}")
    
    # Batch evaluation
    _evaluate_batch_l2_dd(config, workdir, test_data_rollout)
 
 
def _evaluate_batch_l2_dd(config, workdir, test_data_rollout):
    """
    Compute batch-averaged open-loop L2 error per window using test data as ground truth.
    
    Test data has shape (num_ics, num_windows, num_t, N).
    We evaluate the model at the end of each window and compare to ground truth.
    """
    time_steps = 51
    dt_window = float(config.get("dt_window", 0.25))
    num_windows_test = test_data_rollout.shape[1]
    num_ics = test_data_rollout.shape[0]
    N = test_data_rollout.shape[3]
    
    # Load model
    model = models.L96UDON_DD(config, jnp.linspace(0.0, 0.25, time_steps))
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    
    logging.info("Computing batch L2 per window (open-loop, data-driven) …")
    
    l2_per_window: list[float] = []
    u_current_batch = test_data_rollout[:, 0, 0, :]  # (num_ics, N) - initial conditions
    
    # vmapped single-window predictor
    predict_one_window = jax.jit(
        jax.vmap(
            lambda u: model.x_net(params, u, t_star_window[:, None])[-1],
            in_axes=0,
        )
    )
    
    for w in range(num_windows_test - 1):
        # Advance batch by one window
        u_current_batch = predict_one_window(u_current_batch)  # (num_ics, N)
        
        # Ground truth at window boundary w+1 (skip first point)
        x_ref_w = test_data_rollout[:, w + 1, 0, :]  # (num_ics, N) - first point of window w+1
        
        # Per-IC relative L2
        numer = jnp.linalg.norm(u_current_batch - x_ref_w, axis=1)  # (num_ics,)
        denom = jnp.linalg.norm(x_ref_w, axis=1)                    # (num_ics,)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
        
        logging.info(f"  Window {w + 1:>3d} | mean L2: {l2_mean:.3e}")
    
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_dd.pdf")
    _plot_l2_per_window(
        curves    = {"Open-loop (DeepONet DD)": np.array(l2_per_window)},
        dt        = dt_window,
        title     = f"Data-driven open-loop: batch-average L2 per window  (B={num_ics})",
        save_path = save_path,
        colors    = {"Open-loop (DeepONet DD)": "#2196F3"},
    )
 
 
# ── EKF with data-driven model ─────────────────────────────────────────────────
 
def evaluate_with_ekf_dd(config: ml_collections.ConfigDict, workdir: str):
    """
    Data-driven EKF evaluation: combine DeepONet predictions with Kalman filtering
    using test dataset as ground truth reference.
    
    Uses the same EKF machinery as the physics-informed version, but:
      - Predictions come from the data-driven DeepONet
      - Ground truth comes from the test dataset instead of ODE solve
    """
    from examples.KS.kf import make_ekf, run_ekf_smoother, scale_Q_for_fine_steps, build_obs_schedule
    
    # EKF hyperparameters
    obs_every_n = config.ekf.get("obs_every_n",  4)
    sigma_obs   = config.ekf.get("sigma_obs",    0.5)
    sigma_proc  = config.ekf.get("sigma_proc",   0.1)
    P0_sigma    = config.ekf.get("P0_sigma",     1.0)
    dynamic_vars = config.ekf.get("dynamic_vars", False)
    
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
    
    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))
    
    time_steps = 51
    num_windows_test = 31
    
    # Load test data
    test_data_rollout = dd_get_test_data_rollout(
        data_dir=config.training.get("data_dir", "data/"),
        windows_per_traj=num_windows_test,
    )
    
    # Load model
    t_star_window = jnp.linspace(0.0, DT_WINDOW, time_steps)
    model = models.L96UDON_DD(config, t_star_window)
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    N = model.N
    
    # Build EKF propagator using the DeepONet
    # Wrapper that makes DeepONet callable as (u, t_query) -> u
    def predict_fn_wrapper(u: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Wrapped predictor: returns (x_pred, jacobian)."""
        # For data-driven, we predict over a window using the network
        # Map t to appropriate position within window
        t_query = jnp.array([t])
        x_pred = model.x_net(params, u, t_query)  # (1, N) or shape with 1 time point
        x_pred = jnp.squeeze(x_pred)
        
        # Compute Jacobian via jacfwd
        jacobian = jax.jacfwd(lambda u_in: model.x_net(params, u_in, t_query))(u)
        jacobian = jnp.squeeze(jacobian)
        
        return x_pred, jacobian
    
    def update_fn(x_prior: jnp.ndarray, P_prior: jnp.ndarray, 
                  H: jnp.ndarray, y: jnp.ndarray, R: jnp.ndarray) -> tuple:
        """EKF update step (standard Kalman update)."""
        innovation = y - H @ x_prior
        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ jnp.linalg.inv(S)
        x_post = x_prior + K @ innovation
        P_post = (jnp.eye(len(x_prior)) - K @ H) @ P_prior
        return x_post, P_post
    
    # Noise covariances
    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_proc ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
    
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
    
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
    
    # Observation schedule
    total_time = num_windows_test * DT_WINDOW
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )
    
    # Per-IC evaluation
    for ic_idx in range(min(config.saving.total_plots, test_data_rollout.shape[0])):
        logging.info(f"--- EKF (DD) Evaluation for IC {ic_idx} ---")
        u_current_true = test_data_rollout[ic_idx, 0, 0, :]
        
        # Ground truth: concatenate all windows from test data (skip overlaps)
        x_true_windows = []
        for win_idx in range(num_windows_test):
            if win_idx == 0:
                x_true_windows.append(test_data_rollout[ic_idx, win_idx, :, :])
            else:
                x_true_windows.append(test_data_rollout[ic_idx, win_idx, 1:, :])
        
        x_true_full = jnp.concatenate(x_true_windows, axis=0)  # (total_t, N)
        
        # Build time axis matching the test data
        t_eval = np.linspace(0.0, total_time, x_true_full.shape[0])
        
        # Extract ground truth at observation times (interpolate if necessary)
        # For simplicity, we'll assume observations happen at available time steps
        x_true_at_obs_list = []
        for obs_t in obs_times:
            # Find closest time step in t_eval
            closest_idx = np.argmin(np.abs(t_eval - obs_t))
            x_true_at_obs_list.append(x_true_full[closest_idx])
        x_true_at_obs = jnp.stack(x_true_at_obs_list)
        
        # Build observation sequence
        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []
        
        for obs_idx in range(len(obs_times)):
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
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))
        
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
        
        # Perturbed IC
        key, key_ic = jax.random.split(key)
        x0_hat = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        
        # Run EKF smoother
        x_hats, Ps, _ = run_ekf_smoother(
            predict_fn_wrapper, update_fn,
            x0_hat, P0,
            y_obs_seq,
            obs_step_indices,
            H_seq,
            Q_fine,
            R,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
        
        ekf_std = np.sqrt(np.clip(
            np.diagonal(np.array(Ps), axis1=1, axis2=2), 0, None
        ))
        
        # Upsample true trajectory to match filter output if needed
        # For now, use the available test data
        _plot_trajectory_summary(
            t_ax       = t_eval,
            x_true     = np.array(x_true_full),
            x_est      = np.array(x_hats[:x_true_full.shape[0]]),
            x_std      = ekf_std[:x_true_full.shape[0]] if ekf_std.shape[0] >= x_true_full.shape[0] else None,
            ic_idx     = ic_idx,
            est_label  = "EKF (DD) estimate",
            save_path  = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_ekf_dd_ic_{ic_idx}.pdf",
            ),
            N          = N,
            dt_window  = DT_WINDOW,
            obs_coords = obs_coords,
        )
        
        # L2 error at window boundaries
        window_indices = [
            np.argmin(np.abs(t_eval - (w + 1) * DT_WINDOW))
            for w in range(num_windows_test)
        ]
        x_hats_at_windows = x_hats[window_indices]
        x_true_at_windows = x_true_full[window_indices]
        
        l2_ekf = jnp.linalg.norm(x_hats_at_windows - x_true_at_windows) \
               / jnp.linalg.norm(x_true_at_windows)
        print(f"IC {ic_idx} | EKF (DD) L2 (at window boundaries): {l2_ekf:.3e}")
    
    # Batch evaluation
    _evaluate_batch_l2_ekf_dd(config, workdir, test_data_rollout)
 
 
def _evaluate_batch_l2_ekf_dd(config, workdir, test_data_rollout):
    """
    Batch-averaged L2 error comparison between data-driven DeepONet and EKF.
    
    This is a simplified version that focuses on the open-loop comparison.
    Full EKF smoother for all ICs is computationally expensive.
    """
    time_steps = 51
    dt_window = float(config.get("dt_window", 0.25))
    num_windows_test = test_data_rollout.shape[1]
    num_ics = test_data_rollout.shape[0]
    N = test_data_rollout.shape[3]
    
    # Load model
    model = models.L96UDON_DD(config, jnp.linspace(0.0, 0.25, time_steps))
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    
    logging.info("Computing batch L2 per window (data-driven open-loop) …")
    
    l2_per_window: list[float] = []
    u_current_batch = test_data_rollout[:, 0, 0, :]
    
    predict_one_window = jax.jit(
        jax.vmap(
            lambda u: model.x_net(params, u, t_star_window)[-1],
            in_axes=0,
        )
    )
    
    for w in range(num_windows_test - 1):
        u_current_batch = predict_one_window(u_current_batch)
        x_ref_w = test_data_rollout[:, w + 1, 0, :]
        
        numer = jnp.linalg.norm(u_current_batch - x_ref_w, axis=1)
        denom = jnp.linalg.norm(x_ref_w, axis=1)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
        
        logging.info(f"  Window {w + 1:>3d} | mean L2: {l2_mean:.3e}")
    
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_ekf_dd.pdf")
    _plot_l2_per_window(
        curves    = {"Open-loop (DeepONet DD)": np.array(l2_per_window)},
        dt        = dt_window,
        title     = f"Data-driven: batch-average L2 per window  (B={num_ics})",
        save_path = save_path,
        colors    = {"Open-loop (DeepONet DD)": "#2196F3"},
    )
 

def _plot_erf(
    obs_times:  np.ndarray,   # (T_obs,)  absolute observation times
    erf_mean:   np.ndarray,   # (T_obs,)  mean ERF across trajectories
    erf_std:    np.ndarray,   # (T_obs,)  std  ERF across trajectories
    n_traj:     int,          # number of trajectories averaged
    title:      str,
    save_path:  str,
) -> None:
    """
    Plot the Error Reduction Factor (ERF = prior RMSE / posterior RMSE)
    averaged across ``n_traj`` trajectories and over observation times.

    A horizontal reference line at ERF = 1 marks the boundary between
    beneficial assimilation (ERF > 1) and detrimental assimilation (ERF < 1).
    The shaded band shows ±1 std across trajectories.

    Args:
        obs_times:  observation times at which ERF is defined.
        erf_mean:   mean ERF at each observation time.
        erf_std:    standard deviation of ERF across trajectories.
        n_traj:     number of trajectories used for averaging (shown in title).
        title:      figure suptitle.
        save_path:  full path including .pdf extension.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(obs_times, erf_mean,
            color="#FF5722", linewidth=2.0, marker="o", markersize=4,
            label=f"Mean ERF  (n = {n_traj} trajectories)")
    ax.fill_between(
        obs_times,
        erf_mean - erf_std,
        erf_mean + erf_std,
        color="#FF5722", alpha=0.20, linewidth=0,
        label="±1 std across trajectories",
    )

    ax.set_yscale("log")

    # Reference line — ERF = 1 means no error reduction
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"ERF plot saved to: {save_path}")


def _plot_rmse_comparison(
    obs_times:       np.ndarray,   # (T_obs,)  absolute observation times
    prior_rmse_mean: np.ndarray,   # (T_obs,)  mean prior  RMSE across trajectories
    prior_rmse_std:  np.ndarray,   # (T_obs,)  std  prior  RMSE across trajectories
    post_rmse_mean:  np.ndarray,   # (T_obs,)  mean posterior RMSE across trajectories
    post_rmse_std:   np.ndarray,   # (T_obs,)  std  posterior RMSE across trajectories
    sigma_obs:       float,        # measurement noise std — reference level
    n_traj:          int,          # number of trajectories averaged
    title:           str,
    save_path:       str,
) -> None:
    """
    Plot mean prior RMSE, mean posterior RMSE, and the measurement noise
    level (sigma_obs) on a shared log-scale y-axis.

    A well-functioning filter must satisfy two conditions simultaneously:
        (1) posterior RMSE < prior RMSE   — assimilation reduces error
        (2) posterior RMSE < sigma_obs    — filter extracts signal from noise

    Both are immediately visible from this plot.  Shaded ±1-std bands
    across trajectories indicate robustness of the RMSE estimates.

    Args:
        obs_times:       observation times at which RMSE is defined.
        prior_rmse_mean: mean prior  RMSE at each observation time.
        prior_rmse_std:  std  prior  RMSE across trajectories.
        post_rmse_mean:  mean posterior RMSE at each observation time.
        post_rmse_std:   std  posterior RMSE across trajectories.
        sigma_obs:       observation noise standard deviation.
        n_traj:          number of trajectories (shown in legend).
        title:           figure title.
        save_path:       full path including .pdf extension.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Prior RMSE ────────────────────────────────────────────────────────────
    ax.plot(obs_times, prior_rmse_mean,
            color="#2196F3", linewidth=2.0, marker="o", markersize=4,
            label=f"Prior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times,
        prior_rmse_mean - prior_rmse_std,
        prior_rmse_mean + prior_rmse_std,
        color="#2196F3", alpha=0.18, linewidth=0,
        label="Prior ±1 std",
    )

    # ── Posterior RMSE ────────────────────────────────────────────────────────
    ax.plot(obs_times, post_rmse_mean,
            color="#FF5722", linewidth=2.0, marker="s", markersize=4,
            label=f"Posterior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times,
        post_rmse_mean - post_rmse_std,
        post_rmse_mean + post_rmse_std,
        color="#FF5722", alpha=0.18, linewidth=0,
        label="Posterior ±1 std",
    )

    # ── Measurement noise reference ───────────────────────────────────────────
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", linewidth=1.6,
               label=f"Measurement noise  σ_obs = {sigma_obs}")

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
    logging.info(f"RMSE comparison plot saved to: {save_path}")




# ── EnKF with data-driven model ────────────────────────────────────────────────

def evaluate_with_enkf_dd(config: ml_collections.ConfigDict, workdir: str):
    """
    Data-driven EnKF evaluation: ensemble Kalman filtering with DeepONet surrogate.
    
    Uses the same window-aware run_enkf_smoother as the physics-informed version, but:
      - Predictions come from the data-driven DeepONet
      - Ground truth comes from the test dataset instead of ODE solve
    
    The test data has shape (200, 31, 51, 40) with overlapping windows:
      - Window k+1 starts where window k ends
      - We concatenate properly by skipping first point of windows 1-30
    """
    from examples.l96_n40_f6_ics.kf import run_enkf_smoother, init_ensemble
    
    # EKF/EnKF hyperparameters
    obs_every_n  = config.ekf.get("obs_every_n",   4)
    sigma_obs    = config.ekf.get("sigma_obs",      0.5)
    P0_sigma     = config.ekf.get("P0_sigma",       1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",   False)
    N_ens        = config.enkf.get("N_ens",         50)
    sigma_model  = config.enkf.get("sigma_model",   0.1)
    
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
    
    # Timing parameters
    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))
    
    time_steps = 51
    num_windows_test = 31
    
    # Load test data: shape (200, 31, 51, 40)
    test_data_rollout = dd_get_test_data_rollout(
        data_dir=config.training.get("data_dir", "data/"),
        windows_per_traj=num_windows_test,
    )
    logging.info(f"Loaded test data: {test_data_rollout.shape}")
    
    # Load model and checkpoint
    t_star_window = jnp.linspace(0.0, DT_WINDOW, time_steps)
    model = models.L96UDON_DD(config, t_star_window)
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    N = model.N
    
    logging.info(f"Restored data-driven DeepONet model for EnKF evaluation.")
    
    # Build EnKF functions using the DeepONet
    predict_fn, update_fn = model.make_enkf_fns(params, N_ens=N_ens)
    
    # Noise covariances
    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
    
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
    
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
    
    # Observation schedule
    total_time = num_windows_test * DT_WINDOW
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )
    
    logging.info(f"EnKF config: N_ens={N_ens}, obs_every_n={obs_every_n}, "
                 f"sigma_obs={sigma_obs}, sigma_model={sigma_model}")
    
    # Per-IC evaluation
    for ic_idx in range(min(config.saving.total_plots, test_data_rollout.shape[0])):
        logging.info(f"--- EnKF (DD) Evaluation for IC {ic_idx} (N_ens={N_ens}) ---")
        
        u_current_true = test_data_rollout[ic_idx, 0, 0, :]  # (N,)
        
        # Build ground truth by concatenating test data windows
        x_true_windows = []
        for win_idx in range(num_windows_test):
            if win_idx == 0:
                x_true_windows.append(test_data_rollout[ic_idx, win_idx, :, :])
            else:
                # Skip first point to avoid duplication at window boundaries
                x_true_windows.append(test_data_rollout[ic_idx, win_idx, 1:, :])
        
        x_true_full = jnp.concatenate(x_true_windows, axis=0)  # (total_t, N)
        
        # Build time axis matching concatenated data
        # 51 + 50*30 = 1551 total points
        t_eval = np.linspace(0.0, total_time, x_true_full.shape[0])
        
        # Extract ground truth at observation times (nearest neighbor interpolation)
        x_true_at_obs_list = []
        for obs_t in obs_times:
            closest_idx = np.argmin(np.abs(t_eval - obs_t))
            x_true_at_obs_list.append(x_true_full[closest_idx])
        x_true_at_obs = jnp.stack(x_true_at_obs_list)  # (T_obs, N)
        
        # Build observation sequence (indexed over T_obs events)
        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []
        
        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]  # (N,)
            
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
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))
        
        H_seq     = jnp.stack(H_list)       # (T_obs, m, N)
        y_obs_seq = jnp.stack(y_obs_list)   # (T_obs, m)
        
        # Initialize ensemble from perturbed IC
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
        
        # Run EnKF smoother with window awareness
        # dt_fine and dt_window allow the filter to track position within windows
        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices,
            H_seq,
            Q_fine,
            R,
            key,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
        # x_means, x_spreads: (total_fine_steps, N)
        
        # Create trajectory summary plot
        # Upsample x_true_full if needed, or truncate x_means to match
        min_len = min(x_means.shape[0], x_true_full.shape[0])
        
        _plot_trajectory_summary(
            t_ax      = t_eval[:min_len],
            x_true    = np.array(x_true_full[:min_len]),
            x_est     = np.array(x_means[:min_len]),
            x_std     = np.array(x_spreads[:min_len]),
            ic_idx    = ic_idx,
            est_label = "EnKF (DD) mean",
            save_path = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_enkf_dd_ic_{ic_idx}.pdf",
            ),
            N = N,
            dt_window = DT_WINDOW,
            obs_coords = obs_coords,
        )
        
        # Compute L2 error at window boundaries
        window_step_indices = np.array([
            np.argmin(np.abs(t_eval - (w + 1) * DT_WINDOW))
            for w in range(num_windows_test)
        ])
        
        x_means_at_windows = x_means[window_step_indices]
        x_true_at_windows  = x_true_full[window_step_indices]
        
        l2_enkf     = jnp.linalg.norm(x_means_at_windows - x_true_at_windows) \
                    / jnp.linalg.norm(x_true_at_windows)
        mean_spread = float(jnp.mean(x_spreads))
        print(f"IC {ic_idx} | EnKF (DD) L2 (window boundaries): {l2_enkf:.3e} | Mean σ: {mean_spread:.3e}")
    
    # Batch evaluation
    _evaluate_batch_l2_enkf_dd(
        model, params, t_star_window,
        predict_fn, update_fn,
        Q_fine, P0,
        N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config, workdir,
        test_data_rollout,
    )
 
 
def _evaluate_batch_l2_enkf_dd(
    model, params, t_star_window,
    predict_fn, update_fn,
    Q_fine, P0,
    N_ens, obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars,
    dt_fine: float,
    dt_obs: float,
    config, workdir,
    test_data_rollout,
):
    """
    Compute and plot batch-averaged L2 error per window for data-driven EnKF.
    
    Called at the end of evaluate_with_enkf_dd(). For each IC in the pool:
      • Open-loop:  autoregressively roll out the model for k windows.
      • EnKF:       run run_enkf_smoother up to window k, take the final
                    filtered estimate.
    Both errors are averaged over all ICs and plotted together.
    
    Notes
    -----
    For data-driven, we use the test data directly rather than loading from .mat file.
    The batch size is not capped here since test data is small (200 ICs).
    """
    from examples.l96_n40_f6_ics.kf import run_enkf_smoother, init_ensemble
    
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
    
    dt_window     = float(config.get("dt_window", 0.25))
    N             = model.N
    
    num_windows_test = test_data_rollout.shape[1]
    B = test_data_rollout.shape[0]
    
    logging.info("Computing batch L2 per window (open-loop vs EnKF, data-driven) …")
    logging.info(f"  Using {B} ICs from test data for batch L2 evaluation (N_ens={N_ens}).")
    
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
    
    m = len(obs_indices)
    R_fixed = jnp.eye(m) * sigma_obs ** 2
    
    # vmapped single-window predictor for open-loop
    predict_one_window = jax.jit(
        jax.vmap(lambda u: model.x_net(params, u, t_star_window)[-1], in_axes=0)
    )
    
    # Rebuild schedule for the batch duration (all windows)
    total_time_batch = num_windows_test * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch,
        dt_fine=dt_fine,
        dt_obs=dt_obs,
    )
    T_obs = len(obs_step_indices_batch)
    
    # Absolute observation times (for ERF and RMSE x-axis)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])
    
    # Window boundary indices (in fine-step space)
    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(num_windows_test)
    ])
    
    # Accumulators for batch statistics
    enkf_l2_sum     = np.zeros(num_windows_test)
    enkf_spread_sum = np.zeros(num_windows_test)
    enkf_rmse_sum   = np.zeros(num_windows_test)
    
    # ERF accumulators
    erf_sum    = np.zeros(T_obs)
    erf_sq_sum = np.zeros(T_obs)
    
    # RMSE accumulators
    prior_rmse_sum    = np.zeros(T_obs)
    prior_rmse_sq_sum = np.zeros(T_obs)
    post_rmse_sum     = np.zeros(T_obs)
    post_rmse_sq_sum  = np.zeros(T_obs)
    
    # Loop over all ICs in test data
    for ic in range(B):
        key = jax.random.PRNGKey(ic + 77777)
        u_true = test_data_rollout[ic, 0, 0, :]  # (N,)
        
        # Build ground truth by concatenating windows
        x_true_windows = []
        for win_idx in range(num_windows_test):
            if win_idx == 0:
                x_true_windows.append(test_data_rollout[ic, win_idx, :, :])
            else:
                x_true_windows.append(test_data_rollout[ic, win_idx, 1:, :])
        
        x_true_full = jnp.concatenate(x_true_windows, axis=0)  # (total_t, N)
        t_eval = np.linspace(0.0, total_time_batch, x_true_full.shape[0])
        
        # Extract ground truth at observation times
        x_true_at_obs_list = []
        for obs_t in obs_times_batch:
            closest_idx = np.argmin(np.abs(t_eval - obs_t))
            x_true_at_obs_list.append(x_true_full[closest_idx])
        x_true_at_obs = jnp.stack(x_true_at_obs_list)
        
        # Build observation sequence
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
        
        H_seq     = jnp.stack(H_list)      # (T_obs, m, N)
        y_obs_seq = jnp.stack(y_obs_list)  # (T_obs, m)
        
        # Initialize ensemble
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
        
        # Run EnKF smoother
        x_means, x_spreads, prior_means_at_obs = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,
            R_fixed,
            key,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )
        # x_means, x_spreads: (total_fine_steps_batch, N)
        # prior_means_at_obs: (T_obs, N)
        
        # Posterior means at observation steps
        post_means_at_obs = x_means[obs_step_indices_batch]  # (T_obs, N)
        
        # Compute RMSE and ERF for this IC
        prior_rmse = np.sqrt(np.mean(
            (np.array(prior_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))  # (T_obs,)
        post_rmse = np.sqrt(np.mean(
            (np.array(post_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))  # (T_obs,)
        
        erf_ic = prior_rmse / (post_rmse + 1e-12)  # (T_obs,)
        
        erf_sum    += erf_ic
        erf_sq_sum += erf_ic ** 2
        
        prior_rmse_sum    += prior_rmse
        prior_rmse_sq_sum += prior_rmse ** 2
        post_rmse_sum     += post_rmse
        post_rmse_sq_sum  += post_rmse ** 2
        
        # Accumulate L2 at window boundaries
        for k in range(num_windows_test):
            # Get reference state at window k+1 (first point)
            if k == 0:
                ref_k = test_data_rollout[ic, 0, 0, :]
            else:
                ref_k = test_data_rollout[ic, k, 0, :]
            
            step_k = window_step_indices[k]
            x_hat_k = x_means[step_k]
            
            enkf_l2_sum[k] += float(
                jnp.linalg.norm(x_hat_k - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
            enkf_rmse_sum[k] += float(jnp.sqrt(jnp.mean((x_hat_k - ref_k) ** 2)))
            enkf_spread_sum[k] += float(jnp.sqrt(jnp.mean(x_spreads[step_k] ** 2)))
    
    # Open-loop evaluation (vmapped for efficiency)
    ol_l2 = np.zeros(num_windows_test)
    u_current = test_data_rollout[:, 0, 0, :]  # (B, N)
    
    for k in range(num_windows_test):
        u_current = predict_one_window(u_current)  # (B, N)
        
        if k == 0:
            ref_k = test_data_rollout[:, 0, 0, :]
        else:
            ref_k = test_data_rollout[:, k, 0, :]
        
        numer = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom = jnp.linalg.norm(ref_k, axis=1)
        ol_l2[k] = float(jnp.mean(numer / (denom + 1e-12)))
    
    # Normalize by batch size
    l2_enkf = enkf_l2_sum / B
    rmse_enkf = enkf_rmse_sum / B
    spread_mean = enkf_spread_sum / B
    
    erf_mean = erf_sum / B
    erf_std = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))
    
    prior_rmse_mean = prior_rmse_sum / B
    prior_rmse_std = np.sqrt(np.maximum(
        prior_rmse_sq_sum / B - prior_rmse_mean ** 2, 0.0))
    post_rmse_mean = post_rmse_sum / B
    post_rmse_std = np.sqrt(np.maximum(
        post_rmse_sq_sum / B - post_rmse_mean ** 2, 0.0))
    
    # Create output directory
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: L2 per window + Calibration
    save_path = os.path.join(save_dir, "batch_l2_per_window_enkf_dd.pdf")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window_idx = np.arange(1, num_windows_test + 1)
    
    # L2 comparison
    ax = axes[0]
    ax.plot(window_idx, ol_l2, marker="o", markersize=4, linewidth=1.8,
            label="Open-loop (DeepONet DD)", color="#2196F3")
    ax.plot(window_idx, l2_enkf, marker="s", markersize=4, linewidth=1.8,
            label=f"EnKF mean (N_ens={N_ens})", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop: L2 per window", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    
    # Secondary time axis
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time (window × dt)", fontsize=10)
    
    # Calibration (spread vs RMSE)
    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean, marker="^", markersize=4, linewidth=1.8,
             label="RMS ensemble σ", color="#4CAF50")
    ax2.plot(window_idx, rmse_enkf, marker="s", markersize=4, linewidth=1.8,
             linestyle="--", label="EnKF RMSE", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel("Window index", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs RMSE", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    
    # Secondary time axis
    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax2_time.set_xlabel("Simulation time (window × dt)", fontsize=10)
    
    fig.suptitle(
        f"EnKF batch evaluation (DD)  (B={B}, N_ens={N_ens}, "
        f"obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"EnKF batch L2-per-window plot saved to: {save_path}")
    
    # Plot 2: Error Reduction Factor
    erf_save_path = os.path.join(save_dir, "batch_erf_enkf_dd.pdf")
    _plot_erf(
        obs_times=obs_times_batch,
        erf_mean=erf_mean,
        erf_std=erf_std,
        n_traj=B,
        title=(
            f"EnKF (DD) Error Reduction Factor per observation time\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path=erf_save_path,
    )
    
    # Plot 3: Prior vs Posterior RMSE
    rmse_save_path = os.path.join(save_dir, "batch_rmse_enkf_dd.pdf")
    _plot_rmse_comparison(
        obs_times=obs_times_batch,
        prior_rmse_mean=prior_rmse_mean,
        prior_rmse_std=prior_rmse_std,
        post_rmse_mean=post_rmse_mean,
        post_rmse_std=post_rmse_std,
        sigma_obs=sigma_obs,
        n_traj=B,
        title=(
            f"EnKF (DD) prior vs posterior RMSE\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path=rmse_save_path,
    )
    
    logging.info(f"EnKF (DD) batch evaluation complete: B={B}, mean L2 final window = {l2_enkf[-1]:.3e}")