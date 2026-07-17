import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax
from jax.tree_util import tree_map
from flax.jax_utils import replicate
from typing import Callable

from jaxpi.utils import restore_checkpoint
import examples.l96_f.models as models
from examples.l96_f.utils import (
 	    build_obs_schedule,
 	    scale_inflation_for_fine_steps,
 	    scale_Q_for_fine_steps,
        steps_per_window_exact,
 	)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import h5py


def _plot_l2_per_timestep(
    curves:    dict[str, tuple[np.ndarray, np.ndarray]], # label -> (time_axis, l2_array)
    title:     str,
    save_path: str,
    colors:    dict[str, str] | None = None,
) -> None:
    """Plot average L2 error continuously across fine time stamps."""
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for i, (label, (t_axis, l2_arr)) in enumerate(curves.items()):
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])
        # Removed markers to prevent clustering on dense data
        ax.plot(t_axis, l2_arr, linewidth=1.8, label=label, color=color)
 
    ax.set_yscale("log")
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Mean relative L2 error (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Dense batch L2 plot saved to: {save_path}")

def _plot_trajectory_summary(
    t_ax:       np.ndarray,        # (T,)   time axis
    x_true:     np.ndarray,        # (T, N) ground-truth state
    x_est:      np.ndarray,        # (T, N) estimate (prediction / filter mean)
    x_std:      np.ndarray | None, # (T, N) per-variable std, or None
    ic_idx:     int,
    est_label:  str,               # e.g. "DeepONet", "EnKF mean"
    save_path:  str,
    N:          int = 40,
    dt_window:  float | None = None,
    # List of (variable_index, observation_time) pairs built during the filter
    # loop.  Used to mark assimilated observations with an × on each variable
    # panel.  Pass None (or omit) for open-loop evaluations without observations.
    obs_coords: list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single IC.
 
    Layout
    ------
    Row 0 (2-column span):
        Line plot of mean |error| across all N variables vs time.
        Gives a scalar summary of how the error evolves.
        Vertical dashed lines mark every training-window boundary (if
        dt_window is supplied).
 
    Rows 1–20, columns 0–1  (40 panels total):
        Panel for variable i shows:
          • ground-truth trajectory  (solid, dark)
          • estimate trajectory      (dashed, coloured)
          • ±1σ shaded band          (if x_std is not None)
          • × markers at every assimilated observation for that variable
            (if obs_coords is not None)
          • vertical dashed lines at training-window boundaries
            (if dt_window is not None)
 
    Args:
        t_ax:       1-D time array shared by all panels.
        x_true:     Ground-truth states; shape (T, N).
        x_est:      Estimated states; shape (T, N).
        x_std:      Per-variable standard deviation; shape (T, N), or None.
        ic_idx:     Trajectory index, used only for the figure title.
        est_label:  Short name for the estimator shown in legends.
        save_path:  Full output path including filename and .pdf extension.
        N:          State dimension (default 40 for L96).
        dt_window:  Training window length in the same time units as t_ax.
                    When supplied, a vertical dashed line is drawn at each
                    multiple of dt_window in every panel.
        obs_coords: List of (variable_index, time) pairs for all
                    observations that were assimilated.  For each variable i
                    a scatter marker '×' is drawn at the corresponding
                    observation times, interpolated onto the estimate curve.
    """
    x_true = np.asarray(x_true)   # (T, N)
    x_est  = np.asarray(x_est)    # (T, N)
    x_std  = np.asarray(x_std) if x_std is not None else None
 
    abs_error    = np.abs(x_true - x_est)            # (T, N)
    mean_abs_err = abs_error.mean(axis=1)             # (T,)  ← the top-panel curve
 
    n_var_rows = N // 2                               # 20 rows for 40 variables
 
    # ── Pre-compute window-boundary times ────────────────────────────────────
    # Build a sorted array of boundary times within the plotted range so that
    # axvline calls are O(num_windows) rather than O(T).
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        # Start from the first boundary strictly after t_min
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])
 
    # ── Pre-compute per-variable observation times ────────────────────────────
    # Build a dict  {var_idx: sorted array of obs times}
    if obs_coords is not None:
        # dict maps var_idx -> sorted list of (time, observed_value) pairs
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        # sort by time so the scatter x-coords are in order
        obs_by_var = {k: sorted(v, key=lambda x: x[0])
                      for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}
 
    # ── Figure & GridSpec ────────────────────────────────────────────────────
    # Top row is taller (summary plot); variable rows are compact.
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
 
    # ── Top panel: mean absolute error vs time ───────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])   # span both columns
    ax_top.plot(t_ax, mean_abs_err, color="#E53935", linewidth=1.6,
                label="Mean |error| over variables")
 
    # window boundaries on the summary panel
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
 
    # ── Colour palette ───────────────────────────────────────────────────────
    TRUTH_COLOR = "#37474F"   # dark blue-grey — ground truth
    EST_COLOR   = "#1E88E5"   # blue           — estimate
    BAND_COLOR  = "#90CAF9"   # light blue     — ±1σ band
    OBS_COLOR   = "#E53935"   # red            — observation markers
 
    # ── Per-variable panels ──────────────────────────────────────────────────
    # Variable i occupies row (1 + i//2), column (i % 2).
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        # window boundaries — draw first so they sit behind data lines
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.6, alpha=0.45)
 
        # Ground truth
        ax.plot(t_ax, x_true[:, i],
                color=TRUTH_COLOR, linewidth=1.0, label="Truth")
 
        # Estimate
        ax.plot(t_ax, x_est[:, i],
                color=EST_COLOR, linewidth=1.0, linestyle="--",
                label=est_label)
 
        # ±1σ uncertainty band (EnKF only)
        if x_std is not None:
            ax.fill_between(
                t_ax,
                x_est[:, i] - x_std[:, i],
                x_est[:, i] + x_std[:, i],
                color=BAND_COLOR, alpha=0.40, linewidth=0,
                label="±1σ",
            )
 
        # Observation markers — interpolate estimate value at each obs time
        # so the × sits on the estimate curve rather than floating arbitrarily.
        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])   # unzip the pairs
            ax.scatter(obs_times_i, obs_vals_i,              # plot true noisy obs
                       marker="x", s=25, linewidths=0.9,
                       color=OBS_COLOR, zorder=5,
                       label="Observation" if i == min(obs_by_var) else None)
 
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
 
        # Only label axes on the border panels to reduce clutter
        if row == 1 + n_var_rows - 1:          # bottom row
            ax.set_xlabel("t", fontsize=8)
        if col == 0:                            # left column
            ax.set_ylabel("state", fontsize=8)
 
        # Legend only on the first panel (top-left variable)
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

def evaluate(
    config: ml_collections.ConfigDict,
    workdir: str,
    test_h5_path: str = None,
) -> None:
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"
 
    # ── 1. Load the long test trajectories and Forcing parameters ────────────
    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]     # (num_ics, num_test_pts, N)
        t_test = f["t"][:]     # (num_test_pts,) relative time, t_test[0] == 0
        F_test = f["F"][:]     # (num_ics,) per-trajectory F values
 
    dt_window = float(config.get("dt_window", 0.25))

    # Fetch configuration limits with defaults
    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows      = config.eval.get("windows", 200)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
    dt_integration     = config.eval.get("dt_integration", 0.005)
 
    # ── 2. Models & per-window query grid ───────────────────────────────────
    time_steps = int(round(dt_window / dt_integration)) + 1 
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    # T_last is window duration (e.g., 0.25)
    T_last = float(t_star_window[-1])
 
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
    logging.info("Restored trained DeepONet model for long autoregressive rollout.")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
 
    num_plots = min(config.saving.total_plots, u_test.shape[0])
 
    for ic_idx in range(num_plots):
        logging.info(f"--- [long] Evaluating Trajectory for IC index {ic_idx} ---")
 
        # Grab F for this specific IC
        F_i = float(F_test[ic_idx])
        # Append F_i to the initial condition
        u_current = jnp.concatenate([jnp.array(u_test[ic_idx, 0, :]), jnp.array([F_i])])
 
        # ── Autoregressive rollout (using trajectory_windows) ───────────────
        x_pred_list, t_full_list = [], []
        for idx in range(trajectory_windows):
            preds = model.x_pred_fn(params, u_current, t_star_window)
            x_pred_window = jnp.squeeze(preds)
 
            if idx == 0:
                x_pred_list.append(x_pred_window)
                t_full_list.append(t_star_window)
            else:
                x_pred_list.append(x_pred_window[1:])
                t_full_list.append(t_star_window[1:] + idx * T_last)
 
            u_current = jnp.concatenate([x_pred_window[-1, :], jnp.array([F_i])])
 
        x_pred_full = jnp.concatenate(x_pred_list, axis=0)
        t_star_full = jnp.concatenate(t_full_list, axis=0)
 
        # ── Ground truth computed ON THE SPOT ───────────────────────────────
        # Extract the exact F value used for this specific trajectory
        F_i = float(F_test[ic_idx])
        
        def lorenz_96(t, state, F=F_i):
            x_plus_1 = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F
        
        t_eval_np = np.array(t_star_full)
        u0_np = np.array(u_test[ic_idx, 0, :])
        
        # Solve the ODE matching EXACT gen_data.py parameters
        sol = solve_ivp(
            lorenz_96, 
            t_span=[t_eval_np[0], t_eval_np[-1]], 
            y0=u0_np, 
            t_eval=t_eval_np,
            method='LSODA',      # Fix: explicitly set method to LSODA
            rtol=1e-13,          # Fix: match strict relative tolerance
            atol=1e-14           # Fix: match strict absolute tolerance
        )
        x_ref_matched = jnp.array(sol.y.T)
 
        # ── Trajectory summary plot ──────────────────────────────────────────
        _plot_trajectory_summary(
            t_ax       = np.array(t_star_full),
            x_true     = np.array(x_ref_matched),
            x_est      = np.array(x_pred_full),
            x_std      = None,                     
            ic_idx     = ic_idx,
            est_label  = "DeepONet (long rollout)",
            save_path  = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_ic_{ic_idx}.pdf",
            ),
            N          = model.N,
            dt_window  = dt_window,
            obs_coords = None,
        )
 
        # ── Full-rollout relative L2 error ──────────────────────────────────
        total_l2_error = jnp.linalg.norm(x_pred_full - x_ref_matched) / jnp.linalg.norm(x_ref_matched)
        print(
            f"IC {ic_idx} | Long Rollout ({trajectory_windows} windows, "
            f"{trajectory_windows * dt_window:.3g} t.u.) L2 error: {total_l2_error:.3e}"
        )
 
        # ── Heatmaps: exact reference / prediction / absolute error ─────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
 
        im0 = axes[0].pcolormesh(np.arange(model.N), t_star_full, x_ref_matched, cmap='viridis', shading='auto')
        axes[0].set_title(f"Exact L96 Reference (IC {ic_idx}, long)", fontsize=14)
        axes[0].set_ylabel("Time (t)", fontsize=14)
        axes[0].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im0, ax=axes[0])
 
        im1 = axes[1].pcolormesh(np.arange(model.N), t_star_full, x_pred_full, cmap='viridis', shading='auto')
        axes[1].set_title(f"UDON Rollout (IC {ic_idx}, long)", fontsize=14)
        axes[1].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im1, ax=axes[1])
 
        abs_error = jnp.abs(x_ref_matched - x_pred_full)
        im2 = axes[2].pcolormesh(np.arange(model.N), t_star_full, abs_error, cmap='magma', shading='auto')
        axes[2].set_title(f"Absolute Error (IC {ic_idx}, long)", fontsize=14)
        axes[2].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im2, ax=axes[2])
 
        for ax in axes:
            for w in range(1, trajectory_windows):
                ax.axhline(y=w * dt_window, color='white', linestyle=':', alpha=0.5)
 
        fig.tight_layout()
 
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"udon_rollout_analysis_ic_{ic_idx}.pdf")
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
 
    # ── Batch-averaged L2-per-window ────────────────────────────────────────
    # Slice u_test up to num_ics_eval to respect the config, and pass batch_windows
    _evaluate_batch_l2_openloop(
        model, params, t_star_window,
        u_test[:num_ics_eval], t_test, dt_window, batch_windows,
        config, workdir,
    )

def _evaluate_batch_l2_openloop(
    model, params, t_star_window,
    u_test:           np.ndarray,   # (B, num_test_pts, N) dense test trajectories
    t_test:           np.ndarray,   # (num_test_pts,) relative times, t_test[0] == 0
    F_test:           np.ndarray,
    dt_window:        float,
    num_windows_long: int,
    config, workdir,
    curve_label: str = "Open-loop (DeepONet), long rollout",
) -> np.ndarray:
    """
    Batch-averaged open-loop L2 error per window for the long (L-window)
    horizon, computed directly from l96_forcing_test.h5.
 
    This is the long-horizon analogue of `_evaluate_batch_l2_openloop`. The
    two differ only in where the ground truth comes from:
 
      * `_evaluate_batch_l2_openloop` reads a separate augmentation pool
        (a .mat file with keys `u0_original` / `u0_rollout_k`) that was
        purpose-built for a *short* number of windows (`max_additions`).
      * here, every trajectory already stored in `l96_forcing_test.h5` is
        used as its own batch element, and the window-boundary ground truth
        is simply indexed out of the dense trajectory — no separate pool
        file is needed.
 
    For each window k (1-indexed):
        1. Start every trajectory from its true state at t = 0.
        2. Run k autoregressive steps through model.x_pred_fn.
        3. Compare the resulting state to the file's true state at
           t = k * dt_window.
        4. Average the per-trajectory relative L2 norm over the batch.
 
    Returns the (num_windows_long,) array of batch-mean relative L2 errors.
    """
    B = u_test.shape[0]
    dt_test = float(t_test[1] - t_test[0])
    pts_pw  = int(round(dt_window / dt_test))
 
    assert abs(dt_window / dt_test - pts_pw) < 2e-6, (
        f"dt_window ({dt_window}) is not an integer multiple of the test "
        f"file's time step ({dt_test}); cannot index exact window boundaries."
    )
    assert num_windows_long * pts_pw < len(t_test), (
        "num_windows_long exceeds the number of windows available in the "
        "test file."
    )
 
    u0_batch = jnp.concatenate([jnp.array(u_test[:, 0, :]), F_test[:B, None]], axis=-1)
    u_current = u0_batch

    # 1. JIT-compiled, vmapped FULL-window predictor:  (B, N) → (B, T, N)
    predict_full_window = jax.jit(
        jax.vmap(
            lambda u: model.x_pred_fn(params, u, t_star_window),
            in_axes=0,
        )
    )

    x_pred_dense = []

    # 2. Rollout the DeepONet densely
    for k in range(num_windows_long):
        x_pred_window = predict_full_window(u_current)                 
        
        if k == 0:
            x_pred_dense.append(x_pred_window)
        else:
            x_pred_dense.append(x_pred_window[:, 1:, :]) # skip duplicate boundary
            
        u_current = jnp.concatenate([x_pred_window[:, -1, :], F_test[:B, None]], axis=-1)

    # Shape: (B, total_steps, N)
    x_pred_dense = jnp.concatenate(x_pred_dense, axis=1)

    # 3. Dense ground truth is directly available from the test file.
    # We slice u_test to perfectly match the concatenated prediction length.
    total_steps = x_pred_dense.shape[1]
    x_ref_dense = jnp.array(u_test[:, :total_steps, :])
    t_eval_long = t_test[:total_steps]

    # 4. Compute L2 error densely across the batch
    numer = jnp.linalg.norm(x_pred_dense - x_ref_dense, axis=2) # (B, total_steps)
    denom = jnp.linalg.norm(x_ref_dense, axis=2)                # (B, total_steps)
    l2_dense = jnp.mean(numer / (denom + 1e-12), axis=0)        # (total_steps,)
    
    logging.info(f"  [long] Mean L2 at final timestep: {l2_dense[-1]:.3e}")

    # 5. Plot using the dense timestep plotting function
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_timestep_openloop.pdf")
    
    _plot_l2_per_timestep(
        curves    = {curve_label: (t_eval_long, l2_dense)},
        title     = f"Open-loop (long rollout): batch-average L2 per timestep  (B={B})",
        save_path = save_path,
        colors    = {curve_label: "#2196F3"},
    )
    
    return np.array(l2_dense)




# ── DD vs PI ──────────────────────────────────────────────────────────────────

# jax helpers
def build_batched_enkf_compare(
    predict_fn_pi, update_fn_pi, predict_fn_dd, update_fn_dd,
    N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
    dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch
):
    from examples.l96_f.kf import init_ensemble, run_enkf_smoother

    def process_single_ic(key_ic, u_true, F_i, x_true_at_obs, dynamic_vars_static, specify_obs_idx_static):
        T_obs = x_true_at_obs.shape[0]
        keys_t = jax.random.split(key_ic, T_obs)
        
        # 1. Vectorized observation sequence generation
        def single_obs(k, x_t):
            k1, k2 = jax.random.split(k)
            # Static conditions evaluated at JIT-compile time
            if (not specify_obs_idx_static) and dynamic_vars_static:
                idx_vars = jax.random.choice(k1, N, shape=(m,), replace=False)
            else:
                idx_vars = obs_indices
                
            H = jnp.zeros((m, N)).at[jnp.arange(m), idx_vars].set(1.0)
            H_aug = jnp.pad(H, ((0, 0), (0, 1)), mode='constant')
            noise = sigma_obs * jax.random.normal(k2, shape=(m,))
            return H_aug, x_t[idx_vars] + noise, idx_vars
            
        H_seq, y_obs_seq, idx_vars_seq = jax.vmap(single_obs)(keys_t, x_true_at_obs)
        
        # 2. Shared initial ensemble
        k1, k2, k3 = jax.random.split(key_ic, 3)
        x0_hat_40 = u_true + P0_sigma * jax.random.normal(k2, shape=(N,))
        x0_hat_aug = jnp.concatenate([x0_hat_40, jnp.array([F_i])])
        ensemble0 = init_ensemble(x0_hat_aug, P0, N_ens, k3)
        
        # 3. Both estimators run concurrently on the exact same noise/observations
        x_means_pi, x_spreads_pi, prior_means_pi = run_enkf_smoother(
            predict_fn_pi, update_fn_pi,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, alpha_fine, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )
        
        x_means_dd, x_spreads_dd, prior_means_dd = run_enkf_smoother(
            predict_fn_dd, update_fn_dd,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, alpha_fine, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )
        
        return (x_means_pi, x_spreads_pi, prior_means_pi, 
                x_means_dd, x_spreads_dd, prior_means_dd, 
                y_obs_seq, idx_vars_seq)
    
    # Vmap across the batch (in_axes mapped to keys, u_true, F_i, x_true_at_obs)
    vmapped_fn = jax.vmap(process_single_ic, in_axes=(0, 0, 0, 0, None, None))
    # Freeze the boolean flags at compile time to avoid JAX tracer errors on if/else
    return jax.jit(vmapped_fn, static_argnums=(4, 5))


def _plot_trajectory_summary_compare(
    t_ax:       np.ndarray,        
    x_true:     np.ndarray,        
    x_est_pi:   np.ndarray,        
    x_est_dd:   np.ndarray,        
    ic_idx:     int,
    save_path:  str,
    N:          int = 40,
    dt_window:  float | None = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF comparing PI vs DD for a single IC.
    """
    x_true   = np.asarray(x_true)
    x_est_pi = np.asarray(x_est_pi)
    x_est_dd = np.asarray(x_est_dd)
 
    abs_error_pi    = np.abs(x_true - x_est_pi)
    abs_error_dd    = np.abs(x_true - x_est_dd)
    mean_abs_err_pi = abs_error_pi.mean(axis=1) 
    mean_abs_err_dd = abs_error_dd.mean(axis=1) 
 
    n_var_rows = N // 2  
 
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])
 
    # ── Figure & GridSpec ────────────────────────────────────────────────────
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
 
    # ── Top panel: mean absolute error vs time ───────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err_pi, color="#2196F3", linewidth=1.6, label="PI: Mean |error|")
    ax_top.plot(t_ax, mean_abs_err_dd, color="#FF8C00", linewidth=1.6, label="DD: Mean |error|")
 
    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)
 
    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables (PI vs DD)",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ── Colour palette ───────────────────────────────────────────────────────
    TRUTH_COLOR = "#37474F"   
    PI_COLOR    = "#2196F3"   
    DD_COLOR    = "#FF8C00"   
 
    # ── Per-variable panels ──────────────────────────────────────────────────
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.6, alpha=0.45)
 
        ax.plot(t_ax, x_true[:, i], color=TRUTH_COLOR, linewidth=1.2, label="Truth")
        ax.plot(t_ax, x_est_pi[:, i], color=PI_COLOR, linewidth=1.0, linestyle="--", label="PI")
        ax.plot(t_ax, x_est_dd[:, i], color=DD_COLOR, linewidth=1.0, linestyle=":", label="DD")
 
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
 
        if row == 1 + n_var_rows - 1:
            ax.set_xlabel("t", fontsize=8)
        if col == 0:
            ax.set_ylabel("state", fontsize=8)
 
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", handlelength=1.5, framealpha=0.7)
 
    fig.suptitle(
        f"Trajectory comparison — IC {ic_idx} | PI vs DD",
        fontsize=13, fontweight="bold", y=1.002,
    )
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Comparison trajectory summary for IC {ic_idx} saved to: {save_path}")

def _evaluate_batch_l2_openloop_compare(
    model_pi, params_pi, 
    model_dd, params_dd, 
    t_star_window, 
    u_test: np.ndarray, 
    t_test: np.ndarray, 
    F_test: np.ndarray,
    dt_window: float, 
    num_windows_long: int, 
    config, workdir: str
) -> None:
    
    B = u_test.shape[0]
    dt_test = float(t_test[1] - t_test[0])
    pts_pw  = int(round(dt_window / dt_test))
 
    u0_batch = jnp.concatenate([jnp.array(u_test[:, 0, :]), F_test[:B, None]], axis=-1)

    # 1. JIT-compiled, vmapped FULL-window predictors
    predict_full_pi = jax.jit(jax.vmap(lambda u: model_pi.x_pred_fn(params_pi, u, t_star_window), in_axes=0))
    predict_full_dd = jax.jit(jax.vmap(lambda u: model_dd.x_pred_fn(params_dd, u, t_star_window), in_axes=0))

    x_pred_dense_pi, x_pred_dense_dd = [], []
    u_current_pi = u0_batch
    u_current_dd = u0_batch

    # 2. Rollout the DeepONets densely
    for k in range(num_windows_long):
        x_pred_window_pi = predict_full_pi(u_current_pi)                 
        x_pred_window_dd = predict_full_dd(u_current_dd)                 
        
        if k == 0:
            x_pred_dense_pi.append(x_pred_window_pi)
            x_pred_dense_dd.append(x_pred_window_dd)
        else:
            x_pred_dense_pi.append(x_pred_window_pi[:, 1:, :]) 
            x_pred_dense_dd.append(x_pred_window_dd[:, 1:, :]) 
            
        u_current_pi = jnp.concatenate([x_pred_window_pi[:, -1, :], F_test[:B, None]], axis=-1)
        u_current_dd = jnp.concatenate([x_pred_window_dd[:, -1, :], F_test[:B, None]], axis=-1)

    # Shape: (B, total_steps, N)
    x_pred_dense_pi = jnp.concatenate(x_pred_dense_pi, axis=1)
    x_pred_dense_dd = jnp.concatenate(x_pred_dense_dd, axis=1)

    # 3. Reference data
    total_steps = x_pred_dense_pi.shape[1]
    x_ref_dense = jnp.array(u_test[:, :total_steps, :])
    t_eval_long = t_test[:total_steps]

    # 4. Compute L2 error densely across the batch
    denom = jnp.linalg.norm(x_ref_dense, axis=2) + 1e-12
    
    numer_pi = jnp.linalg.norm(x_pred_dense_pi - x_ref_dense, axis=2) 
    l2_dense_pi = jnp.mean(numer_pi / denom, axis=0)        
    
    numer_dd = jnp.linalg.norm(x_pred_dense_dd - x_ref_dense, axis=2) 
    l2_dense_dd = jnp.mean(numer_dd / denom, axis=0)        
    
    logging.info(f"  [long] Mean L2 at final timestep -> PI: {l2_dense_pi[-1]:.3e} | DD: {l2_dense_dd[-1]:.3e}")

    # 5. Plot using the existing dense timestep plotting function
    save_dir  = os.path.join(workdir, "figures", "comparison")
    save_path = os.path.join(save_dir, "batch_l2_per_timestep_compare.pdf")
    
    curves = {
        "PI Model": (t_eval_long, np.array(l2_dense_pi)),
        "DD Model": (t_eval_long, np.array(l2_dense_dd))
    }
    colors = {
        "PI Model": "#2196F3",
        "DD Model": "#FF8C00"
    }
    
    _plot_l2_per_timestep(
        curves    = curves,
        title     = f"PI vs DD (long rollout): batch-average L2 per timestep (B={B})",
        save_path = save_path,
        colors    = colors,
    )

def evaluate_dd_vs_pi(
    config: ml_collections.ConfigDict,
    workdir: str,
    test_h5_path: str = None,
) -> None:
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"
 
    # ── 1. Load the long test trajectories and Forcing parameters ────────────
    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]     
        t_test = f["t"][:]     
        F_test = f["F"][:]     
 
    dt_window = float(config.get("dt_window", 0.25))

    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows      = config.eval.get("windows", 200)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
    dt_integration     = config.eval.get("dt_integration", 0.005)
 
    # ── 2. Models & per-window query grid ───────────────────────────────────
    time_steps = int(round(dt_window / dt_integration)) + 1 
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    # T_last is window duration (e.g., 0.25)
    T_last = float(t_star_window[-1])
 
    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(
        os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model"
    )
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params

    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(
        os.getcwd(), config.wandb.name_dd, "ckpt", "udon_model"
    )
    # Adjust checkpoint name if needed based on DD train loop (udon_dd_model)
    if not os.path.exists(ckpt_path_dd):
         ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params
 
    num_plots = min(config.saving.total_plots, u_test.shape[0])
 
    for ic_idx in range(num_plots):
        logging.info(f"--- [Compare] Evaluating Trajectory for IC index {ic_idx} ---")

        # Grab F for this specific IC
        F_i = float(F_test[ic_idx])
        # Append F_i to the initial condition
        u_current_pi = jnp.concatenate([jnp.array(u_test[ic_idx, 0, :]), jnp.array([F_i])])
        u_current_dd = jnp.concatenate([jnp.array(u_test[ic_idx, 0, :]), jnp.array([F_i])])
 
        # ── Autoregressive rollout ──────────────────────────────────────────
        x_pred_list_pi, x_pred_list_dd, t_full_list = [], [], []
        
        for idx in range(trajectory_windows):
            preds_pi = model_pi.x_pred_fn(params_pi, u_current_pi, t_star_window)
            preds_dd = model_dd.x_pred_fn(params_dd, u_current_dd, t_star_window)
            
            x_pred_window_pi = jnp.squeeze(preds_pi)
            x_pred_window_dd = jnp.squeeze(preds_dd)
 
            if idx == 0:
                x_pred_list_pi.append(x_pred_window_pi)
                x_pred_list_dd.append(x_pred_window_dd)
                t_full_list.append(t_star_window)
            else:
                x_pred_list_pi.append(x_pred_window_pi[1:])
                x_pred_list_dd.append(x_pred_window_dd[1:])
                t_full_list.append(t_star_window[1:] + idx * T_last)
 
            u_current_pi = jnp.concatenate([x_pred_window_pi[-1, :], jnp.array([F_i])])
            u_current_dd = jnp.concatenate([x_pred_window_dd[-1, :], jnp.array([F_i])])
 
        x_pred_full_pi = jnp.concatenate(x_pred_list_pi, axis=0)
        x_pred_full_dd = jnp.concatenate(x_pred_list_dd, axis=0)
        t_star_full    = jnp.concatenate(t_full_list, axis=0)
 
        # ── Ground truth computed ON THE SPOT ───────────────────────────────
        F_i = float(F_test[ic_idx])
        
        def lorenz_96(t, state, F=F_i):
            x_plus_1 = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F
        
        t_eval_np = np.array(t_star_full)
        u0_np = np.array(u_test[ic_idx, 0, :])
        
        sol = solve_ivp(
            lorenz_96, 
            t_span=[t_eval_np[0], t_eval_np[-1]], 
            y0=u0_np, 
            t_eval=t_eval_np,
            method='LSODA',      
            rtol=1e-13,          
            atol=1e-14           
        )
        x_ref_matched = jnp.array(sol.y.T)
 
        # ── Trajectory summary plot (Comparison) ────────────────────────────
        save_path = os.path.join(
            workdir, "figures", "comparison",
            f"trajectory_summary_compare_ic_{ic_idx}.pdf"
        )
        
        _plot_trajectory_summary_compare(
            t_ax       = np.array(t_star_full),
            x_true     = np.array(x_ref_matched),
            x_est_pi   = np.array(x_pred_full_pi),
            x_est_dd   = np.array(x_pred_full_dd),
            ic_idx     = ic_idx,
            save_path  = save_path,
            N          = model_pi.N,
            dt_window  = dt_window,
        )
 
        # ── Full-rollout relative L2 errors ─────────────────────────────────
        norm_ref = jnp.linalg.norm(x_ref_matched)
        total_l2_pi = jnp.linalg.norm(x_pred_full_pi - x_ref_matched) / norm_ref
        total_l2_dd = jnp.linalg.norm(x_pred_full_dd - x_ref_matched) / norm_ref
        
        print(f"IC {ic_idx} | Rollout PI L2: {total_l2_pi:.3e} | DD L2: {total_l2_dd:.3e}")
        # Note: Heatmaps are omitted from this function per your requirements.
 
    # ── Batch-averaged L2-per-window (Comparison) ──────────────────────────
    _evaluate_batch_l2_openloop_compare(
        model_pi, params_pi, 
        model_dd, params_dd, 
        t_star_window,
        u_test[:num_ics_eval], 
        t_test, 
        F_test,
        dt_window, batch_windows,
        config, workdir,
    )



def _binned_spread_skill(
    rmss: np.ndarray,
    rmse: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin raw (RMSS, RMSE) pairs into `n_bins` equal-population bins
    (deciles by default) over RMSS. Raw per-(IC, window) spread/skill pairs
    form an unreadable cloud; binning is the standard fix.

    Returns bin_rmss_mean, bin_rmse_mean, bin_rmse_std, bin_counts,
    each shape (n_bins,) (fewer if some bins end up empty).
    """
    rmss = np.asarray(rmss).ravel()
    rmse = np.asarray(rmse).ravel()
    order = np.argsort(rmss)
    rmss_sorted, rmse_sorted = rmss[order], rmse[order]

    bin_rmss_mean, bin_rmse_mean, bin_rmse_std, bin_counts = [], [], [], []
    for idx in np.array_split(np.arange(len(rmss_sorted)), n_bins):
        if idx.size == 0:
            continue
        bin_rmss_mean.append(rmss_sorted[idx].mean())
        bin_rmse_mean.append(rmse_sorted[idx].mean())
        bin_rmse_std.append(rmse_sorted[idx].std())
        bin_counts.append(idx.size)

    return (np.array(bin_rmss_mean), np.array(bin_rmse_mean),
            np.array(bin_rmse_std), np.array(bin_counts))


# ── PI vs DD + EnKF ──────────────────────────────────────────────────────────

def _plot_trajectory_summary_compare_enkf(
    t_ax:       np.ndarray,        # (T,)   time axis
    x_true:     np.ndarray,        # (T, N) ground-truth state
    x_est_pi:   np.ndarray,        # (T, N) PI + EnKF mean
    x_std_pi:   np.ndarray | None, # (T, N) PI ensemble std, or None
    x_est_dd:   np.ndarray,        # (T, N) DD + EnKF mean
    x_std_dd:   np.ndarray | None, # (T, N) DD ensemble std, or None
    ic_idx:     int,
    save_path:  str,
    N:          int = 40,
    dt_window:  float | None = None,
    obs_coords: list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Trajectory-summary PDF comparing PI+EnKF vs DD+EnKF against the ground
    truth for a single IC.

    This combines the PI-vs-DD side-by-side comparison layout of
    `_plot_trajectory_summary_compare` with the ±1σ ensemble-spread bands
    and assimilated-observation markers of `_plot_trajectory_summary`, so
    the same figure that used to show one filtered estimator now shows two,
    sharing the same observation schedule.
    """
    x_true   = np.asarray(x_true)
    x_est_pi = np.asarray(x_est_pi)
    x_est_dd = np.asarray(x_est_dd)
    x_std_pi = np.asarray(x_std_pi) if x_std_pi is not None else None
    x_std_dd = np.asarray(x_std_dd) if x_std_dd is not None else None

    abs_error_pi    = np.abs(x_true - x_est_pi)
    abs_error_dd    = np.abs(x_true - x_est_dd)
    mean_abs_err_pi = abs_error_pi.mean(axis=1)
    mean_abs_err_dd = abs_error_dd.mean(axis=1)

    n_var_rows = N // 2

    # ── Pre-compute window-boundary times ────────────────────────────────
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])

    # ── Pre-compute per-variable observation times ────────────────────────
    if obs_coords is not None:
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        obs_by_var = {k: sorted(v, key=lambda x: x[0])
                      for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}

    # ── Figure & GridSpec ────────────────────────────────────────────────
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

    # ── Top panel: mean absolute error vs time (PI vs DD) ──────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err_pi, color="#2196F3", linewidth=1.6,
                label="PI + EnKF: Mean |error|")
    ax_top.plot(t_ax, mean_abs_err_dd, color="#FF8C00", linewidth=1.6,
                label="DD + EnKF: Mean |error|")

    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)

    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables  (PI+EnKF vs DD+EnKF)",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ── Colour palette (kept consistent with the rest of the codebase) ────
    TRUTH_COLOR = "#37474F"   # dark blue-grey — ground truth
    PI_COLOR    = "#2196F3"   # blue           — PI + EnKF mean
    DD_COLOR    = "#FF8C00"   # orange         — DD + EnKF mean
    PI_BAND     = "#90CAF9"   # light blue     — PI ±1σ band
    DD_BAND     = "#FFCC80"   # light orange   — DD ±1σ band
    OBS_COLOR   = "#E53935"   # red            — observation markers

    # ── Per-variable panels ──────────────────────────────────────────────
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])

        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.6, alpha=0.45)

        # Ground truth
        ax.plot(t_ax, x_true[:, i],
                color=TRUTH_COLOR, linewidth=1.0, label="Truth")

        # PI + EnKF
        ax.plot(t_ax, x_est_pi[:, i],
                color=PI_COLOR, linewidth=1.0, linestyle="--", label="PI + EnKF")
        if x_std_pi is not None:
            ax.fill_between(
                t_ax,
                x_est_pi[:, i] - x_std_pi[:, i],
                x_est_pi[:, i] + x_std_pi[:, i],
                color=PI_BAND, alpha=0.35, linewidth=0, label="PI ±1σ",
            )

        # DD + EnKF
        ax.plot(t_ax, x_est_dd[:, i],
                color=DD_COLOR, linewidth=1.0, linestyle=":", label="DD + EnKF")
        if x_std_dd is not None:
            ax.fill_between(
                t_ax,
                x_est_dd[:, i] - x_std_dd[:, i],
                x_est_dd[:, i] + x_std_dd[:, i],
                color=DD_BAND, alpha=0.35, linewidth=0, label="DD ±1σ",
            )

        # Observation markers — same cross style/size as the single-model plot
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
            ax.legend(fontsize=6.5, loc="upper right",
                      handlelength=1.3, framealpha=0.7)

    fig.suptitle(
        f"Trajectory summary — IC {ic_idx}  |  PI+EnKF vs DD+EnKF",
        fontsize=13, fontweight="bold", y=1.002,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Comparison EnKF trajectory summary for IC {ic_idx} saved to: {save_path}")

def _plot_erf_compare(
    obs_times:    np.ndarray,   # (T_obs,)
    erf_mean_pi:  np.ndarray,   # (T_obs,)
    erf_std_pi:   np.ndarray,   # (T_obs,)
    erf_mean_dd:  np.ndarray,   # (T_obs,)
    erf_std_dd:   np.ndarray,   # (T_obs,)
    n_traj:       int,
    title:        str,
    save_path:    str,
) -> None:
    """
    ERF comparison for PI+EnKF vs DD+EnKF, same visual conventions as
    `_plot_erf` (log-scale, ±1 std band, ERF=1 reference line).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(obs_times, erf_mean_pi,
            color="#2196F3", linewidth=2.0, marker="o", markersize=4,
            label=f"PI + EnKF  (n = {n_traj} trajectories)")
    ax.fill_between(
        obs_times, erf_mean_pi - erf_std_pi, erf_mean_pi + erf_std_pi,
        color="#2196F3", alpha=0.18, linewidth=0, label="PI ±1 std",
    )

    ax.plot(obs_times, erf_mean_dd,
            color="#FF8C00", linewidth=2.0, marker="s", markersize=4,
            label=f"DD + EnKF  (n = {n_traj} trajectories)")
    ax.fill_between(
        obs_times, erf_mean_dd - erf_std_dd, erf_mean_dd + erf_std_dd,
        color="#FF8C00", alpha=0.18, linewidth=0, label="DD ±1 std",
    )

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"ERF comparison plot (PI vs DD) saved to: {save_path}")

def _plot_rmse_comparison_dd_pi(
    obs_times:           np.ndarray,
    prior_rmse_mean_pi:  np.ndarray, prior_rmse_std_pi: np.ndarray,
    post_rmse_mean_pi:   np.ndarray, post_rmse_std_pi:  np.ndarray,
    prior_rmse_mean_dd:  np.ndarray, prior_rmse_std_dd: np.ndarray,
    post_rmse_mean_dd:   np.ndarray, post_rmse_std_dd:  np.ndarray,
    sigma_obs:           float,
    n_traj:              int,
    title:               str,
    save_path:           str,
) -> None:
    """
    Prior/posterior RMSE comparison for PI+EnKF vs DD+EnKF, on the same
    axes.  Same colour roles as `_plot_rmse_comparison` (blue = prior,
    orange-red = posterior, green = measurement-noise reference); PI is
    solid, DD is dashed so both regimes remain distinguishable at a glance.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ── PI ──────────────────────────────────────────────────────────────
    ax.plot(obs_times, prior_rmse_mean_pi,
            color="#0A36C7", linewidth=2.0, marker="o", markersize=4,
            linestyle="-", label=f"PI prior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, prior_rmse_mean_pi - prior_rmse_std_pi, prior_rmse_mean_pi + prior_rmse_std_pi,
        color="#0A36C7", alpha=0.15, linewidth=0,
    )
    ax.plot(obs_times, post_rmse_mean_pi,
            color="#A30005", linewidth=2.0, marker="s", markersize=4,
            linestyle="-", label=f"PI posterior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, post_rmse_mean_pi - post_rmse_std_pi, post_rmse_mean_pi + post_rmse_std_pi,
        color="#A30005", alpha=0.15, linewidth=0,
    )

    # ── DD ──────────────────────────────────────────────────────────────
    ax.plot(obs_times, prior_rmse_mean_dd,
            color="#2196F3", linewidth=2.0, marker="o", markersize=4,
            linestyle="--", label=f"DD prior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, prior_rmse_mean_dd - prior_rmse_std_dd, prior_rmse_mean_dd + prior_rmse_std_dd,
        color="#2196F3", alpha=0.08, linewidth=0,
    )
    ax.plot(obs_times, post_rmse_mean_dd,
            color="#FF5722", linewidth=2.0, marker="s", markersize=4,
            linestyle="--", label=f"DD posterior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, post_rmse_mean_dd - post_rmse_std_dd, post_rmse_mean_dd + post_rmse_std_dd,
        color="#FF5722", alpha=0.08, linewidth=0,
    )

    # ── Measurement noise reference ─────────────────────────────────────
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.6,
               label=f"Measurement noise  σ_obs = {sigma_obs}")

    ax.set_yscale("log")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("RMSE  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"RMSE comparison plot (PI vs DD) saved to: {save_path}")

def _plot_calibration_compare(
    window_idx:    np.ndarray,
    dt_window:     float,
    spread_pi:     np.ndarray, rmse_pi: np.ndarray,
    spread_dd:     np.ndarray, rmse_dd: np.ndarray,
    spread_pi_raw: np.ndarray, rmse_pi_raw: np.ndarray,
    spread_dd_raw: np.ndarray, rmse_dd_raw: np.ndarray,
    title:         str,
    save_path:     str,
    n_bins:        int = 10,
) -> None:
    """
    Calibration comparison for PI vs DD, one PDF with 3 stacked panels:
      1. DD  — RMS ensemble spread vs EnKF RMSE at window boundaries.
      2. PI  — same, directly below, sharing the window-index x-axis.
      3. Binned spread-skill diagram — raw (RMSS, RMSE) pairs pooled over
         every (IC, window) in the batch, binned into `n_bins`
         equal-population bins against the y=x reference.
    """
    fig = plt.figure(figsize=(9, 13))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.3], hspace=0.55)

    def _timeseries_panel(ax, spread, rmse, c_spread, c_rmse, label):
        ax.plot(window_idx, spread, marker="^", markersize=4, linewidth=1.8,
                linestyle="-", color=c_spread, label=f"{label} RMS ensemble σ")
        ax.plot(window_idx, rmse, marker="s", markersize=4, linewidth=1.8,
                linestyle="--", color=c_rmse, label=f"{label} EnKF RMSE")
        ax.set_yscale("log")
        ax.set_xlabel("Window index", fontsize=11)
        ax.set_ylabel("Log scale", fontsize=11)
        ax.set_title(f"{label}: ensemble spread vs RMSE", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax_time = ax.twiny()
        ax_time.set_xlim(ax.get_xlim())
        ax_time.set_xticks(window_idx)
        ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx],
                                 fontsize=7, rotation=45, ha="left")
        ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=9)

    ax_dd = fig.add_subplot(gs[0])
    _timeseries_panel(ax_dd, spread_dd, rmse_dd, "#8BC34A", "#FF8A65", "DD")

    ax_pi = fig.add_subplot(gs[1])
    _timeseries_panel(ax_pi, spread_pi, rmse_pi, "#4CAF50", "#FF5722", "PI")

    ax_bin = fig.add_subplot(gs[2])
    rmss_dd_b, rmse_dd_b, rmse_dd_s, _ = _binned_spread_skill(spread_dd_raw, rmse_dd_raw, n_bins)
    rmss_pi_b, rmse_pi_b, rmse_pi_s, _ = _binned_spread_skill(spread_pi_raw, rmse_pi_raw, n_bins)

    lim_hi = 1.1 * max(rmss_dd_b.max(), rmse_dd_b.max(), rmss_pi_b.max(), rmse_pi_b.max())
    ax_bin.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.4,
                color="#37474F", label="1:1 (perfect calibration)")
    ax_bin.errorbar(rmss_dd_b, rmse_dd_b, yerr=rmse_dd_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#FF8C00", label=f"DD ({n_bins}-bin)")
    ax_bin.errorbar(rmss_pi_b, rmse_pi_b, yerr=rmse_pi_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#2196F3", label=f"PI ({n_bins}-bin)")

    ax_bin.set_xlim(0, lim_hi); ax_bin.set_ylim(0, lim_hi)
    ax_bin.set_xlabel("RMS ensemble spread (RMSS)", fontsize=11)
    ax_bin.set_ylabel("RMSE of ensemble mean", fontsize=11)
    ax_bin.set_title(f"Binned spread-skill  ({n_bins} equal-population bins, "
                      f"pooled over all ICs × windows)", fontsize=12)
    ax_bin.legend(fontsize=9)
    ax_bin.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bin.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Calibration comparison plot (PI vs DD) saved to: {save_path}")

def _evaluate_batch_enkf_dd_vs_pi(
    model_pi, params_pi, predict_fn_pi, update_fn_pi,
    model_dd, params_dd, predict_fn_dd, update_fn_dd,
    t_star_window, u_test, t_test, F_test, alpha_fine, P0, R, obs_indices,
    N_ens, obs_every_n, sigma_obs, P0_sigma, dynamic_vars,
    specify_obs_idx, obs_idx_list, dt_window, dt_fine, dt_obs,
    num_ics_eval, enkf_batch_size, batch_windows, config, workdir
) -> None:

    B = min(num_ics_eval, enkf_batch_size, u_test.shape[0])
    N = model_pi.N
    u0_batch = u_test[:B, 0, :]
    dt_test  = float(t_test[1] - t_test[0])

    total_time_batch = batch_windows * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=dt_fine, dt_obs=dt_obs
    )
    obs_step_indices_batch = jnp.array(obs_step_indices_batch)

    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    fine_stride = int(round(dt_fine / dt_test))
    n_fine_pts  = total_fine_steps_batch * fine_stride + 1
    
    x_true_fine_batch   = u_test[:B, 0:n_fine_pts:fine_stride, :]
    x_true_at_obs_batch = x_true_fine_batch[:, obs_step_indices_batch + 1, :]
    window_step_indices = np.array([round((k + 1) * dt_window / dt_fine) - 1 for k in range(batch_windows)])
    m = len(obs_indices)

    # ── 1. Vmapped EnKF Execution ──────────────────────────────────────────
    # Generate B distinct keys dynamically mimicking original behavior
    seed = config.training.get("seed", 42)
    master_key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(master_key, B)
    
    batched_enkf = build_batched_enkf_compare(
        predict_fn_pi, update_fn_pi, predict_fn_dd, update_fn_dd,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
        dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch
    )

    (batch_x_means_pi, batch_x_spreads_pi, batch_prior_means_pi,
     batch_x_means_dd, batch_x_spreads_dd, batch_prior_means_dd,
     _, _) = batched_enkf(
         keys_batch, u0_batch, F_test[:B], x_true_at_obs_batch, 
         dynamic_vars, specify_obs_idx
    )

    # ── 2. Vectorized Metric Extraction ────────────────────────────────────
    post_means_pi_obs = batch_x_means_pi[:, obs_step_indices_batch, :N]
    post_means_dd_obs = batch_x_means_dd[:, obs_step_indices_batch, :N]
    prior_means_pi_obs = batch_prior_means_pi[:, :, :N]
    prior_means_dd_obs = batch_prior_means_dd[:, :, :N]

    prior_rmse_pi_ic = jnp.sqrt(jnp.mean((prior_means_pi_obs - x_true_at_obs_batch) ** 2, axis=2))
    post_rmse_pi_ic  = jnp.sqrt(jnp.mean((post_means_pi_obs - x_true_at_obs_batch) ** 2, axis=2))
    prior_rmse_dd_ic = jnp.sqrt(jnp.mean((prior_means_dd_obs - x_true_at_obs_batch) ** 2, axis=2))
    post_rmse_dd_ic  = jnp.sqrt(jnp.mean((post_means_dd_obs - x_true_at_obs_batch) ** 2, axis=2))

    erf_pi_ic = prior_rmse_pi_ic / (post_rmse_pi_ic + 1e-12)
    erf_dd_ic = prior_rmse_dd_ic / (post_rmse_dd_ic + 1e-12)

    erf_pi_mean, erf_pi_std = np.array(jnp.mean(erf_pi_ic, axis=0)), np.array(jnp.std(erf_pi_ic, axis=0))
    erf_dd_mean, erf_dd_std = np.array(jnp.mean(erf_dd_ic, axis=0)), np.array(jnp.std(erf_dd_ic, axis=0))
    prior_rmse_pi_mean, prior_rmse_pi_std = np.array(jnp.mean(prior_rmse_pi_ic, axis=0)), np.array(jnp.std(prior_rmse_pi_ic, axis=0))
    post_rmse_pi_mean, post_rmse_pi_std = np.array(jnp.mean(post_rmse_pi_ic, axis=0)), np.array(jnp.std(post_rmse_pi_ic, axis=0))
    prior_rmse_dd_mean, prior_rmse_dd_std = np.array(jnp.mean(prior_rmse_dd_ic, axis=0)), np.array(jnp.std(prior_rmse_dd_ic, axis=0))
    post_rmse_dd_mean, post_rmse_dd_std = np.array(jnp.mean(post_rmse_dd_ic, axis=0)), np.array(jnp.std(post_rmse_dd_ic, axis=0))

    # Window-boundary processing
    x_true_at_windows = x_true_fine_batch[:, window_step_indices + 1, :]
    x_hat_pi_windows = batch_x_means_pi[:, window_step_indices, :N]
    x_hat_dd_windows = batch_x_means_dd[:, window_step_indices, :N]

    den = jnp.linalg.norm(x_true_at_windows, axis=2) + 1e-12
    l2_enkf_pi = np.array(jnp.mean(jnp.linalg.norm(x_hat_pi_windows - x_true_at_windows, axis=2) / den, axis=0))
    l2_enkf_dd = np.array(jnp.mean(jnp.linalg.norm(x_hat_dd_windows - x_true_at_windows, axis=2) / den, axis=0))

    rmse_pi_ic = jnp.sqrt(jnp.mean((x_hat_pi_windows - x_true_at_windows) ** 2, axis=2))
    rmse_dd_ic = jnp.sqrt(jnp.mean((x_hat_dd_windows - x_true_at_windows) ** 2, axis=2))
    rmse_enkf_pi = np.array(jnp.mean(rmse_pi_ic, axis=0))
    rmse_enkf_dd = np.array(jnp.mean(rmse_dd_ic, axis=0))

    spread_pi_ic = jnp.sqrt(jnp.mean(batch_x_spreads_pi[:, window_step_indices, :N] ** 2, axis=2))
    spread_dd_ic = jnp.sqrt(jnp.mean(batch_x_spreads_dd[:, window_step_indices, :N] ** 2, axis=2))
    spread_pi = np.array(jnp.mean(spread_pi_ic, axis=0))
    spread_dd = np.array(jnp.mean(spread_dd_ic, axis=0))

    rmse_pi_raw, rmse_dd_raw = np.array(rmse_pi_ic.flatten()), np.array(rmse_dd_ic.flatten())
    spread_pi_raw, spread_dd_raw = np.array(spread_pi_ic.flatten()), np.array(spread_dd_ic.flatten())

    # Dense metrics
    x_true_fine_tail = x_true_fine_batch[:, 1:, :]
    den_dense = jnp.linalg.norm(x_true_fine_tail, axis=2) + 1e-12
    l2_enkf_pi_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_pi[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))
    l2_enkf_dd_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_dd[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))

    # ── 3. Open-loop Rollouts (Unchanged & Fast) ───────────────────────────
    u0_batch_j = jnp.array(u0_batch)
    predict_full_pi = jax.jit(jax.vmap(lambda u: model_pi.x_pred_fn(params_pi, u, t_star_window), in_axes=0))
    predict_full_dd = jax.jit(jax.vmap(lambda u: model_dd.x_pred_fn(params_dd, u, t_star_window), in_axes=0))

    x_pred_dense_pi_list, x_pred_dense_dd_list = [], []
    u_current_pi = u_current_dd = jnp.concatenate([u0_batch_j, F_test[:B, None]], axis=-1)

    for k in range(batch_windows):
        x_win_pi = predict_full_pi(u_current_pi)
        x_win_dd = predict_full_dd(u_current_dd)

        if k == 0:
            x_pred_dense_pi_list.append(x_win_pi)
            x_pred_dense_dd_list.append(x_win_dd)
        else:
            x_pred_dense_pi_list.append(x_win_pi[:, 1:, :])
            x_pred_dense_dd_list.append(x_win_dd[:, 1:, :])

        u_current_pi = jnp.concatenate([x_win_pi[:, -1, :], F_test[:B, None]], axis=-1)
        u_current_dd = jnp.concatenate([x_win_dd[:, -1, :], F_test[:B, None]], axis=-1)

    x_pred_dense_pi = jnp.concatenate(x_pred_dense_pi_list, axis=1)
    x_pred_dense_dd = jnp.concatenate(x_pred_dense_dd_list, axis=1)
    total_steps_ol = x_pred_dense_pi.shape[1]
    x_ref_dense_ol = jnp.array(u_test[:B, :total_steps_ol, :])
    
    denom_ol = jnp.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
    l2_ol_pi = np.array(jnp.mean(jnp.linalg.norm(x_pred_dense_pi - x_ref_dense_ol, axis=2) / denom_ol, axis=0))
    l2_ol_dd = np.array(jnp.mean(jnp.linalg.norm(x_pred_dense_dd - x_ref_dense_ol, axis=2) / denom_ol, axis=0))

    t_eval_ol = t_test[:total_steps_ol]
    t_dense_fine = np.arange(1, total_fine_steps_batch + 1) * dt_fine

    logging.info(
        f"  [batch] Final-timestep mean L2 -> "
        f"PI open-loop: {float(l2_ol_pi[-1]):.3e} | PI+EnKF: {l2_enkf_pi_dense[-1]:.3e} | "
        f"DD open-loop: {float(l2_ol_dd[-1]):.3e} | DD+EnKF: {l2_enkf_dd_dense[-1]:.3e}"
    )
    logging.info(
        f"  [batch] Final-window mean L2 (boundary-only, for reference) -> "
        f"PI+EnKF: {l2_enkf_pi[-1]:.3e} | DD+EnKF: {l2_enkf_dd[-1]:.3e}"
    )

    # ── Plotting ──────────────────────────────────────────────────────────
    save_dir = os.path.join(workdir, "figures", "comparison")

    # Plot 1 — dense per-timestamp L2: EnKF vs open-loop, PI vs DD
    curves = {
        "PI Open-loop": (np.array(t_eval_ol), l2_ol_pi),
        "PI + EnKF":    (t_dense_fine,        l2_enkf_pi_dense),
        "DD Open-loop": (np.array(t_eval_ol), l2_ol_dd),
        "DD + EnKF":    (t_dense_fine,        l2_enkf_dd_dense),
    }
    colors = {
        "PI Open-loop": "#90CAF9",
        "PI + EnKF":    "#2196F3",
        "DD Open-loop": "#FFCC80",
        "DD + EnKF":    "#FF8C00",
    }
    _plot_l2_per_timestep(
        curves    = curves,
        title     = f"EnKF vs open-loop: mean relative L2 per timestep  (PI vs DD, B={B})",
        save_path = os.path.join(save_dir, "batch_l2_per_timestep_enkf_compare.pdf"),
        colors    = colors,
    )

    # Plot 2 — calibration: ensemble spread vs RMSE, PI vs DD
    _plot_calibration_compare(
        window_idx = np.arange(1, batch_windows + 1),
        dt_window  = dt_window,
        spread_pi  = spread_pi, rmse_pi = rmse_enkf_pi,
        spread_dd  = spread_dd, rmse_dd = rmse_enkf_dd,
        spread_pi_raw = spread_pi_raw, rmse_pi_raw = rmse_pi_raw,
        spread_dd_raw = spread_dd_raw, rmse_dd_raw = rmse_dd_raw,
        title      = f"Calibration: ensemble spread vs RMSE  (PI vs DD, B={B}, N_ens={N_ens})",
        save_path  = os.path.join(save_dir, "batch_calibration_enkf_compare.pdf"),
    )

    # Plot 3 — Error Reduction Factor, PI vs DD
    _plot_erf_compare(
        obs_times   = obs_times_batch,
        erf_mean_pi = erf_pi_mean, erf_std_pi = erf_pi_std,
        erf_mean_dd = erf_dd_mean, erf_std_dd = erf_dd_std,
        n_traj      = B,
        title       = (
            f"EnKF Error Reduction Factor per observation time  (PI vs DD)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path   = os.path.join(save_dir, "batch_erf_enkf_compare.pdf"),
    )

    # Plot 4 — prior / posterior RMSE, PI vs DD
    _plot_rmse_comparison_dd_pi(
        obs_times          = obs_times_batch,
        prior_rmse_mean_pi = prior_rmse_pi_mean, prior_rmse_std_pi = prior_rmse_pi_std,
        post_rmse_mean_pi  = post_rmse_pi_mean,  post_rmse_std_pi  = post_rmse_pi_std,
        prior_rmse_mean_dd = prior_rmse_dd_mean, prior_rmse_std_dd = prior_rmse_dd_std,
        post_rmse_mean_dd  = post_rmse_dd_mean,  post_rmse_std_dd  = post_rmse_dd_std,
        sigma_obs = sigma_obs, n_traj = B,
        title = (
            f"EnKF prior vs posterior RMSE  (PI vs DD)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path = os.path.join(save_dir, "batch_rmse_enkf_compare.pdf"),
    )

def evaluate_enkf_dd_vs_pi(
    config: ml_collections.ConfigDict,
    workdir: str,
    test_h5_path: str = None,
) -> None:
    """
    EnKF evaluation comparing the Physics-Informed (PI) and Data-Driven (DD)
    DeepONet regimes used as ensemble propagators.

    This is `evaluate_with_enkf` (filter logic: predict/update, ensemble
    init, window-aware smoother, single-trajectory plots marking assimilated
    observations, ERF, and calibration) rewired onto the centralized
    `l96_forcing_test.h5` data source and PI-vs-DD comparison conventions of
    `evaluate_dd_vs_pi`:

      * Test data (initial conditions, forcing F) comes from
        `l96_forcing_test.h5`, exactly as in `evaluate_dd_vs_pi`.
      * Single-trajectory plots run for `config.eval.trajectory_windows`
        windows (longer than the stored test horizon), so their ground
        truth is generated on the fly with the exact solver used for
        trajectory-summary plotting in `evaluate_dd_vs_pi`
        (`LSODA`, `rtol=1e-13`, `atol=1e-14`).
      * The batch evaluation horizon (`config.eval.windows`) always fits
        inside the stored test trajectories, so its ground truth is sliced
        directly out of `l96_forcing_test.h5` rather than re-solved.
      * The EnKF filter is applied to both the PI and the DD propagator
        using the *same* observation schedule (and, for the batch
        evaluation, the same noisy observation draws and initial ensemble)
        so that the comparison isolates the effect of the surrogate model.
      * The batch "EnKF vs open-loop" plot reports mean relative L2 at
        every fine timestep (dense) instead of only at window boundaries.
    """
    from examples.l96_f.kf import run_enkf_smoother, init_ensemble

    # ── EnKF / observation configuration (identical to evaluate_with_enkf) ──
    obs_every_n  = config.kf.get("obs_every_n",   4)
    sigma_obs    = config.kf.get("sigma_obs",      0.5)
    P0_sigma     = config.kf.get("P0_sigma",       1.0)
    dynamic_vars = config.kf.get("dynamic_vars",   False)
    N_ens        = config.kf.get("N_ens",         50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list", None)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.kf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.kf.get("dt_obs",    DT_WINDOW))

    # ── 1. Load the long test trajectories and forcing parameters ─────────
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"

    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]
        t_test = f["t"][:]
        F_test = f["F"][:]

    dt_window = float(config.get("dt_window", 0.25))

    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows      = config.eval.get("windows", 200)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
    dt_integration     = config.eval.get("dt_integration", 0.005)
    enkf_batch_size    = config.kf.get("batch_l2_size", 200)
 
    # ── 2. Models & per-window query grid ───────────────────────────────────
    time_steps = int(round(dt_window / dt_integration)) + 1 
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    # T_last is window duration (e.g., 0.25)
    T_last = float(t_star_window[-1])

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_model")
    if not os.path.exists(ckpt_path_dd):
        ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    # ── 3. EnKF predict/update functions for both regimes ─────────────────
    predict_fn_pi, update_fn_pi = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)

    # Scale multiplicative inflation geometrically for fine timesteps
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine       = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    # ── 4. Per-IC single-trajectory EnKF evaluation (PI vs DD) ────────────
    num_plots  = min(config.saving.total_plots, u_test.shape[0])
    total_time = trajectory_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time = total_time, dt_fine = DT_FINE, dt_obs = DT_OBS,
    )
    obs_step_indices = jnp.array(obs_step_indices)

    # PASS 1: Execute SciPy Ground Truth Sequential Solves
    x_true_fine_list, x_true_at_obs_list = [], []
    t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
    
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
        def lorenz_96(t, state, F=F_i):
            x_plus_1  = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F

        sol = solve_ivp(
            lorenz_96, t_span=[0.0, total_time], y0=np.array(u_test[ic_idx, 0, :]),
            t_eval=t_eval_fine, method='LSODA', rtol=1e-13, atol=1e-14,
        )
        x_true_fine_list.append(sol.y.T)
        x_true_at_obs_list.append(sol.y.T[obs_step_indices + 1])

    # PASS 2: Batched GPU Execution for EnKF
    x_true_fine_batch = jnp.stack(x_true_fine_list)
    x_true_at_obs_batch = jnp.stack(x_true_at_obs_list)
    u0_batch_plots = jnp.array(u_test[:num_plots, 0, :])
    F_batch_plots = jnp.array(F_test[:num_plots])
    keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_plots))

    batched_enkf_plots = build_batched_enkf_compare(
        predict_fn_pi, update_fn_pi, predict_fn_dd, update_fn_dd,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
        DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices
    )

    (batch_x_means_pi, batch_x_spreads_pi, _,
     batch_x_means_dd, batch_x_spreads_dd, _,
     batch_y_obs, batch_idx_vars) = batched_enkf_plots(
         keys_batch_plots, u0_batch_plots, F_batch_plots, 
         x_true_at_obs_batch, dynamic_vars, specify_obs_idx
    )

    # PASS 3: Generate Individual Trajectory PDF plots sequentially
    t_fine_axis = t_eval_fine[1:]
    window_step_indices = np.array([round((w + 1) * DT_WINDOW / DT_FINE) - 1 for w in range(trajectory_windows)])
    
    for ic_idx in range(num_plots):
        x_true_fine = x_true_fine_batch[ic_idx]
        x_means_pi, x_spreads_pi = batch_x_means_pi[ic_idx], batch_x_spreads_pi[ic_idx]
        x_means_dd, x_spreads_dd = batch_x_means_dd[ic_idx], batch_x_spreads_dd[ic_idx]
        y_obs_seq, idx_vars_seq = batch_y_obs[ic_idx], batch_idx_vars[ic_idx]

        # Intercept observation coordinates specifically formatting for the plotting function
        obs_coords = []
        for obs_idx, t_obs in enumerate(obs_times):
            for j, vi in enumerate(idx_vars_seq[obs_idx]):
                obs_coords.append((int(vi), float(t_obs), float(y_obs_seq[obs_idx, j])))

        _plot_trajectory_summary_compare_enkf(
            t_ax=t_fine_axis, x_true=np.array(x_true_fine[1:]),
            x_est_pi=np.array(x_means_pi[:, :N]), x_std_pi=np.array(x_spreads_pi[:, :N]),
            x_est_dd=np.array(x_means_dd[:, :N]), x_std_dd=np.array(x_spreads_dd[:, :N]),
            ic_idx=ic_idx, N=N, dt_window=DT_WINDOW, obs_coords=obs_coords,
            save_path=os.path.join(workdir, "figures", "comparison", f"trajectory_summary_enkf_compare_ic_{ic_idx}.pdf")
        )

        x_true_at_windows = x_true_fine[window_step_indices + 1]
        l2_pi = jnp.linalg.norm(x_means_pi[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
        l2_dd = jnp.linalg.norm(x_means_dd[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)

        print(
            f"IC {ic_idx} | EnKF PI L2: {l2_pi:.3e} | EnKF DD L2: {l2_dd:.3e} "
            f"| Mean σ (PI): {float(jnp.mean(x_spreads_pi)):.3e} "
            f"| Mean σ (DD): {float(jnp.mean(x_spreads_dd)):.3e}"
        )

    # ── 5. Batch-averaged comparison (open-loop & EnKF, PI vs DD) ─────────
    _evaluate_batch_enkf_dd_vs_pi(
        model_pi, params_pi, predict_fn_pi, update_fn_pi,
        model_dd, params_dd, predict_fn_dd, update_fn_dd,
        t_star_window,
        u_test, t_test, F_test,
        alpha_fine, P0, R, obs_indices,
        N_ens, obs_every_n, sigma_obs, P0_sigma, dynamic_vars,
        specify_obs_idx, obs_idx_list,
        DT_WINDOW, DT_FINE, DT_OBS,
        num_ics_eval, enkf_batch_size, batch_windows,
        config, workdir,
    )


# ── PI: classic EnKF vs Route B (residual-scaled covariance) EnKF ────────────
#
# Same propagator (the physics-informed L96UDON surrogate) is used for both
# filters, so any difference between the two regimes below isolates the
# *covariance-inflation strategy* rather than the surrogate itself:
#
#   Classic  : make_enkf            -- fixed geometric multiplicative
#                                       inflation alpha_fine every fine step,
#                                       identical for every ensemble member.
#   Route B  : make_route_b_enkf    -- additive noise xi_i ~ N(0, Q_n^i),
#                                       Q_n^i = (alpha + beta*||rho_n^i||^2) Q0,
#                                       where rho_n^i is member i's own PDE
#                                       residual (see L96UDON.make_residual_fn
#                                       and kf.residual_l2_norm_sq). Members
#                                       whose surrogate currently violates the
#                                       governing PDE more pick up more
#                                       inflation -- physics-driven and
#                                       flow-dependent, with no per-member
#                                       tangent-linear solve.

def build_batched_enkf_pi_compare(
    predict_fn_a, update_fn_a, predict_fn_b, update_fn_b,
    N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, 
    alpha_fine_a, alpha_b, beta_b, Q0,
    dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch
):
    """
    JIT-compiled, batched execution of two EnKF inflation methods (Route A vs Route B)
    sharing the same observations and initial ensemble per trajectory.
    """
    from examples.l96_f.kf import init_ensemble, run_enkf_smoother, run_enkf_smoother_route_b

    def process_single_ic(key_ic, u_true, F_i, x_true_at_obs, dynamic_vars_static, specify_obs_idx_static):
        T_obs = x_true_at_obs.shape[0]
        keys_t = jax.random.split(key_ic, T_obs)
        
        # 1. Vectorized observation sequence generation
        def single_obs(k, x_t):
            k1, k2 = jax.random.split(k)
            # Static conditions evaluated at JIT-compile time
            if (not specify_obs_idx_static) and dynamic_vars_static:
                idx_vars = jax.random.choice(k1, N, shape=(m,), replace=False)
            else:
                idx_vars = obs_indices
                
            H = jnp.zeros((m, N)).at[jnp.arange(m), idx_vars].set(1.0)
            H_aug = jnp.pad(H, ((0, 0), (0, 1)), mode='constant')
            noise = sigma_obs * jax.random.normal(k2, shape=(m,))
            return H_aug, x_t[idx_vars] + noise, idx_vars
            
        H_seq, y_obs_seq, idx_vars_seq = jax.vmap(single_obs)(keys_t, x_true_at_obs)
        
        # 2. Shared initial ensemble
        k1, k2, k3 = jax.random.split(key_ic, 3)
        x0_hat_40 = u_true + P0_sigma * jax.random.normal(k2, shape=(N,))
        x0_hat_aug = jnp.concatenate([x0_hat_40, jnp.array([F_i])])
        ensemble0 = init_ensemble(x0_hat_aug, P0, N_ens, k3)
        
        # 3. Route A: Standard Multiplicative Inflation
        x_means_a, x_spreads_a, prior_means_a = run_enkf_smoother(
            predict_fn_a, update_fn_a,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, alpha_fine_a, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )
        
        # 4. Route B: Residual-Scaled Covariance Inflation
        x_means_b, x_spreads_b, prior_means_b, Q_scale_history = run_enkf_smoother_route_b(
            predict_fn_b, update_fn_b,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, Q0, alpha_b, beta_b, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window, n_quad=3
        )
        
        return (x_means_a, x_spreads_a, prior_means_a, 
                x_means_b, x_spreads_b, prior_means_b, Q_scale_history,
                y_obs_seq, idx_vars_seq)
    
    # Vmap across the batch (in_axes mapped to keys, u_true, F_i, x_true_at_obs)
    vmapped_fn = jax.vmap(process_single_ic, in_axes=(0, 0, 0, 0, None, None))
    # Freeze boolean flags at compile time
    return jax.jit(vmapped_fn, static_argnums=(4, 5))


def _plot_trajectory_summary_compare_enkf_rb(
    t_ax:          np.ndarray,        # (T,)   time axis
    x_true:        np.ndarray,        # (T, N) ground-truth state
    x_est_classic: np.ndarray,        # (T, N) PI + classic EnKF mean
    x_std_classic: np.ndarray | None, # (T, N) classic ensemble std, or None
    x_est_rb:      np.ndarray,        # (T, N) PI + Route B EnKF mean
    x_std_rb:      np.ndarray | None, # (T, N) Route B ensemble std, or None
    ic_idx:        int,
    save_path:     str,
    N:             int = 40,
    dt_window:     float | None = None,
    obs_coords:    list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Trajectory-summary PDF comparing PI+classic-EnKF vs PI+Route-B-EnKF
    against the ground truth for a single IC.  Identical layout to
    `_plot_trajectory_summary_compare_enkf` (top panel: mean |error| vs
    time; per-variable panels with ±1σ bands and assimilated-observation
    markers), just re-labelled for the two *filter* regimes instead of the
    two *propagator* regimes.
    """
    x_true        = np.asarray(x_true)
    x_est_classic = np.asarray(x_est_classic)
    x_est_rb      = np.asarray(x_est_rb)
    x_std_classic = np.asarray(x_std_classic) if x_std_classic is not None else None
    x_std_rb      = np.asarray(x_std_rb) if x_std_rb is not None else None

    abs_error_classic    = np.abs(x_true - x_est_classic)
    abs_error_rb         = np.abs(x_true - x_est_rb)
    mean_abs_err_classic = abs_error_classic.mean(axis=1)
    mean_abs_err_rb      = abs_error_rb.mean(axis=1)

    n_var_rows = N // 2

    # ── Pre-compute window-boundary times ────────────────────────────────
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])

    # ── Pre-compute per-variable observation times ────────────────────────
    if obs_coords is not None:
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        obs_by_var = {k: sorted(v, key=lambda x: x[0])
                      for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}

    # ── Figure & GridSpec ────────────────────────────────────────────────
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

    # ── Top panel: mean absolute error vs time (classic vs Route B) ───────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err_classic, color="#2196F3", linewidth=1.6,
                label="Classic EnKF: Mean |error|")
    ax_top.plot(t_ax, mean_abs_err_rb, color="#8E24AA", linewidth=1.6,
                label="Route B EnKF: Mean |error|")

    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)

    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables  "
        f"(PI + Classic EnKF vs PI + Route B EnKF)",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ── Colour palette ──────────────────────────────────────────────────
    TRUTH_COLOR = "#37474F"   # dark blue-grey — ground truth
    CL_COLOR    = "#2196F3"   # blue           — Classic EnKF mean
    RB_COLOR    = "#8E24AA"   # purple         — Route B EnKF mean
    CL_BAND     = "#90CAF9"   # light blue     — Classic ±1σ band
    RB_BAND     = "#CE93D8"   # light purple   — Route B ±1σ band
    OBS_COLOR   = "#E53935"   # red            — observation markers

    # ── Per-variable panels ──────────────────────────────────────────────
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])

        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.6, alpha=0.45)

        # Ground truth
        ax.plot(t_ax, x_true[:, i],
                color=TRUTH_COLOR, linewidth=1.0, label="Truth")

        # Classic EnKF
        ax.plot(t_ax, x_est_classic[:, i],
                color=CL_COLOR, linewidth=1.0, linestyle="--", label="Classic EnKF")
        if x_std_classic is not None:
            ax.fill_between(
                t_ax,
                x_est_classic[:, i] - x_std_classic[:, i],
                x_est_classic[:, i] + x_std_classic[:, i],
                color=CL_BAND, alpha=0.35, linewidth=0, label="Classic ±1σ",
            )

        # Route B EnKF
        ax.plot(t_ax, x_est_rb[:, i],
                color=RB_COLOR, linewidth=1.0, linestyle=":", label="Route B EnKF")
        if x_std_rb is not None:
            ax.fill_between(
                t_ax,
                x_est_rb[:, i] - x_std_rb[:, i],
                x_est_rb[:, i] + x_std_rb[:, i],
                color=RB_BAND, alpha=0.35, linewidth=0, label="Route B ±1σ",
            )

        # Observation markers — same cross style/size as the single-model plot
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
            ax.legend(fontsize=6.5, loc="upper right",
                      handlelength=1.3, framealpha=0.7)

    fig.suptitle(
        f"Trajectory summary — IC {ic_idx}  |  PI + Classic EnKF vs PI + Route B EnKF",
        fontsize=13, fontweight="bold", y=1.002,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Classic-vs-Route-B EnKF trajectory summary for IC {ic_idx} saved to: {save_path}")


def _plot_erf_compare_rb(
    obs_times:         np.ndarray,   
    erf_mean_classic:  np.ndarray,   
    erf_std_classic:   np.ndarray,   
    erf_mean_rb:       np.ndarray,   
    erf_std_rb:        np.ndarray,   
    n_traj:            int,
    title:             str,
    save_path:         str,
) -> None:
    """
    ERF comparison for PI+classic-EnKF vs PI+Route-B-EnKF.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(obs_times, erf_mean_classic,
            color="#2196F3", linewidth=2.0, marker="o", markersize=4,
            label=f"Classic EnKF  (n = {n_traj} trajectories)")
    ax.fill_between(
        obs_times, erf_mean_classic - erf_std_classic, erf_mean_classic + erf_std_classic,
        color="#2196F3", alpha=0.18, linewidth=0, label="Classic ±1 std",
    )

    ax.plot(obs_times, erf_mean_rb,
            color="#8E24AA", linewidth=2.0, marker="s", markersize=4,
            label=f"Route B EnKF  (n = {n_traj} trajectories)")
    ax.fill_between(
        obs_times, erf_mean_rb - erf_std_rb, erf_mean_rb + erf_std_rb,
        color="#8E24AA", alpha=0.18, linewidth=0, label="Route B ±1 std",
    )

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"ERF comparison plot (Classic vs Route B) saved to: {save_path}")


def _plot_rmse_comparison_rb(
    obs_times:               np.ndarray,
    prior_rmse_mean_classic: np.ndarray, prior_rmse_std_classic: np.ndarray,
    post_rmse_mean_classic:  np.ndarray, post_rmse_std_classic:  np.ndarray,
    prior_rmse_mean_rb:      np.ndarray, prior_rmse_std_rb:      np.ndarray,
    post_rmse_mean_rb:       np.ndarray, post_rmse_std_rb:       np.ndarray,
    sigma_obs:               float,
    n_traj:                  int,
    title:                   str,
    save_path:               str,
) -> None:
    """
    Prior/posterior RMSE comparison for PI+classic-EnKF vs PI+Route-B-EnKF.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Classic ─────────────────────────────────────────────────────────
    ax.plot(obs_times, prior_rmse_mean_classic,
            color="#0A36C7", linewidth=2.0, marker="o", markersize=4,
            linestyle="-", label=f"Classic prior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, prior_rmse_mean_classic - prior_rmse_std_classic,
        prior_rmse_mean_classic + prior_rmse_std_classic,
        color="#0A36C7", alpha=0.15, linewidth=0,
    )
    ax.plot(obs_times, post_rmse_mean_classic,
            color="#A30005", linewidth=2.0, marker="s", markersize=4,
            linestyle="-", label=f"Classic posterior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, post_rmse_mean_classic - post_rmse_std_classic,
        post_rmse_mean_classic + post_rmse_std_classic,
        color="#A30005", alpha=0.15, linewidth=0,
    )

    # ── Route B ─────────────────────────────────────────────────────────
    ax.plot(obs_times, prior_rmse_mean_rb,
            color="#8E24AA", linewidth=2.0, marker="^", markersize=4,
            linestyle="--", label=f"Route B prior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, prior_rmse_mean_rb - prior_rmse_std_rb,
        prior_rmse_mean_rb + prior_rmse_std_rb,
        color="#8E24AA", alpha=0.08, linewidth=0,
    )
    ax.plot(obs_times, post_rmse_mean_rb,
            color="#D81B60", linewidth=2.0, marker="d", markersize=4,
            linestyle="--", label=f"Route B posterior RMSE  (n = {n_traj})")
    ax.fill_between(
        obs_times, post_rmse_mean_rb - post_rmse_std_rb,
        post_rmse_mean_rb + post_rmse_std_rb,
        color="#D81B60", alpha=0.08, linewidth=0,
    )

    # ── Measurement noise reference ─────────────────────────────────────
    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.6,
               label=f"Measurement noise  σ_obs = {sigma_obs}")

    ax.set_yscale("log")
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("RMSE  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"RMSE comparison plot (Classic vs Route B) saved to: {save_path}")


def _plot_calibration_compare_rb(
    window_idx:            np.ndarray,
    dt_window:             float,
    spread_classic:        np.ndarray, rmse_classic:        np.ndarray,
    spread_rb:             np.ndarray, rmse_rb:             np.ndarray,
    spread_classic_raw:    np.ndarray, rmse_classic_raw:    np.ndarray,
    spread_rb_raw:         np.ndarray, rmse_rb_raw:         np.ndarray,
    title:                 str,
    save_path:             str,
    n_bins:                int = 10,
) -> None:
    """
    Calibration comparison for Classic vs Route B EnKF on PI model.
    """
    fig = plt.figure(figsize=(9, 13))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.3], hspace=0.55)

    def _timeseries_panel(ax, spread, rmse, c_spread, c_rmse, label):
        ax.plot(window_idx, spread, marker="^", markersize=4, linewidth=1.8,
                linestyle="-", color=c_spread, label=f"{label} RMS ensemble σ")
        ax.plot(window_idx, rmse, marker="s", markersize=4, linewidth=1.8,
                linestyle="--", color=c_rmse, label=f"{label} EnKF RMSE")
        ax.set_yscale("log")
        ax.set_xlabel("Window index", fontsize=11)
        ax.set_ylabel("Log scale", fontsize=11)
        ax.set_title(f"{label}: ensemble spread vs RMSE", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax_time = ax.twiny()
        ax_time.set_xlim(ax.get_xlim())
        ax_time.set_xticks(window_idx)
        ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx],
                                 fontsize=7, rotation=45, ha="left")
        ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=9)

    ax_rb = fig.add_subplot(gs[0])
    _timeseries_panel(ax_rb, spread_rb, rmse_rb, "#BA68C8", "#D81B60", "Route B")

    ax_classic = fig.add_subplot(gs[1])
    _timeseries_panel(ax_classic, spread_classic, rmse_classic, "#64B5F6", "#0A36C7", "Classic")

    ax_bin = fig.add_subplot(gs[2])
    rmss_rb_b, rmse_rb_b, rmse_rb_s, _ = _binned_spread_skill(spread_rb_raw, rmse_rb_raw, n_bins)
    rmss_cl_b, rmse_cl_b, rmse_cl_s, _ = _binned_spread_skill(spread_classic_raw, rmse_classic_raw, n_bins)

    lim_hi = 1.1 * max(rmss_rb_b.max(), rmse_rb_b.max(), rmss_cl_b.max(), rmse_cl_b.max())
    ax_bin.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.4,
                color="#37474F", label="1:1 (perfect calibration)")
    ax_bin.errorbar(rmss_rb_b, rmse_rb_b, yerr=rmse_rb_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#8E24AA", label=f"Route B ({n_bins}-bin)")
    ax_bin.errorbar(rmss_cl_b, rmse_cl_b, yerr=rmse_cl_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#2196F3", label=f"Classic ({n_bins}-bin)")

    ax_bin.set_xlim(0, lim_hi); ax_bin.set_ylim(0, lim_hi)
    ax_bin.set_xlabel("RMS ensemble spread (RMSS)", fontsize=11)
    ax_bin.set_ylabel("RMSE of ensemble mean", fontsize=11)
    ax_bin.set_title(f"Binned spread-skill  ({n_bins} equal-population bins, "
                      f"pooled over all ICs × windows)", fontsize=12)
    ax_bin.legend(fontsize=9)
    ax_bin.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bin.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Calibration comparison plot (Classic vs Route B) saved to: {save_path}")


def _evaluate_batch_enkf_pi_compare(
    model, params, predict_fn_classic, update_fn_classic,
    predict_fn_rb, update_fn_rb,
    t_star_window, u_test, t_test, F_test, 
    alpha_fine, alpha_b, beta_b, Q0, P0, R, obs_indices,
    N_ens, obs_every_n, sigma_obs, P0_sigma, dynamic_vars,
    specify_obs_idx, obs_idx_list, dt_window, dt_fine, dt_obs,
    num_ics_eval, enkf_batch_size, batch_windows, config, workdir
) -> None:
    """Batch Evaluation Logic mapped for Classic vs Route B."""
    B = min(num_ics_eval, enkf_batch_size, u_test.shape[0])
    N = model.N
    u0_batch = u_test[:B, 0, :]
    dt_test  = float(t_test[1] - t_test[0])

    total_time_batch = batch_windows * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=dt_fine, dt_obs=dt_obs
    )
    obs_step_indices_batch = jnp.array(obs_step_indices_batch)

    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    fine_stride = int(round(dt_fine / dt_test))
    n_fine_pts  = total_fine_steps_batch * fine_stride + 1
    
    x_true_fine_batch   = u_test[:B, 0:n_fine_pts:fine_stride, :]
    x_true_at_obs_batch = x_true_fine_batch[:, obs_step_indices_batch + 1, :]
    window_step_indices = np.array([round((k + 1) * dt_window / dt_fine) - 1 for k in range(batch_windows)])
    m = len(obs_indices)

    # ── 1. Vmapped EnKF Execution ──────────────────────────────────────────
    seed = config.training.get("seed", 42)
    master_key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(master_key, B)
    
    batched_enkf = build_batched_enkf_pi_compare(
        predict_fn_classic, update_fn_classic, predict_fn_rb, update_fn_rb,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, 
        alpha_fine, alpha_b, beta_b, Q0,
        dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch
    )

    (batch_x_means_classic, batch_x_spreads_classic, batch_prior_means_classic,
     batch_x_means_rb, batch_x_spreads_rb, batch_prior_means_rb, Q_scale_history,
     _, _) = batched_enkf(
         keys_batch, u0_batch, F_test[:B], x_true_at_obs_batch, 
         dynamic_vars, specify_obs_idx
    )

    # ── 2. Vectorized Metric Extraction ────────────────────────────────────
    post_means_classic_obs = batch_x_means_classic[:, obs_step_indices_batch, :N]
    post_means_rb_obs = batch_x_means_rb[:, obs_step_indices_batch, :N]
    prior_means_classic_obs = batch_prior_means_classic[:, :, :N]
    prior_means_rb_obs = batch_prior_means_rb[:, :, :N]

    prior_rmse_classic_ic = jnp.sqrt(jnp.mean((prior_means_classic_obs - x_true_at_obs_batch) ** 2, axis=2))
    post_rmse_classic_ic  = jnp.sqrt(jnp.mean((post_means_classic_obs - x_true_at_obs_batch) ** 2, axis=2))
    prior_rmse_rb_ic = jnp.sqrt(jnp.mean((prior_means_rb_obs - x_true_at_obs_batch) ** 2, axis=2))
    post_rmse_rb_ic  = jnp.sqrt(jnp.mean((post_means_rb_obs - x_true_at_obs_batch) ** 2, axis=2))

    erf_classic_ic = prior_rmse_classic_ic / (post_rmse_classic_ic + 1e-12)
    erf_rb_ic = prior_rmse_rb_ic / (post_rmse_rb_ic + 1e-12)

    erf_classic_mean, erf_classic_std = np.array(jnp.mean(erf_classic_ic, axis=0)), np.array(jnp.std(erf_classic_ic, axis=0))
    erf_rb_mean, erf_rb_std = np.array(jnp.mean(erf_rb_ic, axis=0)), np.array(jnp.std(erf_rb_ic, axis=0))
    
    prior_rmse_classic_mean, prior_rmse_classic_std = np.array(jnp.mean(prior_rmse_classic_ic, axis=0)), np.array(jnp.std(prior_rmse_classic_ic, axis=0))
    post_rmse_classic_mean, post_rmse_classic_std = np.array(jnp.mean(post_rmse_classic_ic, axis=0)), np.array(jnp.std(post_rmse_classic_ic, axis=0))
    prior_rmse_rb_mean, prior_rmse_rb_std = np.array(jnp.mean(prior_rmse_rb_ic, axis=0)), np.array(jnp.std(prior_rmse_rb_ic, axis=0))
    post_rmse_rb_mean, post_rmse_rb_std = np.array(jnp.mean(post_rmse_rb_ic, axis=0)), np.array(jnp.std(post_rmse_rb_ic, axis=0))

    x_true_at_windows = x_true_fine_batch[:, window_step_indices + 1, :]
    x_hat_classic_windows = batch_x_means_classic[:, window_step_indices, :N]
    x_hat_rb_windows = batch_x_means_rb[:, window_step_indices, :N]

    den = jnp.linalg.norm(x_true_at_windows, axis=2) + 1e-12
    l2_enkf_classic = np.array(jnp.mean(jnp.linalg.norm(x_hat_classic_windows - x_true_at_windows, axis=2) / den, axis=0))
    l2_enkf_rb = np.array(jnp.mean(jnp.linalg.norm(x_hat_rb_windows - x_true_at_windows, axis=2) / den, axis=0))

    rmse_classic_ic = jnp.sqrt(jnp.mean((x_hat_classic_windows - x_true_at_windows) ** 2, axis=2))
    rmse_rb_ic = jnp.sqrt(jnp.mean((x_hat_rb_windows - x_true_at_windows) ** 2, axis=2))
    rmse_enkf_classic = np.array(jnp.mean(rmse_classic_ic, axis=0))
    rmse_enkf_rb = np.array(jnp.mean(rmse_rb_ic, axis=0))

    spread_classic_ic = jnp.sqrt(jnp.mean(batch_x_spreads_classic[:, window_step_indices, :N] ** 2, axis=2))
    spread_rb_ic = jnp.sqrt(jnp.mean(batch_x_spreads_rb[:, window_step_indices, :N] ** 2, axis=2))
    spread_classic = np.array(jnp.mean(spread_classic_ic, axis=0))
    spread_rb = np.array(jnp.mean(spread_rb_ic, axis=0))

    rmse_classic_raw, rmse_rb_raw = np.array(rmse_classic_ic.flatten()), np.array(rmse_rb_ic.flatten())
    spread_classic_raw, spread_rb_raw = np.array(spread_classic_ic.flatten()), np.array(spread_rb_ic.flatten())

    x_true_fine_tail = x_true_fine_batch[:, 1:, :]
    den_dense = jnp.linalg.norm(x_true_fine_tail, axis=2) + 1e-12
    l2_enkf_classic_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_classic[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))
    l2_enkf_rb_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_rb[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))

    # ── 3. Open-loop Rollouts ───────────────────────────
    u0_batch_j = jnp.array(u0_batch)
    predict_full = jax.jit(jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=0))

    x_pred_dense_list = []
    u_current = jnp.concatenate([u0_batch_j, F_test[:B, None]], axis=-1)

    for k in range(batch_windows):
        x_win = predict_full(u_current)

        if k == 0:
            x_pred_dense_list.append(x_win)
        else:
            x_pred_dense_list.append(x_win[:, 1:, :])

        u_current = jnp.concatenate([x_win[:, -1, :], F_test[:B, None]], axis=-1)

    x_pred_dense = jnp.concatenate(x_pred_dense_list, axis=1)
    total_steps_ol = x_pred_dense.shape[1]
    x_ref_dense_ol = jnp.array(u_test[:B, :total_steps_ol, :])
    
    denom_ol = jnp.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
    l2_ol = np.array(jnp.mean(jnp.linalg.norm(x_pred_dense - x_ref_dense_ol, axis=2) / denom_ol, axis=0))

    t_eval_ol = t_test[:total_steps_ol]
    t_dense_fine = np.arange(1, total_fine_steps_batch + 1) * dt_fine

    logging.info(
        f"  [batch] Final-timestep mean L2 -> "
        f"Open-loop: {float(l2_ol[-1]):.3e} | Classic+EnKF: {l2_enkf_classic_dense[-1]:.3e} | "
        f"RouteB+EnKF: {l2_enkf_rb_dense[-1]:.3e}"
    )
    logging.info(
        f"  [batch] Final-window mean L2 (boundary-only) -> "
        f"Classic+EnKF: {l2_enkf_classic[-1]:.3e} | RouteB+EnKF: {l2_enkf_rb[-1]:.3e}"
    )

    # ── Plotting ──────────────────────────────────────────────────────────
    save_dir = os.path.join(workdir, "figures", "comparison_pi")
    os.makedirs(save_dir, exist_ok=True)

    curves = {
        "Open-loop":    (np.array(t_eval_ol), l2_ol),
        "Classic EnKF": (t_dense_fine,        l2_enkf_classic_dense),
        "Route B EnKF": (t_dense_fine,        l2_enkf_rb_dense),
    }
    colors = {
        "Open-loop":    "#90CAF9",
        "Classic EnKF": "#2196F3",
        "Route B EnKF": "#8E24AA",
    }
    _plot_l2_per_timestep(
        curves    = curves,
        title     = f"EnKF vs open-loop: mean relative L2 per timestep  (Classic vs Route B, B={B})",
        save_path = os.path.join(save_dir, "batch_l2_per_timestep_enkf_pi_compare.pdf"),
        colors    = colors,
    )

    _plot_calibration_compare_rb(
        window_idx = np.arange(1, batch_windows + 1),
        dt_window  = dt_window,
        spread_classic = spread_classic, rmse_classic = rmse_enkf_classic,
        spread_rb      = spread_rb,      rmse_rb      = rmse_enkf_rb,
        spread_classic_raw = spread_classic_raw, rmse_classic_raw = rmse_classic_raw,
        spread_rb_raw      = spread_rb_raw,      rmse_rb_raw      = rmse_rb_raw,
        title      = f"Calibration: ensemble spread vs RMSE  (Classic vs Route B, B={B}, N_ens={N_ens})",
        save_path  = os.path.join(save_dir, "batch_calibration_enkf_pi_compare.pdf"),
    )

    _plot_erf_compare_rb(
        obs_times        = obs_times_batch,
        erf_mean_classic = erf_classic_mean, erf_std_classic = erf_classic_std,
        erf_mean_rb      = erf_rb_mean,      erf_std_rb      = erf_rb_std,
        n_traj           = B,
        title       = (
            f"EnKF Error Reduction Factor per obs time  (Classic vs Route B)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path   = os.path.join(save_dir, "batch_erf_enkf_pi_compare.pdf"),
    )

    _plot_rmse_comparison_rb(
        obs_times               = obs_times_batch,
        prior_rmse_mean_classic = prior_rmse_classic_mean, prior_rmse_std_classic = prior_rmse_classic_std,
        post_rmse_mean_classic  = post_rmse_classic_mean,  post_rmse_std_classic  = post_rmse_classic_std,
        prior_rmse_mean_rb      = prior_rmse_rb_mean,      prior_rmse_std_rb      = prior_rmse_rb_std,
        post_rmse_mean_rb       = post_rmse_rb_mean,       post_rmse_std_rb       = post_rmse_rb_std,
        sigma_obs = sigma_obs, n_traj = B,
        title = (
            f"EnKF prior vs posterior RMSE  (Classic vs Route B)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path = os.path.join(save_dir, "batch_rmse_enkf_pi_compare.pdf"),
    )


def evaluate_enkf_pi_compare(
    config: ml_collections.ConfigDict,
    workdir: str,
    test_h5_path: str = None,
) -> None:
    """
    EnKF evaluation pipeline isolating the covariance-inflation strategy:
    Classic (fixed geometric multiplicative) vs Route B (residual-scaled additive)
    on the physics-informed DeepONet.
    """
    from examples.l96_f.kf import run_enkf_smoother, run_enkf_smoother_route_b, init_ensemble

    # ── EnKF / observation configuration ──
    obs_every_n  = config.kf.get("obs_every_n",   4)
    sigma_obs    = config.kf.get("sigma_obs",      0.5)
    P0_sigma     = config.kf.get("P0_sigma",       1.0)
    dynamic_vars = config.kf.get("dynamic_vars",   False)
    N_ens        = config.kf.get("N_ens",         50)
    
    alpha_coarse = config.kf.get("inflation_factor", 1.05)
    
    alpha_b  = config.kf.get("alpha_b", 1.0)
    beta_b   = config.kf.get("beta_b", 1.0)
    Q0_scale = config.kf.get("Q0_scale", 1.0)

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list", None)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.kf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.kf.get("dt_obs",    DT_WINDOW))

    # ── 1. Load the long test trajectories and forcing parameters ─────────
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"

    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]
        t_test = f["t"][:]
        F_test = f["F"][:]

    dt_window = float(config.get("dt_window", 0.25))

    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows      = config.eval.get("windows", 200)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
    dt_integration     = config.eval.get("dt_integration", 0.005)
    enkf_batch_size    = config.kf.get("batch_l2_size", 200)
 
    # ── 2. Models & per-window query grid ───────────────────────────────────
    time_steps = int(round(dt_window / dt_integration)) + 1 
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
    T_last = float(t_star_window[-1])

    logging.info("Loading PI model for comparison...")
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    N = model.N

    # ── 3. EnKF predict/update functions for both regimes ─────────────────
    predict_fn_classic, update_fn_classic = model.make_enkf_fns(params, N_ens=N_ens)
    predict_fn_rb, update_fn_rb = model.make_route_b_enkf_fns(params, N_ens=N_ens)

    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
    Q0 = jnp.eye(N) * Q0_scale

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    # ── 4. Per-IC single-trajectory EnKF evaluation ────────────
    num_plots  = min(config.saving.total_plots, u_test.shape[0])
    total_time = trajectory_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time = total_time, dt_fine = DT_FINE, dt_obs = DT_OBS,
    )
    obs_step_indices = jnp.array(obs_step_indices)

    x_true_fine_list, x_true_at_obs_list = [], []
    t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
    
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
        def lorenz_96(t, state, F=F_i):
            x_plus_1  = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F

        sol = solve_ivp(
            lorenz_96, t_span=[0.0, total_time], y0=np.array(u_test[ic_idx, 0, :]),
            t_eval=t_eval_fine, method='LSODA', rtol=1e-13, atol=1e-14,
        )
        x_true_fine_list.append(sol.y.T)
        x_true_at_obs_list.append(sol.y.T[obs_step_indices + 1])

    x_true_fine_batch = jnp.stack(x_true_fine_list)
    x_true_at_obs_batch = jnp.stack(x_true_at_obs_list)
    u0_batch_plots = jnp.array(u_test[:num_plots, 0, :])
    F_batch_plots = jnp.array(F_test[:num_plots])
    keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_plots))

    batched_enkf_plots = build_batched_enkf_pi_compare(
        predict_fn_classic, update_fn_classic, predict_fn_rb, update_fn_rb,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, 
        alpha_fine, alpha_b, beta_b, Q0,
        DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices
    )

    (batch_x_means_classic, batch_x_spreads_classic, _,
     batch_x_means_rb, batch_x_spreads_rb, _, _,
     batch_y_obs, batch_idx_vars) = batched_enkf_plots(
         keys_batch_plots, u0_batch_plots, F_batch_plots, 
         x_true_at_obs_batch, dynamic_vars, specify_obs_idx
    )

    t_fine_axis = t_eval_fine[1:]
    window_step_indices = np.array([round((w + 1) * DT_WINDOW / DT_FINE) - 1 for w in range(trajectory_windows)])
    
    for ic_idx in range(num_plots):
        x_true_fine = x_true_fine_batch[ic_idx]
        x_means_classic, x_spreads_classic = batch_x_means_classic[ic_idx], batch_x_spreads_classic[ic_idx]
        x_means_rb, x_spreads_rb = batch_x_means_rb[ic_idx], batch_x_spreads_rb[ic_idx]
        y_obs_seq, idx_vars_seq = batch_y_obs[ic_idx], batch_idx_vars[ic_idx]

        obs_coords = []
        for obs_idx, t_obs in enumerate(obs_times):
            for j, vi in enumerate(idx_vars_seq[obs_idx]):
                obs_coords.append((int(vi), float(t_obs), float(y_obs_seq[obs_idx, j])))

        save_path = os.path.join(workdir, "figures", "comparison_pi", f"trajectory_summary_enkf_pi_compare_ic_{ic_idx}.pdf")
        
        _plot_trajectory_summary_compare_enkf_rb(
            t_ax=t_fine_axis, x_true=np.array(x_true_fine[1:]),
            x_est_classic=np.array(x_means_classic[:, :N]), x_std_classic=np.array(x_spreads_classic[:, :N]),
            x_est_rb=np.array(x_means_rb[:, :N]), x_std_rb=np.array(x_spreads_rb[:, :N]),
            ic_idx=ic_idx, N=N, dt_window=DT_WINDOW, obs_coords=obs_coords,
            save_path=save_path
        )

        x_true_at_windows = x_true_fine[window_step_indices + 1]
        l2_classic = jnp.linalg.norm(x_means_classic[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
        l2_rb = jnp.linalg.norm(x_means_rb[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)

        print(
            f"IC {ic_idx} | EnKF Classic L2: {l2_classic:.3e} | EnKF Route B L2: {l2_rb:.3e} "
            f"| Mean σ (Classic): {float(jnp.mean(x_spreads_classic)):.3e} "
            f"| Mean σ (Route B): {float(jnp.mean(x_spreads_rb)):.3e}"
        )

    # ── 5. Batch-averaged comparison ─────────
    _evaluate_batch_enkf_pi_compare(
        model, params, predict_fn_classic, update_fn_classic,
        predict_fn_rb, update_fn_rb,
        t_star_window,
        u_test, t_test, F_test,
        alpha_fine, alpha_b, beta_b, Q0, P0, R, obs_indices,
        N_ens, obs_every_n, sigma_obs, P0_sigma, dynamic_vars,
        specify_obs_idx, obs_idx_list,
        DT_WINDOW, DT_FINE, DT_OBS,
        num_ics_eval, enkf_batch_size, batch_windows,
        config, workdir,
    )