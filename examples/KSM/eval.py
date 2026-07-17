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
import examples.KSM.models as models
from examples.KSM.utils import build_obs_schedule, scale_Q_for_fine_steps

import numpy as np
from scipy.integrate import solve_ivp
import h5py


def _plot_trajectory_summary(
    t_ax: np.ndarray,
    x_true: np.ndarray,
    x_est: np.ndarray,
    ic_idx: int,
    test_windows: int,
    pts_pw: int,
    save_path: str,
    N: int = 256,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single KS IC.
    Layout: 
    - Row 1: 3 Heatmaps (Pred, True, Diff)
    - Row 2: Relative L2 Error over time
    - Rows 3+: Line plots at every window boundary (t=0, t=1, ...)
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    
    # 1. Compute Errors
    diff = x_est - x_true
    # Relative L2 error for this specific trajectory over time
    norm_err = np.linalg.norm(diff, axis=-1)
    norm_ref = np.linalg.norm(x_true, axis=-1)
    l2_rel = norm_err / (norm_ref + 1e-12)

    # 2. Grid Setup
    # Boundaries are at index w * pts_pw for w in 0...test_windows
    num_boundaries = test_windows + 1
    cols_line_plots = 4 # How many line plots per row
    rows_line_plots = int(np.ceil(num_boundaries / cols_line_plots))
    
    total_rows = 2 + rows_line_plots
    
    # Dynamic figure height based on how many boundary plots we have
    fig = plt.figure(figsize=(18, 5 + 3.5 + 2.5 * rows_line_plots))
    
    # Use 12 columns to easily split by 3 (heatmaps) and by 4 (line plots)
    gs = gridspec.GridSpec(
        nrows=total_rows, 
        ncols=12, 
        figure=fig, 
        hspace=0.6, 
        wspace=0.6,
        height_ratios=[1.5, 0.8] + [1.0] * rows_line_plots
    )

    # --- ROW 0: Heatmaps ---
    ax_heat_pred = fig.add_subplot(gs[0, 0:4])
    ax_heat_true = fig.add_subplot(gs[0, 4:8], sharey=ax_heat_pred)
    ax_heat_diff = fig.add_subplot(gs[0, 8:12], sharey=ax_heat_pred)

    extent = [0, N-1, t_ax[0], t_ax[-1]]
    
    # Prediction
    im_pred = ax_heat_pred.imshow(x_est, aspect='auto', extent=extent, cmap='viridis', origin='lower')
    ax_heat_pred.set_title("DeepONet Prediction", fontsize=12, fontweight='bold')
    ax_heat_pred.set_ylabel("Time (t)")
    fig.colorbar(im_pred, ax=ax_heat_pred, fraction=0.046, pad=0.04)

    # Truth
    im_true = ax_heat_true.imshow(x_true, aspect='auto', extent=extent, cmap='viridis', origin='lower')
    ax_heat_true.set_title("Reference Truth", fontsize=12, fontweight='bold')
    ax_heat_true.set_xlabel("Spatial Points")
    ax_heat_true.tick_params(labelleft=False)
    fig.colorbar(im_true, ax=ax_heat_true, fraction=0.046, pad=0.04)

    # Difference (Centered at 0)
    vmax_diff = np.max(np.abs(diff))
    im_diff = ax_heat_diff.imshow(diff, aspect='auto', extent=extent, cmap='RdBu_r', 
                                  vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
    ax_heat_diff.set_title("Absolute Difference", fontsize=12, fontweight='bold')
    ax_heat_diff.tick_params(labelleft=False)
    fig.colorbar(im_diff, ax=ax_heat_diff, fraction=0.046, pad=0.04)
    
    # --- ROW 1: Mean L2 Error Plot ---
    ax_l2 = fig.add_subplot(gs[1, :])
    ax_l2.plot(t_ax, l2_rel, color="#E53935", linewidth=2.0, label="Relative L2 Error")
    
    # Add vertical lines for window boundaries
    for w in range(num_boundaries):
        wb = t_ax[w * pts_pw]
        ax_l2.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.8, alpha=0.5)
        
    ax_l2.set_title("Trajectory Relative L2 Error Over Time", fontsize=12, fontweight='bold')
    ax_l2.set_xlabel("Time (t)")
    ax_l2.set_ylabel("Error")
    ax_l2.set_yscale("log")
    ax_l2.grid(True, linestyle="--", alpha=0.6)
    ax_l2.legend()

    # --- ROWS 2+: Boundary Line Plots ---
    TRUTH_COLOR = "#37474F"
    EST_COLOR   = "#1E88E5"
    x_nodes = np.arange(N)

    for i in range(num_boundaries):
        row_idx = 2 + (i // cols_line_plots)
        # Each plot takes 3 grid columns (12 / 4 = 3)
        col_start = (i % cols_line_plots) * 3 
        col_end = col_start + 3
        
        ax = fig.add_subplot(gs[row_idx, col_start:col_end])
        
        t_idx = i * pts_pw
        t_val = t_ax[t_idx]
        
        ax.plot(x_nodes, x_true[t_idx], color=TRUTH_COLOR, linewidth=1.5, label="Truth")
        ax.plot(x_nodes, x_est[t_idx], color=EST_COLOR, linewidth=1.5, linestyle="--", label="Pred")
        
        ax.set_title(f"Boundary t = {t_val:.1f}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, N-1)
        
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"KS Trajectory Summary — IC {ic_idx}", fontsize=16, fontweight="bold", y=0.99)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Trajectory summary for IC {ic_idx} saved to: {save_path}")

def _plot_batch_l2_over_time(
    t_ax: np.ndarray,
    overall_mean_l2: np.ndarray,
    save_path: str
) -> None:
    """
    Plots the batch-average L2 error over time for the KS system.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_ax, overall_mean_l2, color="#1E88E5", linewidth=2.5, label="Overall Mean (All Trajectories)")
    ax.set_xlabel("Time (t)", fontsize=11)
    ax.set_ylabel("Mean Relative L2 Error", fontsize=11)
    ax.set_title("KS System: Overall Mean L2 Error Over Time", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # ── 1. Load Dense Test Dataset ──────────────────────────────────────────
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "ks_test_data.h5")

    logging.info(f"Loading test dataset from {test_file}...")
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:])     # Shape: (num_ics, num_test_pts, 256)
        N = f.attrs['N']
        dt = f.attrs['dt']
        test_windows = f.attrs['test_windows']
        
    num_ics, num_test_pts, N_loaded = u_test.shape
    assert N_loaded == 256, f"Expected state dimension 256, got {N_loaded}"

    # Reconstruct time definitions
    # 1 time unit = 1 window
    pts_pw = int(round(1.0 / dt))
    t_ax = np.linspace(0, test_windows, num_test_pts)
    
    # Single-window relative time grid required by the surrogate model
    t_star_window = t_ax[:pts_pw + 1]

    # ── 2. Setup Model & Load Checkpoint ────────────────────────────────────
    if config.mode == "eval":
        model = models.KSUDON(config, t_star_window)
    else:
        model = models.KSUDON_DD(config, t_star_window)
        
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
    
    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor
    # KS is autonomous, so input is just u (shape: 256)
    predict_batch = jax.jit(jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=0))

    # ── 3. Batched Autoregressive Rollout ───────────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")
    
    u_current_batch = u_test[:, 0, :]               # Shape: (num_ics, 256)
    x_pred_list = []
    
    for w in range(test_windows):
        # Predict the full trajectory for the current time window
        pred_window = predict_batch(u_current_batch)    # Shape: (num_ics, pts_pw+1, 256)

        # Avoid duplicating the overlapping boundary states between windows
        if w == 0:
            x_pred_list.append(pred_window)
        else:
            x_pred_list.append(pred_window[:, 1:, :])

        # Advance the initial conditions to the end of the predicted window
        u_current_batch = pred_window[:, -1, :]

    # Reconstruct the continuous dense time series
    x_pred_full = jnp.concatenate(x_pred_list, axis=1)  # Shape: (num_ics, num_test_pts, 256)

    # ── 4. Generate Individual Trajectory Plots ─────────────────────────────
    total_plots = config.saving.get("total_plots", 5)
    for ic_idx in range(min(total_plots, num_ics)):
        logging.info(f"--- Generating detailed summary for IC {ic_idx} ---")
        
        save_path = os.path.join(
            workdir, "figures", config.wandb.name, f"trajectory_summary_ic_{ic_idx}.pdf"
        )
        
        _plot_trajectory_summary(
            t_ax=t_ax,
            x_true=np.array(u_test[ic_idx]),
            x_est=np.array(x_pred_full[ic_idx]),
            ic_idx=ic_idx,
            test_windows=test_windows,
            pts_pw=pts_pw,
            save_path=save_path,
            N=N_loaded
        )

    # ── 5. Generate Batch Error Analysis ─────────────────────────
    logging.info("--- Computing Batch L2 Error Statistics ---")
    
    err = x_pred_full - u_test
    norm_err = jnp.linalg.norm(err, axis=-1)
    norm_ref = jnp.linalg.norm(u_test, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    l2_rel_np = np.array(l2_rel_per_traj_time)

    # Compute overall mean across all test cases (axis 0 is batch)
    overall_mean_l2 = np.mean(l2_rel_np, axis=0)

    # Plot the aggregated analytics
    batch_save_path = os.path.join(
        workdir, "figures", config.wandb.name, "batch_l2_error_analysis.pdf"
    )
    
    _plot_batch_l2_over_time(
        t_ax=t_ax, 
        overall_mean_l2=overall_mean_l2, 
        save_path=batch_save_path
    )
    
    logging.info(f"Batch L2 error plot saved to: {batch_save_path}")