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
import examples.l96_forcing.models as models
from examples.l96_forcing.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps

import numpy as np
from scipy.integrate import solve_ivp
import h5py


def _plot_trajectory_summary(
    t_ax:       np.ndarray,
    x_true:     np.ndarray,
    x_est:      np.ndarray,
    ic_idx:     int,
    F_val:      float,
    save_path:  str,
    N:          int = 40,
    dt_window:  Optional[float] = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single IC.
    """
    x_true = np.asarray(x_true)
    x_est  = np.asarray(x_est)
 
    abs_error    = np.abs(x_true - x_est)
    mean_abs_err = abs_error.mean(axis=1)
 
    n_var_rows = N // 2
 
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])
 
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
    ax_top.plot(t_ax, mean_abs_err, color="#E53935", linewidth=1.6,
                label="Mean |error| over variables")
 
    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)
 
    ax_top.set_xlabel("Time (t)", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} (F = {F_val:.2f}) — Mean absolute error across all {N} variables (DeepONet)",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    TRUTH_COLOR = "#37474F"
    EST_COLOR   = "#1E88E5"
 
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.6, alpha=0.45)
 
        ax.plot(t_ax, x_true[:, i], color=TRUTH_COLOR, linewidth=1.0, label="Truth")
        ax.plot(t_ax, x_est[:, i], color=EST_COLOR, linewidth=1.0, linestyle="--", label="Prediction")
 
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
        f"Trajectory summary — IC {ic_idx}  |  Forcing F = {F_val:.3f}",
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
    """
    Plots the batch-average L2 error over time, followed by a breakdown binned by F values.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # ── Top Panel: Overall Mean ──
    axes[0].plot(t_ax, overall_mean_l2, color="#1E88E5", linewidth=2.5, label="Overall Mean (All Trajectories)")
    axes[0].set_ylabel("Mean Relative L2 Error", fontsize=11)
    axes[0].set_title("Overall Mean L2 Error Over Time", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=11)

    # ── Bottom Panel: Grouped by F ──
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


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # ── 1. Load Dense Test Dataset ──────────────────────────────────────────
    data_dir = config.training.get("data_dir", os.path.join("examples", "l96_forcing", "data"))
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
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
    
    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor
    # The branch network expects a 41-D input [u_0...u_39, F]
    predict_batch = jax.jit(jax.vmap(lambda u_aug: model.x_pred_fn(params, u_aug, t_star_window), in_axes=0))

    # ── 3. Batched Autoregressive Rollout ───────────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")
    
    u_current_batch = u_test[:, 0, :]               # Shape: (num_ics, 40)
    F_batch = F_test[:, None]                       # Shape: (num_ics, 1)

    x_pred_list = []
    
    for w in range(L_windows):
        # Augment current states with their corresponding F parameter
        u_aug_batch = jnp.concatenate([u_current_batch, F_batch], axis=-1)
        
        # Predict the full trajectory for the current time window
        pred_window = predict_batch(u_aug_batch)    # Shape: (num_ics, pts_pw+1, 40)

        # Avoid duplicating the overlapping boundary states between windows
        if w == 0:
            x_pred_list.append(pred_window)
        else:
            x_pred_list.append(pred_window[:, 1:, :])

        # Advance the initial conditions to the end of the predicted window
        u_current_batch = pred_window[:, -1, :]

    # Reconstruct the continuous dense time series
    x_pred_full = jnp.concatenate(x_pred_list, axis=1)  # Shape: (num_ics, num_test_pts, 40)

    # ── 4. Generate Individual Trajectory Plots ─────────────────────────────
    total_plots = config.saving.get("total_plots", 5)
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
            save_path=save_path,
            N=model.N,
            dt_window=window_size
        )

    # ── 5. Generate Batch Error & Binned F Analysis ─────────────────────────
    logging.info("--- Computing Batch L2 Error Statistics ---")
    
    # Calculate the relative L2 error step-by-step for all 500 trajectories
    err = x_pred_full - u_test
    norm_err = jnp.linalg.norm(err, axis=-1)
    norm_ref = jnp.linalg.norm(u_test, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    # Convert to standard numpy for logic masking and plotting
    l2_rel_np = np.array(l2_rel_per_traj_time)
    F_np = np.array(F_test)
    t_test_np = np.array(t_test)

    # Compute overall mean across all test cases
    overall_mean_l2 = np.mean(l2_rel_np, axis=0)

    # Bin trajectories by their independent forcing parameter
    grouped_l2 = {}
    bins = [(5, 6), (6, 7), (7, 8), (8, 9.01)] # 9.01 ensures exactly F=9 is captured
    
    for lower, upper in bins:
        mask = (F_np >= lower) & (F_np < upper)
        if np.any(mask):
            group_mean = np.mean(l2_rel_np[mask], axis=0)
            
            # Format the label nicely
            upper_label = int(upper) if upper > 9 else upper
            label = f"F ∈ [{lower}, {upper_label}) (n={np.sum(mask)})"
            grouped_l2[label] = group_mean

    # Plot the aggregated analytics
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