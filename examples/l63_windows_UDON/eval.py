import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset

def evaluate_prev(config: ml_collections.ConfigDict, workdir: str):
    # Load full dataset
    xyz_ref, t_star = get_dataset()
    
    # Setup windowing logic
    num_windows = config.training.num_time_windows
    num_time_steps = len(t_star) // num_windows
    
    # Initialize model (using the first window's time and initial condition)
    # Note: t must match the training window size for proper internal scaling
    t_window_init = t_star[:num_time_steps]
    xyz0 = xyz_ref[0, :]
    model = models.L63(config, xyz0, t_window_init)

    xyz_pred_list = []

    for idx in range(num_windows):
        # Determine the time slice for this window
        start_idx = idx * num_time_steps
        end_idx = (idx + 1) * num_time_steps
        t_window = t_star[start_idx:end_idx]
        
        # Restore checkpoint for the specific window
        ckpt_path = os.path.join(
            os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1)
        )
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Predict for this window's time segment
        # We pass the specific t_window to the prediction function
        xyz_pred_window = model.xyz_pred_fn(params, t_window)
        xyz_pred_list.append(xyz_pred_window)

        # Optional: Log window-specific error
        window_ref = xyz_ref[start_idx:end_idx, :]
        window_error = jnp.linalg.norm(xyz_pred_window - window_ref) / jnp.linalg.norm(window_ref)
        logging.info(f"Window {idx + 1} L2 Error: {window_error:.3e}")

    # Concatenate all window predictions into one full trajectory
    xyz_pred_full = jnp.concatenate(xyz_pred_list, axis=0)
    
    # Slicing xyz_ref to match xyz_pred_full in case of rounding in num_time_steps
    xyz_ref_matched = xyz_ref[:xyz_pred_full.shape[0], :]
    t_star_matched = t_star[:xyz_pred_full.shape[0]]

    # Compute total L2 error
    total_l2_error = jnp.linalg.norm(xyz_pred_full - xyz_ref_matched) / jnp.linalg.norm(xyz_ref_matched)
    print(f"Full Trajectory L2 error: {total_l2_error:.3e}")

    # --- Plotting Logic (L63 Style) ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    components = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        # Exact vs Predicted Trajectory
        axes[i, 0].plot(t_star_matched, xyz_ref_matched[:, i], label='Exact', color='black', linewidth=1.5)
        axes[i, 0].plot(t_star_matched, xyz_pred_full[:, i], label='Predicted', color=colors[i], linestyle='--', linewidth=1)
        axes[i, 0].set_ylabel(f"{components[i]}(t)", fontsize=14)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="upper right")
        
        # Add vertical lines to show window boundaries
        for w in range(1, num_windows):
            axes[i, 0].axvline(x=t_star[w * num_time_steps], color='gray', linestyle=':', alpha=0.5)

        if i == 0:
            axes[i, 0].set_title("Windowed L63: Exact vs. Predicted", fontsize=16)
        if i == 2:
            axes[i, 0].set_xlabel("Time (t)", fontsize=14)

        # Absolute Error (Log Scale)
        abs_error = jnp.abs(xyz_ref_matched[:, i] - xyz_pred_full[:, i])
        axes[i, 1].plot(t_star_matched, abs_error, color=colors[i], linewidth=1.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_yscale('log') 
        
        if i == 0:
            axes[i, 1].set_title("Absolute Error per Component", fontsize=16)
        if i == 2:
            axes[i, 1].set_xlabel("Time (t)", fontsize=14)

    fig.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "l63_windowed.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


### N

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # 1. Load dataset (xyz_ref is 3D: [num_ics, num_points, 3])
    # Assuming xyz_ref_all contains the full trajectory up to t=20
    xyz_ref_all, u0_ref_all, t_star_window = get_dataset()
    
    # Pick the first trajectory (IC 0) for the long-term rollout evaluation
    u_current = u0_ref_all[0, :] 
    xyz_ref_trajectory = xyz_ref_all[0, :, :] 

    # 2. Setup Model & Load Checkpoint
    # DeepONets are time-invariant for their trained horizon. We just need the single trained model.
    model = models.L63UDON(config, t_star_window)
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_model"
    )
    
    logging.info(f"Restored trained DeepONet model for autoregressive rollout.")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # 3. Rollout Loop
    xyz_pred_list = []
    t_full_list = []
    
    # If trained on [0,1] and rolling out to [0,20], num_windows should be 20
    num_windows = 20
    dt_window = t_star_window[-1] - t_star_window[0]

    for idx in range(num_windows):
        # Predict: Output is likely (1, num_points, 3)
        preds = model.xyz_pred_fn(params, u_current, t_star_window)
        xyz_pred_window = jnp.squeeze(preds)

        # Handle the overlapping boundary (t=1.0, t=2.0, etc.)
        if idx == 0:
            # For the first window, keep all points
            xyz_pred_list.append(xyz_pred_window)
            t_full_list.append(t_star_window)
        else:
            # For subsequent windows, drop the first point to avoid duplication
            xyz_pred_list.append(xyz_pred_window[1:])
            t_offset = idx * dt_window
            t_full_list.append(t_star_window[1:] + t_offset)

        # Autoregressive step: use the LAST point of this prediction as the next IC
        u_current = xyz_pred_window[-1, :]

    # 4. Finalize results
    xyz_pred_full = jnp.concatenate(xyz_pred_list, axis=0)
    t_star_full = jnp.concatenate(t_full_list, axis=0)
    
    # Ensure reference matches the total rollout length
    # This assumes your get_dataset() loaded a long enough reference trajectory
    xyz_ref_matched = xyz_ref_trajectory[:xyz_pred_full.shape[0], :]

    # Compute total L2 error
    total_l2_error = jnp.linalg.norm(xyz_pred_full - xyz_ref_matched) / jnp.linalg.norm(xyz_ref_matched)
    print(f"Full Rollout Trajectory L2 error: {total_l2_error:.3e}")

    # --- Plotting Logic ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    components = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        # Time-series plot
        axes[i, 0].plot(t_star_full, xyz_ref_matched[:, i], label='Exact', color='black', linewidth=1.5)
        axes[i, 0].plot(t_star_full, xyz_pred_full[:, i], label='UDON Rollout', color=colors[i], linestyle='--', linewidth=1.2)
        axes[i, 0].set_ylabel(f"{components[i]}(t)", fontsize=14)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="upper right")
        
        # Window boundaries - adjusted to point to the actual split indices
        for w in range(1, num_windows):
            boundary_time = w * dt_window
            axes[i, 0].axvline(x=boundary_time, color='gray', linestyle=':', alpha=0.4)

        if i == 0:
            axes[i, 0].set_title("Long-term Autoregressive Rollout", fontsize=16)

        # Absolute Error (Log Scale)
        abs_error = jnp.abs(xyz_ref_matched[:, i] - xyz_pred_full[:, i])
        axes[i, 1].plot(t_star_full, abs_error, color=colors[i], linewidth=1.5)
        axes[i, 1].set_yscale('log')
        axes[i, 1].grid(True, which="both", alpha=0.2)
        
        if i == 0:
            axes[i, 1].set_title("Log Absolute Error", fontsize=16)

    axes[2, 0].set_xlabel("Time (t)", fontsize=14)
    axes[2, 1].set_xlabel("Time (t)", fontsize=14)
    fig.tight_layout()

    # Save results
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "udon_rollout_analysis.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    logging.info(f"Evaluation plot saved to: {fig_path}")