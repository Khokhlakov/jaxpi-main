import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset

import numpy as np
from scipy.integrate import solve_ivp

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # 1. Load dataset (x_ref is 3D: [num_ics, num_points, 40])
    x_ref_all, u0_ref_all, t_star_window = get_dataset()

    # 2. Setup Model & Load Checkpoint once (no need to reload per trajectory)
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "udon_model"
    )
    
    logging.info(f"Restored trained DeepONet model for autoregressive rollout.")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Define the Initial Condition (IC) indices you want to plot 
    # (e.g., 0 for the first trajectory, 1 for the second)
    ic_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for ic_idx in ic_indices:
        logging.info(f"--- Evaluating Trajectory for IC index {ic_idx} ---")
        
        # Pick the trajectory for the current IC
        u_current = u0_ref_all[ic_idx, :] 
        logging.info(f"--- The current u is: {u_current} ---")

        # 3. Rollout Loop
        x_pred_list = []
        t_full_list = []
        
        num_windows = config.training.num_time_windows 
        dt_window = t_star_window[-1] - t_star_window[0]

        for idx in range(num_windows):
            # Predict
            preds = model.x_pred_fn(params, u_current, t_star_window)
            x_pred_window = jnp.squeeze(preds)

            # Handle the overlapping boundary 
            if idx == 0:
                x_pred_list.append(x_pred_window)
                t_full_list.append(t_star_window)
            else:
                x_pred_list.append(x_pred_window[1:])
                t_offset = idx * dt_window
                t_full_list.append(t_star_window[1:] + t_offset)

            # Use last point of this prediction as the next IC
            u_current = x_pred_window[-1, :]

        x_pred_full = jnp.concatenate(x_pred_list, axis=0)
        t_star_full = jnp.concatenate(t_full_list, axis=0)
        
        # Generate Exact Reference on the fly for L96
        def lorenz_96(t, state, F=6.0):
            x_plus_1 = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F
        
        t_eval_np = np.array(t_star_full)
        u0_np = np.array(u0_ref_all[ic_idx, :]) # Use the correct IC
        
        # Solve the ODE over the full rollout time
        sol = solve_ivp(
            lorenz_96, 
            t_span=[t_eval_np[0], t_eval_np[-1]], 
            y0=u0_np, 
            t_eval=t_eval_np,
            rtol=1e-8, 
            atol=1e-10
        )
        x_ref_matched = jnp.array(sol.y.T) 
        # ------------------------------------------------

        # Compute total L2 error 
        total_l2_error = jnp.linalg.norm(x_pred_full - x_ref_matched) / jnp.linalg.norm(x_ref_matched)
        print(f"IC {ic_idx} | Full Rollout Trajectory L2 error: {total_l2_error:.3e}")

        # --- Plotting Logic: Heatmaps ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # 1. Exact Reference Heatmap
        im0 = axes[0].pcolormesh(np.arange(model.N), t_star_full, x_ref_matched, cmap='viridis', shading='auto')
        axes[0].set_title(f"Exact L96 Reference (IC {ic_idx})", fontsize=14)
        axes[0].set_ylabel("Time (t)", fontsize=14)
        axes[0].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im0, ax=axes[0])
        
        # 2. UDON Prediction Heatmap
        im1 = axes[1].pcolormesh(np.arange(model.N), t_star_full, x_pred_full, cmap='viridis', shading='auto')
        axes[1].set_title(f"UDON Rollout (IC {ic_idx})", fontsize=14)
        axes[1].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im1, ax=axes[1])
        
        # 3. Absolute Error Heatmap
        abs_error = jnp.abs(x_ref_matched - x_pred_full)
        im2 = axes[2].pcolormesh(np.arange(model.N), t_star_full, abs_error, cmap='magma', shading='auto')
        axes[2].set_title(f"Absolute Error (IC {ic_idx})", fontsize=14)
        axes[2].set_xlabel("Variables (0 to 39)", fontsize=14)
        fig.colorbar(im2, ax=axes[2])

        # Draw Window boundaries across the heatmaps
        for ax in axes:
            for w in range(1, num_windows):
                boundary_time = w * dt_window
                ax.axhline(y=boundary_time, color='white', linestyle=':', alpha=0.5)

        fig.tight_layout()

        # Save results with dynamically named files
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"udon_rollout_analysis_ic_{ic_idx}.pdf")
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        
        logging.info(f"Evaluation plot for IC {ic_idx} saved to: {fig_path}")