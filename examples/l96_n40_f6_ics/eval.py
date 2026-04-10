import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax.tree_util import tree_map
from flax.jax_utils import replicate

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
    for ic_idx in range(config.saving.total_plots):
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
        def lorenz_96(t, state, F=2.0):
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

def evaluate_with_ekf(config: ml_collections.ConfigDict, workdir: str):
    """
    Evaluates the trained UDON model with EKF data assimilation.

    Scenario:
      - obs_every_n: observe only 1-in-N variables at each assimilation step.
      - assimilation_dt: time gap between observations (= one prediction window).
      - Measurement noise std: sigma_obs.
    """
    from ekf import EKFState, run_ekf_smoother

    obs_every_n  = config.ekf.get("obs_every_n",  4)    # observe every 4th variable
    sigma_obs    = config.ekf.get("sigma_obs",    0.5)   # observation noise std
    sigma_proc   = config.ekf.get("sigma_proc",   0.1)   # process noise std
    P0_sigma     = config.ekf.get("P0_sigma",     1.0)   # initial covariance std

    x_ref_all, u0_ref_all, t_star_window = get_dataset()

    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    model.state = restore_checkpoint(model.state, ckpt_path)

    # Unreplicated params for inference
    params = model.state.params


    # Assimilation dt = one window length
    dt = float(t_star_window[-1] - t_star_window[0])

    # Build EKF functions (JIT-compiled, bound to frozen params)
    predict_fn, update_fn = model.make_ekf_fns(params, dt)

    # ── Observation operator ──────────────────────────────────────────────────
    # Observe every obs_every_n-th variable → m = N // obs_every_n
    N = model.N
    obs_indices = jnp.arange(0, N, obs_every_n)
    m = len(obs_indices)
    H = jnp.zeros((m, N))
    H = H.at[jnp.arange(m), obs_indices].set(1.0)   # (m, N)

    # Noise covariances
    Q = jnp.eye(N) * sigma_proc ** 2
    R = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
    # ─────────────────────────────────────────────────────────────────────────

    num_windows = config.training.num_time_windows

    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- EKF Evaluation for IC {ic_idx} ---")

        u_current_true = u0_ref_all[ic_idx, :]  # ground truth IC

        # ── Generate noisy observations by forward-simulating reference ───────
        def lorenz_96(t, state, F=2.0):
            xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
            return (xp1 - xm2) * xm1 - state + F

        t_eval_full = np.array([i * dt for i in range(num_windows + 1)])
        sol = solve_ivp(
            lorenz_96,
            t_span=[t_eval_full[0], t_eval_full[-1]],
            y0=np.array(u_current_true),
            t_eval=t_eval_full, rtol=1e-9, atol=1e-11
        )
        x_true_windows = jnp.array(sol.y.T)   # (num_windows+1, N) — states at each window boundary

        # Add noise to create synthetic observations
        key = jax.random.PRNGKey(ic_idx)
        noise = sigma_obs * jax.random.normal(key, shape=(num_windows, m))
        # Observe every window boundary (skipping t=0) 
        y_obs_seq = x_true_windows[1:, obs_indices] + noise   # (num_windows, m)
        obs_mask  = jnp.ones(num_windows, dtype=bool)
        # ─────────────────────────────────────────────────────────────────────

        # ── Initial EKF state: perturbed IC ───────────────────────────────────
        x0_hat = u_current_true + P0_sigma * jax.random.normal(
            jax.random.PRNGKey(ic_idx + 100), shape=(N,)
        )
        ekf_state_init = EKFState(x_hat=x0_hat, P=P0)
        # ─────────────────────────────────────────────────────────────────────

        # ── Run EKF ───────────────────────────────────────────────────────────
        x_hats, Ps = run_ekf_smoother(
            predict_fn, update_fn,
            x0_hat, P0,
            y_obs_seq, obs_mask,
            H, Q, R
        )
        # x_hats shape: (num_windows, N)
        # ─────────────────────────────────────────────────────────────────────

        # ── Metrics ───────────────────────────────────────────────────────────
        x_true_compare = x_true_windows[1:]           # (num_windows, N)
        l2_open_loop   = jnp.linalg.norm(
            model.x_pred_fn(params, u_current_true, t_star_window)[-1] - x_true_compare[-1]
        ) / jnp.linalg.norm(x_true_compare[-1])

        l2_ekf = jnp.linalg.norm(x_hats - x_true_compare) / jnp.linalg.norm(x_true_compare)
        print(f"IC {ic_idx} | Open-loop L2: {l2_open_loop:.3e} | EKF L2: {l2_ekf:.3e}")

        # ── Plot comparison ───────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        t_ax = t_eval_full[1:]

        for ax, data, title in zip(
            axes,
            [x_true_compare, x_hats, jnp.abs(x_true_compare - x_hats)],
            ["Ground Truth", "EKF Estimate", "Absolute Error"]
        ):
            im = ax.pcolormesh(np.arange(N), t_ax, np.array(data),
                               cmap="viridis" if "Error" not in title else "magma",
                               shading="auto")
            ax.set_title(f"{title} (IC {ic_idx})", fontsize=13)
            ax.set_xlabel("Variable index"); ax.set_ylabel("Time (t)")
            fig.colorbar(im, ax=ax)

        fig.tight_layout()
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"ekf_ic_{ic_idx}.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info(f"EKF plot for IC {ic_idx} saved.")