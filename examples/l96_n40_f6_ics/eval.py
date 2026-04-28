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

def evaluate_with_ekf(config: ml_collections.ConfigDict, workdir: str):
    """
    Evaluates the trained UDON model with EKF data assimilation.

    Scenario:
      - obs_every_n: observe only 1-in-N variables at each assimilation step.
      - assimilation_dt: time gap between observations (= one prediction window).
      - Measurement noise std: sigma_obs.
    """
    from kf import EKFState, run_ekf_smoother

    obs_interval = config.ekf.get("obs_interval", 1)    # 1 = every t*, 2 = every 2t*, etc.
    dynamic_vars = config.ekf.get("dynamic_vars", False)
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

    num_windows = config.training.num_time_windows
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

    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- EKF Evaluation for IC {ic_idx} ---")

        u_current_true = u0_ref_all[ic_idx, :]  # ground truth IC

        # ── Generate noisy observations by forward-simulating reference ───────
        def lorenz_96(t, state, F=6.0):
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

        # 2. Prepare Observation Sequences
        H_list = []
        y_obs_list = []
        obs_mask = []
        obs_coords = [] # To store (time, var_idx) for plotting crosses

        ## Add noise to create synthetic observations
        key = jax.random.PRNGKey(ic_idx)
        # ─────────────────────────────────────────────────────────────────────

        x_true_at_boundaries = x_true_windows[1:] # (num_windows, N)
        for t_idx in range(num_windows):
            is_obs_time = (t_idx + 1) % obs_interval == 0
            obs_mask.append(is_obs_time)
            
            if is_obs_time:
                # Determine which variables to observe
                if dynamic_vars:
                    # Randomly pick m variables each time
                    m = N // obs_every_n
                    key, subkey = jax.random.split(key)
                    obs_indices = jax.random.choice(subkey, N, shape=(m,), replace=False)
                else:
                    obs_indices = jnp.arange(0, N, obs_every_n)
                
                m = len(obs_indices)
                H_t = jnp.zeros((m, N)).at[jnp.arange(m), obs_indices].set(1.0)
                
                # Create noisy observation
                key, subkey = jax.random.split(key)
                noise = sigma_obs * jax.random.normal(subkey, shape=(m,))
                y_t = x_true_at_boundaries[t_idx, obs_indices] + noise
                
                H_list.append(H_t)
                y_obs_list.append(y_t)
                
                # Store coordinates for the cross markers
                current_time = (t_idx + 1) * dt
                for idx in obs_indices:
                    obs_coords.append((idx, current_time))
            else:
                # Padding for JAX consistency (will be ignored by mask)
                m_fixed = N // obs_every_n
                H_list.append(jnp.zeros((m_fixed, N)))
                y_obs_list.append(jnp.zeros((m_fixed,)))

        H_seq = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
        obs_mask = jnp.array(obs_mask)
        
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
            H_seq, Q, R
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
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        t_ax = np.arange(1, num_windows + 1) * dt
        var_ax = np.arange(N)

        data_list = [x_true_at_boundaries, x_hats, jnp.abs(x_true_at_boundaries - x_hats)]
        titles = ["Ground Truth", "EKF Estimate", "Absolute Error"]
        cmaps = ["viridis", "viridis", "magma"]

        for i, (ax, data, title, cmap) in enumerate(zip(axes, data_list, titles, cmaps)):
            im = ax.pcolormesh(var_ax, t_ax, np.array(data), cmap=cmap, shading='auto')
            ax.set_title(title)
            ax.set_xlabel("Variables")
            fig.colorbar(im, ax=ax)
            
            # On the third plot (Error), mark observations with crosses
            if i == 2 and len(obs_coords) > 0:
                obs_vars, obs_times = zip(*obs_coords)
                ax.scatter(obs_vars, obs_times, marker='x', color='cyan', 
                           s=20, linewidths=0.5, label='Observations')
                ax.legend(loc='upper right', fontsize='small')

        axes[0].set_ylabel("Time (t)")
        plt.tight_layout()

        # Save logic
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"ekf_ic_{ic_idx}.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info(f"EKF plot for IC {ic_idx} saved.")

def evaluate_with_enkf(config: ml_collections.ConfigDict, workdir: str):
    """
    Evaluates the trained UDON model with EnKF data assimilation.
 
    The EnKF replaces the EKF's linearised covariance propagation with an
    ensemble of N_ens surrogate forward passes, capturing the non-Gaussian
    error structure of L96 (F=6) without computing any Jacobian.
 
    Config keys read from config.ekf (same as evaluate_with_ekf):
        obs_interval  int   — assimilate every N windows (default 1)
        dynamic_vars  bool  — randomly rotate observed variables (default False)
        obs_every_n   int   — observe every n-th variable (default 4)
        sigma_obs     float — observation noise std (default 0.5)
        sigma_proc    float — process noise std (default 0.1)
        P0_sigma      float — initial ensemble spread std (default 1.0)
 
    Additional config key:
        N_ens         int   — ensemble size (default 50)
 
    Outputs (per IC index):
        enkf_ic_{ic_idx}.pdf — 4-panel heatmap:
            [Ground Truth | EnKF Mean | Absolute Error | Ensemble Std]
    """
    from kf import EnKFState, EnKFState, run_enkf_smoother, init_ensemble
 
    # ── Config ────────────────────────────────────────────────────────────────
    obs_interval  = config.ekf.get("obs_interval",  1)
    dynamic_vars  = config.ekf.get("dynamic_vars",  False)
    obs_every_n   = config.ekf.get("obs_every_n",   4)
    sigma_obs     = config.ekf.get("sigma_obs",      0.5)
    sigma_proc    = config.ekf.get("sigma_proc",     0.1)
    P0_sigma      = config.ekf.get("P0_sigma",       1.0)
    N_ens         = config.ekf.get("N_ens",          50)
 
    # ── Model & checkpoint ────────────────────────────────────────────────────
    x_ref_all, u0_ref_all, t_star_window = get_dataset()
 
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
 
    N  = model.N
    dt = float(t_star_window[-1] - t_star_window[0])   # assimilation window
 
    # ── Build JIT-compiled EnKF functions ─────────────────────────────────────
    # make_enkf_fns vmaps propagator_fn over N_ens members; no jacfwd called.
    predict_fn, update_fn = model.make_enkf_fns(params, dt, N_ens=N_ens)
 
    num_windows = config.training.num_time_windows
 
    # ── Fixed noise covariances ───────────────────────────────────────────────
    Q  = jnp.eye(N) * sigma_proc ** 2
    R_fixed = jnp.eye(N // obs_every_n) * sigma_obs ** 2   # resized per step if dynamic
    P0 = jnp.eye(N) * P0_sigma ** 2
 
    # ── Per-IC evaluation loop ────────────────────────────────────────────────
    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- EnKF Evaluation for IC {ic_idx} (N_ens={N_ens}) ---")
 
        u_current_true = u0_ref_all[ic_idx, :]   # (N,) ground-truth IC
 
        # ── 1. Forward-simulate ground truth at window boundaries ─────────────
        def lorenz_96(t, state, F=6.0):
            xp1 = np.roll(state, -1)
            xm1 = np.roll(state,  1)
            xm2 = np.roll(state,  2)
            return (xp1 - xm2) * xm1 - state + F
 
        t_eval_full = np.array([i * dt for i in range(num_windows + 1)])
        sol = solve_ivp(
            lorenz_96,
            t_span=[t_eval_full[0], t_eval_full[-1]],
            y0=np.array(u_current_true),
            t_eval=t_eval_full,
            rtol=1e-9, atol=1e-11,
        )
        # x_true_windows: (num_windows+1, N) — states at each window boundary
        x_true_windows = jnp.array(sol.y.T)
 
        # ── 2. Build observation sequence ─────────────────────────────────────
        H_list:      list[jnp.ndarray] = []
        y_obs_list:  list[jnp.ndarray] = []
        obs_mask:    list[bool]        = []
        obs_coords:  list[tuple]       = []   # (var_idx, time) for scatter markers
 
        key = jax.random.PRNGKey(ic_idx)
 
        # Observations correspond to the state at each window *end* (index 1..T)
        x_true_at_boundaries = x_true_windows[1:]   # (num_windows, N)
 
        for t_idx in range(num_windows):
            is_obs_time = (t_idx + 1) % obs_interval == 0
            obs_mask.append(is_obs_time)
 
            if is_obs_time:
                # Determine which variables to observe this step
                if dynamic_vars:
                    m_t = N // obs_every_n
                    key, subkey = jax.random.split(key)
                    obs_indices = jax.random.choice(subkey, N, shape=(m_t,), replace=False)
                else:
                    obs_indices = jnp.arange(0, N, obs_every_n)
 
                m_t = len(obs_indices)
                H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_indices].set(1.0)
                R_t = jnp.eye(m_t) * sigma_obs ** 2
 
                # Noisy observation
                key, subkey = jax.random.split(key)
                noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
                y_t   = x_true_at_boundaries[t_idx, obs_indices] + noise
 
                H_list.append(H_t)
                y_obs_list.append(y_t)
 
                current_time = (t_idx + 1) * dt
                for idx in obs_indices:
                    obs_coords.append((int(idx), current_time))
            else:
                # Padding rows (ignored by obs_mask, needed for stack)
                m_fixed = N // obs_every_n
                H_list.append(jnp.zeros((m_fixed, N)))
                y_obs_list.append(jnp.zeros((m_fixed,)))
 
        H_seq     = jnp.stack(H_list)       # (T, m, N)
        y_obs_seq = jnp.stack(y_obs_list)   # (T, m)
        obs_mask  = jnp.array(obs_mask)     # (T,)
 
        # ── 3. Initialise ensemble from prior N(x0_hat, P0) ──────────────────
        # The IC is perturbed to simulate imperfect knowledge of the true state.
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)   # (N_ens, N)
 
        # ── 4. Run EnKF smoother ──────────────────────────────────────────────
        # Returns ensemble mean and per-variable std at every assimilation step.
        x_means, x_spreads = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq, obs_mask,
            H_seq, Q,
            jnp.eye(N // obs_every_n) * sigma_obs ** 2,   # R (fixed-size for smoother)
            key,
        )
        # x_means:   (num_windows, N)
        # x_spreads: (num_windows, N)
 
        # ── 5. Metrics ────────────────────────────────────────────────────────
        x_true_compare = x_true_windows[1:]   # (num_windows, N)
 
        # Open-loop: single DeepONet rollout from the *true* IC, no assimilation
        x_open_loop_end = model.x_pred_fn(params, u_current_true, t_star_window)[-1]
        l2_open_loop    = (
            jnp.linalg.norm(x_open_loop_end - x_true_compare[-1])
            / jnp.linalg.norm(x_true_compare[-1])
        )
        l2_enkf = (
            jnp.linalg.norm(x_means - x_true_compare)
            / jnp.linalg.norm(x_true_compare)
        )
        mean_spread = float(jnp.mean(x_spreads))
 
        print(
            f"IC {ic_idx} | Open-loop L2: {l2_open_loop:.3e} "
            f"| EnKF L2: {l2_enkf:.3e} "
            f"| Mean ensemble σ: {mean_spread:.3e} "
            f"| N_ens: {N_ens}"
        )
 
        # ── 6. Four-panel heatmap ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), sharey=True)
        t_ax   = np.arange(1, num_windows + 1) * dt   # time axis
        var_ax = np.arange(N)                          # variable axis
 
        # Data for each panel
        panels = [
            (x_true_at_boundaries, "Ground Truth",      "viridis", False),
            (x_means,              "EnKF Mean",          "viridis", False),
            (jnp.abs(x_true_at_boundaries - x_means),
                                   "Absolute Error",     "magma",   True),
            (x_spreads,            "Ensemble Std  σ(t)", "plasma",  True),
        ]
 
        for ax, (data, title, cmap, show_obs) in zip(axes, panels):
            im = ax.pcolormesh(var_ax, t_ax, np.array(data), cmap=cmap, shading="auto")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Variables (0 to 39)", fontsize=11)
            fig.colorbar(im, ax=ax, pad=0.02)
 
            # Mark assimilation times as horizontal dashed lines
            for t_idx, is_obs in enumerate(obs_mask):
                if is_obs:
                    ax.axhline(
                        y=(t_idx + 1) * dt,
                        color="white", linestyle="--", linewidth=0.6, alpha=0.4,
                    )
 
            # On Error and Std panels, scatter the observation locations
            if show_obs and len(obs_coords) > 0:
                obs_vars, obs_times = zip(*obs_coords)
                ax.scatter(
                    obs_vars, obs_times,
                    marker="x", color="cyan", s=18, linewidths=0.7,
                    label="Observations",
                )
                if show_obs:
                    ax.legend(loc="upper right", fontsize="small")
 
        axes[0].set_ylabel("Time (t)", fontsize=11)
 
        # Shared title with key hyperparameters
        fig.suptitle(
            f"EnKF Assimilation — IC {ic_idx}  "
            f"(N_ens={N_ens}, obs every {obs_every_n}th var, "
            f"σ_obs={sigma_obs}, σ_proc={sigma_proc})",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
 
        # ── 7. Save ───────────────────────────────────────────────────────────
        save_dir = os.path.join(workdir, "figures", config.wandb.name)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"enkf_ic_{ic_idx}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info(f"EnKF plot for IC {ic_idx} saved to: {out_path}")