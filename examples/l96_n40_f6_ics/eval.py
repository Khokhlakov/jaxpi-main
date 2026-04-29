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
from scipy.io import loadmat


def _load_l2_eval_pool(
    mat_path:      str,
    max_additions: int,
    num_vars:      int,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    Load the pre-computed rollout pool from a .mat file, mirroring the logic
    of _load_rollout_pool used during training.
 
    Args:
        mat_path:      Path to the .mat file (e.g. "data/train_rollouts_025.mat").
        max_additions: Number of rollout slots to read (= config.training.max_additions).
        num_vars:      State dimension N (40 for L96).
 
    Returns:
        u0_original  : (B, N) initial conditions — one row per trajectory.
        rollout_states: list of max_additions arrays, each (B, N).
                        rollout_states[k] is the ground-truth state after
                        k+1 windows (i.e. the key "u0_rollout_{k+1}").
    """
    data = loadmat(mat_path)
 
    u0_original = jnp.array(data["u0_original"].astype(np.float32))  # (B, N)
 
    rollout_states: list[jnp.ndarray] = []
    for k in range(1, max_additions + 1):
        key_name = f"u0_rollout_{k}"
        if key_name not in data:
            raise KeyError(
                f"Key '{key_name}' not found in {mat_path}. "
                f"Regenerate the file with max_additions >= {k}."
            )
        rollout_states.append(jnp.array(data[key_name].astype(np.float32)))
 
    return u0_original, rollout_states
 
 
def _plot_l2_per_window(
    curves:    dict[str, np.ndarray],   # label → (num_windows,) mean L2 array
    dt:        float,                   # window duration (for x-axis labels)
    title:     str,
    save_path: str,
    colors:    dict[str, str] | None = None,
) -> None:
    """
    Plot one or more average-L2-per-window curves on a log-scale y-axis and
    save the figure as a PDF.
 
    Args:
        curves:    Mapping from method label to a 1-D array of length
                   num_windows containing the mean L2 at each window boundary.
        dt:        Assimilation window duration used to label the x-axis.
        title:     Figure suptitle.
        save_path: Full path (including .pdf extension) for the output file.
        colors:    Optional mapping from label to matplotlib colour string.
                   Defaults are applied for unlabelled entries.
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for i, (label, l2_arr) in enumerate(curves.items()):
        num_windows = len(l2_arr)
        window_idx  = np.arange(1, num_windows + 1)
        color       = (colors or {}).get(label, default_colors[i % len(default_colors)])
        ax.plot(window_idx, l2_arr, marker="o", markersize=4,
                linewidth=1.8, label=label, color=color)
 
    ax.set_yscale("log")
    ax.set_xlabel("Window index  (each = {:.3g} time units)".format(dt), fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Batch L2-per-window plot saved to: {save_path}")


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

    _evaluate_batch_l2_openloop(model, params, t_star_window, config, workdir)

def _evaluate_batch_l2_openloop(model, params, t_star_window, config, workdir):
    """
    Compute and plot the batch-averaged open-loop L2 error per window.
 
    Called at the end of evaluate().  Autoregressively rolls out the DeepONet
    from each IC in the pool, comparing the end-of-window prediction to the
    ground-truth state stored in the .mat file.
 
    Strategy
    --------
    For window k (1-indexed):
        1. Start from u0_original.
        2. Run k autoregressive steps through x_pred_fn.
        3. Compare the final state to u0_rollout_k.
        4. Average the relative L2 norm over all ICs in the batch.
 
    The innermost prediction is vmapped over the batch dimension so that all
    ICs are evaluated in a single JIT-compiled call per window.
    """
    dt_window    = float(t_star_window[-1] - t_star_window[0])
    max_additions = config.training.get("max_additions", 5)
    num_vars      = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
 
    logging.info("Computing batch L2 per window (open-loop) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, num_vars)
    B = u0_original.shape[0]
 
    # JIT-compiled, vmapped single-window predictor:  (B, N) → (B, N)
    # x_pred_fn is already vmapped over t; we now also vmap over the IC axis.
    predict_one_window = jax.jit(
        jax.vmap(
            lambda u: model.x_pred_fn(params, u, t_star_window)[-1],
            in_axes=0,
        )
    )
 
    l2_per_window: list[float] = []
    u_current = u0_original  # (B, N)
 
    for k in range(max_additions):
        # Advance every IC by one more window
        u_current = predict_one_window(u_current)        # (B, N)
 
        # Ground truth at window boundary k+1
        x_ref_k = rollout_states[k]                      # (B, N)
 
        # Per-IC relative L2, then batch mean
        numer   = jnp.linalg.norm(u_current - x_ref_k, axis=1)  # (B,)
        denom   = jnp.linalg.norm(x_ref_k,              axis=1)  # (B,)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
 
        logging.info(f"  Window {k+1:>3d} | mean L2: {l2_mean:.3e}")
 
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_openloop.pdf")
    _plot_l2_per_window(
        curves    = {"Open-loop (DeepONet)": np.array(l2_per_window)},
        dt        = dt_window,
        title     = f"Open-loop: batch-average L2 per window  (B={B})",
        save_path = save_path,
        colors    = {"Open-loop (DeepONet)": "#2196F3"},
    )


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
            is_obs_time = (t_idx + 1)*dt % obs_interval == 0
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

    _evaluate_batch_l2_ekf(
         model, params, t_star_window,
         predict_fn, update_fn,
         Q, R, P0,
         obs_every_n, sigma_obs, P0_sigma,
         dynamic_vars, obs_interval,
         config, workdir,
    )
    
def _evaluate_batch_l2_ekf(
    model, params, t_star_window,
    predict_fn, update_fn,
    Q, R, P0,
    obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars, obs_interval,
    config, workdir,
):
    """
    Compute and plot the batch-averaged L2 error per window for both the
    open-loop DeepONet rollout and the EKF estimate.
 
    Called at the end of evaluate_with_ekf().
 
    For each IC in the pool:
      • Open-loop:  autoregressively roll out the model for k windows.
      • EKF:        run run_ekf_smoother up to window k, take the final
                    filtered estimate.
    Both errors are averaged over all ICs and plotted together.
 
    Notes
    -----
    Running the full EKF smoother for every IC in the pool can be expensive.
    The batch size used here is capped at `ekf_batch_size` (default 200) ICs
    to keep wall-clock time manageable; this can be overridden via
    config.ekf.get("batch_l2_size", 200).
    """
    from kf import run_ekf_smoother, EKFState
 
    dt_window    = float(t_star_window[-1] - t_star_window[0])
    max_additions = config.training.get("max_additions", 5)
    num_vars      = model.N
    N             = num_vars
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    ekf_batch_size = config.ekf.get("batch_l2_size", 200)
 
    logging.info("Computing batch L2 per window (open-loop vs EKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, num_vars)
 
    # Cap batch size
    B = min(u0_original.shape[0], ekf_batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  Using {B} ICs from pool for batch L2 evaluation.")
 
    # Open-loop: vmapped single-window predictor
    predict_one_window = jax.jit(
        jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window)[-1], in_axes=0)
    )
 
    # Observation operator (fixed, as in evaluate_with_ekf)
    obs_indices = jnp.arange(0, N, obs_every_n)
    m           = len(obs_indices)
 
    # Accumulators: per-window mean L2 summed over ICs, then divided by B
    ol_l2_sum  = np.zeros(max_additions)   # open-loop
    ekf_l2_sum = np.zeros(max_additions)   # EKF
 
    u_ol = u0_original   # running open-loop state (B, N)
 
    for ic in range(B):
        key = jax.random.PRNGKey(ic + 9999)   # distinct from per-IC eval keys
 
        u_true_ic = u0_original[ic]   # (N,)
 
        # ── Synthesise per-IC observations at each window boundary ──────────
        H_list, y_obs_list, obs_mask_list = [], [], []
 
        for t_idx in range(max_additions):
            x_true_t = rollout_states[t_idx][ic]              # (N,) ground truth
            is_obs   = (t_idx + 1) % obs_interval == 0
            obs_mask_list.append(is_obs)
 
            if is_obs:
                if dynamic_vars:
                    key, subkey = jax.random.split(key)
                    obs_idx = jax.random.choice(subkey, N, shape=(m,), replace=False)
                else:
                    obs_idx = obs_indices
 
                m_t = len(obs_idx)
                H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx].set(1.0)
                key, subkey = jax.random.split(key)
                noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
                y_t   = x_true_t[obs_idx] + noise
 
                H_list.append(H_t)
                y_obs_list.append(y_t)
            else:
                H_list.append(jnp.zeros((m, N)))
                y_obs_list.append(jnp.zeros((m,)))
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
        obs_mask  = jnp.array(obs_mask_list)
 
        # Perturbed IC
        key, key_ic = jax.random.split(key)
        x0_hat = u_true_ic + P0_sigma * jax.random.normal(key_ic, shape=(N,))
 
        # Run EKF smoother — returns (max_additions, N)
        x_hats, _ = run_ekf_smoother(
            predict_fn, update_fn,
            x0_hat, P0,
            y_obs_seq, obs_mask,
            H_seq, Q, R,
        )
 
        # Accumulate per-window L2 for this IC
        for k in range(max_additions):
            ref_k = rollout_states[k][ic]   # (N,)
 
            ekf_err = float(jnp.linalg.norm(x_hats[k] - ref_k)
                            / (jnp.linalg.norm(ref_k) + 1e-12))
            ekf_l2_sum[k] += ekf_err
 
    # Open-loop: much cheaper — run all ICs at once
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_window(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k,              axis=1)
        ol_l2_sum[k] = float(jnp.mean(numer / (denom + 1e-12)))
 
    l2_openloop = ol_l2_sum                  # already mean (computed per-window)
    l2_ekf      = ekf_l2_sum / B            # average over ICs
 
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_ekf.pdf")
    _plot_l2_per_window(
        curves={
            "Open-loop (DeepONet)": l2_openloop,
            "EKF estimate":         l2_ekf,
        },
        dt        = dt_window,
        title     = f"EKF vs open-loop: batch-average L2 per window  (B={B})",
        save_path = save_path,
        colors    = {"Open-loop (DeepONet)": "#2196F3", "EKF estimate": "#FF5722"},
    )


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
    P0_sigma      = config.ekf.get("P0_sigma",       1.0)
    
    N_ens         = config.enkf.get("N_ens",          50)
    sigma_model   = config.enkf.get("sigma_model",     0.1)
 
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
    Q  = jnp.eye(N) * sigma_model ** 2
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
            is_obs_time = (t_idx + 1)*dt % obs_interval == 0
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
            f"σ_obs={sigma_obs}, σ_model={sigma_model})",
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
    
    _evaluate_batch_l2_enkf(
         model, params, t_star_window,
         predict_fn, update_fn,
         Q, P0,
         N_ens, obs_every_n, sigma_obs, P0_sigma,
         dynamic_vars, obs_interval,
         config, workdir,
    )

def _evaluate_batch_l2_enkf(
    model, params, t_star_window,
    predict_fn, update_fn,
    Q, P0,
    N_ens, obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars, obs_interval,
    config, workdir,
):
    """
    Compute and plot the batch-averaged L2 error per window for the open-loop
    DeepONet rollout, the EnKF ensemble mean, and (optionally) the mean
    ensemble spread as a proxy for uncertainty calibration.
 
    Called at the end of evaluate_with_enkf().
 
    For each IC in the pool the EnKF smoother is run with synthetic noisy
    observations constructed from the ground-truth rollout states. The
    ensemble spread (mean σ per window) is collected alongside the L2 error,
    providing a visual check that the filter is well-calibrated: a well-tuned
    EnKF should have spread ≈ RMSE.
 
    Notes
    -----
    Same cost caveat as _evaluate_batch_l2_ekf: capped at
    config.ekf.get("batch_l2_size", 200) ICs for filter evaluation.
    """
    from kf import run_enkf_smoother, init_ensemble, EnKFState
 
    dt_window     = float(t_star_window[-1] - t_star_window[0])
    max_additions = config.training.get("max_additions", 5)
    N             = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    enkf_batch_size = config.ekf.get("batch_l2_size", 200)
 
    logging.info("Computing batch L2 per window (open-loop vs EnKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)
 
    B = min(u0_original.shape[0], enkf_batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  Using {B} ICs from pool for batch L2 evaluation (N_ens={N_ens}).")
 
    obs_indices = jnp.arange(0, N, obs_every_n)
    m           = len(obs_indices)
    R_fixed     = jnp.eye(m) * sigma_obs ** 2
 
    # Open-loop vmapped predictor
    predict_one_window = jax.jit(
        jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window)[-1], in_axes=0)
    )
 
    enkf_l2_sum     = np.zeros(max_additions)
    enkf_spread_sum = np.zeros(max_additions)
 
    for ic in range(B):
        key      = jax.random.PRNGKey(ic + 77777)
        u_true   = u0_original[ic]   # (N,)
 
        # ── Build per-IC synthetic observation sequence ──────────────────────
        H_list, y_obs_list, obs_mask_list = [], [], []
 
        for t_idx in range(max_additions):
            x_true_t = rollout_states[t_idx][ic]
            is_obs   = (t_idx + 1) % obs_interval == 0
            obs_mask_list.append(is_obs)
 
            if is_obs:
                if dynamic_vars:
                    key, subkey = jax.random.split(key)
                    obs_idx = jax.random.choice(subkey, N, shape=(m,), replace=False)
                else:
                    obs_idx = obs_indices
 
                m_t = len(obs_idx)
                H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx].set(1.0)
                key, subkey = jax.random.split(key)
                noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
                y_t   = x_true_t[obs_idx] + noise
 
                H_list.append(H_t)
                y_obs_list.append(y_t)
            else:
                H_list.append(jnp.zeros((m, N)))
                y_obs_list.append(jnp.zeros((m,)))
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
        obs_mask  = jnp.array(obs_mask_list)
 
        # ── Initialise ensemble ──────────────────────────────────────────────
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
 
        # ── Run EnKF smoother — returns (max_additions, N) each ─────────────
        x_means, x_spreads = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq, obs_mask,
            H_seq, Q, R_fixed,
            key,
        )
 
        # ── Accumulate ───────────────────────────────────────────────────────
        for k in range(max_additions):
            ref_k = rollout_states[k][ic]
 
            enkf_l2_sum[k]     += float(
                jnp.linalg.norm(x_means[k] - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
            enkf_spread_sum[k] += float(jnp.mean(x_spreads[k]))
 
    # Open-loop — vectorised over B
    ol_l2 = np.zeros(max_additions)
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_window(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k,              axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))
 
    l2_enkf    = enkf_l2_sum     / B
    spread_mean = enkf_spread_sum / B
 
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_enkf.pdf")
 
    # ── Two-panel figure: L2 error (log) + ensemble spread (log) ────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window_idx = np.arange(1, max_additions + 1)
 
    # Left: L2 error curves
    ax = axes[0]
    ax.plot(window_idx, ol_l2,   marker="o", markersize=4, linewidth=1.8,
            label="Open-loop (DeepONet)", color="#2196F3")
    ax.plot(window_idx, l2_enkf, marker="s", markersize=4, linewidth=1.8,
            label=f"EnKF mean (N_ens={N_ens})", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel(f"Window index  (each = {dt_window:.3g} time units)", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop: L2 per window", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # Right: ensemble spread — a calibration diagnostic
    # Well-calibrated filter: spread ≈ RMSE across the batch.
    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean, marker="^", markersize=4, linewidth=1.8,
             label="Mean ensemble σ", color="#4CAF50")
    ax2.plot(window_idx, l2_enkf,     marker="s", markersize=4, linewidth=1.8,
             linestyle="--", label="EnKF mean L2  (RMSE proxy)", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Window index  (each = {dt_window:.3g} time units)", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs L2 error", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    fig.suptitle(
        f"EnKF batch evaluation  (B={B}, N_ens={N_ens}, "
        f"obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"EnKF batch L2-per-window plot saved to: {save_path}")