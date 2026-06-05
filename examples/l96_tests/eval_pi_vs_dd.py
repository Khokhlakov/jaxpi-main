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
import examples.l96_tests.models as models
from examples.l96_tests.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps
from utils import dd_get_test_data_rollout, build_obs_schedule, scale_Q_for_fine_steps

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat


def _load_l2_eval_pool(
    mat_path:      str,
    max_additions: int,
    num_vars:      int,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    Load the pre-computed rollout pool from a .mat file.
    """
    data = loadmat(mat_path)
    u0_original = jnp.array(data["u0_original"].astype(np.float32))
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


def _plot_l2_comparison(
    curves:    dict[str, np.ndarray],
    dt:        float,
    title:     str,
    save_path: str,
    colors:    dict[str, str] | None = None,
) -> None:
    """
    Plot multiple L2-per-window curves on a shared log-scale y-axis.
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, l2_arr) in enumerate(curves.items()):
        num_windows = len(l2_arr)
        window_idx  = np.arange(1, num_windows + 1)
        color       = (colors or {}).get(label, default_colors[i % len(default_colors)])
        ax.plot(window_idx, l2_arr, marker="o", markersize=5,
                linewidth=2.0, label=label, color=color)

    num_windows = len(next(iter(curves.values())))
    window_idx  = np.arange(1, num_windows + 1)

    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

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
    logging.info(f"Comparison L2-per-window plot saved to: {save_path}")


def _plot_erf_comparison(
    obs_times:  np.ndarray,
    erf_data:   dict[str, tuple[np.ndarray, np.ndarray]],
    n_traj:     int,
    title:      str,
    save_path:  str,
    colors:     dict[str, str] | None = None,
) -> None:
    """
    Plot multiple ERF curves (mean ± std) on the same figure.
    erf_data: {label: (erf_mean, erf_std)}
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, (erf_mean, erf_std)) in enumerate(erf_data.items()):
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])
        ax.plot(obs_times, erf_mean,
                color=color, linewidth=2.0, marker="o", markersize=4,
                label=label)
        ax.fill_between(
            obs_times,
            erf_mean - erf_std,
            erf_mean + erf_std,
            color=color, alpha=0.15, linewidth=0,
        )

    ax.set_yscale("log")
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
    logging.info(f"ERF comparison plot saved to: {save_path}")


def _plot_rmse_comparison(
    obs_times:       np.ndarray,
    rmse_data:       dict[str, dict],
    sigma_obs:       float,
    n_traj:          int,
    title:           str,
    save_path:       str,
    colors:          dict[str, str] | None = None,
) -> None:
    """
    Plot prior and posterior RMSE for multiple models.
    rmse_data: {label: {'prior_mean': ..., 'prior_std': ..., 'post_mean': ..., 'post_std': ...}}
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, rmse_dict) in enumerate(rmse_data.items()):
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])

        ax.plot(obs_times, rmse_dict['post_mean'],
                color=color, linewidth=2.0, marker="s", markersize=4,
                label=f"{label} (posterior)")
        ax.fill_between(
            obs_times,
            rmse_dict['post_mean'] - rmse_dict['post_std'],
            rmse_dict['post_mean'] + rmse_dict['post_std'],
            color=color, alpha=0.15, linewidth=0,
        )

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


def _evaluate_batch_l2_enkf(
    model, params, t_star_window,
    predict_fn, update_fn,
    Q_fine, P0,
    N_ens, obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars,
    dt_fine: float,
    dt_obs:  float,
    config,
    model_name: str,
):
    """
    Compute batch L2, ERF, and RMSE metrics for EnKF with a single model.
    Returns dictionaries of results.
    """
    from examples.KS.kf import run_enkf_smoother, init_ensemble, EnKFState

    specify_obs_idx   = config.kf.get("specify_obs_idx", False)
    obs_idx_list      = config.kf.get("obs_idx_list", None)

    dt_window     = config.get("dt_window", 0.25)
    max_additions = config.training.get("max_additions", 5)
    N             = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    enkf_batch_size = config.ekf.get("batch_l2_size", 200)

    logging.info(f"Computing batch L2/ERF/RMSE for {model_name} …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)

    B = min(u0_original.shape[0], enkf_batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  {model_name}: Using {B} ICs for batch evaluation (N_ens={N_ens}).")

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m           = len(obs_indices)
    R_fixed     = jnp.eye(m) * sigma_obs ** 2

    predict_one_window = jax.jit(
        jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window)[-1], in_axes=0)
    )

    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time = total_time_batch,
        dt_fine    = dt_fine,
        dt_obs     = dt_obs,
    )
    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(max_additions)
    ])

    enkf_l2_sum     = np.zeros(max_additions)
    enkf_spread_sum = np.zeros(max_additions)
    enkf_rmse_sum   = np.zeros(max_additions)

    erf_sum    = np.zeros(T_obs)
    erf_sq_sum = np.zeros(T_obs)

    prior_rmse_sum    = np.zeros(T_obs)
    prior_rmse_sq_sum = np.zeros(T_obs)
    post_rmse_sum     = np.zeros(T_obs)
    post_rmse_sq_sum  = np.zeros(T_obs)

    for ic in range(B):
        key    = jax.random.PRNGKey(ic + 77777)
        u_true = u0_original[ic]

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
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

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

        post_means_at_obs = x_means[obs_step_indices_batch]

        prior_rmse = np.sqrt(np.mean(
            (np.array(prior_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))
        post_rmse  = np.sqrt(np.mean(
            (np.array(post_means_at_obs)  - x_true_at_obs) ** 2, axis=1
        ))

        erf_ic = prior_rmse / (post_rmse + 1e-12)
        erf_sum    += erf_ic
        erf_sq_sum += erf_ic ** 2

        prior_rmse_sum    += prior_rmse
        prior_rmse_sq_sum += prior_rmse ** 2
        post_rmse_sum     += post_rmse
        post_rmse_sq_sum  += post_rmse ** 2

        for k in range(max_additions):
            ref_k      = rollout_states[k][ic]
            step_k     = window_step_indices[k]
            x_hat_k    = x_means[step_k]

            enkf_l2_sum[k] += float(
                jnp.linalg.norm(x_hat_k - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
            enkf_rmse_sum[k] += float(jnp.sqrt(jnp.mean((x_hat_k - ref_k) ** 2)))
            enkf_spread_sum[k] += float(jnp.sqrt(jnp.mean(x_spreads[step_k] ** 2)))

    # Open-loop baseline
    ol_l2     = np.zeros(max_additions)
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_window(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k, axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))

    l2_enkf     = enkf_l2_sum     / B
    rmse_enkf   = enkf_rmse_sum   / B
    spread_mean = enkf_spread_sum / B

    erf_mean = erf_sum    / B
    erf_std  = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))

    prior_rmse_mean = prior_rmse_sum    / B
    prior_rmse_std  = np.sqrt(np.maximum(
        prior_rmse_sq_sum / B - prior_rmse_mean ** 2, 0.0))
    post_rmse_mean  = post_rmse_sum     / B
    post_rmse_std   = np.sqrt(np.maximum(
        post_rmse_sq_sum  / B - post_rmse_mean  ** 2, 0.0))

    results = {
        'ol_l2': ol_l2,
        'enkf_l2': l2_enkf,
        'enkf_rmse': rmse_enkf,
        'spread_mean': spread_mean,
        'erf_mean': erf_mean,
        'erf_std': erf_std,
        'prior_rmse_mean': prior_rmse_mean,
        'prior_rmse_std': prior_rmse_std,
        'post_rmse_mean': post_rmse_mean,
        'post_rmse_std': post_rmse_std,
        'obs_times': obs_times_batch,
        'B': B,
    }

    return results


def _evaluate_batch_l2_openloop(model, params, t_star_window, config, model_name):
    """
    Compute batch-averaged open-loop L2 error per window for a single model.
    Returns l2_per_window array.
    """
    dt_window    = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    num_vars      = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )

    logging.info(f"Computing batch L2 per window (open-loop) for {model_name} …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, num_vars)
    B = u0_original.shape[0]

    predict_one_window = jax.jit(
        jax.vmap(
            lambda u: model.x_pred_fn(params, u, t_star_window)[-1],
            in_axes=0,
        )
    )

    l2_per_window: list[float] = []
    u_current = u0_original

    for k in range(max_additions):
        u_current = predict_one_window(u_current)
        x_ref_k = rollout_states[k]
        numer   = jnp.linalg.norm(u_current - x_ref_k, axis=1)
        denom   = jnp.linalg.norm(x_ref_k, axis=1)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
        logging.info(f"  {model_name} Window {k+1:>3d} | mean L2: {l2_mean:.3e}")

    return np.array(l2_per_window), B


def _evaluate_batch_l2_openloop_dataset2(model, params, t_star_window, test_data_rollout, model_name):
    """
    Compute batch-averaged open-loop L2 error per window for a single model
    using Dataset 2 (rollout tensor). Returns l2_per_window array.
    """
    # Unpack dimensions: (B, num_windows, time_steps, state_dim)
    B, num_windows, time_steps, num_vars = test_data_rollout.shape

    logging.info(f"Computing batch L2 per window (open-loop) for {model_name} (Dataset 2) …")

    predict_one_window = jax.jit(
        jax.vmap(
            lambda u: model.x_pred_fn(params, u, t_star_window)[-1],
            in_axes=0,
        )
    )

    l2_per_window: list[float] = []
    # Initialize with the first time step of the first window
    u_current = test_data_rollout[:, 0, 0, :] 

    for k in range(num_windows):
        u_current = predict_one_window(u_current)
        x_ref_k = test_data_rollout[:, k, -1, :]  # Ground truth at end of window k
        
        numer   = jnp.linalg.norm(u_current - x_ref_k, axis=1)
        denom   = jnp.linalg.norm(x_ref_k, axis=1)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
        logging.info(f"  {model_name} DS2 Window {k+1:>3d} | mean L2: {l2_mean:.3e}")

    return np.array(l2_per_window), B


def evaluate_and_compare_openloop(config: ml_collections.ConfigDict, workdir: str):
    """
    Run open-loop evaluation for both PI and DD models and create comparison plots 
    for both datasets.
    """
    dt_window = float(config.get("dt_window", 0.25))
    
    # -------------------------------------------------------------------------
    # 1. Prepare Data Grids
    # -------------------------------------------------------------------------
    # Dataset 1
    x_ref_all, u0_ref_all, t_star_window = get_dataset()
    time_steps = 50
    t_star_window = t_star_window[0:time_steps]

    # Data set 2
    time_steps_2 = 51
    num_windows_test_2 = 31
    t_star_window_2 = jnp.linspace(0.0, dt_window, time_steps_2)
    
    # Load test data: shape (200, 31, 51, 40)
    test_data_rollout = dd_get_test_data_rollout(
        data_dir=config.training.get("data_dir", "data/"),
        windows_per_traj=num_windows_test_2,
    )
    logging.info(f"Loaded test data (Dataset 2): {test_data_rollout.shape}")

    # -------------------------------------------------------------------------
    # 2. Load Models
    # -------------------------------------------------------------------------
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
        os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model"
    )
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    # Create figure directory
    save_dir = os.path.join(workdir, "figures", "pi_vs_dd")
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3. Evaluate and Plot Dataset 1
    # -------------------------------------------------------------------------
    l2_pi_ds1, B_ds1 = _evaluate_batch_l2_openloop(model_pi, params_pi, t_star_window, config, "PI")
    l2_dd_ds1, _ = _evaluate_batch_l2_openloop(model_dd, params_dd, t_star_window, config, "DD")

    save_path_ds1 = os.path.join(save_dir, "batch_l2_per_window_openloop.pdf")
    _plot_l2_comparison(
        curves={
            "Open-loop (PI DeepONet)": l2_pi_ds1,
            "Open-loop (DD DeepONet)": l2_dd_ds1,
        },
        dt=dt_window,
        title=f"Open-loop: PI vs DD (Dataset 1, B={B_ds1})",
        save_path=save_path_ds1,
        colors={
            "Open-loop (PI DeepONet)": "#2196F3",
            "Open-loop (DD DeepONet)": "#FF5722",
        },
    )

    # -------------------------------------------------------------------------
    # 4. Evaluate and Plot Dataset 2
    # -------------------------------------------------------------------------
    # Note: We pass t_star_window_2 here since Dataset 2 has 51 steps
    l2_pi_ds2, B_ds2 = _evaluate_batch_l2_openloop_dataset2(
        model_pi, params_pi, t_star_window_2, test_data_rollout, "PI"
    )
    l2_dd_ds2, _ = _evaluate_batch_l2_openloop_dataset2(
        model_dd, params_dd, t_star_window_2, test_data_rollout, "DD"
    )

    save_path_ds2 = os.path.join(save_dir, "batch_l2_per_window_openloop_dataset2.pdf")
    _plot_l2_comparison(
        curves={
            "Open-loop (PI DeepONet)": l2_pi_ds2,
            "Open-loop (DD DeepONet)": l2_dd_ds2,
        },
        dt=dt_window,
        title=f"Open-loop: PI vs DD (Dataset 2, B={B_ds2})",
        save_path=save_path_ds2,
        colors={
            "Open-loop (PI DeepONet)": "#2196F3",
            "Open-loop (DD DeepONet)": "#FF5722",
        },
    )


def evaluate_and_compare_with_enkf(config: ml_collections.ConfigDict, workdir: str):
    """
    Run EnKF evaluation for both PI and DD models and create comparison plots.
    """
    from examples.KS.kf import EnKFState, run_enkf_smoother, init_ensemble

    obs_every_n  = config.ekf.get("obs_every_n",   4)
    sigma_obs    = config.ekf.get("sigma_obs",      0.5)
    P0_sigma     = config.ekf.get("P0_sigma",       1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",   False)
    N_ens        = config.enkf.get("N_ens",         50)
    sigma_model  = config.enkf.get("sigma_model",   0.1)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))

    x_ref_all, u0_ref_all, t_star_window = get_dataset()
    time_steps = 50
    t_star_window = t_star_window[0:time_steps]

    # Load PI model
    logging.info("Loading PI model for EnKF...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(
        os.getcwd(), config.wandb.ckpt_name_pi, "ckpt", "udon_model"
    )
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    # Load DD model
    logging.info("Loading DD model for EnKF...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(
        os.getcwd(), config.wandb.ckpt_name_dd, "ckpt", "udon_dd_model"
    )
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    # Build EnKF functions for both models
    predict_fn_pi, update_fn_pi = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)

    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    obs_indices = jnp.arange(0, N, obs_every_n)
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    num_windows = config.training.num_time_windows
    total_time  = num_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time = total_time,
        dt_fine    = DT_FINE,
        dt_obs     = DT_OBS,
    )

    # Evaluate both models
    results_pi = _evaluate_batch_l2_enkf(
        model_pi, params_pi, t_star_window,
        predict_fn_pi, update_fn_pi,
        Q_fine, P0,
        N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config,
        "PI",
    )

    results_dd = _evaluate_batch_l2_enkf(
        model_dd, params_dd, t_star_window,
        predict_fn_dd, update_fn_dd,
        Q_fine, P0,
        N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config,
        "DD",
    )

    dt_window = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    B = results_pi['B']

    save_dir = os.path.join(workdir, "figures", "pi_vs_dd")
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: L2 + Calibration comparison
    save_path = os.path.join(save_dir, "batch_l2_per_window_enkf.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    window_idx = np.arange(1, max_additions + 1)

    # L2 comparison
    ax = axes[0]
    ax.plot(window_idx, results_pi['ol_l2'],   marker="o", markersize=5, linewidth=2.0,
            label="Open-loop (PI)", color="#2196F3")
    ax.plot(window_idx, results_pi['enkf_l2'], marker="s", markersize=5, linewidth=2.0,
            label=f"PI EnKF (N_ens={N_ens})", color="#1976D2")
    ax.plot(window_idx, results_dd['ol_l2'],   marker="^", markersize=5, linewidth=2.0,
            label="Open-loop (DD)", color="#FF5722")
    ax.plot(window_idx, results_dd['enkf_l2'], marker="v", markersize=5, linewidth=2.0,
            label=f"DD EnKF (N_ens={N_ens})", color="#E64A19")
    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop: L2 per window (PI vs DD)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)

    # Calibration comparison
    ax2 = axes[1]
    ax2.plot(window_idx, results_pi['spread_mean'], marker="^", markersize=5, linewidth=2.0,
             label="PI RMS σ", color="#4CAF50")
    ax2.plot(window_idx, results_pi['enkf_rmse'],   marker="s", markersize=5, linewidth=2.0,
             linestyle="--", label="PI RMSE", color="#FF5722")
    ax2.plot(window_idx, results_dd['spread_mean'], marker="^", markersize=5, linewidth=2.0,
             label="DD RMS σ", color="#2196F3")
    ax2.plot(window_idx, results_dd['enkf_rmse'],   marker="s", markersize=5, linewidth=2.0,
             linestyle="--", label="DD RMSE", color="#9C27B0")
    ax2.set_yscale("log")
    ax2.set_xlabel("Window index", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs RMSE", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax2_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)

    fig.suptitle(
        f"EnKF batch evaluation: PI vs DD  (B={B}, N_ens={N_ens}, "
        f"obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"EnKF batch L2 + calibration comparison saved to: {save_path}")

    # Plot 2: ERF comparison
    erf_save_path = os.path.join(save_dir, "batch_erf_enkf.pdf")
    _plot_erf_comparison(
        obs_times=results_pi['obs_times'],
        erf_data={
            "PI EnKF": (results_pi['erf_mean'], results_pi['erf_std']),
            "DD EnKF": (results_dd['erf_mean'], results_dd['erf_std']),
        },
        n_traj=B,
        title=(
            f"EnKF Error Reduction Factor: PI vs DD\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={DT_OBS:.3g})"
        ),
        save_path=erf_save_path,
        colors={
            "PI EnKF": "#2196F3",
            "DD EnKF": "#FF5722",
        },
    )

    # Plot 3: RMSE comparison
    rmse_save_path = os.path.join(save_dir, "batch_rmse_enkf.pdf")
    _plot_rmse_comparison(
        obs_times=results_pi['obs_times'],
        rmse_data={
            "PI": {
                'prior_mean': results_pi['prior_rmse_mean'],
                'prior_std': results_pi['prior_rmse_std'],
                'post_mean': results_pi['post_rmse_mean'],
                'post_std': results_pi['post_rmse_std'],
            },
            "DD": {
                'prior_mean': results_dd['prior_rmse_mean'],
                'prior_std': results_dd['prior_rmse_std'],
                'post_mean': results_dd['post_rmse_mean'],
                'post_std': results_dd['post_rmse_std'],
            },
        },
        sigma_obs=sigma_obs,
        n_traj=B,
        title=(
            f"EnKF prior vs posterior RMSE: PI vs DD\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={DT_OBS:.3g})"
        ),
        save_path=rmse_save_path,
        colors={
            "PI": "#2196F3",
            "DD": "#FF5722",
        },
    )