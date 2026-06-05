"""
Comparison module for Physics-Informed (PI) vs Data-Driven (DD) DeepONets.

This module provides evaluation functions that run both PI and DD models in parallel,
then plot their results superposed for direct comparison. Individual trajectory plots
are skipped in favor of batch-level metrics.
"""

import os
from absl import logging
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from typing import Callable

from jaxpi.utils import restore_checkpoint
import examples.l96_n40_f6_ics.models as models
from examples.l96_n40_f6_ics.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps

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
    curves:    dict[str, np.ndarray],
    dt:        float,
    title:     str,
    save_path: str,
    colors:    dict[str, str] | None = None,
) -> None:
    """
    Plot one or more average-L2-per-window curves on a log-scale y-axis and
    save the figure as a PDF.
 
    A secondary x-axis showing elapsed simulation time (window_index × dt)
    is added above the primary axis.
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
 
    # secondary x-axis — simulation time
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
    logging.info(f"L2-per-window comparison plot saved to: {save_path}")


def _plot_erf(
    obs_times:  np.ndarray,
    erf_mean:   np.ndarray,
    erf_std:    np.ndarray,
    n_traj:     int,
    title:      str,
    save_path:  str,
) -> None:
    """
    Plot the Error Reduction Factor (ERF = prior RMSE / posterior RMSE)
    averaged across ``n_traj`` trajectories.
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
    obs_times:       np.ndarray,
    prior_rmse_mean: np.ndarray,
    prior_rmse_std:  np.ndarray,
    post_rmse_mean:  np.ndarray,
    post_rmse_std:   np.ndarray,
    sigma_obs:       float,
    n_traj:          int,
    title:           str,
    save_path:       str,
) -> None:
    """
    Plot mean prior RMSE, mean posterior RMSE, and the measurement noise
    level (sigma_obs) on a shared log-scale y-axis.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

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


def evaluate_pi_vs_dd(config: ml_collections.ConfigDict, workdir: str):
    """
    Evaluate and compare Physics-Informed (PI) and Data-Driven (DD) DeepONets
    in open-loop mode. Plots batch-level metrics side-by-side without individual
    trajectory plots.
    """
    time_steps = 50
    x_ref_all, u0_ref_all, t_star_window = get_dataset()
    t_star_window = t_star_window[0:time_steps]

    # Load both models
    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(
        os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model"
    )
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params

    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, jnp.linspace(0.0, 0.25, 51))
    ckpt_path_dd = os.path.join(
        os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model"
    )
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    # Run batch evaluation
    _evaluate_batch_l2_openloop_comparison(
        model_pi, params_pi, model_dd, params_dd,
        t_star_window, config, workdir
    )


def _evaluate_batch_l2_openloop_comparison(
    model_pi, params_pi, model_dd, params_dd,
    t_star_window, config, workdir
):
    """
    Compute and plot the batch-averaged open-loop L2 error per window for both
    PI and DD models side-by-side.
    """
    dt_window    = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    num_vars      = model_pi.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )

    logging.info("Computing batch L2 per window (PI vs DD) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, num_vars)
    B = u0_original.shape[0]

    # Create vmapped predictors for both models
    predict_one_window_pi = jax.jit(
        jax.vmap(
            lambda u: model_pi.x_pred_fn(params_pi, u, t_star_window)[-1],
            in_axes=0,
        )
    )

    # DD model uses different time grid
    t_star_window_dd = jnp.linspace(0.0, 0.25, 51)
    predict_one_window_dd = jax.jit(
        jax.vmap(
            lambda u: model_dd.x_pred_fn(params_dd, u, t_star_window_dd)[-1],
            in_axes=0,
        )
    )

    l2_per_window_pi: list[float] = []
    l2_per_window_dd: list[float] = []
    u_current_pi = u0_original
    u_current_dd = u0_original

    for k in range(max_additions):
        # PI prediction
        u_current_pi = predict_one_window_pi(u_current_pi)
        x_ref_k = rollout_states[k]
        numer_pi = jnp.linalg.norm(u_current_pi - x_ref_k, axis=1)
        denom = jnp.linalg.norm(x_ref_k, axis=1)
        l2_mean_pi = float(jnp.mean(numer_pi / (denom + 1e-12)))
        l2_per_window_pi.append(l2_mean_pi)

        # DD prediction
        u_current_dd = predict_one_window_dd(u_current_dd)
        numer_dd = jnp.linalg.norm(u_current_dd - x_ref_k, axis=1)
        l2_mean_dd = float(jnp.mean(numer_dd / (denom + 1e-12)))
        l2_per_window_dd.append(l2_mean_dd)

        logging.info(
            f"  Window {k+1:>3d} | PI L2: {l2_mean_pi:.3e} | DD L2: {l2_mean_dd:.3e}"
        )

    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_pi_vs_dd.pdf")
    _plot_l2_per_window(
        curves={
            "PI (DeepONet)": np.array(l2_per_window_pi),
            "DD (DeepONet)": np.array(l2_per_window_dd),
        },
        dt=dt_window,
        title=f"PI vs DD: batch-average L2 per window  (B={B})",
        save_path=save_path,
        colors={"PI (DeepONet)": "#2196F3", "DD (DeepONet)": "#FF5722"},
    )


def evaluate_with_enkf_pi_vs_dd(config: ml_collections.ConfigDict, workdir: str):
    """
    Evaluate and compare Physics-Informed (PI) and Data-Driven (DD) DeepONets
    with EKF data assimilation. Plots batch-level metrics side-by-side.
    """
    from examples.KS.kf import EKFState, run_ekf_smoother

    obs_every_n = config.ekf.get("obs_every_n", 4)
    sigma_obs   = config.ekf.get("sigma_obs", 0.5)
    sigma_proc  = config.ekf.get("sigma_proc", 0.1)
    P0_sigma    = config.ekf.get("P0_sigma", 1.0)
    dynamic_vars = config.ekf.get("dynamic_vars", False)

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list = config.kf.get("obs_idx_list", None)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.ekf.get("dt_fine", DT_WINDOW))
    DT_OBS = float(config.ekf.get("dt_obs", DT_WINDOW))

    x_ref_all, u0_ref_all, t_star_window = get_dataset()
    t_star_window = t_star_window[0:50]

    # Load PI model
    logging.info("Loading PI model for EKF comparison...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.ckpt_name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    # Load DD model
    logging.info("Loading DD model for EKF comparison...")
    t_star_window_dd = jnp.linspace(0.0, 0.25, 51)
    model_dd = models.L96UDON_DD(config, t_star_window_dd)
    ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.ckpt_name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    # Build EKF functions for both models
    predict_fn_pi, update_fn_pi = model_pi.make_ekf_fns(params_pi)
    predict_fn_dd, update_fn_dd = model_dd.make_ekf_fns(params_dd)

    # Scale process noise
    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_proc ** 2
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m = len(obs_indices)
    R = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    num_windows = config.training.num_time_windows
    total_time = num_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )

    def lorenz_96(t, state, F=6.0):
        xp1 = np.roll(state, -1)
        xm1 = np.roll(state, 1)
        xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F

    # Run batch evaluation
    _evaluate_batch_l2_ekf_comparison(
        model_pi, params_pi, model_dd, params_dd,
        t_star_window, t_star_window_dd,
        predict_fn_pi, update_fn_pi,
        predict_fn_dd, update_fn_dd,
        Q_fine, R, P0,
        obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config, workdir,
    )


def _evaluate_batch_l2_ekf_comparison(
    model_pi, params_pi, model_dd, params_dd,
    t_star_window_pi, t_star_window_dd,
    predict_fn_pi, update_fn_pi,
    predict_fn_dd, update_fn_dd,
    Q_fine, R, P0,
    obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars,
    dt_fine: float,
    dt_obs: float,
    config, workdir,
):
    """
    Compute and plot batch-averaged L2 error per window for both PI and DD models
    with EKF assimilation.
    """
    from examples.KS.kf import run_ekf_smoother, EKFState

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list = config.kf.get("obs_idx_list", None)

    dt_window = config.get("dt_window", 0.25)
    max_additions = config.training.get("max_additions", 5)
    N = model_pi.N
    mat_path = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    ekf_batch_size = config.ekf.get("batch_l2_size", 200)

    logging.info("Computing batch L2 per window (PI vs DD with EKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)

    B = min(u0_original.shape[0], ekf_batch_size)
    u0_original = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  Using {B} ICs from pool for batch L2 evaluation.")

    # Open-loop predictors for both models
    predict_one_window_pi = jax.jit(
        jax.vmap(
            lambda u: model_pi.x_pred_fn(params_pi, u, t_star_window_pi)[-1],
            in_axes=0,
        )
    )

    predict_one_window_dd = jax.jit(
        jax.vmap(
            lambda u: model_dd.x_pred_fn(params_dd, u, t_star_window_dd)[-1],
            in_axes=0,
        )
    )

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m = len(obs_indices)

    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch,
        dt_fine=dt_fine,
        dt_obs=dt_obs,
    )
    T_obs = len(obs_step_indices_batch)

    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(max_additions)
    ])

    # Accumulators for both models
    ekf_l2_sum_pi = np.zeros(max_additions)
    ekf_l2_sum_dd = np.zeros(max_additions)

    erf_sum_pi = np.zeros(T_obs)
    erf_sq_sum_pi = np.zeros(T_obs)
    erf_sum_dd = np.zeros(T_obs)
    erf_sq_sum_dd = np.zeros(T_obs)

    prior_rmse_sum_pi = np.zeros(T_obs)
    prior_rmse_sq_sum_pi = np.zeros(T_obs)
    post_rmse_sum_pi = np.zeros(T_obs)
    post_rmse_sq_sum_pi = np.zeros(T_obs)

    prior_rmse_sum_dd = np.zeros(T_obs)
    prior_rmse_sq_sum_dd = np.zeros(T_obs)
    post_rmse_sum_dd = np.zeros(T_obs)
    post_rmse_sq_sum_dd = np.zeros(T_obs)

    def lorenz_96(t, state, F=6.0):
        xp1 = np.roll(state, -1)
        xm1 = np.roll(state, 1)
        xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F

    for ic in range(B):
        key = jax.random.PRNGKey(ic + 9999)
        u_true = u0_original[ic]

        # Solve reference ODE
        t_eval_fine = np.linspace(0.0, total_time_batch, total_fine_steps_batch + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time_batch],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine = sol.y.T
        x_true_at_obs = x_true_fine[obs_step_indices_batch + 1]

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
            y_t = x_true_t[obs_idx_vars] + noise

            H_list.append(H_t)
            y_obs_list.append(y_t)

        H_seq = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)

        # Perturbed IC
        key, key_ic = jax.random.split(key)
        x0_hat = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))

        # Run EKF for PI model
        x_hats_pi, _, prior_means_at_obs_pi = run_ekf_smoother(
            predict_fn_pi, update_fn_pi,
            x0_hat, P0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,
            R,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )

        # Run EKF for DD model
        x_hats_dd, _, prior_means_at_obs_dd = run_ekf_smoother(
            predict_fn_dd, update_fn_dd,
            x0_hat, P0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,
            R,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )

        # Extract posterior means
        post_means_at_obs_pi = x_hats_pi[obs_step_indices_batch]
        post_means_at_obs_dd = x_hats_dd[obs_step_indices_batch]

        # Compute metrics for PI
        prior_rmse_pi = np.sqrt(np.mean(
            (np.array(prior_means_at_obs_pi) - x_true_at_obs) ** 2, axis=1
        ))
        post_rmse_pi = np.sqrt(np.mean(
            (np.array(post_means_at_obs_pi) - x_true_at_obs) ** 2, axis=1
        ))
        erf_ic_pi = prior_rmse_pi / (post_rmse_pi + 1e-12)

        erf_sum_pi += erf_ic_pi
        erf_sq_sum_pi += erf_ic_pi ** 2
        prior_rmse_sum_pi += prior_rmse_pi
        prior_rmse_sq_sum_pi += prior_rmse_pi ** 2
        post_rmse_sum_pi += post_rmse_pi
        post_rmse_sq_sum_pi += post_rmse_pi ** 2

        # Compute metrics for DD
        prior_rmse_dd = np.sqrt(np.mean(
            (np.array(prior_means_at_obs_dd) - x_true_at_obs) ** 2, axis=1
        ))
        post_rmse_dd = np.sqrt(np.mean(
            (np.array(post_means_at_obs_dd) - x_true_at_obs) ** 2, axis=1
        ))
        erf_ic_dd = prior_rmse_dd / (post_rmse_dd + 1e-12)

        erf_sum_dd += erf_ic_dd
        erf_sq_sum_dd += erf_ic_dd ** 2
        prior_rmse_sum_dd += prior_rmse_dd
        prior_rmse_sq_sum_dd += prior_rmse_dd ** 2
        post_rmse_sum_dd += post_rmse_dd
        post_rmse_sq_sum_dd += post_rmse_dd ** 2

        # Per-window L2 for PI
        for k in range(max_additions):
            ref_k = rollout_states[k][ic]
            step_k = window_step_indices[k]
            x_hat_k_pi = x_hats_pi[step_k]
            ekf_l2_sum_pi[k] += float(
                jnp.linalg.norm(x_hat_k_pi - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )

        # Per-window L2 for DD
        for k in range(max_additions):
            ref_k = rollout_states[k][ic]
            step_k = window_step_indices[k]
            x_hat_k_dd = x_hats_dd[step_k]
            ekf_l2_sum_dd[k] += float(
                jnp.linalg.norm(x_hat_k_dd - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )

    # Open-loop for comparison
    ol_l2_pi = np.zeros(max_additions)
    ol_l2_dd = np.zeros(max_additions)
    u_current_pi = u0_original
    u_current_dd = u0_original

    for k in range(max_additions):
        u_current_pi = predict_one_window_pi(u_current_pi)
        ref_k = rollout_states[k]
        numer = jnp.linalg.norm(u_current_pi - ref_k, axis=1)
        denom = jnp.linalg.norm(ref_k, axis=1)
        ol_l2_pi[k] = float(jnp.mean(numer / (denom + 1e-12)))

        u_current_dd = predict_one_window_dd(u_current_dd)
        numer = jnp.linalg.norm(u_current_dd - ref_k, axis=1)
        ol_l2_dd[k] = float(jnp.mean(numer / (denom + 1e-12)))

    l2_ekf_pi = ekf_l2_sum_pi / B
    l2_ekf_dd = ekf_l2_sum_dd / B

    # ERF statistics
    erf_mean_pi = erf_sum_pi / B
    erf_std_pi = np.sqrt(np.maximum(erf_sq_sum_pi / B - erf_mean_pi ** 2, 0.0))

    erf_mean_dd = erf_sum_dd / B
    erf_std_dd = np.sqrt(np.maximum(erf_sq_sum_dd / B - erf_mean_dd ** 2, 0.0))

    # RMSE statistics
    prior_rmse_mean_pi = prior_rmse_sum_pi / B
    prior_rmse_std_pi = np.sqrt(np.maximum(
        prior_rmse_sq_sum_pi / B - prior_rmse_mean_pi ** 2, 0.0))
    post_rmse_mean_pi = post_rmse_sum_pi / B
    post_rmse_std_pi = np.sqrt(np.maximum(
        post_rmse_sq_sum_pi / B - post_rmse_mean_pi ** 2, 0.0))

    prior_rmse_mean_dd = prior_rmse_sum_dd / B
    prior_rmse_std_dd = np.sqrt(np.maximum(
        prior_rmse_sq_sum_dd / B - prior_rmse_mean_dd ** 2, 0.0))
    post_rmse_mean_dd = post_rmse_sum_dd / B
    post_rmse_std_dd = np.sqrt(np.maximum(
        post_rmse_sq_sum_dd / B - post_rmse_mean_dd ** 2, 0.0))

    # Plotting
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)

    # L2 comparison plot
    save_path = os.path.join(save_dir, "batch_l2_per_window_ekf_pi_vs_dd.pdf")
    fig, ax = plt.subplots(figsize=(10, 6))
    window_idx = np.arange(1, max_additions + 1)

    ax.plot(window_idx, ol_l2_pi, marker="o", markersize=5, linewidth=2.0,
            label="Open-loop PI", color="#2196F3", alpha=0.7)
    ax.plot(window_idx, ol_l2_dd, marker="s", markersize=5, linewidth=2.0,
            label="Open-loop DD", color="#FF5722", alpha=0.7)
    ax.plot(window_idx, l2_ekf_pi, marker="o", markersize=5, linewidth=2.0,
            linestyle="--", label="EKF PI", color="#2196F3")
    ax.plot(window_idx, l2_ekf_dd, marker="s", markersize=5, linewidth=2.0,
            linestyle="--", label="EKF DD", color="#FF5722")

    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title(f"PI vs DD with EKF: batch-average L2 per window  (B={B})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"L2 comparison plot saved to: {save_path}")

    # ERF comparison plot
    erf_save_path = os.path.join(save_dir, "batch_erf_pi_vs_dd.pdf")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(obs_times_batch, erf_mean_pi, marker="o", markersize=4, linewidth=2.0,
            label=f"PI ERF (n={B})", color="#2196F3")
    ax.fill_between(
        obs_times_batch,
        erf_mean_pi - erf_std_pi,
        erf_mean_pi + erf_std_pi,
        color="#2196F3", alpha=0.15, linewidth=0,
    )

    ax.plot(obs_times_batch, erf_mean_dd, marker="s", markersize=4, linewidth=2.0,
            label=f"DD ERF (n={B})", color="#FF5722")
    ax.fill_between(
        obs_times_batch,
        erf_mean_dd - erf_std_dd,
        erf_mean_dd + erf_std_dd,
        color="#FF5722", alpha=0.15, linewidth=0,
    )

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(f"PI vs DD: Error Reduction Factor comparison  (B={B})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(erf_save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"ERF comparison plot saved to: {erf_save_path}")

    # RMSE comparison plot
    rmse_save_path = os.path.join(save_dir, "batch_rmse_pi_vs_dd.pdf")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PI subplot
    ax1.plot(obs_times_batch, prior_rmse_mean_pi, marker="o", markersize=4, linewidth=2.0,
             label=f"Prior RMSE", color="#2196F3")
    ax1.fill_between(
        obs_times_batch,
        prior_rmse_mean_pi - prior_rmse_std_pi,
        prior_rmse_mean_pi + prior_rmse_std_pi,
        color="#2196F3", alpha=0.15, linewidth=0,
    )

    ax1.plot(obs_times_batch, post_rmse_mean_pi, marker="s", markersize=4, linewidth=2.0,
             label=f"Posterior RMSE", color="#FF5722")
    ax1.fill_between(
        obs_times_batch,
        post_rmse_mean_pi - post_rmse_std_pi,
        post_rmse_mean_pi + post_rmse_std_pi,
        color="#FF5722", alpha=0.15, linewidth=0,
    )

    ax1.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", linewidth=1.6,
                label=f"σ_obs = {sigma_obs}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Observation time  t", fontsize=11)
    ax1.set_ylabel("RMSE  (log scale)", fontsize=11)
    ax1.set_title(f"Physics-Informed (PI)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # DD subplot
    ax2.plot(obs_times_batch, prior_rmse_mean_dd, marker="o", markersize=4, linewidth=2.0,
             label=f"Prior RMSE", color="#2196F3")
    ax2.fill_between(
        obs_times_batch,
        prior_rmse_mean_dd - prior_rmse_std_dd,
        prior_rmse_mean_dd + prior_rmse_std_dd,
        color="#2196F3", alpha=0.15, linewidth=0,
    )

    ax2.plot(obs_times_batch, post_rmse_mean_dd, marker="s", markersize=4, linewidth=2.0,
             label=f"Posterior RMSE", color="#FF5722")
    ax2.fill_between(
        obs_times_batch,
        post_rmse_mean_dd - post_rmse_std_dd,
        post_rmse_mean_dd + post_rmse_std_dd,
        color="#FF5722", alpha=0.15, linewidth=0,
    )

    ax2.axhline(y=sigma_obs, color="#4CAF50", linestyle="--", linewidth=1.6,
                label=f"σ_obs = {sigma_obs}")
    ax2.set_yscale("log")
    ax2.set_xlabel("Observation time  t", fontsize=11)
    ax2.set_ylabel("RMSE  (log scale)", fontsize=11)
    ax2.set_title(f"Data-Driven (DD)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle(
        f"PI vs DD: Prior/Posterior RMSE comparison  (B={B})",
        fontsize=13,
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(rmse_save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"RMSE comparison plot saved to: {rmse_save_path}")