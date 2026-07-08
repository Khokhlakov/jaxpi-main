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
import examples.l96_6_larger_data.models as models
from examples.l96_6_larger_data.utils import get_dataset, build_obs_schedule, scale_Q_for_fine_steps

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
 
    A secondary x-axis showing elapsed simulation time (window_index × dt)
    is added above the primary axis so that the absolute temporal scale is
    immediately apparent alongside the window-index ticks.
 
    Args:
        curves:    Mapping from method label to a 1-D array of length
                   num_windows containing the mean L2 at each window boundary.
        dt:        Assimilation window duration used to label the axes.
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
 
    num_windows = len(next(iter(curves.values())))
    window_idx  = np.arange(1, num_windows + 1)
 
    ax.set_yscale("log")
    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ── secondary x-axis — simulation time ──────────────────────────────────
    # Twin the x-axis so the top spine shows elapsed time (window_idx × dt).
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
    # ────────────────────────────────────────────────────────────────────────
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Batch L2-per-window plot saved to: {save_path}")

# evaluate, evaluate_nekf, evaluate_enkf_numerical
def _plot_trajectory_summary(
    t_ax:       np.ndarray,        # (T,)   time axis
    x_true:     np.ndarray,        # (T, N) ground-truth state
    x_est:      np.ndarray,        # (T, N) estimate (prediction / filter mean)
    x_std:      np.ndarray | None, # (T, N) per-variable std, or None
    ic_idx:     int,
    est_label:  str,               # e.g. "DeepONet", "EKF estimate", "EnKF mean"
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
 
        # ±1σ uncertainty band (EKF or EnKF only)
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

def _plot_erf(
    obs_times:  np.ndarray,   # (T_obs,)  absolute observation times
    erf_mean:   np.ndarray,   # (T_obs,)  mean ERF across trajectories
    erf_std:    np.ndarray,   # (T_obs,)  std  ERF across trajectories
    n_traj:     int,          # number of trajectories averaged
    title:      str,
    save_path:  str,
) -> None:
    """
    Plot the Error Reduction Factor (ERF = prior RMSE / posterior RMSE)
    averaged across ``n_traj`` trajectories and over observation times.

    A horizontal reference line at ERF = 1 marks the boundary between
    beneficial assimilation (ERF > 1) and detrimental assimilation (ERF < 1).
    The shaded band shows ±1 std across trajectories.

    Args:
        obs_times:  observation times at which ERF is defined.
        erf_mean:   mean ERF at each observation time.
        erf_std:    standard deviation of ERF across trajectories.
        n_traj:     number of trajectories used for averaging (shown in title).
        title:      figure suptitle.
        save_path:  full path including .pdf extension.
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

    # Reference line — ERF = 1 means no error reduction
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
    obs_times:       np.ndarray,   # (T_obs,)  absolute observation times
    prior_rmse_mean: np.ndarray,   # (T_obs,)  mean prior  RMSE across trajectories
    prior_rmse_std:  np.ndarray,   # (T_obs,)  std  prior  RMSE across trajectories
    post_rmse_mean:  np.ndarray,   # (T_obs,)  mean posterior RMSE across trajectories
    post_rmse_std:   np.ndarray,   # (T_obs,)  std  posterior RMSE across trajectories
    sigma_obs:       float,        # measurement noise std — reference level
    n_traj:          int,          # number of trajectories averaged
    title:           str,
    save_path:       str,
) -> None:
    """
    Plot mean prior RMSE, mean posterior RMSE, and the measurement noise
    level (sigma_obs) on a shared log-scale y-axis.

    A well-functioning filter must satisfy two conditions simultaneously:
        (1) posterior RMSE < prior RMSE   — assimilation reduces error
        (2) posterior RMSE < sigma_obs    — filter extracts signal from noise

    Both are immediately visible from this plot.  Shaded ±1-std bands
    across trajectories indicate robustness of the RMSE estimates.

    Args:
        obs_times:       observation times at which RMSE is defined.
        prior_rmse_mean: mean prior  RMSE at each observation time.
        prior_rmse_std:  std  prior  RMSE across trajectories.
        post_rmse_mean:  mean posterior RMSE at each observation time.
        post_rmse_std:   std  posterior RMSE across trajectories.
        sigma_obs:       observation noise standard deviation.
        n_traj:          number of trajectories (shown in legend).
        title:           figure title.
        save_path:       full path including .pdf extension.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Prior RMSE ────────────────────────────────────────────────────────────
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

    # ── Posterior RMSE ────────────────────────────────────────────────────────
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

    # ── Measurement noise reference ───────────────────────────────────────────
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
    trajectory_windows = config.eval.get("trajectory_windows", 160)
    batch_windows      = config.eval.get("windows", 30)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
 
    # ── 2. Model & per-window query grid ────────────────────────────────────
    time_steps = 50
    _, _, t_star_window = get_dataset()
    t_star_window = t_star_window[0:time_steps]
    T_last = float(t_star_window[-1])
 
    model = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
    logging.info("Restored trained DeepONet model for long autoregressive rollout.")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
 
    num_plots = min(config.saving.total_plots, u_test.shape[0])
 
    for ic_idx in range(num_plots):
        logging.info(f"--- [long] Evaluating Trajectory for IC index {ic_idx} ---")
 
        # Grab IC from the test set
        u_current = jnp.array(u_test[ic_idx, 0, :])   
 
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
 
            u_current = x_pred_window[-1, :]
 
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
 
    u0_batch = jnp.array(u_test[:, 0, :])   # (B, N) — true IC at t = 0

    # 1. JIT-compiled, vmapped FULL-window predictor:  (B, N) → (B, T, N)
    predict_full_window = jax.jit(
        jax.vmap(
            lambda u: model.x_pred_fn(params, u, t_star_window),
            in_axes=0,
        )
    )

    x_pred_dense = []
    u_current = u0_batch

    # 2. Rollout the DeepONet densely
    for k in range(num_windows_long):
        x_pred_window = predict_full_window(u_current)                 
        
        if k == 0:
            x_pred_dense.append(x_pred_window)
        else:
            x_pred_dense.append(x_pred_window[:, 1:, :]) # skip duplicate boundary
            
        u_current = x_pred_window[:, -1, :]

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
    t_star_window, u_test: np.ndarray, t_test: np.ndarray, 
    dt_window: float, num_windows_long: int, 
    config, workdir: str
) -> None:
    
    B = u_test.shape[0]
    dt_test = float(t_test[1] - t_test[0])
    pts_pw  = int(round(dt_window / dt_test))
 
    u0_batch = jnp.array(u_test[:, 0, :])

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
            
        u_current_pi = x_pred_window_pi[:, -1, :]
        u_current_dd = x_pred_window_dd[:, -1, :]

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

    trajectory_windows = config.eval.get("trajectory_windows", 160)
    batch_windows      = config.eval.get("windows", 30)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
 
    # ── 2. Models & per-window query grid ───────────────────────────────────
    time_steps = 50
    _, _, t_star_window = get_dataset()
    t_star_window = t_star_window[0:time_steps]
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
 
        u_current_pi = jnp.array(u_test[ic_idx, 0, :])   
        u_current_dd = jnp.array(u_test[ic_idx, 0, :])   
 
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
 
            u_current_pi = x_pred_window_pi[-1, :]
            u_current_dd = x_pred_window_dd[-1, :]
 
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
        u_test[:num_ics_eval], t_test, dt_window, batch_windows,
        config, workdir,
    )

# ── PI + EnKF ─────────────────────────────────────────────────────────────────

def evaluate_with_enkf(config: ml_collections.ConfigDict, workdir: str):
    from examples.l96_6_larger_data.kf import EnKFState, run_enkf_smoother, init_ensemble

    obs_every_n  = config.ekf.get("obs_every_n",   4)
    sigma_obs    = config.ekf.get("sigma_obs",      0.5)
    P0_sigma     = config.ekf.get("P0_sigma",       1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",   False)
    N_ens        = config.enkf.get("N_ens",         50)
    sigma_model  = config.enkf.get("sigma_model",   0.1)

    specify_obs_idx   = config.kf.get("specify_obs_idx", False)
    obs_idx_list      = config.kf.get("obs_idx_list", None)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))
    
    time_steps = 50

    x_ref_all, u0_ref_all, t_star_window = get_dataset()
    t_star_window = t_star_window[0:time_steps]

    model     = models.L96UDON(config, t_star_window)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.ckpt_name, "ckpt", "udon_model")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    N      = model.N

    # ── Build EnKF functions with variable-t surrogate propagator ─────────────
    # make_enkf_fns no longer takes dt; the propagator is (u, t) -> u and t is
    # supplied at each predict call by run_enkf_smoother.
    predict_fn, update_fn = model.make_enkf_fns(params, N_ens=N_ens)

    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    if (specify_obs_idx and obs_idx_list):
        obs_indices = jnp.array(obs_idx_list)
    else:
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

    def lorenz_96(t, state, F=6.0):
        xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F

    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- EnKF Evaluation for IC {ic_idx} (N_ens={N_ens}) ---")
        u_current_true = u0_ref_all[ic_idx, :]

        t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time],
            y0=np.array(u_current_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine  = jnp.array(sol.y.T)                      # (total_fine_steps+1, N)
        x_true_at_obs = x_true_fine[obs_step_indices + 1]      # (T_obs, N)

        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []

        for obs_idx in range(len(obs_times)):
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
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))

        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)

        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
 
        # ── Window-aware EnKF smoother ────────────────────────────────────────
        # dt_fine and dt_window are passed so run_enkf_smoother can track which
        # fine step we are at within the current DeepONet window and call
        # predict_fn with the correct in-window t_query.
        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices,
            H_seq,
            Q_fine,
            R,
            key,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
 
        t_fine_axis = t_eval_fine[1:]
 
        _plot_trajectory_summary(
            t_ax      = t_fine_axis,
            x_true    = np.array(x_true_fine[1:]),
            x_est     = np.array(x_means),
            x_std     = np.array(x_spreads),
            ic_idx    = ic_idx,
            est_label = "EnKF mean",
            save_path = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_enkf_ic_{ic_idx}.pdf",
            ),
            N = model.N,
            dt_window  = DT_WINDOW,
            obs_coords = obs_coords,
        )

        window_step_indices = np.array([
            round((w + 1) * DT_WINDOW / DT_FINE) - 1
            for w in range(num_windows)
        ])
        x_means_at_windows = x_means[window_step_indices]
        x_true_at_windows  = x_true_fine[window_step_indices + 1]

        l2_enkf     = jnp.linalg.norm(x_means_at_windows - x_true_at_windows) \
                    / jnp.linalg.norm(x_true_at_windows)
        mean_spread = float(jnp.mean(x_spreads))
        print(f"IC {ic_idx} | EnKF L2: {l2_enkf:.3e} | Mean σ: {mean_spread:.3e}")

    _evaluate_batch_l2_enkf(
        model, params, t_star_window,
        predict_fn, update_fn,
        Q_fine, P0,
        N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config, workdir,
    )

def _evaluate_batch_l2_enkf(
    model, params, t_star_window,
    predict_fn, update_fn,
    Q_fine, P0,
    N_ens, obs_every_n, sigma_obs, P0_sigma,
    dynamic_vars,
    dt_fine: float,
    dt_obs:  float,
    config, workdir,
):
    from examples.l96_6_larger_data.kf import run_enkf_smoother, init_ensemble, EnKFState

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

    logging.info("Computing batch L2 per window (open-loop vs EnKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)

    # Cap batch size; l96_udon.mat has 100 reference trajectories
    B = min(u0_original.shape[0], enkf_batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f" Using {B} ICs from pool for batch L2 / ERF evaluation (N_ens={N_ens}).")

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m           = len(obs_indices)
    R_fixed     = jnp.eye(m) * sigma_obs ** 2

    predict_one_window = jax.jit(
        jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window)[-1], in_axes=0)
    )

    
    # ── rebuild schedule for the batch duration ───────────────────────────────
    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time = total_time_batch,
        dt_fine    = dt_fine,
        dt_obs     = dt_obs,
    )
    T_obs = len(obs_step_indices_batch)

    # Absolute observation times (needed for the ERF x-axis)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])

    # ── window boundary indices derived directly from dt_fine ─────────────────
    # Step s is 0-indexed; the state after step s corresponds to time
    # (s + 1) * dt_fine.  Window k (1-indexed) ends at k * dt_window.
    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(max_additions)
    ])

    # Accumulators
    enkf_l2_sum     = np.zeros(max_additions)
    enkf_spread_sum = np.zeros(max_additions)
    enkf_rmse_sum   = np.zeros(max_additions)

    # ERF accumulators — shape (T_obs,); accumulated over B ICs
    erf_sum    = np.zeros(T_obs)
    erf_sq_sum = np.zeros(T_obs)   # for std computation

    # ── RMSE accumulators ────────────────────────────────────────────────────
    prior_rmse_sum    = np.zeros(T_obs)
    prior_rmse_sq_sum = np.zeros(T_obs)
    post_rmse_sum     = np.zeros(T_obs)
    post_rmse_sq_sum  = np.zeros(T_obs)

    for ic in range(B):
        key    = jax.random.PRNGKey(ic + 77777)
        u_true = u0_original[ic]   # (N,)

        def lorenz_96(t, state, F=6.0):
            xp1 = np.roll(state, -1)
            xm1 = np.roll(state,  1)
            xm2 = np.roll(state,  2)
            return (xp1 - xm2) * xm1 - state + F

        # ── Solve ODE for the batch duration (max_additions windows) ──────────
        t_eval_fine = np.linspace(0.0, total_time_batch, total_fine_steps_batch + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time_batch],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = sol.y.T                                   # (total_fine_steps_batch+1, N)
        x_true_at_obs = x_true_fine[obs_step_indices_batch + 1]  # (T_obs, N)

        # ── Build observation sequence indexed over T_obs events ───────────────
        H_list, y_obs_list = [], []

        for obs_idx in range(T_obs):
            x_true_t = x_true_at_obs[obs_idx]   # (N,)

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

        H_seq     = jnp.stack(H_list)      # (T_obs, m, N)
        y_obs_seq = jnp.stack(y_obs_list)  # (T_obs, m)

        # ── Initialise ensemble ───────────────────────────────────────────────
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        # ── Run EnKF smoother ─────────────────────────────────────────────────
        x_means, x_spreads, prior_means_at_obs = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,              # per-fine-step noise
            R_fixed,
            key,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )
        # x_means, x_spreads: (total_fine_steps_batch, N)
        # prior_means_at_obs : (T_obs, N)

        # ── Posterior means at observation steps ─────────────────────────────
        # x_means is indexed by fine step (0-based); obs_step_indices_batch[k]
        # is the fine step at which observation k occurred, so x_means at that
        # index is the *posterior* mean (update has already been applied).
        post_means_at_obs = x_means[obs_step_indices_batch]   # (T_obs, N)

        # ── ERF for this IC ───────────────────────────────────────────────────
        x_true_at_obs_jnp = jnp.array(x_true_at_obs)          # (T_obs, N)

        # RMSE over the N state variables for each observation time
        prior_rmse = np.sqrt(np.mean(
            (np.array(prior_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))  # (T_obs,)
        post_rmse  = np.sqrt(np.mean(
            (np.array(post_means_at_obs)  - x_true_at_obs) ** 2, axis=1
        ))  # (T_obs,)

        erf_ic = prior_rmse / (post_rmse + 1e-12)              # (T_obs,)
        erf_sum    += erf_ic
        erf_sq_sum += erf_ic ** 2

        prior_rmse_sum    += prior_rmse
        prior_rmse_sq_sum += prior_rmse ** 2
        post_rmse_sum     += post_rmse
        post_rmse_sq_sum  += post_rmse ** 2

        # ── Accumulate L2 at window boundaries ────────────────────────────────
        for k in range(max_additions):
            ref_k      = rollout_states[k][ic]             # (N,) ground truth at window k+1
            step_k     = window_step_indices[k]            # fine step index for window k+1
            x_hat_k    = x_means[step_k]                   # (N,) filter mean at that step

            enkf_l2_sum[k] += float(
                jnp.linalg.norm(x_hat_k - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
            enkf_rmse_sum[k] += float(jnp.sqrt(jnp.mean((x_hat_k - ref_k) ** 2)))
            enkf_spread_sum[k] += float(jnp.sqrt(jnp.mean(x_spreads[step_k] ** 2)))

    # ── Open-loop — vectorised over B ─────────────────────────────────────────
    ol_l2     = np.zeros(max_additions)
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_window(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k,              axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))

    l2_enkf     = enkf_l2_sum     / B
    rmse_enkf   = enkf_rmse_sum   / B
    spread_mean = enkf_spread_sum / B
    
    # ── ERF statistics ────────────────────────────────────────────────────────
    erf_mean = erf_sum    / B
    erf_std  = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))

    # ── Plotting ──────────────────────────────────────────────────────────────
    save_dir = os.path.join(workdir, "figures", config.wandb.name)

    # ── Existing L2 + calibration plot ───────────────────────────────────────
    save_path = os.path.join(save_dir, "batch_l2_per_window_enkf.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window_idx = np.arange(1, max_additions + 1)

    ax = axes[0]
    ax.plot(window_idx, ol_l2,   marker="o", markersize=4, linewidth=1.8,
            label="Open-loop (DeepONet)", color="#2196F3")
    ax.plot(window_idx, l2_enkf, marker="s", markersize=4, linewidth=1.8,
            label=f"EnKF mean (N_ens={N_ens})", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel(f"Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("EnKF vs open-loop: L2 per window", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ─────── secondary time axis for the L2 panel ─────────────────────────────
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
    # ─────────────────────────────────────────────────────────────────────────
 
    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean, marker="^", markersize=4, linewidth=1.8,
             label="RMS ensemble σ", color="#4CAF50")
    ax2.plot(window_idx, rmse_enkf,     marker="s", markersize=4, linewidth=1.8,
             linestyle="--", label="EnKF RMSE", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Window index", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs RMSE", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ─────── secondary time axis for the calibration panel ───────────────────
    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax2_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
    # ─────────────────────────────────────────────────────────────────────────

    fig.suptitle(
        f"EnKF batch evaluation  (B={B}, N_ens={N_ens}, "
        f"obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"EnKF batch L2-per-window plot saved to: {save_path}")

    # ─────── Error Reduction Factor plot ──────────────────────────────────────
    erf_save_path = os.path.join(save_dir, "batch_erf_enkf.pdf")
    _plot_erf(
        obs_times  = obs_times_batch,
        erf_mean   = erf_mean,
        erf_std    = erf_std,
        n_traj     = B,
        title      = (
            f"EnKF Error Reduction Factor per observation time\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path  = erf_save_path,
    )

    # ── Prior / Posterior RMSE vs noise level ──────────────────────────────────
    prior_rmse_mean = prior_rmse_sum    / B
    prior_rmse_std  = np.sqrt(np.maximum(
        prior_rmse_sq_sum / B - prior_rmse_mean ** 2, 0.0))
    post_rmse_mean  = post_rmse_sum     / B
    post_rmse_std   = np.sqrt(np.maximum(
        post_rmse_sq_sum  / B - post_rmse_mean  ** 2, 0.0))

    rmse_save_path = os.path.join(save_dir, "batch_rmse_enkf.pdf")
    _plot_rmse_comparison(
        obs_times       = obs_times_batch,
        prior_rmse_mean = prior_rmse_mean,
        prior_rmse_std  = prior_rmse_std,
        post_rmse_mean  = post_rmse_mean,
        post_rmse_std   = post_rmse_std,
        sigma_obs       = sigma_obs,
        n_traj          = B,
        title           = (
            f"EnKF prior vs posterior RMSE\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path       = rmse_save_path,
    )

# ── Numerical propagator ──────────────────────────────────────────────────────

def _make_l96_rk4_propagator(
    dt:         float,
    F:          float = 6.0,
    n_substeps: int   = 10,
) -> Callable:
    """
    Return a pure-JAX fixed-step RK4 propagator for the Lorenz-96 system.

    The returned callable integrates the L96 ODE for exactly ``dt`` time
    units using ``n_substeps`` RK4 steps of size h = dt / n_substeps.

    Because the implementation uses only ``jnp`` operations and
    ``jax.lax.scan``, the function is fully compatible with ``jit``,
    ``vmap`` (used inside ``make_enkf``'s predict step), and ``jacfwd``
    (used inside ``make_ekf``'s predict step).  No Python-level looping
    occurs at call time.

    Accuracy note
    -------------
    For the L96-N40 system with F = 6 the Lyapunov time is ≈ 1/0.9 ≈ 1.1
    time units.  Ten RK4 sub-steps per assimilation window of dt = 0.05
    (h = 0.005) keeps the local truncation error well below observation
    noise for typical sigma_obs values used here.  Increase n_substeps if
    dt is large (e.g. dt = 0.25 → n_substeps = 25 is safer).

    Args:
        dt:         Integration window length, same units as ``t_star``.
        F:          L96 forcing constant (default 6.0).
        n_substeps: Number of RK4 micro-steps within each ``dt`` window.

    Returns:
        propagator: Callable[(N,) -> (N,)] — pure JAX, no side effects.
    """
    h = dt / n_substeps

    def _l96_rhs(x: jnp.ndarray) -> jnp.ndarray:
        x_p1 = jnp.roll(x, -1)
        x_m1 = jnp.roll(x,  1)
        x_m2 = jnp.roll(x,  2)
        return (x_p1 - x_m2) * x_m1 - x + F

    def _rk4_step(x: jnp.ndarray) -> jnp.ndarray:
        k1 = _l96_rhs(x)
        k2 = _l96_rhs(x + 0.5 * h * k1)
        k3 = _l96_rhs(x + 0.5 * h * k2)
        k4 = _l96_rhs(x + h * k3)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagator(u: jnp.ndarray) -> jnp.ndarray:
        # lax.scan avoids Python-level unrolling; all n_substeps fused by XLA.
        x_final, _ = jax.lax.scan(
            lambda x, _: (_rk4_step(x), None),
            u, None, length=n_substeps,
        )
        return x_final

    return propagator

def _make_l96_rk4_variable_propagator(
    dt_fine:    float,
    F:          float = 6.0,
    n_substeps: int   = 10,
) -> Callable:
    """
    Return a variable-time RK4 propagator for use with the window-aware EnKF.
 
    Signature::
 
        propagator(u: (N,), t: float) -> (N,)
 
    where ``t`` is a Python float representing the desired integration
    duration.  ``t`` must be a positive integer multiple of ``dt_fine``.
 
    Internally, the function runs ``round(t / dt_fine)`` applications of
    the fixed-step propagator built by ``_make_l96_rk4_propagator``.
    Because ``make_enkf``'s predict is compiled with
    ``static_argnums=(3,)`` (the ``t_query`` argument), ``t`` is a
    compile-time constant here, so ``round(t / dt_fine)`` is a Python
    int and ``jax.lax.scan`` does not require dynamic shapes.
 
    XLA will retrace at most ``steps_per_window`` distinct values of
    ``t`` (one per distinct in-window offset), after which the JIT cache
    is reused for every subsequent window.
 
    Args:
        dt_fine:    Duration of one fine step (same units as t_star).
        F:          L96 forcing constant (default 6.0).
        n_substeps: Number of RK4 micro-steps *per fine step*.
                    Total micro-steps for a query at t = k * dt_fine
                    is k * n_substeps.
 
    Returns:
        propagator: Callable[(N,), float -> (N,)]
    """
    _single_fine_step = _make_l96_rk4_propagator(dt_fine, F, n_substeps)
 
    def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
        n_fine_steps = round(t / dt_fine)   # Python int — t is static
        x_final, _ = jax.lax.scan(
            lambda x, _: (_single_fine_step(x), None),
            u, None, length=n_fine_steps,
        )
        return x_final
 
    return propagator

# ── EnKF evaluation — numerical propagator ───────────────────────────────────

def evaluate_with_enkf_numerical(config: ml_collections.ConfigDict, workdir: str):
    """
    Per-trajectory EnKF evaluation using a numerical RK4 solver as the
    ensemble propagator instead of the trained DeepONet.
 
    Uses the same window-aware run_enkf_smoother as the DeepONet-based
    EnKF.  The variable-step RK4 propagator is queried at
    t_query = step_in_window * dt_fine, mirroring the DeepONet queries.
    For the RK4 propagator this is equivalent to running exactly
    round(t_query / dt_fine) fine RK4 steps from the window IC, which is
    numerically equivalent to chaining fine steps (RK4 has no surrogate
    error within a window) but uses a consistent interface.
    """
    from examples.l96_6_larger_data.kf import make_enkf, run_enkf_smoother, init_ensemble
 
    obs_every_n  = config.ekf.get("obs_every_n",    4)
    sigma_obs    = config.ekf.get("sigma_obs",       0.5)
    P0_sigma     = config.ekf.get("P0_sigma",        1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",    False)
    N_ens        = config.enkf.get("N_ens",          50)
    sigma_model  = config.enkf.get("sigma_model",    0.1)
    n_substeps   = config.enkf.get("rk4_substeps",   10)
    F            = config.enkf.get("l96_forcing",    6.0)

    specify_obs_idx   = config.kf.get("specify_obs_idx", False)
    obs_idx_list        = config.kf.get("obs_idx_list", None)
 
    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.ekf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.ekf.get("dt_obs",    DT_WINDOW))
 
    _, u0_ref_all, t_star_window = get_dataset()
    N = u0_ref_all.shape[1]
 
    # ── Build variable-time RK4 propagator and EnKF functions ────────────────
    # _make_l96_rk4_variable_propagator returns (u, t) -> u, matching the
    # interface expected by make_enkf and run_enkf_smoother.
    propagator = _make_l96_rk4_variable_propagator(
        dt_fine=DT_FINE, F=F, n_substeps=n_substeps,
    )
    predict_fn, update_fn = make_enkf(propagator, N, N_ens)
 
    logging.info(
        f"Numerical EnKF (window-aware): DT_FINE={DT_FINE}, "
        f"n_substeps={n_substeps}, N_ens={N_ens}, F={F}"
    )

    # ── Noise covariances ────────────────────────────────────────────────────
    # Q is specified at the coarse (window) level, then rescaled to dt_fine.
    steps_per_window = round(DT_WINDOW / DT_FINE)
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2

    # ── Observation schedule ─────────────────────────────────────────────────
    num_windows = config.training.num_time_windows
    total_time  = num_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )

    # ── Reference ODE (scipy, high accuracy) ─────────────────────────────────
    def lorenz_96(t, state, _F=F):
        xp1 = np.roll(state, -1)
        xm1 = np.roll(state,  1)
        xm2 = np.roll(state,  2)
        return (xp1 - xm2) * xm1 - state + _F

    # ── Per-IC evaluation loop ────────────────────────────────────────────────
    for ic_idx in range(config.saving.total_plots):
        logging.info(
            f"--- EnKF (numerical) Evaluation for IC {ic_idx} "
            f"(N_ens={N_ens}, n_substeps={n_substeps}) ---"
        )
        u_current_true = u0_ref_all[ic_idx, :]

        # Ground-truth trajectory at fine resolution (scipy reference)
        t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time],
            y0=np.array(u_current_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = jnp.array(sol.y.T)               # (total_fine_steps+1, N)
        x_true_at_obs = x_true_fine[obs_step_indices + 1] # (T_obs, N)

        # ── Build noisy observation sequence ──────────────────────────────────
        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []

        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]   # (N,)

            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey  = jax.random.split(key)
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
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))

        H_seq     = jnp.stack(H_list)       # (T_obs, m, N)
        y_obs_seq = jnp.stack(y_obs_list)   # (T_obs, m)

        # ── Initialise ensemble from the prior ────────────────────────────────
        # The ensemble mean is a noisy copy of the true IC; the spread is
        # governed by P0_sigma, which should reflect genuine initial uncertainty.
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_current_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)

        # ── Run EnKF smoother ─────────────────────────────────────────────────
        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices,   # fine-step indices of each observation
            H_seq,
            Q_fine,             # per fine-step process noise
            R,
            key,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
        # x_means, x_spreads: (total_fine_steps, N)

        # ── Trajectory summary plot ───────────────────────────────────────────
        t_fine_axis = t_eval_fine[1:]   # exclude t=0 to align with filter output
        _plot_trajectory_summary(
            t_ax       = t_fine_axis,
            x_true     = np.array(x_true_fine[1:]),
            x_est      = np.array(x_means),
            x_std      = np.array(x_spreads),
            ic_idx     = ic_idx,
            est_label  = "EnKF (numerical) mean",
            save_path  = os.path.join(
                workdir, "figures", config.wandb.name,
                f"trajectory_summary_enkf_numerical_ic_{ic_idx}.pdf",
            ),
            N          = N,
            dt_window  = DT_WINDOW,    # NEW
            obs_coords = obs_coords,   # NEW
        )

        # ── L2 error at window boundaries ─────────────────────────────────────
        # Extract filter mean and ground truth at each window boundary so the
        # metric is directly comparable to the DeepONet-based EnKF output.
        window_step_indices = np.array([
            round((w + 1) * DT_WINDOW / DT_FINE) - 1
            for w in range(num_windows)
        ])
        x_means_at_windows = x_means[window_step_indices]           # (num_windows, N)
        x_true_at_windows  = x_true_fine[window_step_indices + 1]   # (num_windows, N)

        l2_enkf     = jnp.linalg.norm(x_means_at_windows - x_true_at_windows) \
                    / jnp.linalg.norm(x_true_at_windows)
        mean_spread = float(jnp.mean(x_spreads))
        print(
            f"IC {ic_idx} | EnKF (numerical) L2 (window boundaries): {l2_enkf:.3e} "
            f"| Mean σ: {mean_spread:.3e}"
        )

    # ── Batch L2 deliberately omitted ────────────────────────────────────────
    # Running the RK4 propagator for every ensemble member and every IC in the
    # large evaluation pool (B × N_ens solves per window, up to max_additions
    # windows) offers no vectorisation shortcut analogous to the vmapped network
    # forward pass and is prohibitively expensive.  Use evaluate_with_enkf (the
    # DeepONet-based mode) for batch-averaged L2 comparisons.
    logging.info(
        "Batch L2 evaluation skipped for numerical EnKF "
        "(no vmapped network shortcut available).  "
        "Use evaluate_with_enkf for batch-averaged comparisons."
    )
