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

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import h5py

from jaxpi.utils import restore_checkpoint
import examples.l96_f.models as models
from examples.l96_f.utils import (
 	    build_obs_schedule,
 	    scale_inflation_for_fine_steps,
 	    scale_Q_for_fine_steps,
        steps_per_window_exact,
 	)

from examples.l96_f.kf import (
    init_ensemble,
    run_enkf_smoother,
    run_enkf_smoother_route_b,
    run_enkf_smoother_rtpp,
)



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



def _binned_spread_skill(
    rmss: np.ndarray,
    rmse: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin raw (RMSS, RMSE) pairs into `n_bins` equal-population bins
    (deciles by default) over RMSS. Raw per-(IC, window) spread/skill pairs
    form an unreadable cloud; binning is the standard fix.

    Returns bin_rmss_mean, bin_rmse_mean, bin_rmse_std, bin_counts,
    each shape (n_bins,) (fewer if some bins end up empty).
    """
    rmss = np.asarray(rmss).ravel()
    rmse = np.asarray(rmse).ravel()
    order = np.argsort(rmss)
    rmss_sorted, rmse_sorted = rmss[order], rmse[order]

    bin_rmss_mean, bin_rmse_mean, bin_rmse_std, bin_counts = [], [], [], []
    for idx in np.array_split(np.arange(len(rmss_sorted)), n_bins):
        if idx.size == 0:
            continue
        bin_rmss_mean.append(rmss_sorted[idx].mean())
        bin_rmse_mean.append(rmse_sorted[idx].mean())
        bin_rmse_std.append(rmse_sorted[idx].std())
        bin_counts.append(idx.size)

    return (np.array(bin_rmss_mean), np.array(bin_rmse_mean),
            np.array(bin_rmse_std), np.array(bin_counts))

# ── PI + DD propagators × Multiplicative + Route B inflation (3-way) ────────
#
# Three concurrently-evaluated strategies, all driven by the
# same fast jit(vmap(...)) batching pattern `build_batched_enkf_compare`
# already uses (no per-IC Python loop):
#
#   1. DD propagator + classic multiplicative inflation   ("DD")
#   2. PI propagator + classic multiplicative inflation   ("PI classic")
#   3. PI propagator + Route B residual-scaled inflation  ("PI Route B")
#
# Strategies 2 and 3 share the PI propagator (only the *filter* differs,
# exactly as in `_evaluate_batch_enkf_pi_compare`); strategy 1 uses its own
# DD propagator. All three share the same per-IC noisy observation draw and
# the same initial ensemble, so the comparison isolates propagator choice
# and inflation strategy rather than differing noise realizations.

# ── Multi-GPU batch execution helper ─────────────────────────────────────────
#
# Every EnKF batch computation here is embarrassingly parallel over the IC
# axis (each trajectory's filter run is independent of every other one).
# `_device_parallel` is a drop-in replacement for the
# `jax.jit(jax.vmap(fn, in_axes=...))` pattern used throughout: it splits the
# batch evenly (zero-padding up to a multiple of `jax.local_device_count()`)
# across every locally visible device via `jax.pmap`, runs the *same*
# vmapped closure per device, and reassembles outputs back into the original
# (B, ...) layout. On a single-device host it falls back to plain
# jit(vmap(...)), so behaviour/numerics are identical -- only the
# distribution across devices changes. `fn` itself (filtering, propagation,
# reconstruction) is passed through untouched.

def _device_parallel(fn, in_axes, static_broadcasted_argnums=()):
    n_devices = jax.local_device_count()
    vmapped_fn = jax.vmap(fn, in_axes=in_axes)

    if n_devices <= 1:
        return jax.jit(vmapped_fn, static_argnums=static_broadcasted_argnums)

    pmapped_fn = jax.pmap(
        vmapped_fn,
        in_axes=in_axes,
        static_broadcasted_argnums=static_broadcasted_argnums,
    )

    batched_idx = [
        i for i, ax in enumerate(in_axes)
        if ax is not None and i not in static_broadcasted_argnums
    ]

    def _shard(x):
        B = x.shape[0]
        per_device = -(-B // n_devices)          # ceil division
        pad = per_device * n_devices - B
        if pad > 0:
            x = jnp.concatenate(
                [x, jnp.zeros((pad,) + x.shape[1:], dtype=x.dtype)], axis=0
            )
        return x.reshape((n_devices, per_device) + x.shape[1:])

    def _unshard(x, B):
        x = x.reshape((-1,) + x.shape[2:])
        return x[:B]

    def wrapped(*args):
        B = args[batched_idx[0]].shape[0]
        sharded_args = tuple(
            _shard(a) if i in batched_idx else a
            for i, a in enumerate(args)
        )
        out = pmapped_fn(*sharded_args)
        return jax.tree_util.tree_map(lambda x: _unshard(x, B), out)

    return wrapped

def build_batched_enkf_3way(
    predict_fn_dd, update_fn_dd,
    predict_fn_cl, update_fn_cl,
    predict_fn_rb, update_fn_rb,
    N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
    Q0, alpha_rb, beta_rb, n_quad_rb,
    dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch,
):
    """
    Three-strategy counterpart of `build_batched_enkf_compare`. Builds ONE
    jit(vmap(...)) closure that, for every IC in the batch, runs all three
    strategies against the SAME noisy observation draw and the SAME initial
    ensemble:
 
      1. DD propagator + classic inflation   -- run_enkf_smoother
      2. PI propagator + classic inflation   -- run_enkf_smoother
      3. PI propagator + Route B inflation   -- run_enkf_smoother_route_b
 
    The Route B filter call is taken directly from the `kf.py` dependencies
    used by `_evaluate_batch_enkf_pi_compare`; only its execution is
    rewired onto the concurrent vmap+jit batching strategy that
    `build_batched_enkf_compare` already uses for PI vs DD.
    """
    from examples.l96_f.kf import init_ensemble, run_enkf_smoother, run_enkf_smoother_route_b
 
    def process_single_ic(key_ic, u_true, F_i, x_true_at_obs, dynamic_vars_static, specify_obs_idx_static):
        T_obs = x_true_at_obs.shape[0]
        keys_t = jax.random.split(key_ic, T_obs)
 
        # 1. Vectorized observation sequence generation, shared by all 3 strategies
        def single_obs(k, x_t):
            k1, k2 = jax.random.split(k)
            # Static conditions evaluated at JIT-compile time
            if (not specify_obs_idx_static) and dynamic_vars_static:
                idx_vars = jax.random.choice(k1, N, shape=(m,), replace=False)
            else:
                idx_vars = obs_indices
 
            H = jnp.zeros((m, N)).at[jnp.arange(m), idx_vars].set(1.0)
            H_aug = jnp.pad(H, ((0, 0), (0, 1)), mode='constant')
            noise = sigma_obs * jax.random.normal(k2, shape=(m,))
            return H_aug, x_t[idx_vars] + noise, idx_vars
 
        H_seq, y_obs_seq, idx_vars_seq = jax.vmap(single_obs)(keys_t, x_true_at_obs)
 
        # 2. Shared initial ensemble
        k1, k2, k3 = jax.random.split(key_ic, 3)
        x0_hat_40 = u_true + P0_sigma * jax.random.normal(k2, shape=(N,))
        x0_hat_aug = jnp.concatenate([x0_hat_40, jnp.array([F_i])])
        ensemble0 = init_ensemble(x0_hat_aug, P0, N_ens, k3)
 
        # 3. All three estimators run concurrently on the exact same
        #    noise/observations/initial ensemble.
        x_means_dd, x_spreads_dd, prior_means_dd = run_enkf_smoother(
            predict_fn_dd, update_fn_dd,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, alpha_fine, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )
 
        x_means_cl, x_spreads_cl, prior_means_cl = run_enkf_smoother(
            predict_fn_cl, update_fn_cl,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, alpha_fine, R, key_ic, total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window,
        )
 
        x_means_rb, x_spreads_rb, prior_means_rb, q_scale_rb = run_enkf_smoother_route_b(
            predict_fn_rb, update_fn_rb,
            ensemble0, y_obs_seq, obs_step_indices_batch,
            H_seq, Q0=Q0, alpha=alpha_rb, beta=beta_rb, R=R, key=key_ic,
            total_fine_steps=total_fine_steps_batch,
            dt_fine=dt_fine, dt_window=dt_window, n_quad=n_quad_rb,
        )
 
        return (x_means_dd, x_spreads_dd, prior_means_dd,
                x_means_cl, x_spreads_cl, prior_means_cl,
                x_means_rb, x_spreads_rb, prior_means_rb, q_scale_rb,
                y_obs_seq, idx_vars_seq)
 
    # Distribute the batch of independent ICs across every locally visible
    # GPU (falls back to single-device jit(vmap(...)) automatically).
    # Freeze the boolean flags at compile time to avoid JAX tracer errors on if/else
    return _device_parallel(
        process_single_ic,
        in_axes=(0, 0, 0, 0, None, None),
        static_broadcasted_argnums=(4, 5),
    )
 
def _plot_trajectory_summary_compare_enkf_3way(
    t_ax:       np.ndarray,        # (T,)   time axis
    x_true:     np.ndarray,        # (T, N) ground-truth state
    x_est_dd:   np.ndarray,        # (T, N) DD + multiplicative-inflation EnKF mean
    x_std_dd:   np.ndarray | None, # (T, N) DD ensemble std, or None
    x_est_cl:   np.ndarray,        # (T, N) PI + classic multiplicative-inflation EnKF mean
    x_std_cl:   np.ndarray | None, # (T, N) PI-classic ensemble std, or None
    x_est_rb:   np.ndarray,        # (T, N) PI + Route B EnKF mean
    x_std_rb:   np.ndarray | None, # (T, N) PI-Route-B ensemble std, or None
    ic_idx:     int,
    F_val:      float,
    save_path:  str,
    N:          int = 40,
    dt_window:  float | None = None,
    obs_coords: list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Trajectory-summary PDF comparing all 3 strategies against the ground
    truth for a single IC. Same layout as
    `_plot_trajectory_summary_compare_enkf` (PI vs DD) and
    `_plot_trajectory_summary_compare_enkf_rb` (Classic vs Route B):
    top panel is the mean |error| vs time for every strategy, followed by
    one panel per state variable with truth, each strategy's mean, ±1σ
    ensemble-spread bands, and assimilated-observation markers.
    """
    x_true   = np.asarray(x_true)
    x_est_dd = np.asarray(x_est_dd)
    x_est_cl = np.asarray(x_est_cl)
    x_est_rb = np.asarray(x_est_rb)
    x_std_dd = np.asarray(x_std_dd) if x_std_dd is not None else None
    x_std_cl = np.asarray(x_std_cl) if x_std_cl is not None else None
    x_std_rb = np.asarray(x_std_rb) if x_std_rb is not None else None
 
    mean_abs_err_dd = np.abs(x_true - x_est_dd).mean(axis=1)
    mean_abs_err_cl = np.abs(x_true - x_est_cl).mean(axis=1)
    mean_abs_err_rb = np.abs(x_true - x_est_rb).mean(axis=1)
 
    n_var_rows = N // 2
 
    # ── Pre-compute window-boundary times ────────────────────────────────
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(first_k * dt_window,
                                      t_max + 1e-12 * dt_window,
                                      dt_window)
    else:
        window_boundaries = np.array([])
 
    # ── Pre-compute per-variable observation times ────────────────────────
    if obs_coords is not None:
        obs_by_var: dict[int, list[tuple[float, float]]] = {}
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(var_idx, []).append((obs_t, obs_val))
        obs_by_var = {k: sorted(v, key=lambda x: x[0])
                      for k, v in obs_by_var.items()}
    else:
        obs_by_var = {}
 
    # ── Figure & GridSpec ────────────────────────────────────────────────
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
 
    # ── Top panel: mean absolute error vs time (all 3 strategies) ─────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err_dd, color="#FF8C00", linewidth=1.6,
                label="DD + Mult. Infl.: Mean |error|")
    ax_top.plot(t_ax, mean_abs_err_cl, color="#2196F3", linewidth=1.6,
                label="PI + Mult. Infl.: Mean |error|")
    ax_top.plot(t_ax, mean_abs_err_rb, color="#8E24AA", linewidth=1.6,
                label="PI + Route B Infl.: Mean |error|")
 
    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.8, alpha=0.55,
                       label="Window boundary" if wb == window_boundaries[0] else None)
 
    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables  "
        f"(DD vs PI-classic vs PI-Route B)",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=9)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ── Colour palette (kept consistent with the rest of the codebase) ────
    TRUTH_COLOR = "#37474F"   # dark blue-grey — ground truth
    DD_COLOR, DD_BAND = "#FF8C00", "#FFCC80"   # orange — DD
    CL_COLOR, CL_BAND = "#2196F3", "#90CAF9"   # blue   — PI classic
    RB_COLOR, RB_BAND = "#8E24AA", "#CE93D8"   # purple — PI Route B
    OBS_COLOR   = "#E53935"   # red — observation markers
 
    # ── Per-variable panels ──────────────────────────────────────────────
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--",
                       linewidth=0.6, alpha=0.45)
 
        # Ground truth
        ax.plot(t_ax, x_true[:, i],
                color=TRUTH_COLOR, linewidth=1.0, label="Truth")
 
        # DD + multiplicative inflation
        ax.plot(t_ax, x_est_dd[:, i],
                color=DD_COLOR, linewidth=1.0, linestyle="--", label="DD + Mult.")
        if x_std_dd is not None:
            ax.fill_between(
                t_ax, x_est_dd[:, i] - x_std_dd[:, i], x_est_dd[:, i] + x_std_dd[:, i],
                color=DD_BAND, alpha=0.30, linewidth=0, label="DD ±1σ",
            )
 
        # PI + classic multiplicative inflation
        ax.plot(t_ax, x_est_cl[:, i],
                color=CL_COLOR, linewidth=1.0, linestyle=":", label="PI + Mult.")
        if x_std_cl is not None:
            ax.fill_between(
                t_ax, x_est_cl[:, i] - x_std_cl[:, i], x_est_cl[:, i] + x_std_cl[:, i],
                color=CL_BAND, alpha=0.30, linewidth=0, label="PI classic ±1σ",
            )
 
        # PI + Route B inflation
        ax.plot(t_ax, x_est_rb[:, i],
                color=RB_COLOR, linewidth=1.0, linestyle="-.", label="PI + Route B")
        if x_std_rb is not None:
            ax.fill_between(
                t_ax, x_est_rb[:, i] - x_std_rb[:, i], x_est_rb[:, i] + x_std_rb[:, i],
                color=RB_BAND, alpha=0.30, linewidth=0, label="PI Route B ±1σ",
            )
 
        # Observation markers
        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])
            ax.scatter(obs_times_i, obs_vals_i,
                       marker="x", s=25, linewidths=0.9,
                       color=OBS_COLOR, zorder=5,
                       label="Observation" if i == min(obs_by_var) else None)
 
        ax.set_title(f"$x_{{{i}}}$", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
 
        if row == 1 + n_var_rows - 1:
            ax.set_xlabel("t", fontsize=8)
        if col == 0:
            ax.set_ylabel("state", fontsize=8)
 
        if i == 0:
            ax.legend(fontsize=5.8, loc="upper right",
                      handlelength=1.2, framealpha=0.7, ncol=2)
 
    fig.suptitle(
        f"Trajectory summary — IC {ic_idx} (F = {F_val:.2f}) |  "
        f"DD vs PI-classic vs PI-Route B",
        fontsize=13, fontweight="bold", y=1.002,
    )
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"3-way EnKF trajectory summary for IC {ic_idx} saved to: {save_path}")

def _plot_erf_compare_3way(
    obs_times:   np.ndarray,   # (T_obs,)
    erf_mean_dd: np.ndarray,   # (T_obs,)
    erf_std_dd:  np.ndarray,   # (T_obs,)
    erf_mean_cl: np.ndarray,   # (T_obs,)
    erf_std_cl:  np.ndarray,   # (T_obs,)
    erf_mean_rb: np.ndarray,   # (T_obs,)
    erf_std_rb:  np.ndarray,   # (T_obs,)
    n_traj:      int,
    title:       str,
    save_path:   str,
) -> None:
    """
    ERF comparison for all three strategies on ONE set of axes.  Unlike the
    RMSE comparison below (two curves per strategy), ERF is a single curve
    per strategy, so all three stay readable together without a pairwise
    split -- 3 lines + 3 light ±1σ bands, same visual density as the
    original two-strategy `_plot_erf_compare` / `_plot_erf_compare_rb`.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
 
    series = [
        ("DD + Mult. Infl.",   erf_mean_dd, erf_std_dd, "#FF8C00", "o"),
        ("PI + Mult. Infl.",   erf_mean_cl, erf_std_cl, "#2196F3", "s"),
        ("PI + Route B Infl.", erf_mean_rb, erf_std_rb, "#8E24AA", "^"),
    ]
    for label, mean, std, color, marker in series:
        ax.plot(obs_times, mean, color=color, linewidth=2.0, marker=marker,
                markersize=4, label=f"{label}  (n = {n_traj} trajectories)")
        ax.fill_between(obs_times, mean - std, mean + std,
                         color=color, alpha=0.15, linewidth=0)
 
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
               label="ERF = 1  (no reduction)")
 
    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"ERF comparison plot (3-way) saved to: {save_path}")
 
def _plot_rmse_comparison_3way(
    obs_times: np.ndarray,
    prior_rmse_mean_dd: np.ndarray, prior_rmse_std_dd: np.ndarray,
    post_rmse_mean_dd:  np.ndarray, post_rmse_std_dd:  np.ndarray,
    prior_rmse_mean_cl: np.ndarray, prior_rmse_std_cl: np.ndarray,
    post_rmse_mean_cl:  np.ndarray, post_rmse_std_cl:  np.ndarray,
    prior_rmse_mean_rb: np.ndarray, prior_rmse_std_rb: np.ndarray,
    post_rmse_mean_rb:  np.ndarray, post_rmse_std_rb:  np.ndarray,
    sigma_obs: float,
    n_traj:    int,
    title:     str,
    save_path: str,
) -> None:
    """
    Prior/posterior RMSE comparison across all three strategies, laid out
    as 3 PAIRWISE panels stacked in one PDF:
 
        Panel 1 -- DD + Mult. Infl.   vs  PI + Mult. Infl.
        Panel 2 -- PI + Mult. Infl.   vs  PI + Route B Infl.
        Panel 3 -- DD + Mult. Infl.   vs  PI + Route B Infl.
 
    Overlaying all three strategies on ONE axes would put 6 lines + 6
    shaded ±1σ bands (prior & posterior x 3 strategies) on a single plot;
    the pairwise split keeps each panel at the same visual density as the
    original two-strategy `_plot_rmse_comparison_dd_pi` /
    `_plot_rmse_comparison_rb`, while still covering every strategy pair.
    """
    fig = plt.figure(figsize=(9, 15))
    gs  = gridspec.GridSpec(3, 1, hspace=0.5)
 
    def _pair_panel(
        ax,
        name_a, prior_a, prior_a_s, post_a, post_a_s, c_prior_a, c_post_a, ls_a,
        name_b, prior_b, prior_b_s, post_b, post_b_s, c_prior_b, c_post_b, ls_b,
    ):
        ax.plot(obs_times, prior_a, color=c_prior_a, linewidth=2.0, marker="o",
                markersize=4, linestyle=ls_a, label=f"{name_a} prior RMSE  (n = {n_traj})")
        #ax.fill_between(obs_times, prior_a - prior_a_s, prior_a + prior_a_s,
        #                 color=c_prior_a, alpha=0.15, linewidth=0)
        ax.plot(obs_times, post_a, color=c_post_a, linewidth=2.0, marker="s",
                markersize=4, linestyle=ls_a, label=f"{name_a} posterior RMSE  (n = {n_traj})")
        #ax.fill_between(obs_times, post_a - post_a_s, post_a + post_a_s,
        #                 color=c_post_a, alpha=0.15, linewidth=0)
 
        ax.plot(obs_times, prior_b, color=c_prior_b, linewidth=2.0, marker="o",
                markersize=4, linestyle=ls_b, label=f"{name_b} prior RMSE  (n = {n_traj})")
        #ax.fill_between(obs_times, prior_b - prior_b_s, prior_b + prior_b_s,
        #                 color=c_prior_b, alpha=0.08, linewidth=0)
        ax.plot(obs_times, post_b, color=c_post_b, linewidth=2.0, marker="s",
                markersize=4, linestyle=ls_b, label=f"{name_b} posterior RMSE  (n = {n_traj})")
        #ax.fill_between(obs_times, post_b - post_b_s, post_b + post_b_s,
        #                 color=c_post_b, alpha=0.08, linewidth=0)
 
        ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.6,
                   label=f"Measurement noise  σ_obs = {sigma_obs}")
 
        ax.set_yscale("log")
        ax.set_xlabel("Observation time  t", fontsize=10)
        ax.set_ylabel("RMSE  (log scale)", fontsize=10)
        ax.set_title(f"{name_a}  vs  {name_b}", fontsize=11)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # Panel 1 — DD vs PI classic (same colour roles as _plot_rmse_comparison_dd_pi)
    ax1 = fig.add_subplot(gs[0])
    _pair_panel(
        ax1,
        "DD + Mult. Infl.", prior_rmse_mean_dd, prior_rmse_std_dd, post_rmse_mean_dd, post_rmse_std_dd,
        "#2196F3", "#FF5722", "--",
        "PI + Mult. Infl.", prior_rmse_mean_cl, prior_rmse_std_cl, post_rmse_mean_cl, post_rmse_std_cl,
        "#0A36C7", "#A30005", "-",
    )
 
    # Panel 2 — PI classic vs PI Route B (same colour roles as _plot_rmse_comparison_rb)
    ax2 = fig.add_subplot(gs[1])
    _pair_panel(
        ax2,
        "PI + Mult. Infl.",   prior_rmse_mean_cl, prior_rmse_std_cl, post_rmse_mean_cl, post_rmse_std_cl,
        "#0A36C7", "#A30005", "-",
        "PI + Route B Infl.", prior_rmse_mean_rb, prior_rmse_std_rb, post_rmse_mean_rb, post_rmse_std_rb,
        "#8E24AA", "#EC407A", "--",
    )
 
    # Panel 3 — DD vs PI Route B (new pairing, kept visually consistent)
    ax3 = fig.add_subplot(gs[2])
    _pair_panel(
        ax3,
        "DD + Mult. Infl.",   prior_rmse_mean_dd, prior_rmse_std_dd, post_rmse_mean_dd, post_rmse_std_dd,
        "#2196F3", "#FF5722", "-",
        "PI + Route B Infl.", prior_rmse_mean_rb, prior_rmse_std_rb, post_rmse_mean_rb, post_rmse_std_rb,
        "#8E24AA", "#EC407A", "--",
    )
 
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"RMSE comparison plot (3-way, pairwise) saved to: {save_path}")
 
def _plot_calibration_compare_3way(
    window_idx:     np.ndarray,
    dt_window:      float,
    spread_dd:      np.ndarray, rmse_dd: np.ndarray,
    spread_cl:      np.ndarray, rmse_cl: np.ndarray,
    spread_rb:      np.ndarray, rmse_rb: np.ndarray,
    spread_dd_raw:  np.ndarray, rmse_dd_raw: np.ndarray,
    spread_cl_raw:  np.ndarray, rmse_cl_raw: np.ndarray,
    spread_rb_raw:  np.ndarray, rmse_rb_raw: np.ndarray,
    title:          str,
    save_path:      str,
    n_bins:         int = 10,
) -> None:
    """
    Calibration comparison across all three strategies, one PDF with 4
    stacked panels:
      1. DD           — RMS ensemble spread vs EnKF RMSE (simulation time).
      2. PI classic    — same, directly below.
      3. PI Route B    — same, directly below. This is the simulation-time
         panel added for the 3rd strategy, alongside the two panels that
         already existed for DD and PI classic.
      4. Binned spread-skill diagram comparing ALL 3 strategies on the same
         axes, pooled over every (IC, window) in the batch.
    Mirrors `_plot_calibration_compare` / `_plot_calibration_compare_rb`.
    """
    fig = plt.figure(figsize=(9, 16.5))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1.3], hspace=0.6)
 
    def _timeseries_panel(ax, spread, rmse, c_spread, c_rmse, label):
        ax.plot(window_idx, spread, marker="^", markersize=4, linewidth=1.8,
                linestyle="-", color=c_spread, label=f"{label} RMS ensemble σ")
        ax.plot(window_idx, rmse, marker="s", markersize=4, linewidth=1.8,
                linestyle="--", color=c_rmse, label=f"{label} EnKF RMSE")
        ax.set_yscale("log")
        ax.set_xlabel("Window index", fontsize=11)
        ax.set_ylabel("Log scale", fontsize=11)
        ax.set_title(f"{label}: ensemble spread vs RMSE  (Simulation time)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
        ax_time = ax.twiny()
        ax_time.set_xlim(ax.get_xlim())
        ax_time.set_xticks(window_idx)
        ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx],
                                 fontsize=7, rotation=45, ha="left")
        ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=9)
 
    ax_dd = fig.add_subplot(gs[0])
    _timeseries_panel(ax_dd, spread_dd, rmse_dd, "#8BC34A", "#FF8A65", "DD + Mult. Infl.")
 
    ax_cl = fig.add_subplot(gs[1])
    _timeseries_panel(ax_cl, spread_cl, rmse_cl, "#4CAF50", "#FF5722", "PI + Mult. Infl.")
 
    # Simulation-time panel added for the 3rd strategy (PI + Route B)
    ax_rb = fig.add_subplot(gs[2])
    _timeseries_panel(ax_rb, spread_rb, rmse_rb, "#AB47BC", "#EC407A", "PI + Route B Infl.")
 
    ax_bin = fig.add_subplot(gs[3])
    rmss_dd_b, rmse_dd_b, rmse_dd_s, _ = _binned_spread_skill(spread_dd_raw, rmse_dd_raw, n_bins)
    rmss_cl_b, rmse_cl_b, rmse_cl_s, _ = _binned_spread_skill(spread_cl_raw, rmse_cl_raw, n_bins)
    rmss_rb_b, rmse_rb_b, rmse_rb_s, _ = _binned_spread_skill(spread_rb_raw, rmse_rb_raw, n_bins)
 
    lim_hi = 1.1 * max(
        rmss_dd_b.max(), rmse_dd_b.max(),
        rmss_cl_b.max(), rmse_cl_b.max(),
        rmss_rb_b.max(), rmse_rb_b.max(),
    )
    ax_bin.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.4,
                color="#37474F", label="1:1 (perfect calibration)")
    ax_bin.errorbar(rmss_dd_b, rmse_dd_b, yerr=rmse_dd_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#FF8C00",
                     label=f"DD + Mult. Infl. ({n_bins}-bin)")
    ax_bin.errorbar(rmss_cl_b, rmse_cl_b, yerr=rmse_cl_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#2196F3",
                     label=f"PI + Mult. Infl. ({n_bins}-bin)")
    ax_bin.errorbar(rmss_rb_b, rmse_rb_b, yerr=rmse_rb_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#8E24AA",
                     label=f"PI + Route B Infl. ({n_bins}-bin)")
 
    ax_bin.set_xlim(0, lim_hi); ax_bin.set_ylim(0, lim_hi)
    ax_bin.set_xlabel("RMS ensemble spread (RMSS)", fontsize=11)
    ax_bin.set_ylabel("RMSE of ensemble mean", fontsize=11)
    ax_bin.set_title(f"Binned spread-skill  ({n_bins} equal-population bins, "
                      f"pooled over all ICs × windows, all 3 strategies)", fontsize=12)
    ax_bin.legend(fontsize=8.5)
    ax_bin.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bin.set_aspect("equal", adjustable="box")
 
    fig.suptitle(title, fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Calibration comparison plot (3-way) saved to: {save_path}")

def _evaluate_batch_enkf_3way(
    model_dd, params_dd, predict_fn_dd, update_fn_dd,
    model_pi, params_pi,
    predict_fn_cl, update_fn_cl,
    predict_fn_rb, update_fn_rb,
    t_star_window,
    u_test:            np.ndarray,   # (num_ics, num_test_pts, N)
    t_test:            np.ndarray,   # (num_test_pts,)
    F_test:            np.ndarray,   # (num_ics,)
    alpha_fine, Q0, alpha_rb, beta_rb, n_quad_rb,
    P0, R, obs_indices,
    N_ens:             int,
    obs_every_n:       int,
    sigma_obs:         float,
    P0_sigma:          float,
    dynamic_vars:      bool,
    specify_obs_idx:   bool,
    obs_idx_list,
    dt_window:         float,
    dt_fine:           float,
    dt_obs:            float,
    num_ics_eval:      int,
    enkf_batch_size:   int,
    batch_windows:     int,
    config, workdir: str,
) -> None:
    """
    Batch-averaged EnKF evaluation across all three strategies:
 
      1. DD propagator + classic multiplicative inflation
      2. PI propagator + classic multiplicative inflation
      3. PI propagator + Route B residual-scaled inflation
 
    Structurally this is `_evaluate_batch_enkf_dd_vs_pi` -- fully
    vmapped/JIT batch execution via `build_batched_enkf_3way`, no per-IC
    Python loop -- extended with the Route B filter logic from
    `_evaluate_batch_enkf_pi_compare`. Strategies 2 and 3 share the PI
    propagator (only their filter differs), so only two open-loop
    reference rollouts (PI, DD) are needed even though three EnKF curves
    are reported, exactly as in the two pipelines this combines.
    """
    B = min(num_ics_eval, enkf_batch_size, u_test.shape[0])
    N = model_pi.N
    logging.info(
        f"Computing batch EnKF 3-way comparison (DD / PI classic / PI Route B) "
        f"over B={B} trajectories from l96_forcing_test.h5 (N_ens={N_ens}) …"
    )
 
    u0_batch = u_test[:B, 0, :]
    dt_test  = float(t_test[1] - t_test[0])
 
    total_time_batch = batch_windows * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=dt_fine, dt_obs=dt_obs
    )
    obs_step_indices_batch = jnp.array(obs_step_indices_batch)
 
    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])
 
    fine_stride = int(round(dt_fine / dt_test))
    n_fine_pts  = total_fine_steps_batch * fine_stride + 1
 
    x_true_fine_batch   = u_test[:B, 0:n_fine_pts:fine_stride, :]
    x_true_at_obs_batch = x_true_fine_batch[:, obs_step_indices_batch + 1, :]
    window_step_indices = np.array([round((k + 1) * dt_window / dt_fine) - 1 for k in range(batch_windows)])
    m = len(obs_indices)
 
    # ── 1. Vmapped EnKF execution, all 3 strategies in ONE JIT call ───────
    seed = config.training.get("seed", 42)
    master_key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(master_key, B)
 
    batched_enkf = build_batched_enkf_3way(
        predict_fn_dd, update_fn_dd,
        predict_fn_cl, update_fn_cl,
        predict_fn_rb, update_fn_rb,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
        Q0, alpha_rb, beta_rb, n_quad_rb,
        dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch,
    )
 
    (batch_x_means_dd, batch_x_spreads_dd, batch_prior_means_dd,
     batch_x_means_cl, batch_x_spreads_cl, batch_prior_means_cl,
     batch_x_means_rb, batch_x_spreads_rb, batch_prior_means_rb, batch_q_scale_rb,
     _, _) = batched_enkf(
         keys_batch, u0_batch, F_test[:B], x_true_at_obs_batch,
         dynamic_vars, specify_obs_idx
    )
 
    # ── 2. Vectorized metric extraction (all 3 strategies) ────────────────
    def _rmse(a, b):
        return jnp.sqrt(jnp.mean((a - b) ** 2, axis=2))
 
    def _mean_std(a):
        return np.array(jnp.mean(a, axis=0)), np.array(jnp.std(a, axis=0))
 
    post_means_dd_obs  = batch_x_means_dd[:, obs_step_indices_batch, :N]
    post_means_cl_obs  = batch_x_means_cl[:, obs_step_indices_batch, :N]
    post_means_rb_obs  = batch_x_means_rb[:, obs_step_indices_batch, :N]
    prior_means_dd_obs = batch_prior_means_dd[:, :, :N]
    prior_means_cl_obs = batch_prior_means_cl[:, :, :N]
    prior_means_rb_obs = batch_prior_means_rb[:, :, :N]
 
    prior_rmse_dd_ic = _rmse(prior_means_dd_obs, x_true_at_obs_batch)
    post_rmse_dd_ic  = _rmse(post_means_dd_obs,  x_true_at_obs_batch)
    prior_rmse_cl_ic = _rmse(prior_means_cl_obs, x_true_at_obs_batch)
    post_rmse_cl_ic  = _rmse(post_means_cl_obs,  x_true_at_obs_batch)
    prior_rmse_rb_ic = _rmse(prior_means_rb_obs, x_true_at_obs_batch)
    post_rmse_rb_ic  = _rmse(post_means_rb_obs,  x_true_at_obs_batch)
 
    erf_dd_ic = prior_rmse_dd_ic / (post_rmse_dd_ic + 1e-12)
    erf_cl_ic = prior_rmse_cl_ic / (post_rmse_cl_ic + 1e-12)
    erf_rb_ic = prior_rmse_rb_ic / (post_rmse_rb_ic + 1e-12)
 
    erf_dd_mean, erf_dd_std = _mean_std(erf_dd_ic)
    erf_cl_mean, erf_cl_std = _mean_std(erf_cl_ic)
    erf_rb_mean, erf_rb_std = _mean_std(erf_rb_ic)
 
    prior_rmse_dd_mean, prior_rmse_dd_std = _mean_std(prior_rmse_dd_ic)
    post_rmse_dd_mean,  post_rmse_dd_std  = _mean_std(post_rmse_dd_ic)
    prior_rmse_cl_mean, prior_rmse_cl_std = _mean_std(prior_rmse_cl_ic)
    post_rmse_cl_mean,  post_rmse_cl_std  = _mean_std(post_rmse_cl_ic)
    prior_rmse_rb_mean, prior_rmse_rb_std = _mean_std(prior_rmse_rb_ic)
    post_rmse_rb_mean,  post_rmse_rb_std  = _mean_std(post_rmse_rb_ic)
 
    # Window-boundary processing
    x_true_at_windows = x_true_fine_batch[:, window_step_indices + 1, :]
    x_hat_dd_windows = batch_x_means_dd[:, window_step_indices, :N]
    x_hat_cl_windows = batch_x_means_cl[:, window_step_indices, :N]
    x_hat_rb_windows = batch_x_means_rb[:, window_step_indices, :N]
 
    den = jnp.linalg.norm(x_true_at_windows, axis=2) + 1e-12
    l2_enkf_dd = np.array(jnp.mean(jnp.linalg.norm(x_hat_dd_windows - x_true_at_windows, axis=2) / den, axis=0))
    l2_enkf_cl = np.array(jnp.mean(jnp.linalg.norm(x_hat_cl_windows - x_true_at_windows, axis=2) / den, axis=0))
    l2_enkf_rb = np.array(jnp.mean(jnp.linalg.norm(x_hat_rb_windows - x_true_at_windows, axis=2) / den, axis=0))
 
    rmse_dd_ic = _rmse(x_hat_dd_windows, x_true_at_windows)
    rmse_cl_ic = _rmse(x_hat_cl_windows, x_true_at_windows)
    rmse_rb_ic = _rmse(x_hat_rb_windows, x_true_at_windows)
    rmse_enkf_dd = np.array(jnp.mean(rmse_dd_ic, axis=0))
    rmse_enkf_cl = np.array(jnp.mean(rmse_cl_ic, axis=0))
    rmse_enkf_rb = np.array(jnp.mean(rmse_rb_ic, axis=0))
 
    spread_dd_ic = jnp.sqrt(jnp.mean(batch_x_spreads_dd[:, window_step_indices, :N] ** 2, axis=2))
    spread_cl_ic = jnp.sqrt(jnp.mean(batch_x_spreads_cl[:, window_step_indices, :N] ** 2, axis=2))
    spread_rb_ic = jnp.sqrt(jnp.mean(batch_x_spreads_rb[:, window_step_indices, :N] ** 2, axis=2))
    spread_dd = np.array(jnp.mean(spread_dd_ic, axis=0))
    spread_cl = np.array(jnp.mean(spread_cl_ic, axis=0))
    spread_rb = np.array(jnp.mean(spread_rb_ic, axis=0))
 
    rmse_dd_raw, rmse_cl_raw, rmse_rb_raw = (
        np.array(rmse_dd_ic.flatten()), np.array(rmse_cl_ic.flatten()), np.array(rmse_rb_ic.flatten())
    )
    spread_dd_raw, spread_cl_raw, spread_rb_raw = (
        np.array(spread_dd_ic.flatten()), np.array(spread_cl_ic.flatten()), np.array(spread_rb_ic.flatten())
    )
 
    # Dense metrics
    x_true_fine_tail = x_true_fine_batch[:, 1:, :]
    den_dense = jnp.linalg.norm(x_true_fine_tail, axis=2) + 1e-12
    l2_enkf_dd_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_dd[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))
    l2_enkf_cl_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_cl[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))
    l2_enkf_rb_dense = np.array(jnp.mean(jnp.linalg.norm(batch_x_means_rb[:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0))
 
    # Route B inflation-scale diagnostic (mean over ensemble, then over ICs)
    q_scale_step_mean = jnp.mean(batch_q_scale_rb, axis=2)          # (B, total_fine_steps_batch)
    q_scale_mean = np.array(jnp.mean(q_scale_step_mean, axis=0))
    q_scale_std  = np.array(jnp.std(q_scale_step_mean, axis=0))
 
    # ── 3. Open-loop rollouts (PI and DD propagators; unchanged & fast) ───
    #    Strategies 2 & 3 share the PI propagator, so only ONE PI open-loop
    #    reference is needed alongside the DD one, even for 3 EnKF curves.
    u0_batch_j = jnp.array(u0_batch)
    predict_full_pi = _device_parallel(
        lambda u: model_pi.x_pred_fn(params_pi, u, t_star_window), in_axes=(0,)
    )
    predict_full_dd = _device_parallel(
        lambda u: model_dd.x_pred_fn(params_dd, u, t_star_window), in_axes=(0,)
    )
 
    x_pred_dense_pi_list, x_pred_dense_dd_list = [], []
    u_current_pi = u_current_dd = jnp.concatenate([u0_batch_j, F_test[:B, None]], axis=-1)
 
    for k in range(batch_windows):
        x_win_pi = predict_full_pi(u_current_pi)
        x_win_dd = predict_full_dd(u_current_dd)
 
        if k == 0:
            x_pred_dense_pi_list.append(x_win_pi)
            x_pred_dense_dd_list.append(x_win_dd)
        else:
            x_pred_dense_pi_list.append(x_win_pi[:, 1:, :])
            x_pred_dense_dd_list.append(x_win_dd[:, 1:, :])
 
        u_current_pi = jnp.concatenate([x_win_pi[:, -1, :], F_test[:B, None]], axis=-1)
        u_current_dd = jnp.concatenate([x_win_dd[:, -1, :], F_test[:B, None]], axis=-1)
 
    x_pred_dense_pi = jnp.concatenate(x_pred_dense_pi_list, axis=1)
    x_pred_dense_dd = jnp.concatenate(x_pred_dense_dd_list, axis=1)
    total_steps_ol = x_pred_dense_pi.shape[1]
    x_ref_dense_ol = jnp.array(u_test[:B, :total_steps_ol, :])
 
    denom_ol = jnp.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
    l2_ol_pi = np.array(jnp.mean(jnp.linalg.norm(x_pred_dense_pi - x_ref_dense_ol, axis=2) / denom_ol, axis=0))
    l2_ol_dd = np.array(jnp.mean(jnp.linalg.norm(x_pred_dense_dd - x_ref_dense_ol, axis=2) / denom_ol, axis=0))
 
    t_eval_ol = t_test[:total_steps_ol]
    t_dense_fine = np.arange(1, total_fine_steps_batch + 1) * dt_fine
 
    logging.info(
        f"  [batch] Final-timestep mean L2 -> "
        f"PI open-loop: {float(l2_ol_pi[-1]):.3e} | DD open-loop: {float(l2_ol_dd[-1]):.3e} | "
        f"DD+EnKF: {l2_enkf_dd_dense[-1]:.3e} | PI classic+EnKF: {l2_enkf_cl_dense[-1]:.3e} | "
        f"PI Route B+EnKF: {l2_enkf_rb_dense[-1]:.3e}"
    )
    logging.info(
        f"  [batch] Final-window mean L2 (boundary-only, for reference) -> "
        f"DD+EnKF: {l2_enkf_dd[-1]:.3e} | PI classic+EnKF: {l2_enkf_cl[-1]:.3e} | "
        f"PI Route B+EnKF: {l2_enkf_rb[-1]:.3e}"
    )
 
    # ── Plotting ────────────────────────────────────────────────────────
    save_dir = os.path.join(workdir, "figures", "three_way")
 
    # Plot 1 — dense per-timestamp L2: all 3 EnKF strategies + both open-loops,
    #          all on the same graph.
    curves = {
        "PI Open-loop":        (np.array(t_eval_ol), l2_ol_pi),
        "DD Open-loop":        (np.array(t_eval_ol), l2_ol_dd),
        "DD + Mult. Infl.":    (t_dense_fine,        l2_enkf_dd_dense),
        "PI + Mult. Infl.":    (t_dense_fine,        l2_enkf_cl_dense),
        "PI + Route B Infl.":  (t_dense_fine,        l2_enkf_rb_dense),
    }
    colors = {
        "PI Open-loop":        "#90CAF9",
        "DD Open-loop":        "#FFCC80",
        "DD + Mult. Infl.":    "#FF8C00",
        "PI + Mult. Infl.":    "#2196F3",
        "PI + Route B Infl.":  "#8E24AA",
    }
    _plot_l2_per_timestep(
        curves    = curves,
        title     = f"EnKF vs open-loop: mean relative L2 per timestep  (3-way, B={B})",
        save_path = os.path.join(save_dir, "batch_l2_per_timestep_3way.pdf"),
        colors    = colors,
    )
 
    # Plot 2 — calibration: spread vs RMSE per strategy (simulation time) +
    #          combined binned spread-skill diagram, all 3 strategies.
    _plot_calibration_compare_3way(
        window_idx = np.arange(1, batch_windows + 1),
        dt_window  = dt_window,
        spread_dd = spread_dd, rmse_dd = rmse_enkf_dd,
        spread_cl = spread_cl, rmse_cl = rmse_enkf_cl,
        spread_rb = spread_rb, rmse_rb = rmse_enkf_rb,
        spread_dd_raw = spread_dd_raw, rmse_dd_raw = rmse_dd_raw,
        spread_cl_raw = spread_cl_raw, rmse_cl_raw = rmse_cl_raw,
        spread_rb_raw = spread_rb_raw, rmse_rb_raw = rmse_rb_raw,
        title      = f"Calibration: ensemble spread vs RMSE  (3-way, B={B}, N_ens={N_ens})",
        save_path  = os.path.join(save_dir, "batch_calibration_enkf_3way.pdf"),
    )
 
    # Plot 3 — Error Reduction Factor, all 3 strategies on the same graph.
    _plot_erf_compare_3way(
        obs_times   = obs_times_batch,
        erf_mean_dd = erf_dd_mean, erf_std_dd = erf_dd_std,
        erf_mean_cl = erf_cl_mean, erf_std_cl = erf_cl_std,
        erf_mean_rb = erf_rb_mean, erf_std_rb = erf_rb_std,
        n_traj      = B,
        title       = (
            f"EnKF Error Reduction Factor per observation time  (3-way)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path   = os.path.join(save_dir, "batch_erf_enkf_3way.pdf"),
    )
 
    # Plot 4 — prior / posterior RMSE, 3 pairwise panels (avoids band clutter).
    _plot_rmse_comparison_3way(
        obs_times = obs_times_batch,
        prior_rmse_mean_dd = prior_rmse_dd_mean, prior_rmse_std_dd = prior_rmse_dd_std,
        post_rmse_mean_dd  = post_rmse_dd_mean,  post_rmse_std_dd  = post_rmse_dd_std,
        prior_rmse_mean_cl = prior_rmse_cl_mean, prior_rmse_std_cl = prior_rmse_cl_std,
        post_rmse_mean_cl  = post_rmse_cl_mean,  post_rmse_std_cl  = post_rmse_cl_std,
        prior_rmse_mean_rb = prior_rmse_rb_mean, prior_rmse_std_rb = prior_rmse_rb_std,
        post_rmse_mean_rb  = post_rmse_rb_mean,  post_rmse_std_rb  = post_rmse_rb_std,
        sigma_obs = sigma_obs, n_traj = B,
        title = (
            f"EnKF prior vs posterior RMSE  (3-way, pairwise)\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path = os.path.join(save_dir, "batch_rmse_enkf_3way.pdf"),
    )
 
    # Plot 5 (bonus) — Route B inflation-scale diagnostic, unique to strategy 3.
    _plot_route_b_scale(
        t_ax       = t_dense_fine,
        scale_mean = q_scale_mean,
        scale_std  = q_scale_std,
        alpha      = float(alpha_rb),
        beta       = float(beta_rb),
        n_traj     = B,
        title      = (
            f"Route B inflation scale  s = α + β‖ρ‖²_L2  over time\n"
            f"(B={B} trajectories, N_ens={N_ens}, α={float(alpha_rb):g}, β={float(beta_rb):g})"
        ),
        save_path  = os.path.join(save_dir, "batch_route_b_scale_3way.pdf"),
    )
  
def _plot_route_b_scale(
    t_ax:        np.ndarray,   # (total_fine_steps,)
    scale_mean:  np.ndarray,   # (total_fine_steps,) mean over ensemble & ICs
    scale_std:   np.ndarray,   # (total_fine_steps,) std over ICs of the ensemble-mean
    alpha:       float,
    beta:        float,
    n_traj:      int,
    title:       str,
    save_path:   str,
) -> None:
    """
    Bonus diagnostic (not present in the Classic/DD comparison plots, since
    it has no Classic-EnKF analogue): the Route B inflation scale factor
    s_i = alpha + beta * ||rho_i||^2_L2 over time, averaged across the
    ensemble and across trajectories.

    This directly visualises the "physics-driven, flow-dependent additive
    inflation" Route B is built to produce -- s tracks how much the
    surrogate's own PDE residual currently pushes the process-noise
    covariance above the alpha floor, e.g. growing near sharp gradients or
    while coasting through observation gaps, and relaxing back toward alpha
    when the surrogate is locally physics-consistent.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(t_ax, scale_mean, color="#8E24AA", linewidth=1.6,
            label=f"Route B scale  s = α + β‖ρ‖²  (n = {n_traj} trajectories)")
    ax.fill_between(
        t_ax, scale_mean - scale_std, scale_mean + scale_std,
        color="#8E24AA", alpha=0.18, linewidth=0, label="±1 std across trajectories",
    )
    ax.axhline(y=alpha, color="#37474F", linestyle="--", linewidth=1.4,
               label=f"α floor = {alpha:g}")

    ax.set_yscale("log")
    ax.set_xlabel("Time  t", fontsize=12)
    ax.set_ylabel("Route B scale factor  s_i  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"Route B inflation-scale diagnostic plot saved to: {save_path}")

def evaluate_enkf_3_way(
    config: ml_collections.ConfigDict,
    workdir: str,
    test_h5_path: str = None,
) -> None:
    """
    Three-way EnKF evaluation combining strategies:
 
      1. Data-driven (DD) propagator      + multiplicative inflation
      2. Physics-informed (PI) propagator + multiplicative inflation
      3. Physics-informed (PI) propagator + Route B residual-scaled
         inflation (novel additive inflation)
 
    Execution strategy
    -------------------
    This uses vmapped/JIT batch
    execution so all three strategies are propagated
    concurrently for every IC in a batch. The Route B filter itself
    (residual-scaled additive inflation via `run_enkf_smoother_route_b`)
    is taken as-is from the `kf.py` dependencies  -- only its *execution* 
    is rewired onto the concurrent batching strategy.
 
    Since strategies 2 and 3 share the PI propagator (only their filter
    differs) while strategy 1 uses its own DD propagator, only two
    open-loop reference rollouts (PI, DD) are needed even though three
    EnKF curves are reported -- exactly as in the two pipelines this
    combines.
 
    Route B hyperparameters:
      * ``route_b_alpha``  (default 1.0)  — variance floor α.
      * ``route_b_beta``   (default 5.0)  — residual sensitivity β.
      * ``Q0_sigma``       (default P0_sigma) — per-window base process-noise
        std used to build ``Q0``.
      * ``route_b_n_quad`` (default 3)    — trapezoidal quadrature points
        per fine step for the residual integral.
 
    Outputs (under ``workdir/figures/three_way/``)
    ------------------------------------------------
      * ``trajectory_summary_enkf_3way_ic_<i>.pdf`` -- per-IC trajectory
        plot, all 3 strategies overlaid against ground truth.
      * ``batch_l2_per_timestep_3way.pdf``    -- all 3 EnKF strategies +
        both open-loop references, on ONE graph.
      * ``batch_calibration_enkf_3way.pdf``   -- one spread-vs-RMSE
        (simulation time) panel per strategy, plus a combined binned
        spread-skill panel comparing all 3 on the same axes.
      * ``batch_erf_enkf_3way.pdf``           -- all 3 ERF curves on the
        same graph.
      * ``batch_rmse_enkf_3way.pdf``          -- 3 pairwise prior/posterior
        RMSE panels (DD-vs-PI-classic, PI-classic-vs-Route-B,
        DD-vs-Route-B) so the ±1σ bands don't crowd a single axes.
      * ``batch_route_b_scale_3way.pdf``      -- bonus Route B
        inflation-scale diagnostic (no analogue for the other 2
        strategies).
    """
    from examples.l96_f.kf import run_enkf_smoother, run_enkf_smoother_route_b, init_ensemble
 
    # ── EnKF / observation configuration (identical to evaluate_enkf_dd_vs_pi) ──
    obs_every_n  = config.kf.get("obs_every_n",   4)
    sigma_obs    = config.kf.get("sigma_obs",      0.5)
    P0_sigma     = config.kf.get("P0_sigma",       1.0)
    dynamic_vars = config.kf.get("dynamic_vars",   False)
    N_ens        = config.kf.get("N_ens",         50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)
 
    # ── Route B-specific configuration ──────────────────────────────────────────
    alpha_rb   = config.kf.get("route_b_alpha", 1.0)
    beta_rb    = config.kf.get("route_b_beta",  5.0)
    Q0_sigma   = config.kf.get("Q0_sigma",       P0_sigma)
    n_quad_rb  = config.kf.get("route_b_n_quad", 3)
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list", None)
 
    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE   = float(config.kf.get("dt_fine",   DT_WINDOW))
    DT_OBS    = float(config.kf.get("dt_obs",    DT_WINDOW))
 
    # ── 1. Load the long test trajectories and forcing parameters ─────────
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"
 
    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]
        t_test = f["t"][:]
        F_test = f["F"][:]

    logging.info(f"JAX sees {jax.local_device_count()} local device(s): {jax.local_devices()}")
 
    dt_window = float(config.get("dt_window", 0.25))
 
    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows      = config.eval.get("windows", 200)
    num_ics_eval       = config.eval.get("num_ics", u_test.shape[0])
    dt_integration     = config.eval.get("dt_integration", 0.005)
    enkf_batch_size    = config.kf.get("batch_l2_size", 200)
 
    # ── 2. Models & per-window query grid ──────────────────────────────────
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)
 
    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N
 
    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_model")
    if not os.path.exists(ckpt_path_dd):
        ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params
 
    # ── 3. EnKF predict/update functions for all three strategies ─────────
    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)
    predict_fn_cl, update_fn_cl = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)
 
    # Scale multiplicative inflation geometrically for fine timesteps
    # (shared by DD and PI classic, exactly as in the 2-way pipelines).
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine       = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
 
    # Scale the Route B base covariance to a per-fine-step value (Q0)
    Q_coarse = jnp.eye(N) * Q0_sigma ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
 
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
 
    # ── 4. Per-IC single-trajectory EnKF evaluation (3-way) ───────────────
    num_plots  = min(config.saving.total_plots, u_test.shape[0])
    total_time = trajectory_windows * DT_WINDOW
 
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time = total_time, dt_fine = DT_FINE, dt_obs = DT_OBS,
    )
    obs_step_indices = jnp.array(obs_step_indices)
 
    # PASS 1: sequential SciPy ground-truth solves — exact gen_data.py solver
    x_true_fine_list, x_true_at_obs_list = [], []
    t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
 
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
 
        def lorenz_96(t, state, F=F_i):
            x_plus_1  = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F
 
        sol = solve_ivp(
            lorenz_96, t_span=[0.0, total_time], y0=np.array(u_test[ic_idx, 0, :]),
            t_eval=t_eval_fine, method='LSODA', rtol=1e-13, atol=1e-14,
        )
        x_true_fine_list.append(sol.y.T)
        x_true_at_obs_list.append(sol.y.T[obs_step_indices + 1])
 
    # PASS 2: batched, concurrent GPU execution for all 3 strategies at once
    x_true_fine_batch   = jnp.stack(x_true_fine_list)
    x_true_at_obs_batch = jnp.stack(x_true_at_obs_list)
    u0_batch_plots = jnp.array(u_test[:num_plots, 0, :])
    F_batch_plots  = jnp.array(F_test[:num_plots])
    keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_plots))
 
    batched_enkf_plots = build_batched_enkf_3way(
        predict_fn_dd, update_fn_dd,
        predict_fn_cl, update_fn_cl,
        predict_fn_rb, update_fn_rb,
        N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R, alpha_fine,
        Q_fine, alpha_rb, beta_rb, n_quad_rb,
        DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices,
    )
 
    (batch_x_means_dd, batch_x_spreads_dd, _,
     batch_x_means_cl, batch_x_spreads_cl, _,
     batch_x_means_rb, batch_x_spreads_rb, _, _,
     batch_y_obs, batch_idx_vars) = batched_enkf_plots(
         keys_batch_plots, u0_batch_plots, F_batch_plots,
         x_true_at_obs_batch, dynamic_vars, specify_obs_idx
    )
 
    # PASS 3: generate individual trajectory PDF plots sequentially
    t_fine_axis = t_eval_fine[1:]
    window_step_indices = np.array([round((w + 1) * DT_WINDOW / DT_FINE) - 1 for w in range(trajectory_windows)])
 
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
        x_true_fine = x_true_fine_batch[ic_idx]
        x_means_dd, x_spreads_dd = batch_x_means_dd[ic_idx], batch_x_spreads_dd[ic_idx]
        x_means_cl, x_spreads_cl = batch_x_means_cl[ic_idx], batch_x_spreads_cl[ic_idx]
        x_means_rb, x_spreads_rb = batch_x_means_rb[ic_idx], batch_x_spreads_rb[ic_idx]
        y_obs_seq, idx_vars_seq  = batch_y_obs[ic_idx], batch_idx_vars[ic_idx]
 
        # Intercept observation coordinates for the plotting function
        obs_coords = []
        for obs_idx, t_obs in enumerate(obs_times):
            for j, vi in enumerate(idx_vars_seq[obs_idx]):
                obs_coords.append((int(vi), float(t_obs), float(y_obs_seq[obs_idx, j])))
 
        _plot_trajectory_summary_compare_enkf_3way(
            t_ax=t_fine_axis, x_true=np.array(x_true_fine[1:]),
            x_est_dd=np.array(x_means_dd[:, :N]), x_std_dd=np.array(x_spreads_dd[:, :N]),
            x_est_cl=np.array(x_means_cl[:, :N]), x_std_cl=np.array(x_spreads_cl[:, :N]),
            x_est_rb=np.array(x_means_rb[:, :N]), x_std_rb=np.array(x_spreads_rb[:, :N]),
            ic_idx=ic_idx, F_val=F_i, N=N, dt_window=DT_WINDOW, obs_coords=obs_coords,
            save_path=os.path.join(
                workdir, "figures", "three_way", f"trajectory_summary_enkf_3way_ic_{ic_idx}.pdf"
            ),
        )
 
        x_true_at_windows = x_true_fine[window_step_indices + 1]
        l2_dd = jnp.linalg.norm(x_means_dd[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
        l2_cl = jnp.linalg.norm(x_means_cl[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
        l2_rb = jnp.linalg.norm(x_means_rb[window_step_indices, :N] - x_true_at_windows) / jnp.linalg.norm(x_true_at_windows)
 
        print(
            f"IC {ic_idx} | EnKF DD L2: {l2_dd:.3e} | EnKF PI(classic) L2: {l2_cl:.3e} | "
            f"EnKF PI(Route B) L2: {l2_rb:.3e} "
            f"| Mean σ (DD): {float(jnp.mean(x_spreads_dd)):.3e} "
            f"| Mean σ (PI classic): {float(jnp.mean(x_spreads_cl)):.3e} "
            f"| Mean σ (PI Route B): {float(jnp.mean(x_spreads_rb)):.3e}"
        )
 
    # ── 5. Batch-averaged 3-way comparison ─────────────────────────────────
    _evaluate_batch_enkf_3way(
        model_dd, params_dd, predict_fn_dd, update_fn_dd,
        model_pi, params_pi,
        predict_fn_cl, update_fn_cl,
        predict_fn_rb, update_fn_rb,
        t_star_window,
        u_test, t_test, F_test,
        alpha_fine, Q_fine, alpha_rb, beta_rb, n_quad_rb,
        P0, R, obs_indices,
        N_ens, obs_every_n, sigma_obs, P0_sigma, dynamic_vars,
        specify_obs_idx, obs_idx_list,
        DT_WINDOW, DT_FINE, DT_OBS,
        num_ics_eval, enkf_batch_size, batch_windows,
        config, workdir,
    )



