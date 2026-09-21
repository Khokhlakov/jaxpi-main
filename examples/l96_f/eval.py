"""
Modular filter evaluation and comparison for the DeepONet + EnKF Lorenz-96
pipeline.

ONE entry point, `run_comparison(config, workdir)`, replaces the earlier
family of `run_3way_comparison` / `run_4way_comparison` /
`run_7way_comparison` / `run_*_inflation_sweep` functions. What is evaluated
and what is plotted is described entirely by the configuration file.

Configuration
-------------
Three keys under `config.eval`. Example 1 -- compare specific strategies::

    config.eval.test_data_name = "l96_forcing_test"   # -> data/l96_forcing_test.h5

    config.eval.strategies = {
        "dd_mult": dict(surrogate="DD", inflation="multiplicative",
                        params=dict(inflation_factor=1.05)),
        "pi_mult": dict(surrogate="PI", inflation="multiplicative",
                        params=dict(inflation_factor=1.10)),
        "pi_add":  dict(surrogate="PI", inflation="additive",
                        params=dict(alpha=3.0)),
        "pi_rb":   dict(surrogate="PI", inflation="route_b",
                        params=dict(alpha=1.0, beta=250.0)),
        "dd_rtpp": dict(surrogate="DD", inflation="rtpp",
                        params=dict(alpha_rtpp=0.5)),
    }

    config.eval.plot_groups = [
        ["dd_mult", "pi_mult"],             # one collection with both on every figure
        ["pi_add", "pi_rb", "dd_rtpp"],
        ["pi_rb"],                          # a collection with a single strategy
    ]

Example 2 -- parameter sweeps (any parameter given as a list; `plot_groups` is
then ignored)::

    config.eval.strategies = {
        "dd_mult": dict(surrogate="DD", inflation="multiplicative",
                        params=dict(inflation_factor=[1.00, 1.02, 1.05, 1.10])),
        "pi_mult": dict(surrogate="PI", inflation="multiplicative",
                        params=dict(inflation_factor=[1.00, 1.02, 1.05, 1.10])),
        "pi_rb":   dict(surrogate="PI", inflation="route_b",
                        params=dict(alpha=0.0, beta=[50.0, 100.0, 250.0])),
    }

Strategy entries (one per identifying name):

    surrogate   "DD" (data-driven DeepONet) or "PI" (physics-informed DeepONet).
    inflation   "multiplicative", "additive", "route_b" or "rtpp".
    params      depends on the inflation type:

        multiplicative   inflation_factor            (> 0; coarse, per-window factor)
        additive         alpha                       (>= 0; constant floor alpha*Q0,
                                                      i.e. Route B with beta = 0)
        route_b          alpha, beta                 (>= 0; scale = alpha + beta*||rho||^2)
        rtpp             alpha_rtpp                  (in [0, 1]; relaxation factor)
                         alpha_fine   (optional, default 1.0; multiplicative inflation
                                       in RTPP's predict step, used as given)

Every parameter listed as "sweepable" above may be a scalar or a list. A list
of n values triggers n evaluations (see "Sweeps"). Anything not listed here
(N_ens, sigma_obs, Q0_sigma, route_b_n_quad, dt_fine, ...) is still read from
`config.kf` / `config.eval` exactly as before and is shared by every strategy.

Evaluation and output files
---------------------------
Each (strategy, parameter version) is evaluated independently and written to
its own HDF5 file in `workdir`:

    <name>.h5                    strategy without a parameter list
    <name>_ver_<i>.h5            i-th value of a parameter list (1-based)
    <name>_ver_<i>_<j>.h5        Route B sweeps: i-th alpha value, j-th beta value

Route B is always indexed by BOTH parameters. If only one of alpha/beta is a
list, the other is treated as a one-element list (index 1); if both are lists,
every (alpha, beta) combination is evaluated (a full grid).

If the file for a name already exists it is NOT recomputed (a partial file is
never left behind: results are written to a temporary file and renamed on
success), and the pipeline goes straight to plotting. A warning is logged if
the stored parameters/settings differ from the current config; delete the
file to force a re-run.

All strategies see the same initial conditions, observation noise draws and
initial ensembles (they are derived from `config.training.seed` only), so
files evaluated at different times remain directly comparable as long as the
shared settings are unchanged.

Plotting
--------
    * Individual-trajectory PDFs are unchanged: one per (IC, strategy file),
      written to `<workdir>/figures/individual_trajectories/`.
    * `config.eval.plot_groups` is a list of lists of identifying names (or a
      dict {group_name: [names]} to choose the folder names). Each inner list
      yields one collection of figures in which those strategies are overlaid
      on the same axes -- calibration, ERF, L2 vs open-loop, prior/posterior
      RMSE and equilibrium variance -- written to
      `<workdir>/figures/comparisons/<group_name>/`. A list with a single
      name plots just that strategy.
    * Sweeps: if ANY strategy has a parameter list, `plot_groups` is ignored
      and one collection per identifying name is written instead, overlaying
      every parameter version of that strategy (a strategy without a list gets
      a collection containing just itself).

HDF5 layout written by `evaluate_filters`
------------------------------------------
    /meta                                   (attrs only, unless noted)
        N, F                                 -- state dim, per-IC forcing (num_ics_traj,)
        dt_window, dt_fine, dt_obs
        sigma_obs, P0_sigma, N_ens, obs_every_n, m
        num_ics_traj, num_ics_batch, trajectory_windows, batch_windows
        test_data                            -- basename of the test-data file used
        obs_indices                          -- dataset, (m,) int, if static
        strategy_keys, strategy_labels        -- string datasets (one entry per file)
        strategy_propagator                   -- which propagator each strategy uses
        strategy_kind                         -- "standard" | "route_b" | "rtpp"
        strategy_config_json                  -- the config entry that produced the file

    /trajectories/t_fine                     (T_fine,)
    /trajectories/ic_{i}  (attrs: F)
        x_true                               (T_fine, N)
        obs_coords                           (n_obs_pts, 3) = [var_idx, t_obs, y_obs]
        strategies/{key}/x_est               (T_fine, N)
        strategies/{key}/x_std               (T_fine, N)
        strategies/{key}/l2_time_avg         scalar attr -- time-avg relative L2
                                              over the 40 vars, for quick titling

    /batch  (attrs: B, obs_every_n, sigma_obs, N_ens, dt_obs, eqvar_burn_in_frac)
        obs_times                            (T_obs,)
        window_idx                           (n_windows,)
        t_dense_fine                         (n_fine,)
        strategies/{key}/prior_rmse_mean     (T_obs,)
        strategies/{key}/prior_rmse_std      (T_obs,)
        strategies/{key}/post_rmse_mean      (T_obs,)
        strategies/{key}/post_rmse_std       (T_obs,)
        strategies/{key}/erf_mean            (T_obs,)
        strategies/{key}/erf_std             (T_obs,)
        strategies/{key}/rmse_window_mean    (n_windows,)   -- for RMSE-with-spread & calibration timeseries
        strategies/{key}/spread_window_mean  (n_windows,)
        strategies/{key}/rmse_raw            (B * n_windows,)  -- for calibration binned scatter
        strategies/{key}/spread_raw          (B * n_windows,)
        strategies/{key}/l2_dense_mean       (n_fine,)      -- batch time-mean L2 error curve
        strategies/{key}/eqvar_mean          (N,)           -- equilibrium (climatological)
        strategies/{key}/eqvar_std           (N,)              variance, see below
        strategies/{key}/route_b_scale_mean  (n_fine,)      -- only present for kind == "route_b"
        strategies/{key}/route_b_scale_std   (n_fine,)
        open_loop/{propagator}/t             (n_ol,)
        open_loop/{propagator}/l2_dense_mean (n_ol,)
        open_loop/{propagator}/eqvar_mean    (N,)
        open_loop/{propagator}/eqvar_std     (N,)
        reference/eqvar_mean                 (N,)           -- ground-truth (unfiltered)
        reference/eqvar_std                  (N,)              equilibrium variance

    `{key}` is the file stem (e.g. "pi_mult_ver_2"). Files written by this
    module hold exactly one strategy, but the plotting code reads any number.

    Equilibrium (climatological/attractor) variance
    -------------------------------------------------
    For a dense state trajectory of B ICs x T time steps x N variables,
    the leading `eqvar_burn_in_frac` fraction of the time axis is
    discarded (letting transients -- assimilation spin-up for filtered
    strategies, or an off-attractor start for an open-loop rollout --
    decay), then the per-variable temporal variance is computed on the
    remaining tail for each IC. `eqvar_mean`/`eqvar_std` are the
    mean/std of that per-IC variance across the B ICs. `reference/*` is
    computed the same way from the true (unfiltered) test trajectories,
    and is the target every open-loop and filtered curve is compared
    against by the equilibrium-variance figure.
"""

import os
import re
import json
import hashlib
import dataclasses
from absl import logging
import ml_collections
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import h5py

import itertools
import colorsys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

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


# ─────────────────────────────────────────────────────────────────────────
# Multi-GPU batch execution helper (unchanged from eval_modular.py)
# ─────────────────────────────────────────────────────────────────────────

def _device_parallel(fn, in_axes, static_broadcasted_argnums=(), to_host=True):
    """
    Same contract as before, except results come back as host numpy
    arrays when `to_host` (the default).
 
    Returning host arrays is what keeps the dense per-strategy outputs
    off the GPU: every consumer of these arrays in `evaluate_filters`
    either writes them to HDF5 or reduces them, so there is no reason for
    them to stay resident on the device.
    """
    n_devices = jax.local_device_count()
    vmapped_fn = jax.vmap(fn, in_axes=in_axes)
 
    def _leaf_to_host(x):
        """Copy one leaf to host and release its device buffer."""
        h = np.asarray(jax.device_get(x))
        try:
            x.delete()          # the leaf is a fresh output, safe to free
        except Exception:       # noqa: BLE001 - not all leaf types support it
            pass
        return h
 
    if n_devices <= 1:
        jitted = jax.jit(vmapped_fn, static_argnums=static_broadcasted_argnums)
        if not to_host:
            return jitted
 
        def wrapped_single(*args):
            out = jitted(*args)
            res = jax.tree_util.tree_map(_leaf_to_host, out)
            del out
            return res
 
        return wrapped_single
 
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
        x = jnp.asarray(x)
        B = x.shape[0]
        per_device = -(-B // n_devices)
        pad = per_device * n_devices - B
        if pad > 0:
            x = jnp.concatenate(
                [x, jnp.zeros((pad,) + x.shape[1:], dtype=x.dtype)], axis=0
            )
        return x.reshape((n_devices, per_device) + x.shape[1:])
 
    def _unshard_host(x, B):
        # Host-side: the reshape is a view, and no single device ever has
        # to hold the full un-sharded array.
        h = _leaf_to_host(x)
        h = h.reshape((-1,) + h.shape[2:])
        return h[:B]
 
    def _unshard_device(x, B):
        x = x.reshape((-1,) + x.shape[2:])
        return x[:B]
 
    def wrapped(*args):
        B = args[batched_idx[0]].shape[0]
        sharded_args = tuple(
            _shard(a) if i in batched_idx else a
            for i, a in enumerate(args)
        )
        out = pmapped_fn(*sharded_args)
        del sharded_args
        unshard = _unshard_host if to_host else _unshard_device
        res = jax.tree_util.tree_map(lambda x: unshard(x, B), out)
        del out
        return res
 
    return wrapped

# ─────────────────────────────────────────────────────────────────────────
# Chunking helpers
# ─────────────────────────────────────────────────────────────────────────
 
def _strategy_groups(strategies, group_size):
    if not group_size or group_size >= len(strategies):
        return [list(strategies)]
    return [
        list(strategies[i:i + group_size])
        for i in range(0, len(strategies), group_size)
    ]
 
 
def _dense_leaves_per_strategy(group):
    """
    Dense (T_fine-long) output arrays each strategy returns:
    x_means + x_spreads for every kind, plus q_scale for Route B
    (`prior_means` is only T_obs long, so it is negligible here).
    """
    return sum(3 if spec["kind"] == "route_b" else 2 for spec in group)
 
 
def _auto_ic_chunk(B, n_devices, bytes_per_ic, budget_bytes):
    if bytes_per_ic <= 0:
        return B
    chunk = int(budget_bytes // bytes_per_ic)
    chunk = min(max(chunk, n_devices), B)
    if chunk > n_devices:
        chunk = (chunk // n_devices) * n_devices   # avoid pmap padding waste
    return max(chunk, 1)


# ─────────────────────────────────────────────────────────────────────────
# Strategy specification (internal, consumed by `evaluate_filters`)
# ─────────────────────────────────────────────────────────────────────────
#
# `run_comparison` turns each entry of `config.eval.strategies` into one or
# more of these plain dicts (one per parameter version):
#
#   {
#     "key":        unique short id, used as the HDF5 group name (= file stem),
#     "label":      human-readable label for legends / titles,
#     "kind":       "standard" (run_enkf_smoother),
#                   "route_b"  (run_enkf_smoother_route_b), or
#                   "rtpp"     (run_enkf_smoother_rtpp),
#     "propagator": "dd" or "pi" -- which surrogate it uses; the open-loop
#                   reference rollout is computed for this propagator,
#     "predict_fn", "update_fn": the EnKF predict/update closures,
#     "config":     (optional) JSON-serializable dict stored in the HDF5 file
#                   so a later run can tell whether a cached file is stale,
#     # kind == "standard":
#     "alpha_fine": scaled multiplicative inflation factor,
#     # kind == "route_b":
#     "Q0", "alpha", "beta", "n_quad": Route-B hyperparameters,
#     #   (inflation="additive" is Route B with "beta": 0.0, i.e. only the
#     #    constant floor alpha*Q0; a residual-only scheme is "alpha": 0.0)
#     # kind == "rtpp":
#     "alpha_fine": multiplicative inflation applied in the (shared)
#                   predict step -- 1.0 isolates RTPP's own relaxation as
#                   the only inflation mechanism,
#     "alpha_rtpp": RTPP relaxation-to-prior factor in [0, 1],
#   }
#
# `propagators` is a dict: propagator_name -> (model, params), used only
# for the open-loop reference rollouts and to read N.


def build_batched_filters(
    strategies, N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R,
    dt_fine, dt_window, total_fine_steps_batch, obs_step_indices_batch,
):
    """
    N-way counterpart of `build_batched_enkf_3way`. Builds ONE
    jit(vmap(...)) (or pmap-sharded) closure that, for every IC in the
    batch, runs every strategy in `strategies` against the SAME noisy
    observation draw and the SAME initial ensemble -- so differences
    between strategies reflect the strategy alone, not differing noise
    realizations.
    """

    def process_single_ic(key_ic, u_true, F_i, x_true_at_obs,
                           dynamic_vars_static, specify_obs_idx_static):
        T_obs = x_true_at_obs.shape[0]
        keys_t = jax.random.split(key_ic, T_obs)

        def single_obs(k, x_t):
            k1, k2 = jax.random.split(k)
            if (not specify_obs_idx_static) and dynamic_vars_static:
                idx_vars = jax.random.choice(k1, N, shape=(m,), replace=False)
            else:
                idx_vars = obs_indices

            H = jnp.zeros((m, N)).at[jnp.arange(m), idx_vars].set(1.0)
            H_aug = jnp.pad(H, ((0, 0), (0, 1)), mode='constant')
            noise = sigma_obs * jax.random.normal(k2, shape=(m,))
            return H_aug, x_t[idx_vars] + noise, idx_vars

        H_seq, y_obs_seq, idx_vars_seq = jax.vmap(single_obs)(keys_t, x_true_at_obs)

        # Shared initial ensemble across every strategy.
        k1, k2, k3 = jax.random.split(key_ic, 3)
        x0_hat_40 = u_true + P0_sigma * jax.random.normal(k2, shape=(N,))
        x0_hat_aug = jnp.concatenate([x0_hat_40, jnp.array([F_i])])
        ensemble0 = init_ensemble(x0_hat_aug, P0, N_ens, k3)

        outputs = {}
        for spec in strategies:
            key = spec["key"]
            if spec["kind"] == "route_b":
                x_means, x_spreads, prior_means, q_scale = run_enkf_smoother_route_b(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices_batch,
                    H_seq, Q0=spec["Q0"], alpha=spec["alpha"], beta=spec["beta"],
                    R=R, key=key_ic, total_fine_steps=total_fine_steps_batch,
                    dt_fine=dt_fine, dt_window=dt_window, n_quad=spec["n_quad"],
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads,
                    prior_means=prior_means, q_scale=q_scale,
                )
            elif spec["kind"] == "rtpp":
                x_means, x_spreads, prior_means = run_enkf_smoother_rtpp(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices_batch,
                    H_seq, spec["alpha_fine"], spec["alpha_rtpp"],
                    R, key_ic, total_fine_steps_batch,
                    dt_fine=dt_fine, dt_window=dt_window,
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads, prior_means=prior_means,
                )
            else:
                x_means, x_spreads, prior_means = run_enkf_smoother(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices_batch,
                    H_seq, spec["alpha_fine"], R, key_ic, total_fine_steps_batch,
                    dt_fine=dt_fine, dt_window=dt_window,
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads, prior_means=prior_means,
                )

        return outputs, y_obs_seq, idx_vars_seq

    return _device_parallel(
        process_single_ic,
        in_axes=(0, 0, 0, 0, None, None),
        static_broadcasted_argnums=(4, 5),
    )


# ─────────────────────────────────────────────────────────────────────────
# Evaluation: one call -> one HDF5 file
# ─────────────────────────────────────────────────────────────────────────
def evaluate_filters(
    config: ml_collections.ConfigDict,
    strategies: list[dict],
    propagators: dict,
    t_star_window,
    out_path: str,
    test_h5_path: str | None = None,
) -> str:
    """
    Runs every strategy in `strategies` on the SAME data (same ICs, noisy
    observation draws and initial ensembles) and writes everything the
    plotting stage needs to ONE HDF5 file at `out_path` (layout: see the
    module docstring). `run_comparison` calls this with a single strategy per
    file; passing several still works and stores them all in the one file.

    `test_h5_path` defaults to `data/<config.eval.test_data_name>.h5`.
    Results are written to `<out_path>.partial` and renamed only once the file
    is complete, so an interrupted run never leaves a half-written file that a
    later run would mistake for a finished evaluation.
    """
    # ── EnKF / observation configuration ───────────────────────────────
    obs_every_n = config.kf.get("obs_every_n", 4)
    sigma_obs = config.kf.get("sigma_obs", 0.5)
    P0_sigma = config.kf.get("P0_sigma", 1.0)
    dynamic_vars = config.kf.get("dynamic_vars", False)
    N_ens = config.kf.get("N_ens", 50)
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list = config.kf.get("obs_idx_list", None)
 
    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    DT_OBS = float(config.kf.get("dt_obs", DT_WINDOW))
 
    n_devices = jax.local_device_count()
    strategy_chunk = int(config.eval.get("strategy_chunk", 0))
    ic_chunk_cfg = int(config.eval.get("ic_chunk", 0))
    budget_bytes = float(config.eval.get("device_budget_gb", 1.5)) * (1024 ** 3)
 
    # ── 1. Load the long test trajectories and forcing parameters ──────
    if test_h5_path is None:
        test_h5_path = resolve_test_h5_path(config)
 
    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]
        t_test = f["t"][:]
        F_test = f["F"][:]
 
    logging.info(f"JAX sees {n_devices} local device(s): {jax.local_devices()}")
 
    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows = config.eval.get("windows", 200)
    num_ics_eval = config.eval.get("num_ics", u_test.shape[0])
    enkf_batch_size = config.kf.get("batch_l2_size", 200)
    eqvar_burn_in_frac = config.eval.get("eqvar_burn_in_frac", 0.5)
 
    if not strategies:
        raise ValueError("evaluate_filters: `strategies` must be a non-empty list.")
    if not propagators or t_star_window is None:
        raise ValueError("evaluate_filters: `propagators` and `t_star_window` are required.")
    N = next(iter(propagators.values()))[0].N
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
 
    m = len(obs_indices)
    R = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
 
    groups = _strategy_groups(strategies, strategy_chunk)
 
    # ── 2. Per-IC single-trajectory data (individual-trajectory plots) ──
    num_plots = min(config.saving.total_plots, u_test.shape[0])
    total_time_traj = trajectory_windows * DT_WINDOW
 
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time_traj, dt_fine=DT_FINE, dt_obs=DT_OBS,
    )
    obs_step_indices = jnp.array(obs_step_indices)
 
    # PASS 1: sequential SciPy ground-truth solves
    x_true_fine_list, x_true_at_obs_list = [], []
    t_eval_fine = np.linspace(0.0, total_time_traj, total_fine_steps + 1)
 
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
 
        def lorenz_96(t, state, F=F_i):
            x_plus_1 = np.roll(state, -1)
            x_minus_1 = np.roll(state, 1)
            x_minus_2 = np.roll(state, 2)
            return (x_plus_1 - x_minus_2) * x_minus_1 - state + F
 
        sol = solve_ivp(
            lorenz_96, t_span=[0.0, total_time_traj], y0=np.array(u_test[ic_idx, 0, :]),
            t_eval=t_eval_fine, method='LSODA', rtol=1e-13, atol=1e-14,
        )
        x_true_fine_list.append(sol.y.T)
        x_true_at_obs_list.append(sol.y.T[obs_step_indices + 1])
 
    # PASS 2: batched GPU execution, one compiled program per strategy group
    x_true_fine_batch = np.stack(x_true_fine_list)          # host
    x_true_at_obs_batch = jnp.stack(x_true_at_obs_list)
    u0_batch_plots = jnp.array(u_test[:num_plots, 0, :])
    F_batch_plots = jnp.array(F_test[:num_plots])
    keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_plots))
 
    traj_bytes_per_ic = (
        total_fine_steps * (N + 1) * 4
        * _dense_leaves_per_strategy(groups[0]) * 2
    )
    traj_chunk = ic_chunk_cfg or _auto_ic_chunk(
        num_plots, n_devices, traj_bytes_per_ic, budget_bytes
    )
    logging.info(
        f"evaluate_filters: trajectory pass -- {num_plots} IC(s) in chunks of "
        f"{traj_chunk}, {len(groups)} strategy group(s)."
    )
 
    outputs_traj = {}
    y_obs_traj = np.zeros((num_plots, len(obs_step_indices), m), dtype=np.float32)
    idx_vars_traj = np.zeros((num_plots, len(obs_step_indices), m), dtype=np.int32)
 
    for g_i, group in enumerate(groups):
        batched_traj_fn = build_batched_filters(  # noqa: F821
            group, N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R,
            DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices,
        )
        for i0 in range(0, num_plots, traj_chunk):
            i1 = min(i0 + traj_chunk, num_plots)
            out_c, y_c, idx_c = batched_traj_fn(
                keys_batch_plots[i0:i1], u0_batch_plots[i0:i1],
                F_batch_plots[i0:i1], x_true_at_obs_batch[i0:i1],
                dynamic_vars, specify_obs_idx,
            )
            if g_i == 0:
                y_obs_traj[i0:i1] = np.asarray(y_c)
                idx_vars_traj[i0:i1] = np.asarray(idx_c)
            for key, d in out_c.items():
                dst = outputs_traj.setdefault(key, {})
                for field, arr in d.items():
                    if field not in dst:
                        dst[field] = np.zeros(
                            (num_plots,) + arr.shape[1:], dtype=arr.dtype
                        )
                    dst[field][i0:i1] = arr
            del out_c, y_c, idx_c
        del batched_traj_fn
        jax.clear_caches()
 
    window_step_indices = np.array(
        [round((w + 1) * DT_WINDOW / DT_FINE) - 1 for w in range(trajectory_windows)]
    )
    t_fine_axis = t_eval_fine[1:]
 
    per_ic_records = []
    for ic_idx in range(num_plots):
        F_i = float(F_test[ic_idx])
        x_true_fine = np.array(x_true_fine_batch[ic_idx][1:])
        x_true_at_windows = x_true_fine[window_step_indices]
 
        idx_vars_seq = idx_vars_traj[ic_idx]
        y_obs_seq = y_obs_traj[ic_idx]
        obs_coords = []
        for obs_idx, t_obs in enumerate(obs_times):
            for j, vi in enumerate(idx_vars_seq[obs_idx]):
                obs_coords.append((int(vi), float(t_obs), float(y_obs_seq[obs_idx, j])))
        obs_coords = np.array(obs_coords, dtype=np.float64) if obs_coords else np.zeros((0, 3))
 
        strat_records = {}
        for spec in strategies:
            key = spec["key"]
            x_means = np.array(outputs_traj[key]["x_means"][ic_idx][:, :N])
            x_spreads = np.array(outputs_traj[key]["x_spreads"][ic_idx][:, :N])
            l2_time_avg = float(
                np.linalg.norm(x_means[window_step_indices] - x_true_at_windows)
                / (np.linalg.norm(x_true_at_windows) + 1e-12)
            )
            strat_records[key] = dict(x_est=x_means, x_std=x_spreads, l2_time_avg=l2_time_avg)
 
        per_ic_records.append(dict(
            F=F_i, x_true=x_true_fine, obs_coords=obs_coords, strategies=strat_records,
        ))
 
    # Everything the trajectory pass produced is now in `per_ic_records`
    # (host numpy). Drop the originals and the compilation cache before the
    # batch pass allocates anything, otherwise both passes are live at once.
    del outputs_traj, y_obs_traj, idx_vars_traj
    del x_true_at_obs_batch, u0_batch_plots, F_batch_plots, keys_batch_plots
    jax.clear_caches()
 
    # ── 3. Batch-averaged metrics ──────────────────────────────────────
    B = min(num_ics_eval, enkf_batch_size, u_test.shape[0])
    u0_batch = u_test[:B, 0, :]
    dt_test = float(t_test[1] - t_test[0])
 
    total_time_batch = batch_windows * DT_WINDOW
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=DT_FINE, dt_obs=DT_OBS,
    )
    obs_step_indices_batch_np = np.asarray(obs_step_indices_batch)
    obs_step_indices_batch = jnp.array(obs_step_indices_batch)
 
    T_obs = len(obs_step_indices_batch_np)
    obs_times_batch = np.array([(k + 1) * DT_OBS for k in range(T_obs)])
 
    fine_stride = int(round(DT_FINE / dt_test))
    n_fine_pts = total_fine_steps_batch * fine_stride + 1
 
    # Kept on the host; only the current chunk is ever touched.
    x_true_fine_batch2 = np.asarray(
        u_test[:B, 0:n_fine_pts:fine_stride, :], dtype=np.float32
    )
    x_true_at_obs_batch2 = x_true_fine_batch2[:, obs_step_indices_batch_np + 1, :]
    window_step_indices_b = np.array(
        [round((k + 1) * DT_WINDOW / DT_FINE) - 1 for k in range(batch_windows)]
    )
 
    seed = config.training.get("seed", 42)
    master_key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(master_key, B)
 
    def _rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2, axis=2))
 
    def _mean_std(a):
        return np.mean(a, axis=0), np.std(a, axis=0)
 
    def _equilibrium_variance_per_ic(x_dense, burn_in_frac: float = 0.5):
        T = x_dense.shape[1]
        if T < 4:
            raise ValueError(
                f"_equilibrium_variance_per_ic needs T >= 4 time steps to "
                f"discard a burn-in and still estimate a variance, got T={T}."
            )
        t0 = min(int(round(burn_in_frac * T)), T - 2)
        return np.var(x_dense[:, t0:, :], axis=1)
 
    x_true_at_windows_b = x_true_fine_batch2[:, window_step_indices_b + 1, :]
    x_true_fine_tail = x_true_fine_batch2[:, 1:, :]
    den_dense = np.linalg.norm(x_true_fine_tail, axis=2) + 1e-12
    t_dense_fine = np.arange(1, total_fine_steps_batch + 1) * DT_FINE
 
    ref_eqvar_ic = _equilibrium_variance_per_ic(x_true_fine_tail, eqvar_burn_in_frac)
    ref_eqvar_mean, ref_eqvar_std = _mean_std(ref_eqvar_ic)
 
    batch_bytes_per_ic = (
        total_fine_steps_batch * (N + 1) * 4
        * max(_dense_leaves_per_strategy(g) for g in groups) * 2
    )
    batch_chunk = ic_chunk_cfg or _auto_ic_chunk(
        B, n_devices, batch_bytes_per_ic, budget_bytes
    )
    logging.info(
        f"evaluate_filters: batch pass -- B={B} IC(s) in chunks of {batch_chunk}, "
        f"{len(strategies)} strategies in {len(groups)} group(s), "
        f"~{batch_bytes_per_ic * batch_chunk / 1024**3:.2f} GiB of dense "
        "outputs per call."
    )
 
    # Per-IC metric accumulators: (B, T)-shaped, i.e. a factor N smaller
    # than the dense (B, T, N) outputs they are computed from.
    acc = {spec["key"]: {} for spec in strategies}
 
    def _push(key, field, value):
        acc[key].setdefault(field, []).append(value)
 
    for group in groups:
        batched_batch_fn = build_batched_filters(  # noqa: F821
            group, N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R,
            DT_FINE, DT_WINDOW, total_fine_steps_batch, obs_step_indices_batch,
        )
 
        for i0 in range(0, B, batch_chunk):
            i1 = min(i0 + batch_chunk, B)
            outputs_c, _, _ = batched_batch_fn(
                keys_batch[i0:i1], jnp.array(u0_batch[i0:i1]), F_test[i0:i1],
                jnp.array(x_true_at_obs_batch2[i0:i1]),
                dynamic_vars, specify_obs_idx,
            )
 
            truth_obs_c = x_true_at_obs_batch2[i0:i1]
            truth_win_c = x_true_at_windows_b[i0:i1]
            truth_tail_c = x_true_fine_tail[i0:i1]
            den_c = den_dense[i0:i1]
 
            for spec in group:
                key = spec["key"]
                out = outputs_c[key]
 
                post_means_obs = out["x_means"][:, obs_step_indices_batch_np, :N]
                prior_means_obs = out["prior_means"][:, :, :N]
 
                prior_rmse_ic = _rmse(prior_means_obs, truth_obs_c)
                post_rmse_ic = _rmse(post_means_obs, truth_obs_c)
                _push(key, "prior_rmse", prior_rmse_ic)
                _push(key, "post_rmse", post_rmse_ic)
                _push(key, "erf", prior_rmse_ic / (post_rmse_ic + 1e-12))
 
                x_hat_windows = out["x_means"][:, window_step_indices_b, :N]
                _push(key, "rmse", _rmse(x_hat_windows, truth_win_c))
                _push(key, "spread", np.sqrt(np.mean(
                    out["x_spreads"][:, window_step_indices_b, :N] ** 2, axis=2
                )))
 
                _push(key, "l2_dense", np.linalg.norm(
                    out["x_means"][:, :, :N] - truth_tail_c, axis=2
                ) / den_c)
 
                _push(key, "eqvar", _equilibrium_variance_per_ic(
                    out["x_means"][:, :, :N], eqvar_burn_in_frac
                ))
 
                if spec["kind"] == "route_b":
                    _push(key, "q_scale", np.mean(out["q_scale"], axis=2))
 
                del out
                outputs_c[key] = None
 
            del outputs_c, truth_obs_c, truth_win_c, truth_tail_c, den_c
 
        del batched_batch_fn
        jax.clear_caches()
 
    batch_strat_records = {}
    for spec in strategies:
        key = spec["key"]
        a = {f: np.concatenate(v, axis=0) for f, v in acc[key].items()}
        acc[key] = None
 
        prior_rmse_mean, prior_rmse_std = _mean_std(a["prior_rmse"])
        post_rmse_mean, post_rmse_std = _mean_std(a["post_rmse"])
        erf_mean, erf_std = _mean_std(a["erf"])
        eqvar_mean, eqvar_std = _mean_std(a["eqvar"])
 
        rec = dict(
            label=spec["label"], kind=spec["kind"], propagator=spec["propagator"],
            prior_rmse_mean=prior_rmse_mean, prior_rmse_std=prior_rmse_std,
            post_rmse_mean=post_rmse_mean, post_rmse_std=post_rmse_std,
            erf_mean=erf_mean, erf_std=erf_std,
            rmse_window_mean=np.mean(a["rmse"], axis=0),
            spread_window_mean=np.mean(a["spread"], axis=0),
            rmse_raw=a["rmse"].flatten(),
            spread_raw=a["spread"].flatten(),
            l2_dense_mean=np.mean(a["l2_dense"], axis=0),
            eqvar_mean=eqvar_mean, eqvar_std=eqvar_std,
        )
 
        if spec["kind"] == "route_b":
            rec["route_b_scale_mean"] = np.mean(a["q_scale"], axis=0)
            rec["route_b_scale_std"] = np.std(a["q_scale"], axis=0)
            rec["route_b_alpha"] = float(spec["alpha"])
            rec["route_b_beta"] = float(spec["beta"])
 
        batch_strat_records[key] = rec
        del a
 
    del acc
 
    # ── 4. Open-loop reference rollouts (one per unique propagator) ────
    used_propagators = sorted({spec["propagator"] for spec in strategies})
    open_loop_records = {}
    if propagators is not None:
        for prop_key in used_propagators:
            if prop_key not in propagators:
                continue
            model, params = propagators[prop_key]
            predict_full = _device_parallel(
                lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=(0,)
            )
 
            l2_ol_chunks, eqvar_ol_chunks = [], []
            total_steps_ol = None
            for i0 in range(0, B, batch_chunk):
                i1 = min(i0 + batch_chunk, B)
                u_current = jnp.concatenate(
                    [jnp.array(u0_batch[i0:i1]), jnp.array(F_test[i0:i1])[:, None]],
                    axis=-1,
                )
                x_pred_list = []
                for k in range(batch_windows):
                    x_win = predict_full(u_current)            # host numpy
                    x_pred_list.append(x_win if k == 0 else x_win[:, 1:, :])
                    u_current = jnp.concatenate(
                        [jnp.array(x_win[:, -1, :]),
                         jnp.array(F_test[i0:i1])[:, None]], axis=-1,
                    )
                x_pred_dense = np.concatenate(x_pred_list, axis=1)
                del x_pred_list
                total_steps_ol = x_pred_dense.shape[1]
 
                x_ref_dense_ol = np.asarray(
                    u_test[i0:i1, :total_steps_ol, :], dtype=np.float32
                )
                denom_ol = np.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
                l2_ol_chunks.append(
                    np.linalg.norm(x_pred_dense[..., :N] - x_ref_dense_ol, axis=2) / denom_ol
                )
                eqvar_ol_chunks.append(
                    _equilibrium_variance_per_ic(x_pred_dense[..., :N], eqvar_burn_in_frac)
                )
                del x_pred_dense, x_ref_dense_ol, denom_ol
 
            l2_ol = np.mean(np.concatenate(l2_ol_chunks, axis=0), axis=0)
            eqvar_mean_ol, eqvar_std_ol = _mean_std(
                np.concatenate(eqvar_ol_chunks, axis=0)
            )
            del l2_ol_chunks, eqvar_ol_chunks, predict_full
            jax.clear_caches()
 
            open_loop_records[prop_key] = dict(
                t=np.array(t_test[:total_steps_ol]), l2_dense_mean=l2_ol,
                eqvar_mean=eqvar_mean_ol, eqvar_std=eqvar_std_ol,
            )
 
    # ── 5. Write everything to HDF5 ────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".partial"
 
    with h5py.File(tmp_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["N"] = N
        meta.attrs["dt_window"] = DT_WINDOW
        meta.attrs["dt_fine"] = DT_FINE
        meta.attrs["dt_obs"] = DT_OBS
        meta.attrs["sigma_obs"] = sigma_obs
        meta.attrs["P0_sigma"] = P0_sigma
        meta.attrs["N_ens"] = N_ens
        meta.attrs["obs_every_n"] = obs_every_n
        meta.attrs["m"] = m
        meta.attrs["num_ics_traj"] = num_plots
        meta.attrs["num_ics_batch"] = B
        meta.attrs["trajectory_windows"] = trajectory_windows
        meta.attrs["batch_windows"] = batch_windows
        meta.attrs["test_data"] = os.path.basename(test_h5_path)
        meta.create_dataset("obs_indices", data=np.array(obs_indices))
        meta.create_dataset("F_traj", data=F_test[:num_plots])
        meta.create_dataset(
            "strategy_keys",
            data=np.array([s["key"] for s in strategies], dtype=h5py.string_dtype()),
        )
        meta.create_dataset(
            "strategy_labels",
            data=np.array([s["label"] for s in strategies], dtype=h5py.string_dtype()),
        )
        meta.create_dataset(
            "strategy_kind",
            data=np.array([s["kind"] for s in strategies], dtype=h5py.string_dtype()),
        )
        meta.create_dataset(
            "strategy_propagator",
            data=np.array([s["propagator"] for s in strategies], dtype=h5py.string_dtype()),
        )
        meta.create_dataset(
            "strategy_config_json",
            data=np.array(
                [json.dumps(s.get("config", {}), sort_keys=True) for s in strategies],
                dtype=h5py.string_dtype(),
            ),
        )
 
        traj_grp = f.create_group("trajectories")
        traj_grp.create_dataset("t_fine", data=t_fine_axis)
        for ic_idx, rec in enumerate(per_ic_records):
            ic_grp = traj_grp.create_group(f"ic_{ic_idx}")
            ic_grp.attrs["F"] = rec["F"]
            ic_grp.create_dataset("x_true", data=rec["x_true"], compression="gzip")
            ic_grp.create_dataset("obs_coords", data=rec["obs_coords"])
            strat_grp = ic_grp.create_group("strategies")
            for key, srec in rec["strategies"].items():
                sg = strat_grp.create_group(key)
                sg.attrs["l2_time_avg"] = srec["l2_time_avg"]
                sg.create_dataset("x_est", data=srec["x_est"], compression="gzip")
                sg.create_dataset("x_std", data=srec["x_std"], compression="gzip")
 
        batch_grp = f.create_group("batch")
        batch_grp.attrs["B"] = B
        batch_grp.attrs["eqvar_burn_in_frac"] = eqvar_burn_in_frac
        batch_grp.create_dataset("obs_times", data=obs_times_batch)
        batch_grp.create_dataset("window_idx", data=np.arange(1, batch_windows + 1))
        batch_grp.create_dataset("t_dense_fine", data=t_dense_fine)
 
        ref_grp = batch_grp.create_group("reference")
        ref_grp.create_dataset("eqvar_mean", data=ref_eqvar_mean)
        ref_grp.create_dataset("eqvar_std", data=ref_eqvar_std)
 
        strat_grp_b = batch_grp.create_group("strategies")
        for key, rec in batch_strat_records.items():
            sg = strat_grp_b.create_group(key)
            sg.attrs["label"] = rec["label"]
            sg.attrs["kind"] = rec["kind"]
            sg.attrs["propagator"] = rec["propagator"]
            for field in (
                "prior_rmse_mean", "prior_rmse_std", "post_rmse_mean", "post_rmse_std",
                "erf_mean", "erf_std", "rmse_window_mean", "spread_window_mean",
                "rmse_raw", "spread_raw", "l2_dense_mean",
                "eqvar_mean", "eqvar_std",
            ):
                sg.create_dataset(field, data=rec[field])
            if rec["kind"] == "route_b":
                sg.create_dataset("route_b_scale_mean", data=rec["route_b_scale_mean"])
                sg.create_dataset("route_b_scale_std", data=rec["route_b_scale_std"])
                sg.attrs["route_b_alpha"] = rec["route_b_alpha"]
                sg.attrs["route_b_beta"] = rec["route_b_beta"]
 
        ol_grp = batch_grp.create_group("open_loop")
        for prop_key, rec in open_loop_records.items():
            og = ol_grp.create_group(prop_key)
            og.create_dataset("t", data=rec["t"])
            og.create_dataset("l2_dense_mean", data=rec["l2_dense_mean"])
            og.create_dataset("eqvar_mean", data=rec["eqvar_mean"])
            og.create_dataset("eqvar_std", data=rec["eqvar_std"])
 
    os.replace(tmp_path, out_path)          # publish only a fully written file
    logging.info(f"evaluate_filters: wrote all evaluation data to {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────
# Strategy configuration:  config.eval.strategies  ->  list of jobs
# ─────────────────────────────────────────────────────────────────────────

SURROGATES = ("dd", "pi")

# Per inflation type: which EnKF machinery it uses (`kind`), which params are
# required, which are optional (with defaults), and which may be lists.
_INFLATION_SCHEMA = {
    "multiplicative": dict(
        kind="standard", required=("inflation_factor",), optional={},
        sweepable=("inflation_factor",),
    ),
    "additive": dict(
        kind="route_b", required=("alpha",), optional={},
        sweepable=("alpha",),
    ),
    "route_b": dict(
        kind="route_b", required=("alpha", "beta"), optional={},
        sweepable=("alpha", "beta"),
    ),
    "rtpp": dict(
        kind="rtpp", required=("alpha_rtpp",), optional={"alpha_fine": 1.0},
        sweepable=("alpha_rtpp",),
    ),
}

# (inflation, param) -> (predicate, description of the valid range)
_PARAM_RANGES = {
    ("multiplicative", "inflation_factor"): (lambda v: v > 0, "> 0"),
    ("additive", "alpha"): (lambda v: v >= 0, ">= 0"),
    ("route_b", "alpha"): (lambda v: v >= 0, ">= 0"),
    ("route_b", "beta"): (lambda v: v >= 0, ">= 0"),
    ("rtpp", "alpha_rtpp"): (lambda v: 0.0 <= v <= 1.0, "in [0, 1]"),
    ("rtpp", "alpha_fine"): (lambda v: v > 0, "> 0"),
}

# EnKF-closure factory on the model, per `kind`.
_CLOSURE_METHOD = {
    "standard": "make_enkf_fns",
    "route_b": "make_route_b_enkf_fns",
    "rtpp": "make_rtpp_enkf_fns",
}

_SURROGATE_LABEL = {"dd": "DD", "pi": "PI"}
_INFLATION_LABEL = {
    "multiplicative": "Mult. Infl.", "additive": "Add. Infl.",
    "route_b": "Route B", "rtpp": "RTPP",
}

# Identifying names become file names, and ConfigDict rejects dots in keys.
_NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _plain(obj):
    """ConfigDict / FrozenConfigDict -> plain nested dict; anything else unchanged."""
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def _is_seq(v) -> bool:
    # FrozenConfigDict turns lists into tuples; configs may also hold arrays.
    return isinstance(v, (list, tuple, np.ndarray))


def _norm_token(s) -> str:
    """'Route B' / 'route-b' / 'ROUTE_B' -> 'route_b'."""
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


@dataclasses.dataclass(frozen=True)
class _StrategyDef:
    """One validated entry of `config.eval.strategies` (params may hold lists)."""
    name: str
    surrogate: str          # "dd" | "pi"
    inflation: str          # key of _INFLATION_SCHEMA
    params: dict            # param -> float | list[float], schema order, defaults filled in


@dataclasses.dataclass(frozen=True)
class _Job:
    """One evaluation = one HDF5 file: a strategy at one concrete parameter value."""
    name: str               # identifying name from the config
    stem: str               # file stem: `name`, `name_ver_i` or `name_ver_i_j`
    surrogate: str
    inflation: str
    params: dict            # scalars only
    version: tuple | None   # 1-based indices of the swept parameters; None if not swept
    label: str

    def config_dict(self) -> dict:
        return dict(surrogate=self.surrogate, inflation=self.inflation,
                    params=dict(self.params))


def _check_number(v, where: str, inflation: str, param: str) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(f"{where}: parameter '{param}' must be a number (or a list of "
                        f"numbers), got {v!r}.")
    f = float(v)
    ok, desc = _PARAM_RANGES[(inflation, param)]
    if not np.isfinite(f) or not ok(f):
        raise ValueError(f"{where}: parameter '{param}' must be {desc}, got {f:g}.")
    return f


def _parse_strategy_entry(name, entry) -> _StrategyDef:
    where = f"config.eval.strategies['{name}']"
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"{where}: identifying names become file names and may only "
                         "contain letters, digits, '_' and '-'.")
    if not isinstance(entry, dict):
        raise TypeError(f"{where} must be a dict with keys 'surrogate', 'inflation', "
                        f"'params'; got {type(entry).__name__}.")
    expected = {"surrogate", "inflation", "params"}
    if set(entry) != expected:
        raise ValueError(f"{where}: expected exactly the keys {sorted(expected)}, "
                         f"got {sorted(entry)}.")

    surrogate = _norm_token(entry["surrogate"])
    if surrogate not in SURROGATES:
        raise ValueError(f"{where}: surrogate must be 'DD' or 'PI', got {entry['surrogate']!r}.")

    inflation = _norm_token(entry["inflation"])
    if inflation not in _INFLATION_SCHEMA:
        raise ValueError(f"{where}: inflation must be one of {sorted(_INFLATION_SCHEMA)}, "
                         f"got {entry['inflation']!r}.")
    schema = _INFLATION_SCHEMA[inflation]

    raw = entry["params"]
    if not isinstance(raw, dict):
        raise TypeError(f"{where}['params'] must be a dict, got {type(raw).__name__}.")
    allowed = tuple(schema["required"]) + tuple(schema["optional"])
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown parameter(s) {sorted(unknown)} for "
                         f"inflation '{inflation}'; allowed: {list(allowed)}.")
    missing = set(schema["required"]) - set(raw)
    if missing:
        raise ValueError(f"{where}: missing required parameter(s) {sorted(missing)} for "
                         f"inflation '{inflation}'.")

    params = {}
    for p in allowed:                                   # schema order -> stable JSON / labels
        if p not in raw:
            params[p] = float(schema["optional"][p])
            continue
        v = raw[p]
        if _is_seq(v):
            if p not in schema["sweepable"]:
                raise ValueError(f"{where}: parameter '{p}' cannot be a list.")
            if len(v) == 0:
                raise ValueError(f"{where}: parameter '{p}' is an empty list.")
            params[p] = [_check_number(x, where, inflation, p) for x in v]
        else:
            params[p] = _check_number(v, where, inflation, p)
    return _StrategyDef(name=name, surrogate=surrogate, inflation=inflation, params=params)


def parse_strategy_config(config) -> dict:
    """Validate `config.eval.strategies` -> {name: _StrategyDef}, in the config's order."""
    raw = _plain(config.eval.get("strategies", None))
    if not raw:
        raise ValueError(
            "config.eval.strategies is missing or empty. Expected e.g. "
            "{'pi_mult': dict(surrogate='PI', inflation='multiplicative', "
            "params=dict(inflation_factor=1.05))}."
        )
    if not isinstance(raw, dict):
        raise TypeError("config.eval.strategies must be a dict keyed by identifying name, "
                        f"got {type(raw).__name__}.")
    return {name: _parse_strategy_entry(name, entry) for name, entry in raw.items()}


def _make_label(surrogate: str, inflation: str, params: dict) -> str:
    if inflation == "multiplicative":
        ptxt = f"α={params['inflation_factor']:g}"
    elif inflation == "additive":
        ptxt = f"α={params['alpha']:g}"
    elif inflation == "route_b":
        ptxt = f"α={params['alpha']:g}, β={params['beta']:g}"
    else:  # rtpp
        ptxt = f"α={params['alpha_rtpp']:g}"
        if params["alpha_fine"] != 1.0:
            ptxt += f", α_fine={params['alpha_fine']:g}"
    return f"{_SURROGATE_LABEL[surrogate]} + {_INFLATION_LABEL[inflation]} ({ptxt})"


def expand_jobs(defs: dict) -> list:
    """
    One `_Job` per HDF5 file. A strategy without a parameter list gives one
    job named `<name>`; with a list (n values) it gives n jobs `<name>_ver_<i>`.
    Route B is indexed by both parameters, `<name>_ver_<i>_<j>` (alpha index,
    beta index); a scalar next to a list counts as a one-element list, and two
    lists give every (alpha, beta) combination.
    """
    jobs, stem_owner = [], {}
    for d in defs.values():
        schema = _INFLATION_SCHEMA[d.inflation]
        swept = [p for p in schema["sweepable"] if _is_seq(d.params[p])]

        if not swept:
            versions = [(None, dict(d.params))]
        else:
            axes = schema["sweepable"] if d.inflation == "route_b" else tuple(swept)
            grids = [
                [(p, i, v) for i, v in enumerate(
                    d.params[p] if _is_seq(d.params[p]) else [d.params[p]], start=1)]
                for p in axes
            ]
            versions = []
            for combo in itertools.product(*grids):
                params = dict(d.params)
                for p, _, v in combo:
                    params[p] = v
                versions.append((tuple(i for _, i, _ in combo), params))

        for version, params in versions:
            stem = d.name if version is None else f"{d.name}_ver_" + "_".join(map(str, version))
            if stem in stem_owner:
                raise ValueError(
                    f"Output-name collision: '{stem_owner[stem]}' and '{d.name}' would both "
                    f"write '{stem}.h5'. Rename one of the strategies."
                )
            stem_owner[stem] = d.name
            jobs.append(_Job(
                name=d.name, stem=stem, surrogate=d.surrogate, inflation=d.inflation,
                params=params, version=version,
                label=_make_label(d.surrogate, d.inflation, params),
            ))
    return jobs


def resolve_test_h5_path(config, data_dir: str = "data") -> str:
    """`config.eval.test_data_name` -> `<data_dir>/<name>.h5` (default: l96_forcing_test)."""
    name = str(config.eval.get("test_data_name", "l96_forcing_test"))
    if name.endswith(".h5"):
        name = name[:-3]
    return os.path.join(data_dir, f"{name}.h5")


# ─────────────────────────────────────────────────────────────────────────
# Surrogate loading / EnKF closures / strategy specs
# ─────────────────────────────────────────────────────────────────────────

class _Surrogates:
    """
    Lazily loads the DD / PI checkpoints and builds their EnKF closures.

    Each surrogate is loaded at most once, and each (surrogate, kind) closure
    pair is built at most once, however many parameter versions use it:
    inflation parameters are runtime scalars carried in the strategy spec, not
    baked into the closures, so a sweep costs no extra model calls.
    """

    def __init__(self, config, N_ens: int):
        self.config = config
        self.N_ens = N_ens
        dt_window = float(config.get("dt_window", 0.25))
        dt_integration = config.eval.get("dt_integration", 0.005)
        time_steps = int(round(dt_window / dt_integration)) + 1
        self.t_star_window = jnp.linspace(0.0, dt_window, time_steps)
        self._models = {}      # surrogate -> (model, params)
        self._fns = {}         # (surrogate, kind) -> (predict_fn, update_fn)
        self.N = None

    def propagator(self, surrogate: str):
        if surrogate not in self._models:
            self._models[surrogate] = self._load(surrogate)
            N = self._models[surrogate][0].N
            if self.N is None:
                self.N = N
            elif N != self.N:
                raise ValueError(
                    f"DD and PI checkpoints disagree on state dimension N "
                    f"({self.N} vs {N}); they must share one L96 system."
                )
        return self._models[surrogate]

    def _load(self, surrogate: str):
        cfg = self.config
        if surrogate == "pi":
            logging.info("Loading PI model...")
            model = models.L96UDON(cfg, self.t_star_window)
            ckpt_path = os.path.join(os.getcwd(), cfg.wandb.name_pi, "ckpt", "udon_model")
        else:
            logging.info("Loading DD model...")
            model = models.L96UDON_DD(cfg, self.t_star_window)
            ckpt_path = os.path.join(os.getcwd(), cfg.wandb.name_dd, "ckpt", "udon_model")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(os.getcwd(), cfg.wandb.name_dd, "ckpt", "udon_dd_model")
        model.state = restore_checkpoint(model.state, ckpt_path)
        return model, model.state.params

    def enkf_fns(self, surrogate: str, kind: str):
        if (surrogate, kind) not in self._fns:
            model, params = self.propagator(surrogate)
            method = _CLOSURE_METHOD[kind]
            factory = getattr(model, method, None)
            if factory is None:
                raise NotImplementedError(
                    f"The {_SURROGATE_LABEL[surrogate]} surrogate ({type(model).__name__}) "
                    f"does not expose `{method}`, which is needed for the '{kind}' "
                    "EnKF variant. Implement it on the model or use the other surrogate."
                )
            self._fns[(surrogate, kind)] = factory(params, N_ens=self.N_ens)
        return self._fns[(surrogate, kind)]

    def make_spec(self, job: _Job) -> dict:
        """The `evaluate_filters` strategy dict for one job."""
        cfg = self.config
        kind = _INFLATION_SCHEMA[job.inflation]["kind"]
        predict_fn, update_fn = self.enkf_fns(job.surrogate, kind)
        spec = dict(
            key=job.stem, label=job.label, kind=kind, propagator=job.surrogate,
            predict_fn=predict_fn, update_fn=update_fn, config=job.config_dict(),
        )
        p = job.params

        dt_window = float(cfg.get("dt_window", 0.25))
        dt_fine = float(cfg.kf.get("dt_fine", dt_window))
        steps_per_window = steps_per_window_exact(dt_window, dt_fine)

        if job.inflation == "multiplicative":
            spec["alpha_fine"] = scale_inflation_for_fine_steps(
                p["inflation_factor"], steps_per_window)
        elif job.inflation == "rtpp":
            spec["alpha_fine"] = p["alpha_fine"]
            spec["alpha_rtpp"] = p["alpha_rtpp"]
        else:  # additive / route_b share the Route B machinery
            P0_sigma = cfg.kf.get("P0_sigma", 1.0)
            Q0_sigma = cfg.kf.get("Q0_sigma", P0_sigma)
            Q_coarse = jnp.eye(self.N) * Q0_sigma ** 2
            spec["Q0"] = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
            spec["n_quad"] = cfg.kf.get("route_b_n_quad", 3)
            spec["alpha"] = p["alpha"]
            spec["beta"] = 0.0 if job.inflation == "additive" else p["beta"]
        return spec


def _shared_settings(config) -> dict:
    """The config-derived settings stored in every file's /meta (mirrors the
    defaults `evaluate_filters` uses); compared against cached files."""
    dt_window = float(config.get("dt_window", 0.25))
    return dict(
        N_ens=int(config.kf.get("N_ens", 50)),
        sigma_obs=float(config.kf.get("sigma_obs", 0.5)),
        P0_sigma=float(config.kf.get("P0_sigma", 1.0)),
        obs_every_n=int(config.kf.get("obs_every_n", 4)),
        dt_window=dt_window,
        dt_fine=float(config.kf.get("dt_fine", dt_window)),
        dt_obs=float(config.kf.get("dt_obs", dt_window)),
        trajectory_windows=int(config.eval.get("trajectory_windows", 200)),
        batch_windows=int(config.eval.get("windows", 200)),
    )


def _py(x):
    """numpy scalar -> plain Python value (for readable messages)."""
    return x.item() if isinstance(x, np.generic) else x


def _same_value(a, b) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return a == b


def _warn_if_stale(path: str, job: _Job, settings: dict) -> None:
    """A cached file is reused as-is (by design); just say so if it looks outdated."""
    problems = []
    with h5py.File(path, "r") as f:
        meta = f["meta"]
        if "strategy_config_json" in meta:
            stored = json.loads(_decode(meta["strategy_config_json"][:])[0])
            if stored != json.loads(json.dumps(job.config_dict())):
                problems.append(f"strategy config differs (file: {stored}, "
                                f"config: {job.config_dict()})")
        else:
            problems.append("file has no stored strategy config")
        for k, v in settings.items():
            if k in meta.attrs and not _same_value(meta.attrs[k], v):
                problems.append(f"{k}: file={_py(meta.attrs[k])!r}, config={v!r}")
    if problems:
        logging.warning(
            f"{path} already exists and will NOT be re-evaluated, but it looks out of date "
            f"relative to the current config: {'; '.join(problems)}. "
            "Delete the file to force a re-run."
        )


def _evaluate_pending(config, jobs: list, h5_path_of: dict) -> None:
    """Evaluate every job whose HDF5 file does not exist yet."""
    settings = _shared_settings(config)
    pending = []
    for job in jobs:
        path = h5_path_of[job.stem]
        if os.path.exists(path):
            logging.info(f"'{job.stem}': {path} exists -- skipping evaluation.")
            _warn_if_stale(path, job, settings)
        else:
            pending.append(job)
    if not pending:
        logging.info("All requested evaluation files already exist; nothing to evaluate.")
        return

    test_h5_path = resolve_test_h5_path(config)
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(
            f"Test data '{test_h5_path}' not found (from config.eval.test_data_name = "
            f"{config.eval.get('test_data_name', 'l96_forcing_test')!r})."
        )

    # Load the needed surrogates and build EVERY closure/spec up front, so an
    # unsupported combination (e.g. a surrogate without a Route B closure)
    # fails immediately instead of after earlier jobs have already run.
    surrogates = _Surrogates(config, N_ens=settings["N_ens"])
    specs = [(job, surrogates.make_spec(job)) for job in pending]

    for i, (job, spec) in enumerate(specs, start=1):
        logging.info(f"Evaluating {i}/{len(specs)}: '{job.stem}'  [{job.label}] ...")
        evaluate_filters(
            config=config,
            strategies=[spec],
            propagators={job.surrogate: surrogates.propagator(job.surrogate)},
            t_star_window=surrogates.t_star_window,
            out_path=h5_path_of[job.stem],
            test_h5_path=test_h5_path,
        )


# ─────────────────────────────────────────────────────────────────────────
# Small shared helpers
# ─────────────────────────────────────────────────────────────────────────
def _decode(arr):
    """h5py string datasets may come back as bytes; normalize to str."""
    out = []
    for v in arr:
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out

def _binned_spread_skill(rmss, rmse, n_bins=10):
    """
    Bin raw (RMSS, RMSE) pairs into `n_bins` equal-population bins
    (deciles by default) over RMSS. Raw per-(IC, window) spread/skill
    pairs form an unreadable cloud; binning is the standard fix.

    Returns bin_rmss_mean, bin_rmse_mean, bin_rmse_std, bin_counts, each
    shape (n_bins,) (fewer if some bins end up empty).
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

def _save(fig, save_path, dpi=300):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def _save_pdf_pages(figs, save_path, dpi=300):
    """Save a list of already-built figures as successive pages of ONE
    PDF file, then close them. Used wherever a single logical plot needs
    a companion page (e.g. the same curves with some series omitted)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with PdfPages(save_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 1. Individual-trajectory plot (single strategy vs ground truth)
# ─────────────────────────────────────────────────────────────────────────
def _plot_trajectory_individual(
    t_ax: np.ndarray,          # (T,)   time axis
    x_true: np.ndarray,        # (T, N) ground-truth state
    x_est: np.ndarray,         # (T, N) strategy's EnKF mean
    x_std: np.ndarray | None,  # (T, N) strategy's ensemble std, or None
    ic_idx: int,
    F_val: float,
    strategy_label: str,
    l2_time_avg: float | None,
    save_path: str,
    N: int = 40,
    dt_window: float | None = None,
    obs_coords=None,           # iterable of (var_idx, t_obs, y_obs)
) -> None:
    """
    Trajectory-summary PDF for ONE strategy on ONE IC: top panel is the
    mean |error| vs time, followed by one panel per state variable with
    truth, the strategy's mean, a +/-1 sigma ensemble-spread band, and
    assimilated-observation markers. Same layout as the historical
    N-way trajectory-summary plots, collapsed to a single strategy.
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    x_std = np.asarray(x_std) if x_std is not None else None

    mean_abs_err = np.abs(x_true - x_est).mean(axis=1)
    n_var_rows = N // 2

    # ── Window-boundary times ──────────────────────────────────────────
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(
            first_k * dt_window, t_max + 1e-12 * dt_window, dt_window
        )
    else:
        window_boundaries = np.array([])

    # ── Per-variable observation times ───────────────────────────────────
    obs_by_var: dict[int, list[tuple[float, float]]] = {}
    if obs_coords is not None:
        for var_idx, obs_t, obs_val in obs_coords:
            obs_by_var.setdefault(int(var_idx), []).append((float(obs_t), float(obs_val)))
        obs_by_var = {k: sorted(v, key=lambda x: x[0]) for k, v in obs_by_var.items()}

    # ── Figure & GridSpec ────────────────────────────────────────────────
    top_height = 3.2
    var_row_h = 1.9
    total_height = top_height + n_var_rows * var_row_h

    fig = plt.figure(figsize=(14, total_height))
    gs = gridspec.GridSpec(
        nrows=1 + n_var_rows, ncols=2, figure=fig,
        height_ratios=[top_height] + [var_row_h] * n_var_rows,
        hspace=0.55, wspace=0.32,
    )

    # ── Top panel: mean absolute error vs time ─────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t_ax, mean_abs_err, color="#2196F3", linewidth=1.1,
                label=f"{strategy_label}: Mean |error|")

    for wb in window_boundaries:
        ax_top.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.8,
                        alpha=0.55,
                        label="Window boundary" if wb == window_boundaries[0] else None)

    l2_str = f"  |  time-avg L2 = {l2_time_avg:.3e}" if l2_time_avg is not None else ""
    ax_top.set_xlabel("Time  t", fontsize=11)
    ax_top.set_ylabel("Mean absolute error", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"IC {ic_idx} — Mean absolute error across all {N} variables  "
        f"({strategy_label}){l2_str}",
        fontsize=12, fontweight="bold",
    )
    ax_top.legend(fontsize=9)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    TRUTH_COLOR = "#37474F"
    EST_COLOR, EST_BAND = "#2196F3", "#90CAF9"
    OBS_COLOR = "#E53935"

    # ── Per-variable panels ──────────────────────────────────────────────
    for i in range(N):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])

        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.6, alpha=0.45)

        ax.plot(t_ax, x_true[:, i], color=TRUTH_COLOR, linewidth=0.8, label="Truth")
        ax.plot(t_ax, x_est[:, i], color=EST_COLOR, linewidth=0.8, linestyle="--",
                label=strategy_label)
        if x_std is not None:
            ax.fill_between(
                t_ax, x_est[:, i] - x_std[:, i], x_est[:, i] + x_std[:, i],
                color=EST_BAND, alpha=0.30, linewidth=0, label=f"{strategy_label} ±1σ",
            )

        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])
            ax.scatter(obs_times_i, obs_vals_i, marker="x", s=14, linewidths=0.6,
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
            ax.legend(fontsize=6.5, loc="upper right", handlelength=1.2,
                      framealpha=0.7, ncol=2)

    fig.suptitle(
        f"Trajectory summary — IC {ic_idx} (F = {F_val:.2f})  |  {strategy_label}",
        fontsize=13, fontweight="bold", y=1.002,
    )
    _save(fig, save_path, dpi=150)
    logging.info(f"Individual trajectory plot (IC {ic_idx}, {strategy_label}) saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Overlay ("bulk") plots: every strategy of a collection on the same axes
# ─────────────────────────────────────────────────────────────────────────
#
# Decluttering strategy (keeps 1..~16+ overlaid strategies legible):
#
#   * Calibration's spread/RMSE timeseries uses small multiples: one row
#     per strategy, stacked vertically, each pairing that strategy's own
#     RMS ensemble spread and EnKF RMSE on a single (small) axes.
#   * The prior/posterior-RMSE figure is split into two side-by-side
#     subplots (one metric each), so each only ever has S lines.
#   * Every strategy gets a single, STABLE color from `_strategy_colors`
#     (backed by `_distinct_colors`: tab20/tab20b/tab20c, 60 distinguishable
#     swatches before any hue repeats), reused across every panel and plot.
#   * Legends switch to two columns above ~4-5 strategies, and error-bar /
#     spread-band decorations fade (or are dropped, for ERF bands beyond 4
#     strategies) as S grows, so the trend lines stay dominant.
#   * The EnKF-vs-open-loop L2 plot writes a multi-page PDF: all curves, then
#     a page without the open-loop curves (which have a much larger scale).
#   * The L2, ERF and prior/posterior-RMSE PDFs each carry an extra "best
#     strategies only" page (see `_rank_by_mean` / `TOP_K_BEST`) holding at
#     most the 3 top-ranked strategies by that plot's own headline metric.
#     Skipped when S <= 3, where it would only repeat the first page -- so a
#     sweep over many parameter values shows both the whole sweep and which
#     settings actually won.
#
# All of these work for S = 1 (a collection with a single strategy).

def _distinct_colors(n: int) -> list[str]:
    """
    Generate `n` hex colors that stay visually distinguishable well past
    the old fixed 10-color palette, which started repeating once plots
    like the 16-line (DD/PI x alpha) multiplicative-inflation sweep
    overlaid more strategies than it had colors for. Draws first from
    matplotlib's qualitative tab20/tab20b/tab20c colormaps (60 swatches
    designed to be pairwise distinguishable); if more than 60 strategies
    are ever plotted at once, falls back to additional evenly spaced HSV
    hues rather than repeating a color.
    """
    swatches = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        swatches.extend(plt.get_cmap(cmap_name).colors)
    if n > len(swatches):
        n_extra = n - len(swatches)
        swatches = swatches + [
            colorsys.hsv_to_rgb(h, 0.65, 0.85)
            for h in np.linspace(0.0, 1.0, n_extra, endpoint=False)
        ]
    return [mcolors.to_hex(c) for c in swatches[:n]]


def _strategy_colors(strategy_keys):
    """Stable, deterministic strategy -> color mapping shared by every
    bulk plot, so the same strategy always gets the same color
    everywhere, with no repeats up to `_distinct_colors`'s 60-swatch
    palette (see above)."""
    palette = _distinct_colors(len(strategy_keys))
    return {k: palette[i] for i, k in enumerate(strategy_keys)}


# ─────────────────────────────────────────────────────────────────────────
# Shared ranking helper for the "best strategies only" companion pages
# ─────────────────────────────────────────────────────────────────────────
TOP_K_BEST = 3   # max curves kept on the "best strategies only" PDF pages


def _rank_by_mean(metric_of, keys=None, largest=False, k=TOP_K_BEST):
    """
    Rank strategies by the (finite) mean of a per-strategy metric array
    and return the best `k` of them.

    Used by the bulk plots to build an extra, decluttered PDF page that
    shows only the handful of best-performing strategies -- handy when
    calibrating a large sweep (e.g. the 16-line multiplicative-inflation
    scan), where the all-strategies page answers "what does the whole
    sweep look like?" and this page answers "which settings actually
    won?".

    Parameters
    ----------
    metric_of : dict
        key -> 1-D array of the metric sampled over time (e.g. posterior
        RMSE per observation time, ERF per observation time, relative L2
        per fine timestep). Each array is collapsed to one scalar by
        taking the mean over its finite entries.
    keys : iterable, optional
        Which keys of `metric_of` to rank (defaults to all of them).
        Lets callers exclude curves that aren't strategies, e.g. the
        open-loop reference rollouts in the L2 plot.
    largest : bool
        False -> "lower is better" (RMSE, L2); True -> "higher is
        better" (ERF).
    k : int
        Maximum number of keys to return.

    Returns
    -------
    (ranked_keys, scores)
        `ranked_keys` holds at most `k` keys, best first; `scores` maps
        EVERY ranked key to its scalar summary, so callers can quote the
        number in a subtitle. Keys whose metric is entirely non-finite
        sort last and are given a NaN score.
    """
    keys = list(metric_of.keys() if keys is None else keys)
    scores, sortable = {}, []
    for key in keys:
        arr = np.asarray(metric_of[key], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            score = float(finite.mean())
            scores[key] = score
            # Negate for "largest is best" so a single ascending sort
            # handles both directions; non-finite metrics get +inf and
            # therefore always sort to the very end.
            sortable.append(((-score if largest else score), key))
        else:
            scores[key] = float("nan")
            sortable.append((float("inf"), key))

    order = [key for _, key in sorted(sortable, key=lambda item: item[0])]
    return order[:max(0, int(k))], scores


# ─────────────────────────────────────────────────────────────────────────
# 2a. EnKF vs open-loop, time-mean relative L2 (curve-count-agnostic --
#     called with S filtered curves plus the open-loop references).
# ─────────────────────────────────────────────────────────────────────────
def _plot_l2_per_timestep(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],  # label -> (t_axis, l2_array)
    title: str,
    save_path: str,
    colors: dict[str, str] | None = None,
) -> None:
    """
    Plot average L2 error continuously across fine time stamps, as a
    multi-page PDF:

      page 1 -- every curve in `curves` (matches the original behavior).
      page 2 -- the same plot with the pure-propagator open-loop curves
        (any label ending in "open-loop", e.g. "dd open-loop") omitted,
        so the filtered-strategy curves aren't dwarfed by the open-loop
        curves' much larger error scale. Skipped if there are no
        open-loop curves to drop.
      pages 3-4 -- duplicates of pages 1 and 2 restricted to the (at
        most) `TOP_K_BEST` filtered strategies with the LOWEST
        time-mean relative L2, i.e. the best performers of the sweep,
        with the open-loop references kept on the first of the two for
        scale. Both are skipped when there are no more than
        `TOP_K_BEST` filtered curves to begin with, since the pages
        would just repeat pages 1-2 (this is what keeps the pairwise
        two-strategy caller's output unchanged).
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    open_loop_labels = {label for label in curves if label.endswith("open-loop")}
    filtered_curves = {k: v for k, v in curves.items() if k not in open_loop_labels}

    def _draw(curve_subset, subtitle):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, (label, (t_axis, l2_arr)) in enumerate(curve_subset.items()):
            color = (colors or {}).get(label, default_colors[i % len(default_colors)])
            ax.plot(t_axis, l2_arr, linewidth=1.1, label=label, color=color)

        ax.set_yscale("log")
        ax.set_xlabel("Time (t)", fontsize=12)
        ax.set_ylabel("Mean relative L2 error (log scale)", fontsize=12)
        ax.set_title(subtitle, fontsize=13)
        ax.legend(fontsize=9, ncol=(2 if len(curve_subset) > 5 else 1))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        return fig

    figs = [_draw(curves, title)]
    if open_loop_labels and filtered_curves:
        figs.append(_draw(
            filtered_curves,
            title + "\n(pure-propagator open-loop curves omitted)",
        ))

    # -- Best-performing strategies only (lowest time-mean relative L2) --
    if len(filtered_curves) > TOP_K_BEST:
        best, scores = _rank_by_mean(
            {label: l2_arr for label, (_, l2_arr) in filtered_curves.items()},
            largest=False, k=TOP_K_BEST,
        )
        best_curves = {label: filtered_curves[label] for label in best}
        ranking = ", ".join(f"{label} ({scores[label]:.3g})" for label in best)
        best_note = (
            f"\nBest {len(best)} strategies by time-mean relative L2: {ranking}"
        )
        if open_loop_labels:
            with_ol = {label: val for label, val in curves.items()
                       if label in open_loop_labels}
            with_ol.update(best_curves)
            figs.append(_draw(with_ol, title + best_note))
        figs.append(_draw(
            best_curves,
            title + best_note + "\n(pure-propagator open-loop curves omitted)",
        ))

    _save_pdf_pages(figs, save_path)
    logging.info(
        f"L2-vs-open-loop comparison plot ({len(figs)}-page PDF) saved to: {save_path}"
    )

# ─────────────────────────────────────────────────────────────────────────
# 2b. Calibration, ALL strategies on one PDF
# ─────────────────────────────────────────────────────────────────────────
def _plot_calibration_bulk(
    strategy_keys,
    label_of,
    window_idx: np.ndarray,
    dt_window: float,
    spread_window_of: dict,   # key -> (n_window,) sim-time-mean RMS spread
    rmse_window_of: dict,     # key -> (n_window,) sim-time-mean EnKF RMSE
    spread_raw_of: dict,      # key -> raw (IC, window) spread pairs, flattened
    rmse_raw_of: dict,        # key -> raw (IC, window) RMSE pairs, flattened
    colors: dict,
    title: str,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Single calibration PDF covering every strategy at once:

      * one small row per strategy, stacked vertically, each pairing
        that strategy's own RMS ensemble spread and EnKF RMSE
        (simulation time) on the SAME axes -- S small line plots
        instead of two "every strategy overlaid" panels, so a given
        strategy's own spread/RMSE relationship stays legible even when
        S is large (e.g. the 16-line multiplicative-inflation sweep).
      * a final row with the pooled binned spread-skill scatter, shown
        in both linear and log/log axes, side by side.

    The old secondary "simulation time (window x dt)" tick strip is
    dropped -- with many strategies/windows it just overcrowded the
    axes; the window-to-time conversion (`dt_window`) is instead
    reported once, in the figure title.
    """
    S = len(strategy_keys)
    row_h = 1.15
    bin_row_h = 4.4
    fig_h = row_h * S + bin_row_h + 1.0
    fig = plt.figure(figsize=(9.5, fig_h))
    gs = gridspec.GridSpec(
        S + 1, 2, height_ratios=[row_h] * S + [bin_row_h], hspace=0.65, wspace=0.3,
    )

    # -- One row per strategy: spread + RMSE paired on the same axes ----
    ax_prev = None
    for i, key in enumerate(strategy_keys):
        ax = fig.add_subplot(gs[i, :], sharex=ax_prev)
        ax_prev = ax
        c = colors[key]
        # Marker-free lines: with many windows the per-point markers just
        # merged into a solid band and hid the curve shape. The solid /
        # dashed linestyle still separates spread from RMSE.
        ax.plot(window_idx, spread_window_of[key],
                 linewidth=1.0, linestyle="-", color=c, label="RMS ensemble σ")
        ax.plot(window_idx, rmse_window_of[key],
                 linewidth=1.0, linestyle="--", color=c, label="EnKF RMSE")
        ax.set_yscale("log")
        ax.set_title(label_of[key], fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
        if i == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7)
            ax.set_ylabel("Spread / RMSE\n(log)", fontsize=6.5)
        if i < S - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Window index", fontsize=9)

    # -- Final row: pooled binned spread-skill, linear + log/log --------
    ax_lin = fig.add_subplot(gs[S, 0])
    ax_log = fig.add_subplot(gs[S, 1])

    binned = {}
    lim_hi, lim_lo = 0.0, np.inf
    for key in strategy_keys:
        rmss_b, rmse_b, rmse_s, _ = _binned_spread_skill(
            spread_raw_of[key], rmse_raw_of[key], n_bins)
        binned[key] = (rmss_b, rmse_b, rmse_s)
        vals = np.concatenate([rmss_b, rmse_b])
        lim_hi = max(lim_hi, float(vals.max()))
        pos_vals = vals[vals > 0]
        if pos_vals.size:
            lim_lo = min(lim_lo, float(pos_vals.min()))
    lim_hi *= 1.1
    lim_lo = lim_lo / 1.5 if np.isfinite(lim_lo) else lim_hi * 1e-3

    # Error bars get busy fast with many strategies pooled on one axes;
    # fade just the bars/caps as S grows while keeping the trend line and
    # markers fully opaque, so shapes stay readable.
    eb_alpha = max(0.25, 0.9 - 0.12 * S)
    for ax, log_scale in ((ax_lin, False), (ax_log, True)):
        lo = lim_lo if log_scale else 0.0
        ax.plot([lo, lim_hi], [lo, lim_hi], linestyle="--", linewidth=1.0,
                 color="#37474F", label="1:1 (perfect calibration)", zorder=1)
        for key in strategy_keys:
            rmss_b, rmse_b, rmse_s = binned[key]
            container = ax.errorbar(
                rmss_b, rmse_b, yerr=rmse_s, fmt="o-", markersize=1.4, capsize=1.8,
                linewidth=1.0, color=colors[key],
                label=f"{label_of[key]} ({n_bins}-bin)", zorder=3,
            )
            for cap in container[1]:
                cap.set_alpha(eb_alpha)
            for barcol in container[2]:
                barcol.set_alpha(eb_alpha)

        ax.set_xlabel("RMS ensemble spread (RMSS)", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_title(f"Binned spread-skill (log/log, {n_bins}-bin)", fontsize=10.5)
        else:
            ax.set_xlim(0, lim_hi)
            ax.set_ylim(0, lim_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Binned spread-skill (linear, {n_bins}-bin)", fontsize=10.5)

    ax_lin.set_ylabel("RMSE of ensemble mean", fontsize=10)
    ax_lin.legend(fontsize=6.5, ncol=(2 if S > 4 else 1))

    fig.suptitle(
        f"{title}\n(window index × dt_window={dt_window:g} = simulation time)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=[0.04, 0, 1, 1 - 0.85 / fig_h])
    _save(fig, save_path)
    logging.info(
        f"Bulk calibration plot ({S} strategies, stacked small-multiples) saved to: {save_path}"
    )

# ─────────────────────────────────────────────────────────────────────────
# 2c. Error Reduction Factor, ALL strategies on one PDF
# ─────────────────────────────────────────────────────────────────────────
def _plot_erf_bulk(
    strategy_keys,
    label_of,
    obs_times: np.ndarray,
    erf_mean_of: dict,
    erf_std_of: dict,
    colors: dict,
    n_traj: int,
    title: str,
    save_path: str,
    show_bands: bool | None = None,
) -> None:
    """
    ERF comparison for every strategy on ONE set of axes, written as a
    multi-page PDF:

      page 1 -- every strategy (the original behavior).
      page 2 -- the same axes restricted to the (at most) `TOP_K_BEST`
        strategies with the GREATEST time-mean ERF, i.e. the ones whose
        filtering reduces the error the most. Skipped when there are no
        more than `TOP_K_BEST` strategies, since it would just repeat
        page 1.

    +/-1 sigma bands are auto-dropped once there are more than 4
    strategies, since overlapping fills stop conveying anything once
    they stack that deep -- evaluated per page, so the decluttered
    second page usually gets its bands back.
    """
    S = len(strategy_keys)

    def _draw(keys, subtitle):
        n = len(keys)
        bands = (n <= 4) if show_bands is None else show_bands
        band_alpha = 0.15 if n <= 3 else 0.08

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key in keys:
            c = colors[key]
            mean, std = erf_mean_of[key], erf_std_of[key]
            ax.plot(obs_times, mean, color=c, linewidth=1.1, marker="o", markersize=2.2,
                    label=label_of[key])
            if bands:
                ax.fill_between(obs_times, mean - std, mean + std, color=c,
                                 alpha=band_alpha, linewidth=0)

        ax.set_yscale("log")
        ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.1,
                   label="ERF = 1  (no reduction)")

        ax.set_xlabel("Observation time  t", fontsize=12)
        ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
        ax.set_title(f"{subtitle}  (n = {n_traj} trajectories)", fontsize=13)
        ax.legend(fontsize=8, ncol=(2 if n > 5 else 1))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if not bands:
            ax.text(0.99, 0.02, "±1σ bands omitted for legibility (n strategies > 4)",
                    transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                    color="#666666", style="italic")

        fig.tight_layout()
        return fig

    figs = [_draw(strategy_keys, title)]

    # -- Best-performing strategies only (greatest time-mean ERF) -------
    if S > TOP_K_BEST:
        best, scores = _rank_by_mean(
            erf_mean_of, keys=strategy_keys, largest=True, k=TOP_K_BEST,
        )
        ranking = ", ".join(f"{label_of[key]} ({scores[key]:.3g})" for key in best)
        figs.append(_draw(
            best,
            f"{title}\nBest {len(best)} strategies by mean ERF: {ranking}",
        ))

    _save_pdf_pages(figs, save_path)
    logging.info(
        f"Bulk ERF plot ({S} strategies, {len(figs)}-page PDF) saved to: {save_path}"
    )

# ─────────────────────────────────────────────────────────────────────────
# 2d. Prior vs posterior RMSE, ALL strategies on one PDF
# ─────────────────────────────────────────────────────────────────────────
def _plot_rmse_bulk(
    strategy_keys,
    label_of,
    obs_times: np.ndarray,
    prior_mean_of: dict,
    post_mean_of: dict,
    sigma_obs: float,
    n_traj: int,
    colors: dict,
    title: str,
    save_path: str,
) -> None:
    """
    Prior/posterior RMSE for every strategy, split into two side-by-side
    panels (prior, posterior) rather than 2*S lines on one axes; both
    panels share a y-axis and a single strategy-color legend. Both panels
    use small dot markers (rather than squares) and thin lines so the
    S-way overlay stays legible.

    Written as a multi-page PDF:

      page 1 -- every strategy (the original behavior).
      page 2 -- the same two panels restricted to the (at most)
        `TOP_K_BEST` strategies with the LOWEST time-mean POSTERIOR
        RMSE. Note both panels are filtered by that single posterior
        ranking, so the prior panel still shows where those same
        strategies started from. Skipped when there are no more than
        `TOP_K_BEST` strategies, since it would just repeat page 1.
    """
    S = len(strategy_keys)

    def _draw(keys, subtitle):
        n = len(keys)
        fig, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

        for key in keys:
            c = colors[key]
            ax_prior.plot(obs_times, prior_mean_of[key], color=c, linewidth=1.1,
                           marker="o", markersize=2.2, label=label_of[key])
            ax_post.plot(obs_times, post_mean_of[key], color=c, linewidth=1.1,
                          marker="o", markersize=2.2, label=label_of[key])

        for ax, panel_title in ((ax_prior, "Prior RMSE"), (ax_post, "Posterior RMSE")):
            ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.2,
                       label=f"σ_obs = {sigma_obs}")
            ax.set_yscale("log")
            ax.set_xlabel("Observation time  t", fontsize=11)
            ax.set_title(panel_title, fontsize=12)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax_prior.set_ylabel("RMSE  (log scale)", fontsize=11)
        plt.setp(ax_post.get_yticklabels(), visible=False)

        ax_prior.legend(fontsize=8, ncol=(2 if n > 4 else 1))

        fig.suptitle(f"{subtitle}  (n = {n_traj} trajectories)", fontsize=13, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    figs = [_draw(strategy_keys, title)]

    # -- Best-performing strategies only (lowest mean posterior RMSE) ---
    if S > TOP_K_BEST:
        best, scores = _rank_by_mean(
            post_mean_of, keys=strategy_keys, largest=False, k=TOP_K_BEST,
        )
        ranking = ", ".join(f"{label_of[key]} ({scores[key]:.3g})" for key in best)
        figs.append(_draw(
            best,
            f"{title}\nBest {len(best)} strategies by mean posterior RMSE: {ranking}",
        ))

    _save_pdf_pages(figs, save_path)
    logging.info(
        f"Bulk prior/posterior RMSE plot ({S} strategies, {len(figs)}-page PDF) "
        f"saved to: {save_path}"
    )

# ─────────────────────────────────────────────────────────────────────────
# 2e. Equilibrium (climatological/attractor) variance: reference truth +
#     static (open-loop) physics vs every filtered strategy, one PDF
# ─────────────────────────────────────────────────────────────────────────
def _plot_equilibrium_variance_bulk(
    strategy_keys,
    label_of,
    eqvar_mean_of: dict,          # key -> (N,) per-variable equilibrium variance
    eqvar_std_of: dict,           # key -> (N,) IC-to-IC std of that estimate
    reference_mean: np.ndarray,   # (N,) reference (unfiltered truth) equilibrium variance
    reference_std: np.ndarray,
    open_loop_eqvar: dict,        # propagator -> dict(mean=(N,), std=(N,))
    colors: dict,
    title: str,
    save_path: str,
    log_scale: bool = True,
) -> None:
    """
    Per-state-variable equilibrium (climatological/attractor) variance —
    the long-term temporal variance of each variable once transients
    have decayed — comparing:

      * the reference/truth trajectory (the intrinsic variability of the
        unforced, unfiltered system; the target every curve below is
        compared against),
      * each propagator's open-loop, "static" (unfiltered) physics
        rollout, and
      * every filtered EnKF strategy's posterior-mean trajectory,

    all on ONE set of axes, regardless of strategy count S (mirrors the
    other bulk "*_all" plots above). A well-calibrated filter's curve
    should track the reference closely; collapsing well below it signals
    ensemble/variance collapse (over-confident filtering, insufficient
    inflation), while sitting well above it signals over-inflation.
    """
    N = len(reference_mean)
    var_idx = np.arange(N)
    S = len(strategy_keys)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(var_idx, reference_mean, color="#37474F", linewidth=1.6,
            marker="o", markersize=2.2, label="Reference (unfiltered truth)", zorder=5)
    ax.fill_between(var_idx, reference_mean - reference_std, reference_mean + reference_std,
                     color="#37474F", alpha=0.15, linewidth=0, zorder=1)

    ol_palette = ["#B0BEC5", "#78909C", "#546E7A"]
    for i, prop_key in enumerate(sorted(open_loop_eqvar)):
        rec = open_loop_eqvar[prop_key]
        c = ol_palette[i % len(ol_palette)]
        ax.plot(var_idx, rec["mean"], color=c, linewidth=1.1, linestyle="--",
                 marker="^", markersize=2.2, zorder=4,
                 label=f"{prop_key} open-loop (static physics)")

    for key in strategy_keys:
        c = colors[key]
        ax.plot(var_idx, eqvar_mean_of[key], color=c, linewidth=1.1,
                 marker="o", markersize=2.2, label=label_of[key], zorder=3)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("State variable index", fontsize=12)
    ax.set_ylabel("Equilibrium variance" + ("  (log scale)" if log_scale else ""), fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=(2 if S > 5 else 1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _save(fig, save_path)
    logging.info(f"Bulk equilibrium-variance plot ({S} strategies) saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Individual-trajectory plots: one PDF per (IC, strategy file)
# ─────────────────────────────────────────────────────────────────────────
def plot_individual_trajectories(h5_path: str, indiv_dir: str) -> int:
    """
    Writes `trajectory_ic_<i>_<key>.pdf` into `indiv_dir` for every trajectory
    IC and every strategy stored in `h5_path`. Returns the number of PDFs.
    """
    os.makedirs(indiv_dir, exist_ok=True)
    n_pdfs = 0
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        N = int(meta.attrs["N"])
        dt_window = float(meta.attrs["dt_window"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])
        strategy_keys = _decode(meta["strategy_keys"][:])
        label_of = dict(zip(strategy_keys, _decode(meta["strategy_labels"][:])))
        t_fine_traj = f["trajectories/t_fine"][:]

        for ic_idx in range(num_ics_traj):
            ic_grp = f[f"trajectories/ic_{ic_idx}"]
            F_val = float(ic_grp.attrs["F"])
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = list(map(tuple, obs_coords_raw)) if obs_coords_raw.size else []

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=sg["x_est"][:], x_std=sg["x_std"][:],
                    ic_idx=ic_idx, F_val=F_val, strategy_label=label_of[key],
                    l2_time_avg=float(sg.attrs["l2_time_avg"]),
                    save_path=os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf"),
                    N=N, dt_window=dt_window, obs_coords=obs_coords,
                )
                n_pdfs += 1
    logging.info(f"plot_individual_trajectories: wrote {n_pdfs} PDF(s) from {h5_path} to {indiv_dir}")
    return n_pdfs


# ─────────────────────────────────────────────────────────────────────────
# Collection plots: several strategy files overlaid on the same axes
# ─────────────────────────────────────────────────────────────────────────

_BATCH_FIELDS = (
    "prior_rmse_mean", "post_rmse_mean", "erf_mean", "erf_std",
    "rmse_window_mean", "spread_window_mean", "rmse_raw", "spread_raw",
    "l2_dense_mean",
)

# /meta and /batch attributes that must agree for files to be overlaid: they
# fix the axes' shapes and the ICs / noise / ensemble the strategies saw.
_CONSISTENCY_ATTRS = (
    "N", "dt_window", "dt_fine", "dt_obs", "sigma_obs", "P0_sigma", "N_ens",
    "obs_every_n", "m", "num_ics_traj", "num_ics_batch", "trajectory_windows",
    "batch_windows",
)


def _read_group(h5_paths: list) -> dict:
    """
    Reads the batch-level data of several strategy files into one dict shaped
    for the `_plot_*_bulk` helpers. Raises if the files were produced with
    different shared settings (overlaying them would be meaningless).
    """
    if not h5_paths:
        raise ValueError("_read_group: no HDF5 files given.")

    g = dict(strategy_keys=[], label_of={}, propagator_of={}, batch_strat={},
             eqvar_mean_of={}, eqvar_std_of={}, open_loop={}, open_loop_eqvar={})
    ref_path, ref_attrs = None, None

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            meta, batch = f["meta"], f["batch"]
            attrs = {k: meta.attrs[k] for k in _CONSISTENCY_ATTRS}
            attrs["B"] = batch.attrs["B"]
            attrs["test_data"] = meta.attrs.get("test_data", None)

            if ref_attrs is None:
                ref_path, ref_attrs = path, attrs
                g.update(
                    N=int(attrs["N"]), dt_window=float(attrs["dt_window"]),
                    obs_every_n=int(attrs["obs_every_n"]), sigma_obs=float(attrs["sigma_obs"]),
                    N_ens=int(attrs["N_ens"]), B=int(attrs["B"]),
                    obs_times=batch["obs_times"][:], window_idx=batch["window_idx"][:],
                    t_dense_fine=batch["t_dense_fine"][:],
                    burn_in_frac=float(batch.attrs["eqvar_burn_in_frac"]),
                )
                if "reference" not in batch:
                    raise KeyError(f"{path} has no 'batch/reference' (equilibrium-variance "
                                   "data); delete it and re-run to regenerate it.")
                g["reference_mean"] = batch["reference/eqvar_mean"][:]
                g["reference_std"] = batch["reference/eqvar_std"][:]
            else:
                bad = [k for k in attrs
                       if attrs[k] is not None and ref_attrs[k] is not None
                       and not _same_value(attrs[k], ref_attrs[k])]
                if bad:
                    raise ValueError(
                        "Cannot overlay strategies evaluated with different settings: "
                        + "; ".join(f"{k}: {ref_path}={_py(ref_attrs[k])!r} vs {path}={_py(attrs[k])!r}"
                                    for k in bad)
                        + ". Delete the outdated file(s) and re-run."
                    )

            keys = _decode(meta["strategy_keys"][:])
            labels = _decode(meta["strategy_labels"][:])
            props = _decode(meta["strategy_propagator"][:])
            for key, label, prop in zip(keys, labels, props):
                if key in g["label_of"]:
                    raise ValueError(f"Strategy key '{key}' appears in more than one file.")
                sg = batch[f"strategies/{key}"]
                if "eqvar_mean" not in sg:
                    raise KeyError(f"{path} has no equilibrium-variance data for '{key}'; "
                                   "delete it and re-run to regenerate it.")
                g["strategy_keys"].append(key)
                g["label_of"][key] = label
                g["propagator_of"][key] = prop
                g["batch_strat"][key] = {fld: sg[fld][:] for fld in _BATCH_FIELDS}
                g["eqvar_mean_of"][key] = sg["eqvar_mean"][:]
                g["eqvar_std_of"][key] = sg["eqvar_std"][:]

            if "open_loop" in batch:
                for prop_key in batch["open_loop"]:
                    if prop_key in g["open_loop"]:
                        continue            # same propagator in another file: identical rollout
                    og = batch[f"open_loop/{prop_key}"]
                    g["open_loop"][prop_key] = dict(t=og["t"][:], l2_dense_mean=og["l2_dense_mean"][:])
                    if "eqvar_mean" in og:
                        g["open_loop_eqvar"][prop_key] = dict(
                            mean=og["eqvar_mean"][:], std=og["eqvar_std"][:])

    # The L2 plot keys curves (and colors) by label, so labels must be unique.
    seen = {}
    for key in g["strategy_keys"]:
        lbl = g["label_of"][key]
        if lbl in seen:
            g["label_of"][key] = f"{lbl} [{key}]"
            if g["label_of"][seen[lbl]] == lbl:
                g["label_of"][seen[lbl]] = f"{lbl} [{seen[lbl]}]"
        else:
            seen[lbl] = key
    return g


def plot_group(h5_paths: list, save_dir: str, title_tag: str, n_bins: int = 10) -> str:
    """
    Overlays every strategy stored in `h5_paths` (any number >= 1) on the same
    axes and writes five PDFs into `save_dir`:

        calibration.pdf            ensemble spread vs RMSE + binned spread-skill
        erf.pdf                    Error Reduction Factor per observation time
        l2.pdf                     EnKF vs open-loop time-mean relative L2
        rmse.pdf                   prior vs posterior RMSE
        equilibrium_variance.pdf   per-variable equilibrium variance vs reference

    `title_tag` is appended to every figure title to say what is being compared.
    Returns `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)
    g = _read_group(h5_paths)
    keys, label_of, batch_strat = g["strategy_keys"], g["label_of"], g["batch_strat"]
    colors = _strategy_colors(keys)
    B, N_ens = g["B"], g["N_ens"]
    obs_every_n, sigma_obs = g["obs_every_n"], g["sigma_obs"]
    S = len(keys)

    _plot_calibration_bulk(
        strategy_keys=keys, label_of=label_of,
        window_idx=g["window_idx"], dt_window=g["dt_window"],
        spread_window_of={k: batch_strat[k]["spread_window_mean"] for k in keys},
        rmse_window_of={k: batch_strat[k]["rmse_window_mean"] for k in keys},
        spread_raw_of={k: batch_strat[k]["spread_raw"] for k in keys},
        rmse_raw_of={k: batch_strat[k]["rmse_raw"] for k in keys},
        colors=colors,
        title=(f"Calibration: ensemble spread vs RMSE — {title_tag}\n"
               f"(B={B} trajectories, N_ens={N_ens})"),
        save_path=os.path.join(save_dir, "calibration.pdf"),
        n_bins=n_bins,
    )

    _plot_erf_bulk(
        strategy_keys=keys, label_of=label_of, obs_times=g["obs_times"],
        erf_mean_of={k: batch_strat[k]["erf_mean"] for k in keys},
        erf_std_of={k: batch_strat[k]["erf_std"] for k in keys},
        colors=colors, n_traj=B,
        title=(f"EnKF Error Reduction Factor per observation time — {title_tag}\n"
               f"(N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})"),
        save_path=os.path.join(save_dir, "erf.pdf"),
    )

    curves, curve_colors = {}, {}
    ol_palette = ["#B0BEC5", "#78909C", "#546E7A"]
    for i, prop_key in enumerate(sorted({g["propagator_of"][k] for k in keys})):
        if prop_key in g["open_loop"]:
            lbl = f"{prop_key} open-loop"
            curves[lbl] = (g["open_loop"][prop_key]["t"], g["open_loop"][prop_key]["l2_dense_mean"])
            curve_colors[lbl] = ol_palette[i % len(ol_palette)]
    for key in keys:
        curves[label_of[key]] = (g["t_dense_fine"], batch_strat[key]["l2_dense_mean"])
        curve_colors[label_of[key]] = colors[key]
    _plot_l2_per_timestep(
        curves=curves,
        title=f"EnKF vs open-loop: mean relative L2 per timestep — {title_tag}  (B={B})",
        save_path=os.path.join(save_dir, "l2.pdf"),
        colors=curve_colors,
    )

    _plot_rmse_bulk(
        strategy_keys=keys, label_of=label_of, obs_times=g["obs_times"],
        prior_mean_of={k: batch_strat[k]["prior_rmse_mean"] for k in keys},
        post_mean_of={k: batch_strat[k]["post_rmse_mean"] for k in keys},
        sigma_obs=sigma_obs, n_traj=B, colors=colors,
        title=(f"EnKF prior vs posterior RMSE — {title_tag}\n"
               f"(N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})"),
        save_path=os.path.join(save_dir, "rmse.pdf"),
    )

    _plot_equilibrium_variance_bulk(
        strategy_keys=keys, label_of=label_of,
        eqvar_mean_of=g["eqvar_mean_of"], eqvar_std_of=g["eqvar_std_of"],
        reference_mean=g["reference_mean"], reference_std=g["reference_std"],
        open_loop_eqvar=g["open_loop_eqvar"], colors=colors,
        title=(f"Equilibrium variance — static (open-loop) physics vs filtered "
               f"strategies — {title_tag}\n(B={B} trajectories, N_ens={N_ens}, obs every "
               f"{obs_every_n}th var, σ_obs={sigma_obs}, "
               f"burn-in={g['burn_in_frac']:.0%} of window discarded)"),
        save_path=os.path.join(save_dir, "equilibrium_variance.pdf"),
    )

    logging.info(f"plot_group: wrote 5 PDFs ({S} strategy file(s)) for '{title_tag}' to {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# config.eval.plot_groups
# ─────────────────────────────────────────────────────────────────────────

def _safe_dirname(name: str, max_len: int = 100) -> str:
    """Shorten over-long auto-generated folder names, keeping them unique."""
    if len(name) <= max_len:
        return name
    return f"{name[:max_len - 9]}_{hashlib.md5(name.encode()).hexdigest()[:8]}"


def _normalize_plot_groups(raw, valid_names: list) -> list:
    """
    `config.eval.plot_groups` -> [(folder_name, [names], title_tag)].

    Accepts a list of lists of identifying names (folders are named by joining
    the names) or a dict {group_name: [names]}. Every name must be an entry of
    `config.eval.strategies`.
    """
    raw = _plain(raw)
    if raw is None or len(raw) == 0:
        return []

    if isinstance(raw, dict):
        items = [(str(k), v) for k, v in raw.items()]
    elif _is_seq(raw):
        items = [(None, v) for v in raw]
    else:
        raise TypeError("config.eval.plot_groups must be a list of lists of names "
                        f"(or a dict of them), got {type(raw).__name__}.")

    groups, seen = [], set()
    for gname, members in items:
        if isinstance(members, (set, frozenset)):
            members = sorted(members)            # sets have no order; make it deterministic
        elif isinstance(members, str) or not _is_seq(members):
            raise TypeError(
                "config.eval.plot_groups: each group must be a list of names, e.g. "
                f"[['a', 'b'], ['c']]; got {members!r}. (A single-strategy group is ['c'].)"
            )
        members = [str(m) for m in members]
        where = f"config.eval.plot_groups[{gname if gname else members}]"
        if not members:
            raise ValueError(f"{where} is empty.")
        unknown = [m for m in members if m not in valid_names]
        if unknown:
            raise ValueError(f"{where}: unknown strategy name(s) {unknown}; "
                             f"available: {valid_names}.")
        if len(set(members)) != len(members):
            raise ValueError(f"{where} lists a strategy more than once.")

        if tuple(members) in seen:
            logging.warning(f"{where}: duplicate of an earlier group; skipping.")
            continue
        seen.add(tuple(members))

        if gname is None:
            joined_vs = " vs ".join(members)
            folder = _safe_dirname("__".join(members))
            tag = joined_vs if len(joined_vs) <= 70 else f"{len(members)} strategies"
        else:
            if not _NAME_RE.fullmatch(gname):
                raise ValueError(f"{where}: group names become folder names and may only "
                                 "contain letters, digits, '_' and '-'.")
            folder, tag = gname, gname
        groups.append((folder, members, tag))
    return groups


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────
def run_comparison(config, workdir: str, n_bins: int = 10) -> str:
    """
    Evaluates every strategy in `config.eval.strategies` (skipping any whose
    HDF5 file already exists in `workdir`) and writes all figures. See the
    module docstring for the configuration format and output layout.

    Output (under `workdir`):

        <name>.h5 | <name>_ver_<i>.h5 | <name>_ver_<i>_<j>.h5    evaluation data
        figures/individual_trajectories/trajectory_ic_<i>_<file stem>.pdf
        figures/comparisons/<collection>/{calibration,erf,l2,rmse,equilibrium_variance}.pdf

    Returns the path of the `figures` directory.
    """
    os.makedirs(workdir, exist_ok=True)
    figures_dir = os.path.join(workdir, "figures")

    # ── Parse and validate everything before any expensive work ─────────
    defs = parse_strategy_config(config)
    jobs = expand_jobs(defs)
    h5_path_of = {job.stem: os.path.join(workdir, f"{job.stem}.h5") for job in jobs}

    sweep_mode = any(job.version is not None for job in jobs)
    raw_groups = config.eval.get("plot_groups", None)
    if sweep_mode:
        if raw_groups is not None and len(raw_groups) > 0:
            logging.warning(
                "config.eval.plot_groups is ignored because at least one strategy has a "
                "parameter list; writing one collection per strategy name instead."
            )
        collections = []
        for name in defs:
            stems = [j.stem for j in jobs if j.name == name]
            swept = any(j.version is not None for j in jobs if j.name == name)
            collections.append((name, stems, f"{name} (parameter sweep)" if swept else name))
    else:
        collections = _normalize_plot_groups(raw_groups, valid_names=list(defs))
        if not collections:
            logging.warning("config.eval.plot_groups is empty: only individual-trajectory "
                            "plots will be written.")

    logging.info(
        f"run_comparison: {len(defs)} strateg{'y' if len(defs) == 1 else 'ies'} -> "
        f"{len(jobs)} evaluation file(s); {len(collections)} plot collection(s) "
        f"({'sweep' if sweep_mode else 'plot_groups'} mode)."
    )

    # ── Stage 1: evaluation (only what is missing) ───────────────────────
    _evaluate_pending(config, jobs, h5_path_of)

    # ── Stage 2: individual-trajectory plots, one set per strategy file ──
    indiv_dir = os.path.join(figures_dir, "individual_trajectories")
    for job in jobs:
        plot_individual_trajectories(h5_path_of[job.stem], indiv_dir)

    # ── Stage 3: one collection of overlaid plots per group / sweep ──────
    for folder, stems, tag in collections:
        plot_group(
            h5_paths=[h5_path_of[s] for s in stems],
            save_dir=os.path.join(figures_dir, "comparisons", folder),
            title_tag=tag, n_bins=n_bins,
        )

    logging.info(f"run_comparison: done. Figures in {figures_dir}")
    return figures_dir