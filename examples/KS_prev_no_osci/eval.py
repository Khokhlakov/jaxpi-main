"""
Modular filter evaluation and comparison for the DeepONet + EnKF
Kuramoto-Sivashinsky pipeline.

KS translation of ``examples.l96_f.eval`` (same design, three surrogates
instead of two).

ONE entry point, `run_comparison(config, workdir)`, replaces the earlier
family of `run_3way_comparison` / `run_4way_comparison` /
`run_mult_inflation_sweep` / `run_add_inflation_sweep` /
`run_rtpp_inflation_sweep` / `run_route_b_inflation_sweep` functions (and
`plot_comparisons`, `plot_comparisons_bulk`, `plot_equilibrium_variance`).
What is evaluated and what is plotted is described entirely by the
configuration file.

Configuration
-------------
Three keys under `config.eval`. Example 1 -- compare specific strategies::

    config.eval.test_data_name = "ks_test_data"      # -> data/ks_test_data.h5

    config.eval.strategies = {
        "dd_mult": dict(surrogate="DD", inflation="multiplicative",
                        params=dict(inflation_factor=1.05)),
        "pi_mult": dict(surrogate="PI", inflation="multiplicative",
                        params=dict(inflation_factor=1.10)),
        "hy_mult": dict(surrogate="HY", inflation="multiplicative",
                        params=dict(inflation_factor=1.10)),
        "pi_add":  dict(surrogate="PI", inflation="additive",
                        params=dict(alpha=0.5)),
        "hy_rb":   dict(surrogate="HY", inflation="route_b",
                        params=dict(alpha=0.0, beta=1.0)),
        "dd_rtpp": dict(surrogate="DD", inflation="rtpp",
                        params=dict(alpha_rtpp=0.5)),
    }

    config.eval.plot_groups = [
        ["dd_mult", "pi_mult", "hy_mult"],   # one collection, all three on every figure
        ["pi_add", "hy_rb", "dd_rtpp"],
        ["hy_rb"],                           # a collection with a single strategy
    ]

Example 2 -- parameter sweeps (any parameter given as a list; `plot_groups` is
then ignored)::

    config.eval.strategies = {
        "dd_mult": dict(surrogate="DD", inflation="multiplicative",
                        params=dict(inflation_factor=[1.00, 1.02, 1.05, 1.10])),
        "hy_mult": dict(surrogate="HY", inflation="multiplicative",
                        params=dict(inflation_factor=[1.00, 1.02, 1.05, 1.10])),
        "pi_rb":   dict(surrogate="PI", inflation="route_b",
                        params=dict(alpha=0.0, beta=[1e-2, 1e-1, 1.0, 10.0])),
    }

Strategy entries (one per identifying name):

    surrogate   "DD" (data-driven DeepONet), "PI" (physics-informed DeepONet)
                or "HY" / "hybrid" (PI + a fraction of DD). The checkpoint of
                each surrogate that is actually used is read from
                `config.wandb.name_dd` / `name_pi` / `name_hy`.
    inflation   "multiplicative", "additive", "route_b" or "rtpp".
                "additive" and "route_b" need the PDE residual (`r_net`), which
                the data-driven DD surrogate does not have, so DD + additive /
                Route B is rejected up front. Multiplicative and RTPP run on
                every surrogate.
    params      depends on the inflation type:

        multiplicative   inflation_factor            (> 0; coarse, per-window factor)
        additive         alpha                       (>= 0; constant floor alpha*Q0,
                                                      i.e. Route B with beta = 0)
        route_b          alpha, beta                 (>= 0; scale = alpha + beta*||rho||^2)
        rtpp             alpha_rtpp                  (in [0, 1]; relaxation factor)
                         alpha_fine   (optional, default 1.0; multiplicative
                                       inflation in RTPP's predict step, used as given)

Every parameter listed above may be a scalar or a list. A list of n values
triggers n evaluations (see "Sweeps"). Anything not listed here (N_ens,
sigma_obs, P0_sigma, P0_corr_len, Q0_sigma, Q0_corr_len, route_b_n_quad,
dt_fine, ...) is still read from `config.kf` / `config.eval` exactly as before
and is shared by every strategy.

The per-scheme knobs the old runners read from `config.kf` (`inflation_factor`,
`inflation_factor_list`, `inflation_alpha_list`, `route_b_alpha`,
`route_b_beta`, `route_b_beta_list`, `rtpp_alpha`, `rtpp_alpha_list`,
`rtpp_alpha_fine`, `compare_propagators`, `sweep_propagators*`) are NO LONGER
read: their values now live in `config.eval.strategies`. Likewise
`config.wandb.name` no longer names the results file.

Test data
---------
`config.eval.test_data_name` (default "ks_test_data") names the test set:
`<config.training.data_dir>/<name>.h5` (`data_dir` defaults to "data"). A
trailing ".h5" in the name is accepted. The same lookup is used by the
unfiltered `evaluate`.

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

What is KS-specific (read this before porting anything else from l96_f)
=======================================================================

1. **No augmented parameter.** L96 carries a 41-D state (40 variables +
   the per-trajectory forcing F) so every H is zero-padded by one column,
   every dense output is sliced `[..., :N]`, and the HDF5 records a per-IC
   `F`. KS has no free parameter: the state is exactly the N-point
   periodic field, so all of that drops out. What replaces `F` in the
   metadata is the domain length `L`, which is a property of the dataset
   rather than of each IC.

2. **Ground truth comes from the reference ETDRK4 solver, not SciPy.**
   L96 re-solves each trajectory with `solve_ivp(..., method='LSODA')` on
   an arbitrary output grid. The KS reference solver
   (`data.gen_data.KuramotoSivashinskyAdvanced`) is an exponential
   time-differencing RK4 scheme whose coefficients are precomputed for one
   fixed `dt`, so the truth can only be sampled on multiples of that step:
   `DT_FINE` must be an exact integer multiple of the data `dt` (enforced
   by `steps_per_fine_exact`). `_make_ks_truth_batch` below advances the
   spectral solver on the GPU, batched over ICs, and returns real-space
   states on the filter's fine grid.

   The batch pass reuses the dense trajectories already stored in the test
   file (saved at every solver step by `gen_data.py`) rather than
   re-integrating -- same trajectories, read by stride.

3. **The state is a field, so the per-variable plots change shape.**
   L96's individual-trajectory PDF draws one time-series panel per state
   variable (40 variables -> 20 rows). At N=256 grid points that layout is
   unusable, and it would also be the wrong picture: neighbouring entries
   are neighbouring points of one smooth profile. The KS version draws
   space-time heatmaps (truth / estimate / difference), the error and
   spread as time series, a handful of fixed-probe time series, and
   profile snapshots with the +/-1 sigma band -- see
   `_plot_trajectory_individual`.

4. **Different natural scales.** The stored KS field is nondimensionalised
   (`c_u = 3.5` in the solver), so it is O(1) rather than L96's O(5-10).
   The `sigma_obs` / `P0_sigma` / `Q0_sigma` defaults below are set
   accordingly, and Route B's `beta` in particular needs its own
   calibration: the KS residual involves fourth-order spatial derivatives
   summed over 256 grid points, so `||rho||^2` lives on a completely
   different scale from L96's. Sweep `beta` with a wide logarithmic list
   (e.g. `beta=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]`) and read
   `batch/strategies/<key>/route_b_scale_mean` out of the results files --
   that is the realised `alpha + beta*||rho||^2` per fine step -- before
   trusting any inherited default. Keep a `beta = 0.0` entry: it recovers
   whatever `alpha` alone gives.

5. **Spatially correlated P0/Q0 are available.** White noise on a 256-point
   grid is dominated by the wavenumbers KS damps hardest, so a diagonal P0
   produces an ensemble that looks well-spread at t=0 and collapses almost
   at once. Set `config.kf.P0_corr_len` (physical units, i.e. the same
   units as `L`; roughly one KS cell, ~4-9 for L=64) to draw the initial
   ensemble from a smooth Gaussian-correlated covariance instead. The
   default is 0.0, which reproduces the L96 diagonal behaviour exactly.
   `config.kf.Q0_corr_len` does the same for the Route B / additive `Q0`
   (defaulting to `P0_corr_len`).

HDF5 layout written by `evaluate_filters`
------------------------------------------
    /meta                                   (attrs, plus datasets where noted)
        N, L, dt_data                        -- grid size, domain length, solver dt
        dt_window, dt_fine, dt_obs
        sigma_obs, P0_sigma, P0_corr_len, N_ens, obs_every_n, m
        num_ics_traj, num_ics_batch, trajectory_windows, batch_windows
        test_data                            -- basename of the test-data file used
        obs_indices                          -- dataset, (m,) int, if static
        x_grid                               -- dataset, (N,) physical coordinates
        strategy_keys, strategy_labels        -- string datasets (one entry per file)
        strategy_propagator                   -- which propagator each strategy uses
        strategy_kind                         -- "standard" | "route_b" | "rtpp"
        strategy_config_json                  -- the config entry that produced the file

    /trajectories/t_fine                     (T_fine,)
    /trajectories/ic_{i}
        x_true                               (T_fine, N)
        obs_coords                           (n_obs_pts, 3) = [grid_idx, t_obs, y_obs]
        strategies/{key}/x_est               (T_fine, N)
        strategies/{key}/x_std               (T_fine, N)
        strategies/{key}/l2_time_avg         scalar attr -- time-avg relative L2

    /batch  (attrs: B, eqvar_burn_in_frac)
        obs_times                            (T_obs,)
        window_idx                           (n_windows,)
        t_dense_fine                         (n_fine,)
        strategies/{key}/prior_rmse_mean     (T_obs,)
        strategies/{key}/prior_rmse_std      (T_obs,)
        strategies/{key}/post_rmse_mean      (T_obs,)
        strategies/{key}/post_rmse_std       (T_obs,)
        strategies/{key}/erf_mean            (T_obs,)
        strategies/{key}/erf_std             (T_obs,)
        strategies/{key}/rmse_window_mean    (n_windows,)
        strategies/{key}/spread_window_mean  (n_windows,)
        strategies/{key}/rmse_raw            (B * n_windows,)
        strategies/{key}/spread_raw          (B * n_windows,)
        strategies/{key}/l2_dense_mean       (n_fine,)
        strategies/{key}/eqvar_mean          (N,)
        strategies/{key}/eqvar_std           (N,)
        strategies/{key}/route_b_scale_mean  (n_fine,)  -- kind == "route_b" only
        strategies/{key}/route_b_scale_std   (n_fine,)
        open_loop/{propagator}/t             (n_ol,)
        open_loop/{propagator}/l2_dense_mean (n_ol,)
        open_loop/{propagator}/eqvar_mean    (N,)
        open_loop/{propagator}/eqvar_std     (N,)
        reference/eqvar_mean                 (N,)
        reference/eqvar_std                  (N,)

    `{key}` is the file stem (e.g. "pi_mult_ver_2"). Files written by this
    module hold exactly one strategy, but the plotting code reads any number.

    Equilibrium (climatological/attractor) variance
    -------------------------------------------------
    For a dense state trajectory of B ICs x T time steps x N grid points,
    the leading `eqvar_burn_in_frac` fraction of the time axis is
    discarded (letting transients -- assimilation spin-up for filtered
    strategies, or an off-attractor start for an open-loop rollout --
    decay), then the per-grid-point temporal variance is computed on the
    remaining tail for each IC. `eqvar_mean`/`eqvar_std` are the mean/std
    of that per-IC variance across the B ICs. For KS the truth curve is
    close to flat in x (the attractor is statistically homogeneous on a
    periodic domain), which makes it an unusually clean target: a filtered
    curve that dips at the un-observed grid points is showing you exactly
    where the analysis is over-confident.
"""

import os
import re
import json
import hashlib
import dataclasses
import itertools
import colorsys
import warnings

from absl import logging
import ml_collections
import jax
import jax.numpy as jnp
import numpy as np
import h5py

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from jaxpi.utils import restore_checkpoint
import examples.KS_prev_no_osci.models as models
from examples.KS_prev_no_osci.utils import (
    build_obs_schedule,
    check_divisible,
    scale_inflation_for_fine_steps,
    scale_Q_for_fine_steps,
    steps_per_fine_exact,
    steps_per_window_exact,
)
from examples.KS_prev_no_osci.kf import (
    build_cov,
    init_ensemble,
    safe_cholesky,
    run_enkf_smoother,
    run_enkf_smoother_route_b,
    run_enkf_smoother_rtpp,
)
from data.gen_data import KuramotoSivashinskyAdvanced

# ─────────────────────────────────────────────────────────────────────────
# Multi-GPU batch execution helper
# ─────────────────────────────────────────────────────────────────────────

def _device_parallel(fn, in_axes, static_broadcasted_argnums=(), to_host=True):
    """
    jit(vmap(fn)) on one device, pmap(vmap(fn)) sharded across many.

    Results come back as host numpy arrays when `to_host` (the default).
    That is what keeps the dense per-strategy outputs off the GPU: every
    consumer of these arrays in `evaluate_filters` either writes them to
    HDF5 or reduces them, so there is no reason for them to stay resident
    on the device. This matters more for KS than it did for L96 -- the
    dense outputs are 256/40 = 6.4x larger per time step.
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
# Dataset / model loading
# ─────────────────────────────────────────────────────────────────────────

def resolve_test_h5_path(config) -> str:
    """
    `config.eval.test_data_name` -> `<data_dir>/<name>.h5`.

    `data_dir` is `config.training.data_dir` (default "data") and the name
    defaults to "ks_test_data"; a trailing ".h5" in the name is accepted
    and stripped. This is the single place the test set is named, shared
    by `evaluate_filters`, `run_comparison` and the unfiltered `evaluate`.
    """
    data_dir = config.training.get("data_dir", "data") if "training" in config else "data"
    name = str(config.eval.get("test_data_name", "ks_test_data"))
    if name.endswith(".h5"):
        name = name[:-3]
    return os.path.join(data_dir, f"{name}.h5")


def _read_test_meta(test_h5_path: str) -> dict:
    """
    Read the grid/time metadata `gen_data.generate_datasets` wrote onto
    ks_test_data.h5, without pulling the (multi-GB) trajectory array into
    memory.
    """
    with h5py.File(test_h5_path, "r") as f:
        dset = f["u"]
        num_ics, num_test_pts, N = dset.shape
        meta = dict(
            num_ics=int(num_ics),
            num_test_pts=int(num_test_pts),
            N=int(N),
            L=float(f.attrs["L"]),
            dt=float(f.attrs["dt"]),
            test_windows=int(f.attrs["test_windows"]),
        )
    return meta


_CKPT_CANDIDATES_BY_KIND = {
    "pi": ("ks_udon_model", "udon_model"),
    "dd": ("ks_udon_dd_model", "udon_dd_model"),
    "hy": ("ks_udon_hybrid_model", "udon_hybrid_model"),
}

def _resolve_ckpt(run_name: str, kind: str | None = None) -> str:
    """
    Locate <cwd>/<run_name>/ckpt/<basename>. If `kind` is given, only that
    kind's basenames are tried (so a PI checkpoint can never be silently
    loaded into a DD/Hybrid model shape or vice versa). If `kind` is None,
    falls back to trying every known basename, in kind order.
    """
    base = os.path.join(os.getcwd(), run_name, "ckpt")
    candidates = (
        _CKPT_CANDIDATES_BY_KIND[kind] if kind is not None
        else sum(_CKPT_CANDIDATES_BY_KIND.values(), ())
    )
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No '{kind or 'any'}' checkpoint found under {base} -- tried "
        f"{list(candidates)}. Set config.wandb.name_pi / name_dd / name_hy "
        "to the training run directory that holds ckpt/<model>."
    )


def _load_ks_model(config, kind: str, t_star_window, grid: dict):
    """
    Instantiate and restore one KS surrogate.

    Args:
        kind: "pi" (KSUDON), "dd" (KSUDON_DD) or "hy" (KSUDON_Hybrid).
        grid: dict from `_read_test_meta` -- supplies N, L and the solver dt
              so the model's spectral operators match the dataset exactly
              rather than relying on the class defaults.

    Returns (model, params).
    """
    name_key = {"pi": "name_pi", "dd": "name_dd", "hy": "name_hy"}[kind]
    run_name = config.wandb.get(name_key, None)
    if run_name is None:
        raise KeyError(
            f"config.wandb.{name_key} is not set, but a '{kind}' propagator "
            "was requested."
        )

    logging.info(f"Loading {kind.upper()} model from run '{run_name}' ...")
    if kind == "dd":
        model = models.KSUDON_DD(config, t_star_window, N=grid["N"], L=grid["L"])
    elif kind == "hy":
        model = models.KSUDON_Hybrid(
            config, t_star_window, L=grid["L"], N=grid["N"], dt=grid["dt"]
        )
    else:
        model = models.KSUDON(
            config, t_star_window, L=grid["L"], N=grid["N"], dt=grid["dt"]
        )

    model.state = restore_checkpoint(model.state, _resolve_ckpt(run_name, kind))
    return model, model.state.params


def _window_grid(config, grid: dict):
    """
    (dt_window, t_star_window) -- the single-window time grid every
    surrogate is queried on. `dt_integration` defaults to the dataset's own
    solver dt so that open-loop rollouts land on the stored trajectory's
    time stamps.
    """
    dt_window = float(config.get("dt_window", 1.0))
    dt_integration = float(config.eval.get("dt_integration", grid["dt"]))
    time_steps = check_divisible(dt_window, dt_integration,
                                 "dt_window / dt_integration") + 1
    return dt_window, dt_integration, jnp.linspace(0.0, dt_window, time_steps)


# ─────────────────────────────────────────────────────────────────────────
# Reference (ground-truth) trajectories from the ETDRK4 spectral solver
# ─────────────────────────────────────────────────────────────────────────

def _make_ks_truth_batch(solver, steps_per_fine: int, n_fine: int):
    """
    Build a jitted, IC-batched reference rollout on the filter's fine grid.

    Returns a callable ``(u0: (B, N) real) -> (B, n_fine, N) real`` giving
    the true field at t = k * dt_fine for k = 1 .. n_fine, obtained by
    running ``steps_per_fine`` ETDRK4 steps of the reference solver per
    fine step. This is the KS stand-in for L96's per-IC
    ``solve_ivp(..., method='LSODA')`` loop -- but batched and on device,
    so `num_plots` trajectories cost one compiled scan rather than
    `num_plots` sequential SciPy solves.

    The solver integrates in Fourier space, so the IC is transformed once
    up front and only the saved fine-step states are transformed back.
    """
    N = solver.N

    def single(u0_real):
        u_hat0 = jnp.fft.rfft(u0_real)

        def fine_step(u_hat, _):
            def inner(uh, __):
                return solver.step_pure(uh), None
            u_hat_next, _ = jax.lax.scan(inner, u_hat, None, length=steps_per_fine)
            return u_hat_next, jnp.fft.irfft(u_hat_next, n=N)

        _, traj = jax.lax.scan(fine_step, u_hat0, None, length=n_fine)
        return traj                                    # (n_fine, N)

    return jax.jit(jax.vmap(single))
# ─────────────────────────────────────────────────────────────────────────
# Strategy specification
# ─────────────────────────────────────────────────────────────────────────
#
# Each strategy is a plain dict:
#
#   {
#     "key":        unique short id, used as the HDF5 group name,
#     "label":      human-readable label for legends / titles,
#     "kind":       "standard" (run_enkf_smoother),
#                   "route_b"  (run_enkf_smoother_route_b), or
#                   "rtpp"     (run_enkf_smoother_rtpp),
#     "propagator": name of the propagator it uses -- strategies sharing a
#                   propagator name share ONE open-loop rollout,
#     "predict_fn", "update_fn": the EnKF predict/update closures,
#     # kind == "standard":
#     "alpha_fine": scaled multiplicative inflation factor,
#     # kind == "route_b":
#     "Q0", "alpha", "beta", "n_quad": Route-B hyperparameters,
#     #   (plain additive inflation is just Route B with "beta": 0.0,
#     #    i.e. the flow-dependent term zeroed out, leaving only the
#     #    constant alpha * Q0 floor)
#     # kind == "rtpp":
#     "alpha_fine": multiplicative inflation applied in the (shared)
#                   predict step -- set to 1.0 to isolate RTPP's own
#                   relaxation as the only inflation mechanism,
#     "alpha_rtpp": RTPP relaxation-to-prior factor in [0, 1],
#   }
#
# `propagators` is a dict: propagator_name -> (model, params), used only
# for the open-loop reference rollouts and to read N.


def build_batched_filters(
    strategies, N, m, obs_indices, P0, N_ens, sigma_obs, R,
    dt_fine, dt_window, total_fine_steps, obs_step_indices,
):
    """
    Builds ONE jit(vmap(...)) (or pmap-sharded) closure that, for every IC
    in the batch, runs every strategy in `strategies` against the SAME
    noisy observation draw and the SAME initial ensemble -- so differences
    between strategies reflect the strategy alone, not differing noise
    realizations.

    Unlike the L96 version there is no augmented forcing parameter, so H is
    (m, N) with no zero-padded column and the initial ensemble mean is just
    the perturbed true IC.
    """

    # Factorised once, outside the per-IC closure: P0 is shared by every IC
    # and every strategy, and for KS it is a dense (N, N) matrix whenever
    # `P0_corr_len > 0`.
    L_P0 = safe_cholesky(P0)

    def process_single_ic(key_ic, u_true, x_true_at_obs,
                          dynamic_vars_static, specify_obs_idx_static):
        T_obs = x_true_at_obs.shape[0]
        keys_t = jax.random.split(key_ic, T_obs)

        def single_obs(k, x_t):
            k1, k2 = jax.random.split(k)
            if (not specify_obs_idx_static) and dynamic_vars_static:
                # Re-drawn observation locations at every observation time.
                # For a field this is the "roving sensor network" setup:
                # each analysis sees a different random subset of grid
                # points, so information reaches unobserved regions through
                # the ensemble covariance rather than through fixed sensors.
                idx_pts = jax.random.choice(k1, N, shape=(m,), replace=False)
            else:
                idx_pts = obs_indices

            H = jnp.zeros((m, N)).at[jnp.arange(m), idx_pts].set(1.0)
            noise = sigma_obs * jax.random.normal(k2, shape=(m,))
            return H, x_t[idx_pts] + noise, idx_pts

        H_seq, y_obs_seq, idx_pts_seq = jax.vmap(single_obs)(keys_t, x_true_at_obs)

        # Shared initial ensemble across every strategy.
        _, k2, k3 = jax.random.split(key_ic, 3)
        # P0 carries the marginal variance (and, if configured, the spatial
        # correlation), so the mean perturbation is drawn from it too rather
        # than from an independent P0_sigma * white-noise draw. With the
        # default diagonal P0 this is exactly `P0_sigma * normal`, i.e. the
        # L96 behaviour.
        x0_hat = u_true + jax.random.normal(k2, shape=(N,)) @ L_P0.T
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, k3)

        outputs = {}
        for spec in strategies:
            key = spec["key"]
            if spec["kind"] == "route_b":
                x_means, x_spreads, prior_means, q_scale = run_enkf_smoother_route_b(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices,
                    H_seq, Q0=spec["Q0"], alpha=spec["alpha"], beta=spec["beta"],
                    R=R, key=key_ic, total_fine_steps=total_fine_steps,
                    dt_fine=dt_fine, dt_window=dt_window, n_quad=spec["n_quad"],
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads,
                    prior_means=prior_means, q_scale=q_scale,
                )
            elif spec["kind"] == "rtpp":
                x_means, x_spreads, prior_means = run_enkf_smoother_rtpp(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices,
                    H_seq, spec["alpha_fine"], spec["alpha_rtpp"],
                    R, key_ic, total_fine_steps,
                    dt_fine=dt_fine, dt_window=dt_window,
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads, prior_means=prior_means,
                )
            else:
                x_means, x_spreads, prior_means = run_enkf_smoother(
                    spec["predict_fn"], spec["update_fn"],
                    ensemble0, y_obs_seq, obs_step_indices,
                    H_seq, spec["alpha_fine"], R, key_ic, total_fine_steps,
                    dt_fine=dt_fine, dt_window=dt_window,
                )
                outputs[key] = dict(
                    x_means=x_means, x_spreads=x_spreads, prior_means=prior_means,
                )

        return outputs, y_obs_seq, idx_pts_seq

    return _device_parallel(
        process_single_ic,
        in_axes=(0, 0, 0, None, None),
        static_broadcasted_argnums=(3, 4),
    )
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
    module docstring). `run_comparison` calls this with a single strategy
    per file; passing several still works and stores them all in one file.

    `test_h5_path` defaults to `<data_dir>/<config.eval.test_data_name>.h5`.
    Results are written to `<out_path>.partial` and renamed only once the
    file is complete, so an interrupted run never leaves a half-written
    file that a later run would mistake for a finished evaluation.

    Memory-chunked over ICs and (optionally) over strategy groups; see
    `config.eval.ic_chunk`, `config.eval.strategy_chunk` and
    `config.eval.device_budget_gb`. KS dense outputs are 6.4x larger per
    time step than L96's, so the automatic chunk sizing matters more here
    -- if you hit OOM, lower `device_budget_gb` first.

    Note that the per-scheme inflation knobs are NOT read here: every
    inflation parameter travels inside the strategy specs built by
    `_Surrogates.make_spec` (so a sweep is just several specs), and only
    the settings shared by every strategy are read from `config.kf`.
    """
    # ── EnKF / observation configuration ───────────────────────────────
    obs_every_n = config.kf.get("obs_every_n", 4)
    sigma_obs = config.kf.get("sigma_obs", 0.1)
    P0_sigma = config.kf.get("P0_sigma", 0.5)
    P0_corr_len = float(config.kf.get("P0_corr_len", 0.0))
    dynamic_vars = config.kf.get("dynamic_vars", False)
    N_ens = config.kf.get("N_ens", 50)

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list = config.kf.get("obs_idx_list", None)

    n_devices = jax.local_device_count()
    strategy_chunk = int(config.eval.get("strategy_chunk", 0))
    ic_chunk_cfg = int(config.eval.get("ic_chunk", 0))
    budget_bytes = float(config.eval.get("device_budget_gb", 1.5)) * (1024 ** 3)

    # ── 1. Dataset metadata and time grids ─────────────────────────────
    if test_h5_path is None:
        test_h5_path = resolve_test_h5_path(config)
    grid = _read_test_meta(test_h5_path)
    N_grid, L_dom, dt_data = grid["N"], grid["L"], grid["dt"]

    DT_WINDOW, dt_integration, t_star_default = _window_grid(config, grid)
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    DT_OBS = float(config.kf.get("dt_obs", DT_WINDOW))

    # The reference solver can only be sampled on multiples of its own dt.
    steps_per_fine = steps_per_fine_exact(DT_FINE, dt_data)
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    logging.info(
        f"JAX sees {n_devices} local device(s): {jax.local_devices()}"
    )
    logging.info(
        f"KS grid: N={N_grid}, L={L_dom:g}, solver dt={dt_data:g}  |  "
        f"dt_window={DT_WINDOW:g} ({steps_per_window} fine steps), "
        f"dt_fine={DT_FINE:g} ({steps_per_fine} solver steps), dt_obs={DT_OBS:g}"
    )

    trajectory_windows = config.eval.get("trajectory_windows", 20)
    batch_windows = config.eval.get("windows", 20)
    num_ics_eval = config.eval.get("num_ics", grid["num_ics"])
    enkf_batch_size = config.kf.get("batch_l2_size", 50)
    eqvar_burn_in_frac = config.eval.get("eqvar_burn_in_frac", 0.5)

    # ── 2. Strategies ──────────────────────────────────────────────────
    # P0 is shared by every strategy (Q0 is not: it is a Route B / additive
    # hyperparameter and travels inside the individual strategy specs).
    P0 = build_cov(N_grid, L_dom, P0_sigma, P0_corr_len)

    if not strategies:
        raise ValueError("evaluate_filters: `strategies` must be a non-empty list.")
    if not propagators:
        raise ValueError("evaluate_filters: `propagators` is required.")
    N = next(iter(propagators.values()))[0].N
    if t_star_window is None:
        t_star_window = t_star_default

    assert N == N_grid, (
        f"Checkpoint grid size ({N}) != test-dataset grid size ({N_grid}). "
        "The surrogate and the reference data must share a spatial grid."
    )

    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)

    m = len(obs_indices)
    R = jnp.eye(m) * sigma_obs ** 2
    x_grid = np.arange(N) * (L_dom / N)

    logging.info(
        f"Observation network: m={m} of N={N} grid points "
        f"({'roving' if dynamic_vars and not specify_obs_idx else 'fixed'}), "
        f"sigma_obs={sigma_obs:g}; P0_sigma={P0_sigma:g}, "
        f"P0_corr_len={P0_corr_len:g} "
        f"({'correlated' if P0_corr_len > 0 else 'diagonal'})."
    )

    groups = _strategy_groups(strategies, strategy_chunk)

    solver = KuramotoSivashinskyAdvanced(L=L_dom, N=N, dt=dt_data)

    with h5py.File(test_h5_path, "r") as f_test:
        dset = f_test["u"]

        # ── 3. Per-IC single-trajectory data ───────────────────────────
        num_plots = min(config.saving.get("total_plots", 3), grid["num_ics"])
        total_time_traj = trajectory_windows * DT_WINDOW

        obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
            total_time=total_time_traj, dt_fine=DT_FINE, dt_obs=DT_OBS,
        )
        obs_step_indices_np = np.asarray(obs_step_indices)
        obs_step_indices = jnp.array(obs_step_indices)

        # PASS 1: reference ETDRK4 rollout on the fine grid, batched over ICs.
        u0_batch_plots = jnp.asarray(
            np.asarray(dset[:num_plots, 0, :], dtype=np.float32)
        )
        truth_fn = _make_ks_truth_batch(solver, steps_per_fine, total_fine_steps)
        logging.info(
            f"evaluate_filters: integrating {num_plots} reference "
            f"trajectory(ies) for {total_fine_steps} fine steps "
            f"({total_fine_steps * steps_per_fine} ETDRK4 steps each) ..."
        )
        x_true_fine_batch = np.asarray(truth_fn(u0_batch_plots))   # (P, T_fine, N)
        x_true_at_obs_batch = jnp.asarray(
            x_true_fine_batch[:, obs_step_indices_np, :]
        )
        keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(
            jnp.arange(num_plots)
        )

        # PASS 2: batched GPU filtering, one compiled program per group.
        traj_bytes_per_ic = (
            total_fine_steps * N * 4
            * _dense_leaves_per_strategy(groups[0]) * 2
        )
        traj_chunk = ic_chunk_cfg or _auto_ic_chunk(
            num_plots, n_devices, traj_bytes_per_ic, budget_bytes
        )
        logging.info(
            f"evaluate_filters: trajectory pass -- {num_plots} IC(s) in chunks "
            f"of {traj_chunk}, {len(groups)} strategy group(s)."
        )

        outputs_traj = {}
        y_obs_traj = np.zeros((num_plots, len(obs_step_indices_np), m), dtype=np.float32)
        idx_pts_traj = np.zeros((num_plots, len(obs_step_indices_np), m), dtype=np.int32)

        for g_i, group in enumerate(groups):
            batched_traj_fn = build_batched_filters(
                group, N, m, obs_indices, P0, N_ens, sigma_obs, R,
                DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices,
            )
            for i0 in range(0, num_plots, traj_chunk):
                i1 = min(i0 + traj_chunk, num_plots)
                out_c, y_c, idx_c = batched_traj_fn(
                    keys_batch_plots[i0:i1], u0_batch_plots[i0:i1],
                    x_true_at_obs_batch[i0:i1], dynamic_vars, specify_obs_idx,
                )
                if g_i == 0:
                    y_obs_traj[i0:i1] = np.asarray(y_c)
                    idx_pts_traj[i0:i1] = np.asarray(idx_c)
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
        t_fine_axis = np.arange(1, total_fine_steps + 1) * DT_FINE

        per_ic_records = []
        for ic_idx in range(num_plots):
            x_true_fine = np.asarray(x_true_fine_batch[ic_idx])
            x_true_at_windows = x_true_fine[window_step_indices]

            idx_pts_seq = idx_pts_traj[ic_idx]
            y_obs_seq = y_obs_traj[ic_idx]
            obs_coords = []
            for obs_idx, t_obs in enumerate(obs_times):
                for j, pi_ in enumerate(idx_pts_seq[obs_idx]):
                    obs_coords.append((int(pi_), float(t_obs), float(y_obs_seq[obs_idx, j])))
            obs_coords = (np.array(obs_coords, dtype=np.float64)
                          if obs_coords else np.zeros((0, 3)))

            strat_records = {}
            for spec in strategies:
                key = spec["key"]
                x_means = np.asarray(outputs_traj[key]["x_means"][ic_idx])
                x_spreads = np.asarray(outputs_traj[key]["x_spreads"][ic_idx])
                l2_time_avg = float(
                    np.linalg.norm(x_means[window_step_indices] - x_true_at_windows)
                    / (np.linalg.norm(x_true_at_windows) + 1e-12)
                )
                strat_records[key] = dict(
                    x_est=x_means, x_std=x_spreads, l2_time_avg=l2_time_avg
                )

            per_ic_records.append(dict(
                x_true=x_true_fine, obs_coords=obs_coords, strategies=strat_records,
            ))

        # Everything the trajectory pass produced is now host numpy. Drop the
        # originals and the compilation cache before the batch pass allocates.
        del outputs_traj, y_obs_traj, idx_pts_traj
        del x_true_at_obs_batch, u0_batch_plots, keys_batch_plots
        jax.clear_caches()

        # ── 4. Batch-averaged metrics ──────────────────────────────────
        B = min(num_ics_eval, enkf_batch_size, grid["num_ics"])

        total_time_batch = batch_windows * DT_WINDOW
        _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
            total_time=total_time_batch, dt_fine=DT_FINE, dt_obs=DT_OBS,
        )
        obs_step_indices_batch_np = np.asarray(obs_step_indices_batch)
        obs_step_indices_batch = jnp.array(obs_step_indices_batch)

        T_obs = len(obs_step_indices_batch_np)
        obs_times_batch = np.array([(k + 1) * DT_OBS for k in range(T_obs)])

        n_fine_pts = total_fine_steps_batch * steps_per_fine + 1
        if n_fine_pts > grid["num_test_pts"]:
            raise ValueError(
                f"config.eval.windows={batch_windows} needs {n_fine_pts} stored "
                f"time points at dt={dt_data:g}, but {test_h5_path} only holds "
                f"{grid['num_test_pts']} (test_windows={grid['test_windows']}). "
                "Lower config.eval.windows or regenerate the test set with more "
                "test_windows."
            )

        # Stride the stored dense trajectories onto the filter's fine grid.
        # (B, T_fine + 1, N); index 0 is the IC, 1.. are the fine steps.
        x_true_fine_batch2 = np.asarray(
            dset[:B, 0:n_fine_pts:steps_per_fine, :], dtype=np.float32
        )
        u0_batch = x_true_fine_batch2[:, 0, :]
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
            total_fine_steps_batch * N * 4
            * max(_dense_leaves_per_strategy(g) for g in groups) * 2
        )
        batch_chunk = ic_chunk_cfg or _auto_ic_chunk(
            B, n_devices, batch_bytes_per_ic, budget_bytes
        )
        logging.info(
            f"evaluate_filters: batch pass -- B={B} IC(s) in chunks of "
            f"{batch_chunk}, {len(strategies)} strategies in {len(groups)} "
            f"group(s), ~{batch_bytes_per_ic * batch_chunk / 1024**3:.2f} GiB "
            "of dense outputs per call."
        )

        # Per-IC metric accumulators: (B, T)-shaped, i.e. a factor N smaller
        # than the dense (B, T, N) outputs they are computed from.
        acc = {spec["key"]: {} for spec in strategies}

        def _push(key, field, value):
            acc[key].setdefault(field, []).append(value)

        for group in groups:
            batched_batch_fn = build_batched_filters(
                group, N, m, obs_indices, P0, N_ens, sigma_obs, R,
                DT_FINE, DT_WINDOW, total_fine_steps_batch, obs_step_indices_batch,
            )

            for i0 in range(0, B, batch_chunk):
                i1 = min(i0 + batch_chunk, B)
                outputs_c, _, _ = batched_batch_fn(
                    keys_batch[i0:i1], jnp.array(u0_batch[i0:i1]),
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

                    post_means_obs = out["x_means"][:, obs_step_indices_batch_np, :]
                    prior_means_obs = out["prior_means"]

                    prior_rmse_ic = _rmse(prior_means_obs, truth_obs_c)
                    post_rmse_ic = _rmse(post_means_obs, truth_obs_c)
                    _push(key, "prior_rmse", prior_rmse_ic)
                    _push(key, "post_rmse", post_rmse_ic)
                    _push(key, "erf", prior_rmse_ic / (post_rmse_ic + 1e-12))

                    x_hat_windows = out["x_means"][:, window_step_indices_b, :]
                    _push(key, "rmse", _rmse(x_hat_windows, truth_win_c))
                    _push(key, "spread", np.sqrt(np.mean(
                        out["x_spreads"][:, window_step_indices_b, :] ** 2, axis=2
                    )))

                    _push(key, "l2_dense", np.linalg.norm(
                        out["x_means"] - truth_tail_c, axis=2
                    ) / den_c)

                    _push(key, "eqvar", _equilibrium_variance_per_ic(
                        out["x_means"], eqvar_burn_in_frac
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

        acc.clear()

        # ── 5. Open-loop reference rollouts (one per unique propagator) ─
        used_propagators = sorted({spec["propagator"] for spec in strategies})
        open_loop_records = {}
        ol_stride = check_divisible(dt_integration, dt_data,
                                    "dt_integration / dt_solver")
        pts_per_window_ol = int(t_star_window.shape[0]) - 1

        if propagators is not None:
            for prop_key in used_propagators:
                if prop_key not in propagators:
                    continue
                model, params = propagators[prop_key]
                predict_full = _device_parallel(
                    lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=(0,)
                )

                l2_ol_chunks, eqvar_ol_chunks = [], []
                total_steps_ol = batch_windows * pts_per_window_ol + 1
                n_pts_needed = (total_steps_ol - 1) * ol_stride + 1
                if n_pts_needed > grid["num_test_pts"]:
                    raise ValueError(
                        f"Open-loop rollout needs {n_pts_needed} stored time "
                        f"points, but {test_h5_path} holds "
                        f"{grid['num_test_pts']}."
                    )

                for i0 in range(0, B, batch_chunk):
                    i1 = min(i0 + batch_chunk, B)
                    u_current = jnp.array(u0_batch[i0:i1])
                    x_pred_list = []
                    for k in range(batch_windows):
                        x_win = predict_full(u_current)            # host numpy
                        x_pred_list.append(x_win if k == 0 else x_win[:, 1:, :])
                        u_current = jnp.array(x_win[:, -1, :])
                    x_pred_dense = np.concatenate(x_pred_list, axis=1)
                    del x_pred_list

                    x_ref_dense_ol = np.asarray(
                        dset[i0:i1, 0:n_pts_needed:ol_stride, :], dtype=np.float32
                    )
                    denom_ol = np.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
                    l2_ol_chunks.append(
                        np.linalg.norm(x_pred_dense - x_ref_dense_ol, axis=2) / denom_ol
                    )
                    eqvar_ol_chunks.append(
                        _equilibrium_variance_per_ic(x_pred_dense, eqvar_burn_in_frac)
                    )
                    del x_pred_dense, x_ref_dense_ol, denom_ol

                l2_ol = np.mean(np.concatenate(l2_ol_chunks, axis=0), axis=0)
                eqvar_mean_ol, eqvar_std_ol = _mean_std(
                    np.concatenate(eqvar_ol_chunks, axis=0)
                )
                del l2_ol_chunks, eqvar_ol_chunks, predict_full
                jax.clear_caches()

                open_loop_records[prop_key] = dict(
                    t=np.arange(total_steps_ol) * dt_integration,
                    l2_dense_mean=l2_ol,
                    eqvar_mean=eqvar_mean_ol, eqvar_std=eqvar_std_ol,
                )

    # ── 6. Write everything to HDF5 ────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".partial"

    with h5py.File(tmp_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["N"] = N
        meta.attrs["L"] = L_dom
        meta.attrs["dt_data"] = dt_data
        meta.attrs["dt_window"] = DT_WINDOW
        meta.attrs["dt_fine"] = DT_FINE
        meta.attrs["dt_obs"] = DT_OBS
        meta.attrs["sigma_obs"] = sigma_obs
        meta.attrs["P0_sigma"] = P0_sigma
        meta.attrs["P0_corr_len"] = P0_corr_len
        meta.attrs["N_ens"] = N_ens
        meta.attrs["obs_every_n"] = obs_every_n
        meta.attrs["m"] = m
        meta.attrs["num_ics_traj"] = num_plots
        meta.attrs["num_ics_batch"] = B
        meta.attrs["trajectory_windows"] = trajectory_windows
        meta.attrs["batch_windows"] = batch_windows
        meta.attrs["test_data"] = os.path.basename(test_h5_path)
        meta.create_dataset("obs_indices", data=np.array(obs_indices))
        meta.create_dataset("x_grid", data=x_grid)
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

# Canonical surrogate tokens. KS has three (L96 has two): the hybrid
# checkpoint is PI plus a fraction of DD, and is spelled "HY" or "hybrid".
SURROGATES = ("dd", "pi", "hy")

# Accepted spellings -> canonical token.
_SURROGATE_ALIASES = {
    "dd": "dd", "data_driven": "dd", "datadriven": "dd",
    "pi": "pi", "physics_informed": "pi", "physicsinformed": "pi",
    "hy": "hy", "hybrid": "hy",
}

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

# Inflation schemes that need the PDE residual `r_net` (Route B's
# flow-dependent scale, and the additive scheme that shares its machinery).
# The data-driven surrogate has no residual, so those combinations are
# rejected while parsing the config rather than after the checkpoints have
# been loaded and earlier jobs have already run.
_RESIDUAL_INFLATIONS = ("additive", "route_b")

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

_SURROGATE_LABEL = {"dd": "DD", "pi": "PI", "hy": "Hybrid"}
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
    surrogate: str          # "dd" | "pi" | "hy"
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

    surrogate = _SURROGATE_ALIASES.get(_norm_token(entry["surrogate"]), None)
    if surrogate is None:
        raise ValueError(f"{where}: surrogate must be 'DD', 'PI' or 'HY'/'hybrid', "
                         f"got {entry['surrogate']!r}.")

    inflation = _norm_token(entry["inflation"])
    if inflation not in _INFLATION_SCHEMA:
        raise ValueError(f"{where}: inflation must be one of {sorted(_INFLATION_SCHEMA)}, "
                         f"got {entry['inflation']!r}.")
    schema = _INFLATION_SCHEMA[inflation]

    # DD has no PDE residual (`r_net`), so it cannot drive Route B's
    # flow-dependent scale -- nor the additive scheme, which runs on the same
    # machinery. Reject here, before any checkpoint is loaded.
    if surrogate == "dd" and inflation in _RESIDUAL_INFLATIONS:
        raise ValueError(
            f"{where}: the data-driven surrogate has no PDE residual (r_net), so it "
            f"cannot run '{inflation}' inflation (Route B and additive share that "
            "machinery). Use surrogate 'PI' or 'HY', or inflation 'multiplicative' "
            "or 'rtpp' with 'DD'."
        )

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
            "{'hy_mult': dict(surrogate='HY', inflation='multiplicative', "
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


# ─────────────────────────────────────────────────────────────────────────
# Surrogate loading / EnKF closures / strategy specs
# ─────────────────────────────────────────────────────────────────────────

class _Surrogates:
    """
    Lazily loads the DD / PI / Hybrid checkpoints and builds their EnKF
    closures.

    Each surrogate is loaded at most once, and each (surrogate, kind) closure
    pair is built at most once, however many parameter versions use it:
    inflation parameters are runtime scalars carried in the strategy spec, not
    baked into the closures, so a sweep costs no extra model calls.

    The spectral operators of the PI and Hybrid models are built from the test
    dataset's own (N, L, dt) rather than from class defaults, which is why
    `grid` (from `_read_test_meta`) is required here -- the same `grid` also
    fixes the single-window time axis `t_star_window` every surrogate is
    queried on, and the shape of the Route B / additive `Q0`.
    """

    def __init__(self, config, N_ens: int, grid: dict):
        self.config = config
        self.N_ens = N_ens
        self.grid = grid
        dt_window, _, t_star_window = _window_grid(config, grid)
        self.dt_window = dt_window
        self.t_star_window = t_star_window
        self._models = {}      # surrogate -> (model, params)
        self._fns = {}         # (surrogate, kind) -> (predict_fn, update_fn)
        self._Q0 = None        # built once, shared by every Route B / additive spec
        self.N = None

    def propagator(self, surrogate: str):
        if surrogate not in self._models:
            self._models[surrogate] = _load_ks_model(
                self.config, surrogate, self.t_star_window, self.grid
            )
            N = self._models[surrogate][0].N
            if self.N is None:
                self.N = N
            elif N != self.N:
                raise ValueError(
                    f"The {_SURROGATE_LABEL[surrogate]} checkpoint has grid size {N}, "
                    f"but an earlier one has {self.N}; every surrogate compared here "
                    "must share one KS grid."
                )
        return self._models[surrogate]

    def enkf_fns(self, surrogate: str, kind: str):
        if (surrogate, kind) not in self._fns:
            model, params = self.propagator(surrogate)
            method = _CLOSURE_METHOD[kind]
            factory = getattr(model, method, None)
            if factory is None:
                raise NotImplementedError(
                    f"The {_SURROGATE_LABEL[surrogate]} surrogate ({type(model).__name__}) "
                    f"does not expose `{method}`, which is needed for the '{kind}' "
                    "EnKF variant. Implement it on the model or use another surrogate."
                )
            self._fns[(surrogate, kind)] = factory(params, N_ens=self.N_ens)
        return self._fns[(surrogate, kind)]

    def Q0_fine(self):
        """
        The per-fine-step additive covariance shared by every Route B /
        additive strategy: a `Q0_sigma`-scaled (optionally spatially
        correlated) covariance on the KS grid, divided down so that
        accumulating it over one window reproduces the coarse `Q0`.
        """
        if self._Q0 is None:
            cfg = self.config
            P0_sigma = cfg.kf.get("P0_sigma", 0.5)
            Q0_sigma = cfg.kf.get("Q0_sigma", P0_sigma)
            Q0_corr_len = float(
                cfg.kf.get("Q0_corr_len", cfg.kf.get("P0_corr_len", 0.0))
            )
            dt_fine = float(cfg.kf.get("dt_fine", self.dt_window))
            steps_per_window = steps_per_window_exact(self.dt_window, dt_fine)
            Q_coarse = build_cov(
                self.grid["N"], self.grid["L"], Q0_sigma, Q0_corr_len
            )
            self._Q0 = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
        return self._Q0

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

        dt_fine = float(cfg.kf.get("dt_fine", self.dt_window))
        steps_per_window = steps_per_window_exact(self.dt_window, dt_fine)

        if job.inflation == "multiplicative":
            spec["alpha_fine"] = scale_inflation_for_fine_steps(
                p["inflation_factor"], steps_per_window)
        elif job.inflation == "rtpp":
            spec["alpha_fine"] = p["alpha_fine"]
            spec["alpha_rtpp"] = p["alpha_rtpp"]
        else:  # additive / route_b share the Route B machinery
            spec["Q0"] = self.Q0_fine()
            spec["n_quad"] = cfg.kf.get("route_b_n_quad", 3)
            spec["alpha"] = p["alpha"]
            spec["beta"] = 0.0 if job.inflation == "additive" else p["beta"]
        return spec


def _shared_settings(config) -> dict:
    """The config-derived settings stored in every file's /meta (mirrors the
    defaults `evaluate_filters` uses); compared against cached files."""
    dt_window = float(config.get("dt_window", 1.0))
    return dict(
        N_ens=int(config.kf.get("N_ens", 50)),
        sigma_obs=float(config.kf.get("sigma_obs", 0.1)),
        P0_sigma=float(config.kf.get("P0_sigma", 0.5)),
        P0_corr_len=float(config.kf.get("P0_corr_len", 0.0)),
        obs_every_n=int(config.kf.get("obs_every_n", 4)),
        dt_window=dt_window,
        dt_fine=float(config.kf.get("dt_fine", dt_window)),
        dt_obs=float(config.kf.get("dt_obs", dt_window)),
        trajectory_windows=int(config.eval.get("trajectory_windows", 20)),
        batch_windows=int(config.eval.get("windows", 20)),
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
            f"{config.eval.get('test_data_name', 'ks_test_data')!r})."
        )
    grid = _read_test_meta(test_h5_path)

    # Load the needed surrogates and build EVERY closure/spec up front, so an
    # unsupported combination (e.g. a surrogate without a Route B closure)
    # fails immediately instead of after earlier jobs have already run.
    surrogates = _Surrogates(config, N_ens=settings["N_ens"], grid=grid)
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


def _tight_layout(fig, **kwargs):
    """
    `fig.tight_layout()` on a figure that contains an equal-aspect axes
    warns that the result "might be incorrect" while still producing an
    acceptable layout. The calibration figures need that 1:1 aspect for the
    spread-skill panel to be readable, so the warning would otherwise fire
    once per calibration PDF -- C(S, 2) times for a sweep. Suppress just
    that message, nothing else.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*not compatible with tight_layout.*")
        fig.tight_layout(**kwargs)


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


def _window_average(t_axis: np.ndarray, values: np.ndarray, dt_window: float):
    """
    Bucket a (t_axis, values) curve into non-overlapping windows of width
    `dt_window` (assumed to start at t_axis[0]) and return the per-window
    mean time and mean value.
 
    Used to turn a per-fine-timestep curve (e.g. the L2-vs-time curves)
    into a per-time-window curve for a decluttered comparison page, the
    same way the calibration/ERF/RMSE plots already work in window- or
    observation-indexed units rather than raw fine timesteps.
    """
    t_axis = np.asarray(t_axis, dtype=float)
    values = np.asarray(values, dtype=float)
    t0 = float(t_axis[0])
    win_idx = np.floor((t_axis - t0) / dt_window + 1e-9).astype(int)
    uniq = np.unique(win_idx)
    # Window-center times, purely for plotting on a continuous time axis.
    w_t = t0 + (uniq + 0.5) * dt_window
    w_v = np.array([np.nanmean(values[win_idx == w]) for w in uniq])
    return w_t, w_v
 
 
# ═══════════════════════════════════════════════════════════════════════
# Used by the individual-trajectory snapshot selection
# ═══════════════════════════════════════════════════════════════════════
def _snapshot_window_indices(t_ax: np.ndarray, dt_window: float) -> np.ndarray:
    """
    Choose which fine-timestep indices of `t_ax` to use as spatial-profile
    snapshots, based on WINDOW-BOUNDARY times rather than uniform spacing:
 
      * the first 20 windows' limits:   t = 0, 1, 2, ..., 20   (x dt_window)
      * 10-window blocks, repeating with a 50-window gap between them
        (i.e. a new block starts every 60 windows), starting at window 70:
        t = 70..80, 130..140, 190..200, ...  (x dt_window)
      * the last 20 windows' limits of the run
 
    Overlaps between these three groups (e.g. a short run where the
    periodic blocks run into the final-20 region) are deduplicated
    automatically. Returns the sorted, deduplicated indices into `t_ax`
    nearest each of those times.
    """
    t_ax = np.asarray(t_ax, dtype=float)
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    n_windows = int(round((t_max - t_min) / dt_window))
 
    win_idx = set(range(0, min(20, n_windows) + 1))          # first 20 window limits
 
    start = 70
    while start <= n_windows:                                 # periodic 10-window blocks
        win_idx.update(range(start, min(start + 10, n_windows) + 1))
        start += 60                                           # 10-window span + 50-window gap
 
    win_idx.update(range(max(0, n_windows - 20), n_windows + 1))  # last 20 window limits
 
    times = sorted(t_min + w * dt_window for w in win_idx if 0 <= w <= n_windows)
    idx = np.unique(np.array([int(np.argmin(np.abs(t_ax - t))) for t in times]))
    return idx

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
    calibrating a large sweep, where the all-strategies page answers
    "what does the whole sweep look like?" and this page answers "which
    settings actually won?".

    Parameters
    ----------
    metric_of : dict
        key -> 1-D array of the metric sampled over time. Each array is
        collapsed to one scalar by taking the mean over its finite entries.
    keys : iterable, optional
        Which keys of `metric_of` to rank (defaults to all of them). Lets
        callers exclude curves that aren't strategies, e.g. the open-loop
        reference rollouts in the L2 plot.
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
# Observation bookkeeping for the trajectory plots
# ─────────────────────────────────────────────────────────────────────────
def _index_observations(obs_coords):
    """
    Split the flat (grid_idx, t_obs, y_obs) observation list into two
    lookups: by grid point (for the probe time series) and by observation
    time (for the profile snapshots).

    Returns (obs_by_point, obs_by_time, obs_points), where obs_points is
    the sorted set of grid indices that were EVER observed -- which, when
    `config.kf.dynamic_vars` is on and the sensor network roves, is larger
    than the m points observed at any single time.
    """
    obs_by_point: dict[int, list[tuple[float, float]]] = {}
    obs_by_time: dict[float, list[tuple[int, float]]] = {}
    if obs_coords is None:
        return {}, {}, []

    for row in obs_coords:
        p, t, y = int(row[0]), float(row[1]), float(row[2])
        obs_by_point.setdefault(p, []).append((t, y))
        obs_by_time.setdefault(round(t, 9), []).append((p, y))

    obs_by_point = {k: sorted(v) for k, v in obs_by_point.items()}
    return obs_by_point, obs_by_time, sorted(obs_by_point)


def _pick_probe_points(N, obs_points, n_probes):
    """
    Choose which grid points get their own time-series panel.

    Half are drawn from the observed points and half from the gaps
    between them (when the network is static and sparse), because that
    contrast is the whole question for a field: an EnKF will almost always
    look good where it is told the answer, and the unobserved points are
    where the ensemble covariance either does or doesn't carry the
    information across.

    Falls back to evenly spaced points when everything (or nothing) is
    observed.
    """
    obs_set = set(int(p) for p in obs_points)
    unobs = [i for i in range(N) if i not in obs_set]
    if not obs_set or not unobs:
        return list(np.linspace(0, N - 1, n_probes, dtype=int)), obs_set

    n_o = max(1, n_probes // 2)
    n_u = max(1, n_probes - n_o)
    obs_sorted = sorted(obs_set)
    pick_o = [obs_sorted[i] for i in np.linspace(0, len(obs_sorted) - 1, n_o, dtype=int)]
    pick_u = [unobs[i] for i in np.linspace(0, len(unobs) - 1, n_u, dtype=int)]
    return sorted(set(pick_o) | set(pick_u)), obs_set


# ─────────────────────────────────────────────────────────────────────────
# 1. Individual-trajectory plot (single strategy vs ground truth)
# ─────────────────────────────────────────────────────────────────────────
def _plot_trajectory_individual(
    t_ax: np.ndarray,
    x_true: np.ndarray,
    x_est: np.ndarray,
    x_std: np.ndarray,
    ic_idx: int,
    strategy_label: str,
    l2_time_avg: float,
    save_path: str,
    x_grid: np.ndarray = None,
    dt_window: float = None,
    obs_coords=None,
    n_probes: int = 10,     # 5 obs + 5 unobs
    n_snapshots: int = 8,   # fallback only; overridden by the window scheme when dt_window is set
) -> None:
    """
    Trajectory-summary PDF for ONE strategy on ONE IC, now written as a
    2-page PDF:
 
      page 1
        row 0            error/spread time series (relative L2, mean
                         |error|, RMSE vs. ensemble spread)
        probe rows       `n_probes` single-grid-point time series (now 10
                         by default: 5 observed + 5 unobserved, evenly
                         spaced -- 2 more of each than before)
        snapshot rows     spatial profiles (truth vs. estimate, ±1σ band)
                         at window-boundary times: the first 20 windows,
                         then 10-window blocks every 60 windows (50-window
                         gap) starting at window 70, then the last 20
                         windows (see `_snapshot_window_indices`); falls
                         back to `n_snapshots` evenly-spaced snapshots if
                         `dt_window` is not given.
 
      page 2             the three space-time heatmaps -- reference truth,
                         ensemble mean, and (estimate - truth) -- stacked
                         VERTICALLY instead of side-by-side. The
                         "observed x" markers on the difference heatmap
                         are dots instead of tick lines, plus a second
                         dot-scatter showing the actual observation times
                         of ONE representative observed grid point (a
                         vertical strip of dots), to show how densely
                         that point was sampled in time.
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    x_std = np.asarray(x_std) if x_std is not None else None
    T, N = x_true.shape
    if x_grid is None:
        x_grid = np.arange(N, dtype=float)
 
    diff = x_est - x_true
    mean_abs_err = np.abs(diff).mean(axis=1)
    rmse_t = np.sqrt(np.mean(diff ** 2, axis=1))
    rel_l2_t = np.linalg.norm(diff, axis=1) / (np.linalg.norm(x_true, axis=1) + 1e-12)
    spread_t = (np.sqrt(np.mean(x_std ** 2, axis=1)) if x_std is not None else None)
 
    t_min, t_max = float(t_ax[0]), float(t_ax[-1])
    if dt_window is not None and dt_window > 0:
        first_k = int(np.floor(t_min / dt_window)) + 1
        window_boundaries = np.arange(
            first_k * dt_window, t_max + 1e-12 * dt_window, dt_window
        )
    else:
        window_boundaries = np.array([])
 
    obs_by_point, obs_by_time, obs_points = _index_observations(obs_coords)
    probes, obs_set = _pick_probe_points(N, obs_points, n_probes)
    n_probes = len(probes)
 
    # ── Window-boundary snapshot selection instead of linspace ─
    if dt_window is not None and dt_window > 0:
        snap_idx = _snapshot_window_indices(t_ax, dt_window)
    else:
        snap_idx = np.unique(np.linspace(0, T - 1, min(n_snapshots, T)).astype(int))
    n_snapshots = len(snap_idx)
 
    TRUTH_COLOR = "#37474F"
    EST_COLOR, EST_BAND = "#2196F3", "#90CAF9"
    OBS_COLOR = "#E53935"
 
    # ═════════════════════ PAGE 1: metrics / probes / snapshots ════════
    probe_cols, snap_cols = 2, 4
    probe_rows = int(np.ceil(n_probes / probe_cols))
    snap_rows = int(np.ceil(n_snapshots / snap_cols))
 
    metric_h, probe_h, snap_h = 2.6, 1.9, 2.0
    # NOTE: fig_h no longer includes heat_h -- heatmaps moved to page 2.
    fig_h = metric_h + probe_rows * probe_h + snap_rows * snap_h + 1.0
    fig = plt.figure(figsize=(16, fig_h))
    gs = gridspec.GridSpec(
        nrows=1 + probe_rows + snap_rows, ncols=12, figure=fig,
        height_ratios=[metric_h] + [probe_h] * probe_rows + [snap_h] * snap_rows,
        hspace=0.75, wspace=0.55,
    )
 
    # ── row 0: error and spread time series (was row 1) ─────────────────
    ax_err = fig.add_subplot(gs[0, 0:6])
    ax_err.plot(t_ax, rel_l2_t, color="#E53935", linewidth=1.2,
                label="Relative L2 error")
    ax_err.plot(t_ax, mean_abs_err, color="#1E88E5", linewidth=1.0,
                linestyle="--", label="Mean |error|")
    for wb in window_boundaries:
        ax_err.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.7, alpha=0.45)
    l2_str = f"  |  time-avg L2 = {l2_time_avg:.3e}" if l2_time_avg is not None else ""
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Time  t", fontsize=10)
    ax_err.set_ylabel("Error (log)", fontsize=10)
    ax_err.set_title(f"Error vs time{l2_str}", fontsize=11, fontweight="bold")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    ax_sp = fig.add_subplot(gs[0, 6:12])
    ax_sp.plot(t_ax, rmse_t, color="#FF8A65", linewidth=1.2, linestyle="--",
               label="EnKF RMSE")
    if spread_t is not None:
        ax_sp.plot(t_ax, spread_t, color="#8BC34A", linewidth=1.2,
                   label="RMS ensemble σ")
    for wb in window_boundaries:
        ax_sp.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.7, alpha=0.45)
    ax_sp.set_yscale("log")
    ax_sp.set_xlabel("Time  t", fontsize=10)
    ax_sp.set_ylabel("Log scale", fontsize=10)
    ax_sp.set_title("Spread vs skill (this trajectory)", fontsize=11, fontweight="bold")
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    # ── probe rows (row offset now 1, was 2) ─────────────────────────────
    for i, p in enumerate(probes):
        row = 1 + i // probe_cols
        col0 = (i % probe_cols) * 6
        ax = fig.add_subplot(gs[row, col0:col0 + 6])
 
        for wb in window_boundaries:
            ax.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.5, alpha=0.35)
 
        ax.plot(t_ax, x_true[:, p], color=TRUTH_COLOR, linewidth=0.9, label="Truth")
        ax.plot(t_ax, x_est[:, p], color=EST_COLOR, linewidth=0.9, linestyle="--",
                label=strategy_label)
        if x_std is not None:
            ax.fill_between(
                t_ax, x_est[:, p] - x_std[:, p], x_est[:, p] + x_std[:, p],
                color=EST_BAND, alpha=0.30, linewidth=0, label="±1σ",
            )
        if p in obs_by_point:
            ot, ov = zip(*obs_by_point[p])
            ax.scatter(ot, ov, marker="x", s=14, linewidths=0.6,
                       color=OBS_COLOR, zorder=5, label="Observation")
 
        tag = "observed" if p in obs_set else "gap"
        ax.set_title(f"x = {x_grid[p]:.2f}  (grid point {p}, {tag})",
                     fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_ylabel("u", fontsize=8)
        if row == 1 + probe_rows - 1:
            ax.set_xlabel("t", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper right", handlelength=1.2,
                      framealpha=0.7, ncol=2)
 
    # ── snapshot rows (row offset now 1 + probe_rows, was 2 + probe_rows) ─
    for i, ti in enumerate(snap_idx):
        row = 1 + probe_rows + i // snap_cols
        col0 = (i % snap_cols) * 3
        ax = fig.add_subplot(gs[row, col0:col0 + 3])
 
        ax.plot(x_grid, x_true[ti], color=TRUTH_COLOR, linewidth=1.2, label="Truth")
        ax.plot(x_grid, x_est[ti], color=EST_COLOR, linewidth=1.2, linestyle="--",
                label="Est.")
        if x_std is not None:
            ax.fill_between(x_grid, x_est[ti] - x_std[ti], x_est[ti] + x_std[ti],
                            color=EST_BAND, alpha=0.30, linewidth=0, label="±1σ")
 
        dt_fine = float(t_ax[1] - t_ax[0]) if T > 1 else 0.0
        if obs_by_time and dt_fine > 0:
            t_snap = float(t_ax[ti])
            near = min(obs_by_time, key=lambda tt: abs(tt - t_snap))
            if abs(near - t_snap) <= 0.5 * dt_fine:
                pts, vals = zip(*obs_by_time[near])
                ax.scatter(x_grid[np.asarray(pts)], vals, marker="x", s=12,
                           linewidths=0.6, color=OBS_COLOR, zorder=5, label="Obs")
 
        ax.set_title(f"t = {t_ax[ti]:.3g}", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        if i % snap_cols == 0:
            ax.set_ylabel("u", fontsize=8)
        ax.set_xlabel("x", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper right", handlelength=1.2,
                      framealpha=0.7, ncol=2)
 
    fig.suptitle(
        f"KS trajectory summary — IC {ic_idx}  |  {strategy_label}",
        fontsize=14, fontweight="bold", y=1.001,
    )
 
    # ═════════════════ PAGE 2: heatmaps, stacked vertically
    extent = [float(x_grid[0]), float(x_grid[-1]), t_min, t_max]
    vmax = float(np.max(np.abs(x_true)))
    dmax = float(np.max(np.abs(diff))) or 1.0
 
    fig2 = plt.figure(figsize=(6.5, 12.5))
    gs2 = gridspec.GridSpec(3, 1, figure=fig2, hspace=0.32)
 
    ax_t2 = fig2.add_subplot(gs2[0, 0])
    ax_e2 = fig2.add_subplot(gs2[1, 0], sharex=ax_t2)
    ax_d2 = fig2.add_subplot(gs2[2, 0], sharex=ax_t2)
 
    im_t = ax_t2.imshow(x_true, aspect="auto", extent=extent, origin="lower",
                        cmap="viridis", vmin=-vmax, vmax=vmax)
    ax_t2.set_title("Reference truth", fontsize=11, fontweight="bold")
    ax_t2.set_ylabel("Time  t", fontsize=10)
    fig2.colorbar(im_t, ax=ax_t2, fraction=0.046, pad=0.04)
    plt.setp(ax_t2.get_xticklabels(), visible=False)
 
    im_e = ax_e2.imshow(x_est, aspect="auto", extent=extent, origin="lower",
                        cmap="viridis", vmin=-vmax, vmax=vmax)
    ax_e2.set_title(f"{strategy_label}: ensemble mean", fontsize=11, fontweight="bold")
    ax_e2.set_ylabel("Time  t", fontsize=10)
    fig2.colorbar(im_e, ax=ax_e2, fraction=0.046, pad=0.04)
    plt.setp(ax_e2.get_xticklabels(), visible=False)
 
    im_d = ax_d2.imshow(diff, aspect="auto", extent=extent, origin="lower",
                        cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    ax_d2.set_title("Estimate − truth", fontsize=11, fontweight="bold")
    ax_d2.set_xlabel("x", fontsize=10)
    ax_d2.set_ylabel("Time  t", fontsize=10)
    fig2.colorbar(im_d, ax=ax_d2, fraction=0.046, pad=0.04)
 
    # -- Dots instead of tick lines for "which x are observed"
    if obs_points and len(obs_points) < N:
        ax_d2.scatter(x_grid[np.asarray(obs_points)],
                      np.full(len(obs_points), t_min), marker="o", s=10,
                      color=OBS_COLOR, clip_on=False, zorder=6,
                      label="observed x")
 
    # -- temporal density of ONE representative observed grid point
    if obs_points:
        rep_point = int(obs_points[len(obs_points) // 2])
        if rep_point in obs_by_point:
            rep_t, _rep_y = zip(*obs_by_point[rep_point])
            ax_d2.scatter(
                np.full(len(rep_t), x_grid[rep_point]), rep_t,
                marker="o", s=6, color=OBS_COLOR, alpha=0.55,
                edgecolors="white", linewidths=0.3, zorder=7,
                label=f"obs density @ x={x_grid[rep_point]:.2f}",
            )
    if obs_points and len(obs_points) < N:
        ax_d2.legend(fontsize=7, loc="upper right", framealpha=0.7)
 
    fig2.suptitle(
        f"KS heatmaps — IC {ic_idx}  |  {strategy_label}",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
 
    _save_pdf_pages([fig, fig2], save_path, dpi=140)
    logging.info(
        f"Individual trajectory plot (IC {ic_idx}, {strategy_label}, 2-page PDF: "
        f"metrics/probes/snapshots + heatmaps) saved to: {save_path}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2a. EnKF vs open-loop, time-mean relative L2 (curve-count-agnostic;
#     shared by the pairwise and bulk entry points)
# ─────────────────────────────────────────────────────────────────────────
def _plot_l2_per_timestep(
    curves: dict,             # label -> (t_axis, l2_array)
    title: str,
    save_path: str,
    colors: dict = None,
    dt_window: float = None,  # required to draw the two windowed pages
) -> None:
    """
    Plot average L2 error continuously across fine time stamps, as a
    multi-page PDF:
 
      page 1 -- every curve in `curves`.                                  (unchanged)
      page 2 -- open-loop curves omitted.                                 (unchanged)
      pages 3-4 -- best-TOP_K_BEST filtered strategies only.              (unchanged)
      page 5 --- same as page 1, but every curve is averaged into
        dt_window-wide time buckets, and each legend entry reports that
        strategy's overall (all-time, full-resolution) mean relative L2.
      page 6 -- same as page 5, with open-loop curves omitted, like
        page 2. Pages 5-6 are skipped if `dt_window` is not given.
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
 
    # ── pages 5-6: window-averaged versions of pages 1-2 ──────────────
    if dt_window is not None and dt_window > 0:
 
        def _draw_windowed(curve_subset, subtitle):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for i, (label, (t_axis, l2_arr)) in enumerate(curve_subset.items()):
                color = (colors or {}).get(label, default_colors[i % len(default_colors)])
                t_axis = np.asarray(t_axis)
                l2_arr = np.asarray(l2_arr)
                w_t, w_l2 = _window_average(t_axis, l2_arr, dt_window)
                finite = l2_arr[np.isfinite(l2_arr)]
                overall_avg = float(finite.mean()) if finite.size else float("nan")
                ax.plot(w_t, w_l2, linewidth=1.2, marker="o", markersize=3,
                        color=color, label=f"{label}  (overall avg={overall_avg:.3g})")
            ax.set_yscale("log")
            ax.set_xlabel("Time (t)  [window-averaged]", fontsize=12)
            ax.set_ylabel("Mean relative L2 error (log scale)", fontsize=12)
            ax.set_title(subtitle, fontsize=13)
            ax.legend(fontsize=8, ncol=(2 if len(curve_subset) > 5 else 1))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            fig.tight_layout()
            return fig
 
        figs.append(_draw_windowed(curves, title + "\n(window-averaged)"))
        if open_loop_labels and filtered_curves:
            figs.append(_draw_windowed(
                filtered_curves,
                title + "\n(window-averaged; pure-propagator open-loop curves omitted)",
            ))
 
    _save_pdf_pages(figs, save_path)
    logging.info(
        f"L2-vs-open-loop comparison plot ({len(figs)}-page PDF) saved to: {save_path}"
    )

# ─────────────────────────────────────────────────────────────────────────
# Bulk (all-strategies-at-once) comparison plots
# ─────────────────────────────────────────────────────────────────────────
"""
Instead of one PDF per strategy PAIR per category (4 * C(S, 2) PDFs), the
bulk plots produce exactly ONE PDF per category, with every strategy
overlaid (4 PDFs total, regardless of S). Individual-trajectory plots are
unchanged -- still one PDF per (IC, strategy).

Decluttering strategy
---------------------
  * Calibration's spread/RMSE timeseries uses small multiples: one row
    per strategy, each pairing that strategy's own RMS ensemble spread
    and EnKF RMSE on a single axes.
  * The prior/posterior-RMSE panels are split into two side-by-side
    subplots (one metric each), so each subplot only ever has S lines.
  * Every strategy gets a single STABLE color from `_strategy_colors`,
    reused across every panel and every bulk plot.
  * Legends switch to two columns past ~4-5 strategies; error-bar and
    spread-band decorations fade (or drop, for ERF beyond 4 strategies)
    as S grows.
  * The L2, ERF and prior/posterior-RMSE PDFs each carry an extra
    "best strategies only" page (see `_rank_by_mean` / `TOP_K_BEST`).
"""


def _distinct_colors(n: int) -> list[str]:
    """
    Generate `n` hex colors that stay visually distinguishable well past
    a fixed 10-color palette. Draws first from matplotlib's qualitative
    tab20/tab20b/tab20c colormaps (60 swatches designed to be pairwise
    distinguishable); beyond that, falls back to evenly spaced HSV hues
    rather than repeating a color.
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
    everywhere."""
    palette = _distinct_colors(len(strategy_keys))
    return {k: palette[i] for i, k in enumerate(strategy_keys)}


# ─────────────────────────────────────────────────────────────────────────
# 3a. Calibration, ALL strategies on one PDF
# ─────────────────────────────────────────────────────────────────────────
def _plot_calibration_bulk(
    strategy_keys,
    label_of,
    window_idx: np.ndarray,
    dt_window: float,
    spread_window_of: dict,
    rmse_window_of: dict,
    spread_raw_of: dict,
    rmse_raw_of: dict,
    colors: dict,
    title: str,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Calibration, written as a 2-page PDF covering every strategy:
 
      page 1 -- one small row per strategy pairing that
        strategy's own RMS ensemble spread and EnKF RMSE, plus the pooled
        binned spread-skill scatter (linear + log/log).
      page 2 -- |RMS ensemble sigma - EnKF RMSE| vs. window index,
        every strategy overlaid on ONE set of axes (like the binned
        spread-skill panel on page 1, rather than the per-strategy small
        multiples), with each strategy's overall time-mean absolute
        difference shown in its legend entry and in a summary textbox.
    """
    S = len(strategy_keys)
    row_h = 1.15
    bin_row_h = 4.4
    fig_h = row_h * S + bin_row_h + 1.0
    fig = plt.figure(figsize=(9.5, fig_h))
    gs = gridspec.GridSpec(
        S + 1, 2, height_ratios=[row_h] * S + [bin_row_h], hspace=0.65, wspace=0.3,
    )
 
    # -- One row per strategy: spread + RMSE paired on the same axes ---- (unchanged)
    ax_prev = None
    for i, key in enumerate(strategy_keys):
        ax = fig.add_subplot(gs[i, :], sharex=ax_prev)
        ax_prev = ax
        c = colors[key]
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
 
    # -- Final row: pooled binned spread-skill, linear + log/log -------- (unchanged)
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
    _tight_layout(fig, rect=[0.04, 0, 1, 1 - 0.85 / fig_h])
 
    # Page 2: |spread - RMSE| over time, all strategies overlaid
    fig2, ax2 = plt.subplots(figsize=(9.5, 5.5))
    avg_abs_diff = {}
    for key in strategy_keys:
        diff_t = np.abs(np.asarray(spread_window_of[key]) - np.asarray(rmse_window_of[key]))
        avg_abs_diff[key] = float(np.nanmean(diff_t))
        ax2.plot(window_idx, diff_t, linewidth=1.1, color=colors[key],
                 label=f"{label_of[key]}  (avg={avg_abs_diff[key]:.3g})")
    ax2.set_yscale("log")
    ax2.set_xlabel("Window index", fontsize=11)
    ax2.set_ylabel("|RMS ensemble σ − EnKF RMSE|  (log scale)", fontsize=11)
    ax2.set_title(
        f"Spread–skill absolute difference over time — {title}\n"
        f"(window index × dt_window={dt_window:g} = simulation time)",
        fontsize=12,
    )
    ax2.legend(fontsize=7.5, ncol=(2 if S > 5 else 1))
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    summary_txt = "\n".join(f"{label_of[k]}: {avg_abs_diff[k]:.3g}" for k in strategy_keys)
    ax2.text(
        1.02, 0.5, "Overall avg |Δ|:\n" + summary_txt, transform=ax2.transAxes,
        fontsize=7, va="center", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    fig2.tight_layout(rect=[0, 0, 0.82, 1])
 
    _save_pdf_pages([fig, fig2], save_path)
    logging.info(
        f"Bulk calibration plot (2-page PDF: {S} strategies, stacked small-multiples + "
        f"spread-skill abs-diff) saved to: {save_path}"
    )
 

# ─────────────────────────────────────────────────────────────────────────
# 3b. Error Reduction Factor, ALL strategies on one PDF
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
    show_bands: bool = None,
) -> None:
    """Same 1-2 page structure as before; each legend entry now also
    reports that strategy's time-mean ERF."""
    S = len(strategy_keys)
 
    def _draw(keys, subtitle):
        n = len(keys)
        bands = (n <= 4) if show_bands is None else show_bands
        band_alpha = 0.15 if n <= 3 else 0.08
 
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key in keys:
            c = colors[key]
            mean, std = erf_mean_of[key], erf_std_of[key]
            avg_erf = float(np.nanmean(mean))          
            ax.plot(obs_times, mean, color=c, linewidth=1.1, marker="o", markersize=2.2,
                    label=f"{label_of[key]}  (avg={avg_erf:.3g})")
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
# 3c. Prior vs posterior RMSE, ALL strategies on one PDF
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
    """Same 1-2 page structure as before; each legend entry (on both the
    prior and posterior panels, separately) now also reports that
    strategy's time-mean RMSE for that panel's metric."""
    S = len(strategy_keys)
 
    def _draw(keys, subtitle):
        n = len(keys)
        fig, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
 
        for key in keys:
            c = colors[key]
            prior_vals = np.asarray(prior_mean_of[key])
            post_vals = np.asarray(post_mean_of[key])
            prior_avg = float(np.nanmean(prior_vals))
            post_avg = float(np.nanmean(post_vals))
            ax_prior.plot(obs_times, prior_vals, color=c, linewidth=1.1,
                          marker="o", markersize=2.2,
                          label=f"{label_of[key]}  (avg={prior_avg:.3g})")
            ax_post.plot(obs_times, post_vals, color=c, linewidth=1.1,
                         marker="o", markersize=2.2,
                         label=f"{label_of[key]}  (avg={post_avg:.3g})")
 
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
        ax_post.legend(fontsize=8, ncol=(2 if n > 4 else 1))
 
        fig.suptitle(f"{subtitle}  (n = {n_traj} trajectories)", fontsize=13, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
 
    figs = [_draw(strategy_keys, title)]
 
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
# 3d. Equilibrium (climatological/attractor) variance
# ─────────────────────────────────────────────────────────────────────────
def _plot_equilibrium_variance_bulk(
    strategy_keys,
    label_of,
    eqvar_mean_of: dict,          # key -> (N,) per-grid-point equilibrium variance
    eqvar_std_of: dict,           # key -> (N,) IC-to-IC std of that estimate
    reference_mean: np.ndarray,   # (N,) reference (unfiltered truth) equilibrium variance
    reference_std: np.ndarray,
    open_loop_eqvar: dict,        # propagator -> dict(mean=(N,), std=(N,))
    colors: dict,
    title: str,
    save_path: str,
    x_grid: np.ndarray | None = None,
    obs_indices: np.ndarray | None = None,
    log_scale: bool = True,
) -> None:
    """
    Per-grid-point equilibrium (climatological/attractor) variance -- the
    long-term temporal variance of the field at each x once transients
    have decayed -- comparing:

      * the reference/truth trajectory (the intrinsic variability of the
        unfiltered system; the target every curve below is compared
        against),
      * each propagator's open-loop, "static" (unfiltered) physics
        rollout, and
      * every filtered EnKF strategy's posterior-mean trajectory,

    all on ONE set of axes regardless of strategy count S.

    Reading this plot for KS
    ------------------------
    On a periodic domain the KS attractor is statistically homogeneous, so
    the reference curve should be close to FLAT in x. That makes two
    failure modes immediately visible:

      * a curve sitting uniformly below the reference -> variance
        collapse (over-confident filtering / insufficient inflation);
      * a curve that is flat at the observed grid points but sags between
        them -> the analysis is only tracking where it is told the answer,
        and the ensemble covariance is not spreading information into the
        gaps. Observed points are marked along the axis, so this pattern
        is easy to spot.
    """
    N = len(reference_mean)
    xs = np.arange(N, dtype=float) if x_grid is None else np.asarray(x_grid)
    S = len(strategy_keys)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(xs, reference_mean, color="#37474F", linewidth=1.6,
            marker="o", markersize=2.2, label="Reference (unfiltered truth)", zorder=5)
    ax.fill_between(xs, reference_mean - reference_std, reference_mean + reference_std,
                    color="#37474F", alpha=0.15, linewidth=0, zorder=1)

    ol_palette = ["#B0BEC5", "#78909C", "#546E7A"]
    for i, prop_key in enumerate(sorted(open_loop_eqvar)):
        rec = open_loop_eqvar[prop_key]
        c = ol_palette[i % len(ol_palette)]
        ax.plot(xs, rec["mean"], color=c, linewidth=1.1, linestyle="--",
                marker="^", markersize=2.2, zorder=4,
                label=f"{prop_key} open-loop (static physics)")

    for key in strategy_keys:
        c = colors[key]
        ax.plot(xs, eqvar_mean_of[key], color=c, linewidth=1.1,
                marker="o", markersize=2.2, label=label_of[key], zorder=3)

    if obs_indices is not None and len(obs_indices) and len(obs_indices) < N:
        obs_idx = np.asarray(obs_indices, dtype=int)
        ymin = ax.get_ylim()[0]
        ax.scatter(xs[obs_idx], np.full(obs_idx.shape, ymin), marker="|", s=18,
                   color="#E53935", clip_on=False, zorder=6,
                   label="observed x")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Equilibrium variance" + ("  (log scale)" if log_scale else ""),
                  fontsize=11)
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
        dt_window = float(meta.attrs["dt_window"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])
        x_grid = meta["x_grid"][:]
        strategy_keys = _decode(meta["strategy_keys"][:])
        label_of = dict(zip(strategy_keys, _decode(meta["strategy_labels"][:])))
        t_fine_traj = f["trajectories/t_fine"][:]

        for ic_idx in range(num_ics_traj):
            ic_grp = f[f"trajectories/ic_{ic_idx}"]
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = obs_coords_raw if obs_coords_raw.size else None

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=sg["x_est"][:],
                    x_std=sg["x_std"][:], ic_idx=ic_idx,
                    strategy_label=label_of[key],
                    l2_time_avg=float(sg.attrs["l2_time_avg"]),
                    save_path=os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf"),
                    x_grid=x_grid, dt_window=dt_window, obs_coords=obs_coords,
                )
                n_pdfs += 1
    logging.info(
        f"plot_individual_trajectories: wrote {n_pdfs} PDF(s) from {h5_path} to {indiv_dir}"
    )
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
# fix the axes' shapes and the grid / ICs / noise / ensemble the strategies
# saw. `L` and `dt_data` are KS additions: the domain length and the
# reference solver's own step, which together decide what the truth even is.
_CONSISTENCY_ATTRS = (
    "N", "L", "dt_data", "dt_window", "dt_fine", "dt_obs", "sigma_obs",
    "P0_sigma", "P0_corr_len", "N_ens", "obs_every_n", "m", "num_ics_traj",
    "num_ics_batch", "trajectory_windows", "batch_windows",
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
                    N=int(attrs["N"]), L=float(attrs["L"]),
                    dt_window=float(attrs["dt_window"]),
                    obs_every_n=int(attrs["obs_every_n"]),
                    sigma_obs=float(attrs["sigma_obs"]),
                    N_ens=int(attrs["N_ens"]), B=int(attrs["B"]),
                    obs_times=batch["obs_times"][:], window_idx=batch["window_idx"][:],
                    t_dense_fine=batch["t_dense_fine"][:],
                    burn_in_frac=float(batch.attrs["eqvar_burn_in_frac"]),
                    x_grid=meta["x_grid"][:] if "x_grid" in meta else None,
                    obs_indices=meta["obs_indices"][:] if "obs_indices" in meta else None,
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
        equilibrium_variance.pdf   per-grid-point equilibrium variance vs reference

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
               f"(N_ens={N_ens}, obs every {obs_every_n}th grid point, σ_obs={sigma_obs})"),
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
        dt_window=g["dt_window"],
    )

    _plot_rmse_bulk(
        strategy_keys=keys, label_of=label_of, obs_times=g["obs_times"],
        prior_mean_of={k: batch_strat[k]["prior_rmse_mean"] for k in keys},
        post_mean_of={k: batch_strat[k]["post_rmse_mean"] for k in keys},
        sigma_obs=sigma_obs, n_traj=B, colors=colors,
        title=(f"EnKF prior vs posterior RMSE — {title_tag}\n"
               f"(N_ens={N_ens}, obs every {obs_every_n}th grid point, σ_obs={sigma_obs})"),
        save_path=os.path.join(save_dir, "rmse.pdf"),
    )

    _plot_equilibrium_variance_bulk(
        strategy_keys=keys, label_of=label_of,
        eqvar_mean_of=g["eqvar_mean_of"], eqvar_std_of=g["eqvar_std_of"],
        reference_mean=g["reference_mean"], reference_std=g["reference_std"],
        open_loop_eqvar=g["open_loop_eqvar"], colors=colors,
        title=(f"Equilibrium variance — static (open-loop) physics vs filtered "
               f"strategies — {title_tag}\n(B={B} trajectories, N_ens={N_ens}, obs every "
               f"{obs_every_n}th grid point, σ_obs={sigma_obs}, "
               f"burn-in={g['burn_in_frac']:.0%} of window discarded)"),
        save_path=os.path.join(save_dir, "equilibrium_variance.pdf"),
        x_grid=g["x_grid"], obs_indices=g["obs_indices"],
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


# ─────────────────────────────────────────────────────────────────────────
# Unfiltered (open-loop) surrogate evaluation
# ─────────────────────────────────────────────────────────────────────────
#
# This is the original KS `evaluate`: no data assimilation at all, just an
# autoregressive rollout of the surrogate against the stored reference
# trajectories. Keep running it first -- it answers "is the surrogate any
# good on its own?", which is the baseline every filtered curve in
# `evaluate_filters` is trying to beat, and it is far cheaper to iterate on.

def _plot_trajectory_summary(
    t_ax: np.ndarray,
    x_true: np.ndarray,
    x_est: np.ndarray,
    ic_idx: int,
    test_windows: int,
    pts_pw: int,
    save_path: str,
    N: int = 256,
    x_grid: np.ndarray | None = None,
) -> None:
    """
    Generate and save the trajectory-summary PDF for a single KS IC.
    Layout:
    - Row 1: 3 heatmaps (Pred, True, Diff)
    - Row 2: Relative L2 error over time
    - Rows 3+: Line plots at every window boundary (t=0, t=1, ...)
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    if x_grid is None:
        x_grid = np.arange(N, dtype=float)

    # 1. Compute Errors
    diff = x_est - x_true
    norm_err = np.linalg.norm(diff, axis=-1)
    norm_ref = np.linalg.norm(x_true, axis=-1)
    l2_rel = norm_err / (norm_ref + 1e-12)

    # 2. Grid Setup -- boundaries at index w * pts_pw for w in 0...test_windows
    num_boundaries = test_windows + 1
    cols_line_plots = 4
    rows_line_plots = int(np.ceil(num_boundaries / cols_line_plots))
    total_rows = 2 + rows_line_plots

    fig = plt.figure(figsize=(18, 5 + 3.5 + 2.5 * rows_line_plots))
    gs = gridspec.GridSpec(
        nrows=total_rows, ncols=12, figure=fig, hspace=0.6, wspace=0.6,
        height_ratios=[1.5, 0.8] + [1.0] * rows_line_plots,
    )

    # --- ROW 0: Heatmaps ---
    ax_heat_pred = fig.add_subplot(gs[0, 0:4])
    ax_heat_true = fig.add_subplot(gs[0, 4:8], sharey=ax_heat_pred)
    ax_heat_diff = fig.add_subplot(gs[0, 8:12], sharey=ax_heat_pred)

    extent = [float(x_grid[0]), float(x_grid[-1]), t_ax[0], t_ax[-1]]

    im_pred = ax_heat_pred.imshow(x_est, aspect='auto', extent=extent,
                                  cmap='viridis', origin='lower')
    ax_heat_pred.set_title("DeepONet Prediction", fontsize=12, fontweight='bold')
    ax_heat_pred.set_ylabel("Time (t)")
    ax_heat_pred.set_xlabel("x")
    fig.colorbar(im_pred, ax=ax_heat_pred, fraction=0.046, pad=0.04)

    im_true = ax_heat_true.imshow(x_true, aspect='auto', extent=extent,
                                  cmap='viridis', origin='lower')
    ax_heat_true.set_title("Reference Truth", fontsize=12, fontweight='bold')
    ax_heat_true.set_xlabel("x")
    ax_heat_true.tick_params(labelleft=False)
    fig.colorbar(im_true, ax=ax_heat_true, fraction=0.046, pad=0.04)

    vmax_diff = np.max(np.abs(diff)) or 1.0
    im_diff = ax_heat_diff.imshow(diff, aspect='auto', extent=extent, cmap='RdBu_r',
                                  vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
    ax_heat_diff.set_title("Absolute Difference", fontsize=12, fontweight='bold')
    ax_heat_diff.set_xlabel("x")
    ax_heat_diff.tick_params(labelleft=False)
    fig.colorbar(im_diff, ax=ax_heat_diff, fraction=0.046, pad=0.04)

    # --- ROW 1: Relative L2 Error ---
    ax_l2 = fig.add_subplot(gs[1, :])
    ax_l2.plot(t_ax, l2_rel, color="#E53935", linewidth=2.0, label="Relative L2 Error")
    for w in range(num_boundaries):
        wb = t_ax[w * pts_pw]
        ax_l2.axvline(x=wb, color="#78909C", linestyle="--", linewidth=0.8, alpha=0.5)

    ax_l2.set_title("Trajectory Relative L2 Error Over Time", fontsize=12, fontweight='bold')
    ax_l2.set_xlabel("Time (t)")
    ax_l2.set_ylabel("Error")
    ax_l2.set_yscale("log")
    ax_l2.grid(True, linestyle="--", alpha=0.6)
    ax_l2.legend()

    # --- ROWS 2+: Boundary Line Plots ---
    TRUTH_COLOR = "#37474F"
    EST_COLOR = "#1E88E5"

    for i in range(num_boundaries):
        row_idx = 2 + (i // cols_line_plots)
        col_start = (i % cols_line_plots) * 3
        ax = fig.add_subplot(gs[row_idx, col_start:col_start + 3])

        t_idx = i * pts_pw
        t_val = t_ax[t_idx]

        ax.plot(x_grid, x_true[t_idx], color=TRUTH_COLOR, linewidth=1.5, label="Truth")
        ax.plot(x_grid, x_est[t_idx], color=EST_COLOR, linewidth=1.5,
                linestyle="--", label="Pred")

        ax.set_title(f"Boundary t = {t_val:.1f}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        ax.set_xlabel("x", fontsize=8)

        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"KS Trajectory Summary — IC {ic_idx}", fontsize=16,
                 fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Trajectory summary for IC {ic_idx} saved to: {save_path}")


def _plot_batch_l2_over_time(
    t_ax: np.ndarray,
    overall_mean_l2: np.ndarray,
    save_path: str,
) -> None:
    """Plots the batch-average L2 error over time for the KS system."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_ax, overall_mean_l2, color="#1E88E5", linewidth=2.5,
            label="Overall Mean (All Trajectories)")
    ax.set_xlabel("Time (t)", fontsize=11)
    ax.set_ylabel("Mean Relative L2 Error", fontsize=11)
    ax.set_title("KS System: Overall Mean L2 Error Over Time", fontsize=13,
                 fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """
    Unfiltered autoregressive rollout of a single surrogate against the
    stored reference trajectories.

    `config.mode` selects the surrogate: "eval" loads the physics-informed
    `KSUDON`, "hybrid" loads `KSUDON_Hybrid`, anything else loads the
    data-driven `KSUDON_DD`. The checkpoint comes from `config.wandb.name`.
    """
    # ── 1. Load the dense test dataset ──────────────────────────────────
    # Same lookup as the filtered pipeline: config.eval.test_data_name.
    test_file = resolve_test_h5_path(config)

    logging.info(f"Loading test dataset from {test_file}...")
    max_ics = config.eval.get("num_ics", 100)
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:max_ics, :, :])
        N = int(f.attrs['N'])
        L_dom = float(f.attrs['L'])
        dt = float(f.attrs['dt'])
        test_windows = int(f.attrs['test_windows'])

    num_ics, num_test_pts, N_loaded = u_test.shape
    assert N_loaded == N, f"Expected state dimension {N}, got {N_loaded}"

    # Reconstruct time definitions: 1 time unit = 1 window.
    w_dt = float(config.get("dt_window", 1.0))
    pts_pw = check_divisible(w_dt, dt, "dt_window / dt")
    t_ax = np.arange(num_test_pts) * dt
    x_grid = np.arange(N) * (L_dom / N)

    # Single-window relative time grid required by the surrogate model
    t_star_window = t_ax[:pts_pw + 1]

    # ── 2. Set up model & load checkpoint ───────────────────────────────
    if config.mode == "eval":
        model = models.KSUDON(config, t_star_window, L=L_dom, N=N, dt=dt)
        kind = "pi"
    elif config.mode == "eval_hybrid":
        model = models.KSUDON_Hybrid(config, t_star_window, L=L_dom, N=N, dt=dt)
        kind = "hy"
    else:
        model = models.KSUDON_DD(config, t_star_window, N=N, L=L_dom)
        kind = "dd"

    ckpt_path = _resolve_ckpt(config.wandb.name, kind)

    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor. KS is autonomous, so the
    # branch input is just the current profile.
    predict_batch = jax.jit(jax.vmap(
        lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=0
    ))

    # ── 3. Batched autoregressive rollout ───────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")

    u_current_batch = u_test[:, 0, :]               # (num_ics, N)
    x_pred_list = []   # host-side (numpy) chunks -- keeps GPU memory O(1 window)

    rollout_windows = min(test_windows, (num_test_pts - 1) // pts_pw)
    for w in range(rollout_windows):
        pred_window = predict_batch(u_current_batch)    # (num_ics, pts_pw+1, N), on device

        # Advance the autoregressive state from the device array first...
        u_current_batch = pred_window[:, -1, :]

        # ...then copy this window to host immediately. This is the fix:
        # `pred_window` was previously appended to x_pred_list as a *device*
        # array. That list is a local variable, so it stays alive for the
        # rest of the function (Python doesn't free it just because later
        # code stops using it) -- by the time we reached the error stats we
        # were holding all `rollout_windows` per-window buffers AND the
        # concatenated x_pred_full AND the full u_test tensor on the GPU
        # simultaneously. That's what exhausted the allocator on what looked
        # like a trivial 18MB request: it's cumulative retention/fragmentation
        # from the whole rollout, not the norm op itself.
        chunk = np.asarray(pred_window if w == 0 else pred_window[:, 1:, :])
        x_pred_list.append(chunk)
        del pred_window, chunk

    x_pred_full = np.concatenate(x_pred_list, axis=1)   # host array
    del x_pred_list
    n_pts = x_pred_full.shape[1]

    u_ref = np.asarray(u_test[:, :n_pts, :])   # copy what's needed to host...
    del u_test                                  # ...then free the (large) device tensor
    t_ax = t_ax[:n_pts]

    # ── 4. Individual trajectory plots ──────────────────────────────────
    total_plots = config.saving.get("total_plots", 5)
    for ic_idx in range(min(total_plots, num_ics)):
        logging.info(f"--- Generating detailed summary for IC {ic_idx} ---")

        save_path = os.path.join(
            workdir, "figures", config.wandb.name, f"trajectory_summary_ic_{ic_idx}.pdf"
        )

        _plot_trajectory_summary(
            t_ax=t_ax,
            x_true=u_ref[ic_idx],
            x_est=x_pred_full[ic_idx],
            ic_idx=ic_idx,
            test_windows=rollout_windows,
            pts_pw=pts_pw,
            save_path=save_path,
            N=N,
            x_grid=x_grid,
        )

    # ── 5. Batch error analysis ─────────────────────────────────────────
    logging.info("--- Computing Batch L2 Error Statistics ---")

    err = x_pred_full - u_ref
    norm_err = np.linalg.norm(err, axis=-1)
    norm_ref = np.linalg.norm(u_ref, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    overall_mean_l2 = np.mean(l2_rel_per_traj_time, axis=0)

    batch_save_path = os.path.join(
        workdir, "figures", config.wandb.name, "batch_l2_error_analysis.pdf"
    )

    _plot_batch_l2_over_time(
        t_ax=t_ax,
        overall_mean_l2=overall_mean_l2,
        save_path=batch_save_path,
    )