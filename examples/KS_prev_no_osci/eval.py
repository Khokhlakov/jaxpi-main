"""
Modular filter evaluation for the DeepONet + EnKF Kuramoto-Sivashinsky pipeline.

KS translation of ``examples.l96_f.eval``. Same two-stage split:

    1. `evaluate_filters(...)`  -- runs every requested filtering strategy
       on the SAME data (same ICs, same noisy-observation draws, same
       initial ensembles) and stores every number a downstream plotting
       function could need into a single HDF5 file, keyed by
       `config.wandb.name`.

    2. `plot_comparisons` / `plot_comparisons_bulk` /
       `plot_equilibrium_variance` later read that HDF5 file and draw the
       figures: individual trajectories, RMSE with spread, calibration,
       Error Reduction Factor (ERF), batch time-mean L2 error, and the
       steady-state equilibrium-variance diagnostic.

`evaluate_filters` takes an arbitrary list of strategy specifications, so
new filters can be added or removed without touching the evaluation code.
`build_default_3way_strategies` reconstructs the classic DD-mult /
PI-mult / PI-RouteB set.

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
   by `steps_per_fine_exact`). `_ks_truth_batch` below advances the
   spectral solver on the GPU, batched over ICs, and returns real-space
   states on the filter's fine grid.

   The batch pass reuses the dense trajectories already stored in
   `ks_test_data.h5` (saved at every solver step by `gen_data.py`) rather
   than re-integrating -- same trajectories, read by stride.

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
   different scale from L96's. Run `run_route_b_inflation_sweep` before
   trusting any inherited default.

5. **Spatially correlated P0/Q0 are available.** White noise on a 256-point
   grid is dominated by the wavenumbers KS damps hardest, so a diagonal P0
   produces an ensemble that looks well-spread at t=0 and collapses almost
   at once. Set `config.kf.P0_corr_len` (physical units, i.e. the same
   units as `L`; roughly one KS cell, ~4-9 for L=64) to draw the initial
   ensemble from a smooth Gaussian-correlated covariance instead. The
   default is 0.0, which reproduces the L96 diagonal behaviour exactly.

HDF5 layout written by `evaluate_filters`
------------------------------------------
    /meta                                   (attrs only, plus datasets)
        N, L, dt_data                        -- grid size, domain length, solver dt
        dt_window, dt_fine, dt_obs
        sigma_obs, P0_sigma, P0_corr_len, N_ens, obs_every_n, m
        num_ics_traj, num_ics_batch, trajectory_windows, batch_windows
        obs_indices                          -- dataset, (m,) int, if static
        x_grid                               -- dataset, (N,) physical coordinates
        strategy_keys, strategy_labels        -- ordered, parallel string arrays
        strategy_propagator                   -- which propagator each strategy uses
        strategy_kind                         -- "standard" | "route_b" | "rtpp"

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
from absl import logging
import ml_collections
import jax
import jax.numpy as jnp
import numpy as np
import h5py

import itertools
import colorsys
import warnings

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

def _default_test_path(config) -> str:
    data_dir = config.training.get("data_dir", "data") if "training" in config else "data"
    return os.path.join(data_dir, "ks_test_data.h5")


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


# ─────────────────────────────────────────────────────────────────────────
# Default 3-way strategy set (DD-mult / PI-mult / PI-RouteB)
# ─────────────────────────────────────────────────────────────────────────

def build_default_3way_strategies(config, N_ens, alpha_fine, Q_fine,
                                  alpha_rb, beta_rb, n_quad_rb, grid):
    """
    Loads the PI and DD checkpoints named in `config.wandb.name_pi` /
    `config.wandb.name_dd` and builds the classic 3 strategies:

        dd_mult    -- data-driven propagator + multiplicative inflation
        pi_mult    -- physics-informed propagator + multiplicative inflation
        pi_route_b -- physics-informed propagator + Route B inflation

    Returns (strategies, propagators, N, t_star_window).
    """
    _, _, t_star_window = _window_grid(config, grid)

    model_pi, params_pi = _load_ks_model(config, "pi", t_star_window, grid)
    model_dd, params_dd = _load_ks_model(config, "dd", t_star_window, grid)
    N = model_pi.N
    assert model_dd.N == N, (
        f"DD checkpoint grid size ({model_dd.N}) != PI checkpoint grid size "
        f"({N}); can't share a strategy/propagator set across them."
    )

    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)
    predict_fn_pi, update_fn_pi = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)

    strategies = [
        dict(key="dd_mult", label="DD + Mult. Infl.", kind="standard",
             propagator="dd", predict_fn=predict_fn_dd, update_fn=update_fn_dd,
             alpha_fine=alpha_fine),
        dict(key="pi_mult", label="PI + Mult. Infl.", kind="standard",
             propagator="pi", predict_fn=predict_fn_pi, update_fn=update_fn_pi,
             alpha_fine=alpha_fine),
        dict(key="pi_route_b", label="PI + Route B Infl.", kind="route_b",
             propagator="pi", predict_fn=predict_fn_rb, update_fn=update_fn_rb,
             Q0=Q_fine, alpha=alpha_rb, beta=beta_rb, n_quad=n_quad_rb),
    ]
    propagators = {
        "dd": (model_dd, params_dd),
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window


# ─────────────────────────────────────────────────────────────────────────
# Default 4-way strategy set, crossed with the propagators: for each
# propagator in `propagator_kinds`, the same 4 inflation schemes are run,
# so differences ACROSS inflation schemes (at fixed propagator) isolate
# the inflation choice, and differences ACROSS propagators (at fixed
# inflation scheme) isolate the surrogate choice (unlike the 3-way set,
# which conflates the two by swapping both DD vs PI and mult vs Route B
# at once).
# ─────────────────────────────────────────────────────────────────────────

def build_default_4way_strategies(
    config, N_ens, alpha_fine, alpha_rb, beta_rb, n_quad_rb, alpha_rtpp,
    grid, alpha_fine_rtpp: float = 1.0, propagator_kinds=("pi", "hy"),
):
    """
    Builds the 4-way inflation-strategy comparison set, crossed with each
    propagator in `propagator_kinds` (default: the physics-informed "pi"
    checkpoint named in `config.wandb.name_pi` and the hybrid "hy"
    checkpoint named in `config.wandb.name_hy`; "dd" has no PDE residual
    so it can't run Route B / additive inflation and is not a valid entry
    here):

        1. <kind>_mult     -- standard multiplicative inflation (`make_enkf`).
        2. <kind>_route_b  -- Route B: residual-scaled additive inflation,
                              scale = alpha_rb + beta_rb * ||rho||^2
                              (`make_route_b_enkf`).
        3. <kind>_additive -- plain additive inflation, obtained from the
                              SAME Route B machinery with the flow-dependent
                              term zeroed out (beta=0.0), leaving only the
                              constant alpha_rb * Q0 floor.
        4. <kind>_rtpp     -- Relaxation-to-Prior Perturbations
                              (`make_rtpp_enkf`). `alpha_fine_rtpp` controls
                              the (optional, shared) multiplicative
                              inflation in RTPP's predict step; it defaults
                              to 1.0 so `alpha_rtpp` (the relaxation factor,
                              in [0, 1]) is the only active inflation
                              mechanism for this strategy.

    With the default two-propagator set this produces an 8-way comparison
    (4 inflation schemes x {PI, Hybrid}); pass `propagator_kinds=("pi",)`
    to recover the original PI-only 4-way comparison.

    Note that (2) and (3), for the SAME propagator, differ ONLY in beta,
    so a flat gap between them is the cleanest available read on whether
    Route B's physics-driven term is doing anything for KS at the beta you
    passed in.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window, _, t_star_window = _window_grid(config, grid)

    loaded = {}
    N = None
    for kind in propagator_kinds:
        if kind == "dd":
            raise ValueError(
                "'dd' has no PDE residual (r_net), so it cannot run Route "
                "B / additive inflation and is not eligible for the 4-way "
                "comparison. Use 'pi' and/or 'hy' in propagator_kinds."
            )
        model, params = _load_ks_model(config, kind, t_star_window, grid)
        loaded[kind] = (model, params)
        if N is None:
            N = model.N
        assert model.N == N, (
            f"'{kind}' checkpoint grid size ({model.N}) != {N}; can't share a "
            "strategy/propagator set across them."
        )

    # Route B / additive both need Q_fine.
    P0_sigma = config.kf.get("P0_sigma", 0.5)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    Q0_corr_len = config.kf.get("Q0_corr_len", config.kf.get("P0_corr_len", 0.0))
    DT_FINE = float(config.kf.get("dt_fine", dt_window))
    steps_per_window = steps_per_window_exact(dt_window, DT_FINE)
    Q_coarse = build_cov(N, grid["L"], Q0_sigma, Q0_corr_len)
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    label_of_kind = {"pi": "PI", "hy": "Hybrid"}

    strategies = []
    for kind in propagator_kinds:
        model, params = loaded[kind]
        label = label_of_kind.get(kind, kind.upper())

        predict_fn_mult, update_fn_mult = model.make_enkf_fns(params, N_ens=N_ens)
        predict_fn_rb, update_fn_rb = model.make_route_b_enkf_fns(params, N_ens=N_ens)
        predict_fn_rtpp, update_fn_rtpp = model.make_rtpp_enkf_fns(params, N_ens=N_ens)

        strategies.extend([
            dict(key=f"{kind}_mult", label=f"{label} + Mult. Infl.", kind="standard",
                 propagator=kind, predict_fn=predict_fn_mult, update_fn=update_fn_mult,
                 alpha_fine=alpha_fine),
            dict(key=f"{kind}_route_b", label=f"{label} + Route B Infl.", kind="route_b",
                 propagator=kind, predict_fn=predict_fn_rb, update_fn=update_fn_rb,
                 Q0=Q_fine, alpha=alpha_rb, beta=beta_rb, n_quad=n_quad_rb),
            dict(key=f"{kind}_additive", label=f"{label} + Additive Infl.", kind="route_b",
                 propagator=kind, predict_fn=predict_fn_rb, update_fn=update_fn_rb,
                 Q0=Q_fine, alpha=alpha_rb, beta=0.0, n_quad=n_quad_rb),
            dict(key=f"{kind}_rtpp", label=f"{label} + RTPP", kind="rtpp",
                 propagator=kind, predict_fn=predict_fn_rtpp, update_fn=update_fn_rtpp,
                 alpha_fine=alpha_fine_rtpp, alpha_rtpp=alpha_rtpp),
        ])

    return strategies, loaded, N, t_star_window


# ─────────────────────────────────────────────────────────────────────────
# Main entry point -- filtered evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate_filters(
    config: ml_collections.ConfigDict,
    workdir: str,
    strategies: list[dict] | None = None,
    propagators: dict | None = None,
    t_star_window=None,
    test_h5_path: str | None = None,
) -> str:
    """
    Runs every strategy on shared data and writes one HDF5 results file.

    Memory-chunked over ICs and (optionally) over strategy groups; see
    `config.eval.ic_chunk`, `config.eval.strategy_chunk` and
    `config.eval.device_budget_gb`. KS dense outputs are 6.4x larger per
    time step than L96's, so the automatic chunk sizing matters more here
    -- if you hit OOM, lower `device_budget_gb` first.
    """
    # ── EnKF / observation configuration ───────────────────────────────
    obs_every_n = config.kf.get("obs_every_n", 4)
    sigma_obs = config.kf.get("sigma_obs", 0.1)
    P0_sigma = config.kf.get("P0_sigma", 0.5)
    P0_corr_len = float(config.kf.get("P0_corr_len", 0.0))
    dynamic_vars = config.kf.get("dynamic_vars", False)
    N_ens = config.kf.get("N_ens", 50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)

    alpha_rb = config.kf.get("route_b_alpha", 1.0)
    beta_rb = config.kf.get("route_b_beta", 1.0)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    Q0_corr_len = float(config.kf.get("Q0_corr_len", P0_corr_len))
    n_quad_rb = config.kf.get("route_b_n_quad", 3)

    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list = config.kf.get("obs_idx_list", None)

    n_devices = jax.local_device_count()
    strategy_chunk = int(config.eval.get("strategy_chunk", 0))
    ic_chunk_cfg = int(config.eval.get("ic_chunk", 0))
    budget_bytes = float(config.eval.get("device_budget_gb", 1.5)) * (1024 ** 3)

    # ── 1. Dataset metadata and time grids ─────────────────────────────
    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)
    N_grid, L_dom, dt_data = grid["N"], grid["L"], grid["dt"]

    DT_WINDOW, dt_integration, t_star_default = _window_grid(config, grid)
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    DT_OBS = float(config.kf.get("dt_obs", DT_WINDOW))

    # The reference solver can only be sampled on multiples of its own dt.
    steps_per_fine = steps_per_fine_exact(DT_FINE, dt_data)
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)

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
    P0 = build_cov(N_grid, L_dom, P0_sigma, P0_corr_len)
    Q_coarse = build_cov(N_grid, L_dom, Q0_sigma, Q0_corr_len)
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    if strategies is None or propagators is None:
        strategies, propagators, N, t_star_window = build_default_3way_strategies(
            config, N_ens, alpha_fine, Q_fine, alpha_rb, beta_rb, n_quad_rb, grid,
        )
    else:
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
    out_path = os.path.join(workdir, f"{config.wandb.name}.h5")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with h5py.File(out_path, "w") as f:
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

    logging.info(f"evaluate_filters: wrote all evaluation data to {out_path}")
    return out_path


"""
Plotting stage for the modular DeepONet + EnKF Kuramoto-Sivashinsky
evaluation pipeline.

Reads the HDF5 file written by `evaluate_filters` -- keyed only by strategy
`key`/`label`, so it works for an arbitrary number of strategies -- and
writes:

    1. Individual-trajectory PDFs
       One PDF per (IC, strategy): space-time heatmaps of truth, estimate
       and difference; error/spread time series; a few fixed-probe time
       series (observed grid points vs gaps); and profile snapshots with
       the +/-1 sigma ensemble band and assimilated observations.

       This is where the KS version departs most from L96's. L96 draws one
       time-series panel per state variable, which is both readable (40
       variables) and meaningful (the variables are separate dynamical
       quantities). At N=256 grid points that layout would be 128 rows of
       near-identical curves, and it would also hide the thing that
       actually matters for a field: whether the analysis is good
       *between* the observed points. Heatmaps + snapshots show that
       directly.

    2. Pairwise batch-comparison PDFs (`plot_comparisons`)
       For every unordered pair of strategies (A, B) -- C(S, 2) pairs --
       four PDFs: calibration, ERF, EnKF-vs-open-loop L2, and prior/
       posterior RMSE.

    3. Bulk batch-comparison PDFs (`plot_comparisons_bulk`)
       Exactly four PDFs total, each overlaying every strategy -- the
       right choice for the inflation sweeps, where S gets large.

    4. Equilibrium-variance PDF (`plot_equilibrium_variance`)
       Per-grid-point steady-state variance of truth, each propagator's
       open-loop rollout, and every filtered strategy.
"""


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
    t_ax: np.ndarray,          # (T,)   time axis
    x_true: np.ndarray,        # (T, N) ground-truth field
    x_est: np.ndarray,         # (T, N) strategy's EnKF mean
    x_std: np.ndarray | None,  # (T, N) strategy's ensemble std, or None
    ic_idx: int,
    strategy_label: str,
    l2_time_avg: float | None,
    save_path: str,
    x_grid: np.ndarray | None = None,
    dt_window: float | None = None,
    obs_coords=None,           # iterable of (grid_idx, t_obs, y_obs)
    n_probes: int = 6,
    n_snapshots: int = 8,
) -> None:
    """
    Trajectory-summary PDF for ONE strategy on ONE IC.

    Layout
    ------
      row 0            truth / estimate / difference space-time heatmaps
      row 1            left: relative L2 and mean |error| vs time;
                       right: RMSE vs RMS ensemble spread vs time
      probe rows       `n_probes` single-grid-point time series (observed
                       points and gaps, see `_pick_probe_points`) with the
                       +/-1 sigma band and the assimilated observations
      snapshot rows    `n_snapshots` spatial profiles (truth vs estimate,
                       +/-1 sigma band) at evenly spaced times, with the
                       observations assimilated at that time overlaid
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

    # ── Window-boundary times ──────────────────────────────────────────
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

    # Snapshot times, deduplicated: short runs (few fine steps) would
    # otherwise repeat the same index several times and draw identical
    # panels.
    snap_idx = np.unique(np.linspace(0, T - 1, min(n_snapshots, T)).astype(int))
    n_snapshots = len(snap_idx)

    # ── Figure & GridSpec ────────────────────────────────────────────────
    probe_cols, snap_cols = 2, 4
    probe_rows = int(np.ceil(n_probes / probe_cols))
    snap_rows = int(np.ceil(n_snapshots / snap_cols))

    heat_h, metric_h, probe_h, snap_h = 4.0, 2.6, 1.9, 2.0
    fig_h = heat_h + metric_h + probe_rows * probe_h + snap_rows * snap_h + 1.0
    fig = plt.figure(figsize=(16, fig_h))
    gs = gridspec.GridSpec(
        nrows=2 + probe_rows + snap_rows, ncols=12, figure=fig,
        height_ratios=[heat_h, metric_h] + [probe_h] * probe_rows + [snap_h] * snap_rows,
        hspace=0.75, wspace=0.55,
    )

    TRUTH_COLOR = "#37474F"
    EST_COLOR, EST_BAND = "#2196F3", "#90CAF9"
    OBS_COLOR = "#E53935"

    # ── Row 0: space-time heatmaps ───────────────────────────────────────
    extent = [float(x_grid[0]), float(x_grid[-1]), t_min, t_max]
    vmax = float(np.max(np.abs(x_true)))
    ax_t = fig.add_subplot(gs[0, 0:4])
    ax_e = fig.add_subplot(gs[0, 4:8], sharey=ax_t)
    ax_d = fig.add_subplot(gs[0, 8:12], sharey=ax_t)

    im_t = ax_t.imshow(x_true, aspect="auto", extent=extent, origin="lower",
                       cmap="viridis", vmin=-vmax, vmax=vmax)
    ax_t.set_title("Reference truth", fontsize=11, fontweight="bold")
    ax_t.set_ylabel("Time  t", fontsize=10)
    ax_t.set_xlabel("x", fontsize=10)
    fig.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

    im_e = ax_e.imshow(x_est, aspect="auto", extent=extent, origin="lower",
                       cmap="viridis", vmin=-vmax, vmax=vmax)
    ax_e.set_title(f"{strategy_label}: ensemble mean", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("x", fontsize=10)
    ax_e.tick_params(labelleft=False)
    fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)

    dmax = float(np.max(np.abs(diff))) or 1.0
    im_d = ax_d.imshow(diff, aspect="auto", extent=extent, origin="lower",
                       cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    ax_d.set_title("Estimate − truth", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("x", fontsize=10)
    ax_d.tick_params(labelleft=False)
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

    # Mark the observed grid points along the bottom of the truth panel --
    # with a static network this immediately shows which stripes of the
    # difference panel are "free" and which the filter had to infer.
    if obs_points and len(obs_points) < N:
        ax_d.scatter(x_grid[np.asarray(obs_points)],
                     np.full(len(obs_points), t_min), marker="|", s=18,
                     color=OBS_COLOR, clip_on=False, zorder=6,
                     label="observed x")
        ax_d.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # ── Row 1: error and spread time series ──────────────────────────────
    ax_err = fig.add_subplot(gs[1, 0:6])
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

    ax_sp = fig.add_subplot(gs[1, 6:12])
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

    # ── Probe rows: single-grid-point time series ────────────────────────
    for i, p in enumerate(probes):
        row = 2 + i // probe_cols
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
        if row == 2 + probe_rows - 1:
            ax.set_xlabel("t", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper right", handlelength=1.2,
                      framealpha=0.7, ncol=2)

    # ── Snapshot rows: spatial profiles ──────────────────────────────────
    dt_fine = float(t_ax[1] - t_ax[0]) if T > 1 else 0.0
    for i, ti in enumerate(snap_idx):
        row = 2 + probe_rows + i // snap_cols
        col0 = (i % snap_cols) * 3
        ax = fig.add_subplot(gs[row, col0:col0 + 3])

        ax.plot(x_grid, x_true[ti], color=TRUTH_COLOR, linewidth=1.2, label="Truth")
        ax.plot(x_grid, x_est[ti], color=EST_COLOR, linewidth=1.2, linestyle="--",
                label="Est.")
        if x_std is not None:
            ax.fill_between(x_grid, x_est[ti] - x_std[ti], x_est[ti] + x_std[ti],
                            color=EST_BAND, alpha=0.30, linewidth=0, label="±1σ")

        # Observations assimilated at (or within half a fine step of) this time.
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
    _save(fig, save_path, dpi=140)
    logging.info(
        f"Individual trajectory plot (IC {ic_idx}, {strategy_label}) saved to: {save_path}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2a. EnKF vs open-loop, time-mean relative L2 (curve-count-agnostic;
#     shared by the pairwise and bulk entry points)
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

      page 1 -- every curve in `curves`.
      page 2 -- the same plot with the pure-propagator open-loop curves
        (any label ending in "open-loop") omitted, so the filtered-strategy
        curves aren't dwarfed by the open-loop curves' much larger error
        scale. For KS this page is close to mandatory: an unfiltered
        surrogate rollout saturates at relative L2 ~ 1 within a couple of
        Lyapunov times, several orders above a working filter.
      pages 3-4 -- duplicates of pages 1 and 2 restricted to the (at most)
        `TOP_K_BEST` filtered strategies with the LOWEST time-mean relative
        L2, with the open-loop references kept on the first of the two for
        scale. Both are skipped when there are no more than `TOP_K_BEST`
        filtered curves to begin with, since the pages would just repeat
        pages 1-2.
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
# 2b. Calibration (pairwise): spread-vs-RMSE timeseries + binned scatter
# ─────────────────────────────────────────────────────────────────────────
def _plot_calibration_pair(
    window_idx: np.ndarray,
    dt_window: float,
    spread_a: np.ndarray, rmse_a: np.ndarray,
    spread_b: np.ndarray, rmse_b: np.ndarray,
    spread_a_raw: np.ndarray, rmse_a_raw: np.ndarray,
    spread_b_raw: np.ndarray, rmse_b_raw: np.ndarray,
    label_a: str, label_b: str,
    title: str,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Two-panel calibration PDF for one strategy pair:
      1. RMS ensemble spread vs EnKF RMSE (simulation time), both
         strategies overlaid on one graph.
      2. Binned spread-skill scatter for both strategies, pooled over
         every (IC, window) in the batch, on one graph.
    """
    fig = plt.figure(figsize=(9, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.5)

    # -- Panel 1: simulation-time spread/RMSE timeseries, both strategies --
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.plot(window_idx, spread_a, marker="^", markersize=2.5, linewidth=1.1,
               linestyle="-", color="#8BC34A", label=f"{label_a} RMS ensemble σ")
    ax_ts.plot(window_idx, rmse_a, marker="s", markersize=2.5, linewidth=1.1,
               linestyle="--", color="#FF8A65", label=f"{label_a} EnKF RMSE")
    ax_ts.plot(window_idx, spread_b, marker="^", markersize=2.5, linewidth=1.1,
               linestyle="-", color="#4CAF50", label=f"{label_b} RMS ensemble σ")
    ax_ts.plot(window_idx, rmse_b, marker="s", markersize=2.5, linewidth=1.1,
               linestyle="--", color="#EC407A", label=f"{label_b} EnKF RMSE")
    ax_ts.set_yscale("log")
    ax_ts.set_xlabel("Window index", fontsize=11)
    ax_ts.set_ylabel("Log scale", fontsize=11)
    ax_ts.set_title(f"Ensemble spread vs RMSE (simulation time) — {label_a} vs {label_b}",
                    fontsize=12)
    ax_ts.legend(fontsize=8.5, ncol=2)
    ax_ts.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_time = ax_ts.twiny()
    ax_time.set_xlim(ax_ts.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels([f"{k * dt_window:.3g}" for k in window_idx],
                            fontsize=7, rotation=45, ha="left")
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=9)

    # -- Panel 2: binned spread-skill scatter, both strategies -----------
    ax_bin = fig.add_subplot(gs[1])
    rmss_a_b, rmse_a_b, rmse_a_s, _ = _binned_spread_skill(spread_a_raw, rmse_a_raw, n_bins)
    rmss_b_b, rmse_b_b, rmse_b_s, _ = _binned_spread_skill(spread_b_raw, rmse_b_raw, n_bins)

    lim_hi = 1.1 * max(rmss_a_b.max(), rmse_a_b.max(), rmss_b_b.max(), rmse_b_b.max())
    ax_bin.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.0,
                color="#37474F", label="1:1 (perfect calibration)")
    ax_bin.errorbar(rmss_a_b, rmse_a_b, yerr=rmse_a_s, fmt="o", markersize=3.5,
                    capsize=2, linewidth=1.0, color="#FF8C00",
                    label=f"{label_a} ({n_bins}-bin)")
    ax_bin.errorbar(rmss_b_b, rmse_b_b, yerr=rmse_b_s, fmt="o", markersize=3.5,
                    capsize=2, linewidth=1.0, color="#2196F3",
                    label=f"{label_b} ({n_bins}-bin)")

    ax_bin.set_xlim(0, lim_hi)
    ax_bin.set_ylim(0, lim_hi)
    ax_bin.set_xlabel("RMS ensemble spread (RMSS)", fontsize=11)
    ax_bin.set_ylabel("RMSE of ensemble mean", fontsize=11)
    ax_bin.set_title(
        f"Binned spread-skill  ({n_bins} equal-population bins, pooled over all "
        f"ICs × windows) — {label_a} vs {label_b}", fontsize=12,
    )
    ax_bin.legend(fontsize=9)
    ax_bin.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bin.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=13, y=0.997)
    _tight_layout(fig, rect=[0, 0, 1, 0.98])
    _save(fig, save_path)
    logging.info(f"Calibration comparison plot ({label_a} vs {label_b}) saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# 2c. Error Reduction Factor (pairwise, one graph)
# ─────────────────────────────────────────────────────────────────────────
def _plot_erf_pair(
    obs_times: np.ndarray,
    erf_mean_a: np.ndarray, erf_std_a: np.ndarray,
    erf_mean_b: np.ndarray, erf_std_b: np.ndarray,
    label_a: str, label_b: str,
    n_traj: int,
    title: str,
    save_path: str,
) -> None:
    """ERF comparison for a strategy pair on ONE set of axes: 2 lines +
    2 light +/-1 sigma bands, plus the ERF=1 reference line."""
    fig, ax = plt.subplots(figsize=(9, 5))

    series = [
        (label_a, erf_mean_a, erf_std_a, "#FF8C00", "o"),
        (label_b, erf_mean_b, erf_std_b, "#2196F3", "s"),
    ]
    for label, mean, std, color, marker in series:
        ax.plot(obs_times, mean, color=color, linewidth=1.2, marker=marker,
                markersize=2.5, label=f"{label}  (n = {n_traj} trajectories)")
        ax.fill_between(obs_times, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0)

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.1,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _save(fig, save_path)
    logging.info(f"ERF comparison plot ({label_a} vs {label_b}) saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# 2d. Prior vs posterior RMSE (pairwise, one graph, no spread bands)
# ─────────────────────────────────────────────────────────────────────────
def _plot_rmse_pair(
    obs_times: np.ndarray,
    prior_mean_a: np.ndarray, post_mean_a: np.ndarray,
    prior_mean_b: np.ndarray, post_mean_b: np.ndarray,
    sigma_obs: float,
    n_traj: int,
    label_a: str, label_b: str,
    title: str,
    save_path: str,
) -> None:
    """Prior/posterior RMSE for a strategy pair on ONE graph (4 lines,
    no spread bands), plus the sigma_obs measurement-noise reference line."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(obs_times, prior_mean_a, color="#0A36C7", linewidth=1.2, marker="o",
            markersize=2.5, linestyle="-", label=f"{label_a} prior RMSE  (n = {n_traj})")
    ax.plot(obs_times, post_mean_a, color="#A30005", linewidth=1.2, marker="s",
            markersize=2.5, linestyle="-", label=f"{label_a} posterior RMSE  (n = {n_traj})")
    ax.plot(obs_times, prior_mean_b, color="#8E24AA", linewidth=1.2, marker="o",
            markersize=2.5, linestyle="--", label=f"{label_b} prior RMSE  (n = {n_traj})")
    ax.plot(obs_times, post_mean_b, color="#EC407A", linewidth=1.2, marker="s",
            markersize=2.5, linestyle="--", label=f"{label_b} posterior RMSE  (n = {n_traj})")

    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.2,
               label=f"Measurement noise  σ_obs = {sigma_obs}")

    ax.set_yscale("log")
    ax.set_xlabel("Observation time  t", fontsize=11)
    ax.set_ylabel("RMSE  (log scale)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _save(fig, save_path)
    logging.info(f"Prior/posterior RMSE comparison plot ({label_a} vs {label_b}) saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Entry point -- pairwise comparisons
# ─────────────────────────────────────────────────────────────────────────
def plot_comparisons(h5_path: str, workdir: str | None = None, n_bins: int = 10) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and generates:

      * one individual-trajectory PDF per (IC, strategy) -- S * P PDFs
        for S strategies and P trajectory ICs,
      * four pairwise batch-comparison PDFs per strategy pair -- calibration
        (2 graphs), ERF (1 graph), EnKF-vs-open-loop L2 (multi-page), and
        prior/posterior RMSE (1 graph) -- 4 * C(S, 2) PDFs total.

    Output layout (under ``workdir/figures/comparisons/``):

        individual_trajectories/trajectory_ic_<i>_<strategy_key>.pdf
        calibration_<keyA>_vs_<keyB>.pdf
        erf_<keyA>_vs_<keyB>.pdf
        l2_<keyA>_vs_<keyB>.pdf
        rmse_<keyA>_vs_<keyB>.pdf

    Use `plot_comparisons_bulk` instead for the inflation sweeps: the
    pairwise count grows as C(S, 2).

    Returns the path of the ``figures/comparisons`` directory written.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(h5_path))

    save_dir = os.path.join(workdir, "figures", "comparisons")
    indiv_dir = os.path.join(save_dir, "individual_trajectories")
    os.makedirs(indiv_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        dt_window = float(meta.attrs["dt_window"])
        obs_every_n = int(meta.attrs["obs_every_n"])
        sigma_obs = float(meta.attrs["sigma_obs"])
        N_ens = int(meta.attrs["N_ens"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])
        x_grid = meta["x_grid"][:]

        strategy_keys = _decode(meta["strategy_keys"][:])
        strategy_labels = _decode(meta["strategy_labels"][:])
        strategy_propagator = _decode(meta["strategy_propagator"][:])
        label_of = dict(zip(strategy_keys, strategy_labels))
        propagator_of = dict(zip(strategy_keys, strategy_propagator))

        if len(strategy_keys) < 2:
            logging.warning(
                "plot_comparisons: fewer than 2 strategies found in "
                f"{h5_path}; individual-trajectory plots will still be "
                "written, but no pairwise comparison PDFs will be generated."
            )

        t_fine_traj = f["trajectories/t_fine"][:]

        # ── 1. Individual-trajectory PDFs: one per (IC, strategy) ───────
        n_traj_pdfs = 0
        for ic_idx in range(num_ics_traj):
            ic_grp = f[f"trajectories/ic_{ic_idx}"]
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = obs_coords_raw if obs_coords_raw.size else None

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                x_est = sg["x_est"][:]
                x_std = sg["x_std"][:]
                l2_time_avg = float(sg.attrs["l2_time_avg"])

                save_path = os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf")
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=x_est, x_std=x_std,
                    ic_idx=ic_idx, strategy_label=label_of[key],
                    l2_time_avg=l2_time_avg, save_path=save_path,
                    x_grid=x_grid, dt_window=dt_window, obs_coords=obs_coords,
                )
                n_traj_pdfs += 1

        logging.info(
            f"plot_comparisons: wrote {n_traj_pdfs} individual trajectory "
            f"PDFs ({num_ics_traj} ICs × {len(strategy_keys)} strategies) to {indiv_dir}"
        )

        # ── 2. Batch data needed for pairwise comparisons ────────────────
        batch = f["batch"]
        B = int(batch.attrs["B"])
        obs_times_batch = batch["obs_times"][:]
        window_idx = batch["window_idx"][:]
        t_dense_fine = batch["t_dense_fine"][:]

        batch_fields = (
            "prior_rmse_mean", "post_rmse_mean", "erf_mean", "erf_std",
            "rmse_window_mean", "spread_window_mean", "rmse_raw", "spread_raw",
            "l2_dense_mean",
        )
        batch_strat = {}
        for key in strategy_keys:
            sg = batch[f"strategies/{key}"]
            batch_strat[key] = {field: sg[field][:] for field in batch_fields}

        open_loop = {}
        if "open_loop" in batch:
            for prop_key in batch["open_loop"]:
                og = batch[f"open_loop/{prop_key}"]
                open_loop[prop_key] = dict(t=og["t"][:], l2_dense_mean=og["l2_dense_mean"][:])

    # ── 3. Pairwise batch-comparison PDFs ────────────────────────────────
    strategy_pairs = list(itertools.combinations(strategy_keys, 2))
    n_comparison_pdfs = 0

    for key_a, key_b in strategy_pairs:
        label_a, label_b = label_of[key_a], label_of[key_b]
        pair_name = f"{key_a}_vs_{key_b}"
        rec_a, rec_b = batch_strat[key_a], batch_strat[key_b]

        # -- Calibration (2 graphs) --------------------------------------
        _plot_calibration_pair(
            window_idx=window_idx, dt_window=dt_window,
            spread_a=rec_a["spread_window_mean"], rmse_a=rec_a["rmse_window_mean"],
            spread_b=rec_b["spread_window_mean"], rmse_b=rec_b["rmse_window_mean"],
            spread_a_raw=rec_a["spread_raw"], rmse_a_raw=rec_a["rmse_raw"],
            spread_b_raw=rec_b["spread_raw"], rmse_b_raw=rec_b["rmse_raw"],
            label_a=label_a, label_b=label_b,
            title=(
                f"Calibration: ensemble spread vs RMSE — {label_a} vs {label_b}\n"
                f"(B={B} trajectories, N_ens={N_ens})"
            ),
            save_path=os.path.join(save_dir, f"calibration_{pair_name}.pdf"),
            n_bins=n_bins,
        )

        # -- ERF (1 graph) --------------------------------------------------
        _plot_erf_pair(
            obs_times=obs_times_batch,
            erf_mean_a=rec_a["erf_mean"], erf_std_a=rec_a["erf_std"],
            erf_mean_b=rec_b["erf_mean"], erf_std_b=rec_b["erf_std"],
            label_a=label_a, label_b=label_b, n_traj=B,
            title=(
                f"EnKF Error Reduction Factor per observation time — {label_a} vs {label_b}\n"
                f"(B={B} trajectories, N_ens={N_ens}, obs every {obs_every_n}th grid "
                f"point, σ_obs={sigma_obs})"
            ),
            save_path=os.path.join(save_dir, f"erf_{pair_name}.pdf"),
        )

        # -- EnKF vs open-loop, time-mean relative L2 -----------------------
        curves, colors = {}, {}
        used_props = {propagator_of[key_a], propagator_of[key_b]}
        ol_palette = ["#B0BEC5", "#78909C"]
        for i, prop_key in enumerate(sorted(used_props)):
            if prop_key in open_loop:
                lbl = f"{prop_key} open-loop"
                curves[lbl] = (open_loop[prop_key]["t"], open_loop[prop_key]["l2_dense_mean"])
                colors[lbl] = ol_palette[i % len(ol_palette)]
        curves[label_a] = (t_dense_fine, rec_a["l2_dense_mean"])
        curves[label_b] = (t_dense_fine, rec_b["l2_dense_mean"])
        colors[label_a] = "#FF8C00"
        colors[label_b] = "#2196F3"

        _plot_l2_per_timestep(
            curves=curves,
            title=(
                f"EnKF vs open-loop: mean relative L2 per timestep — "
                f"{label_a} vs {label_b}  (B={B})"
            ),
            save_path=os.path.join(save_dir, f"l2_{pair_name}.pdf"),
            colors=colors,
        )

        # -- Prior vs posterior RMSE, no spread bands (1 graph) -----------
        _plot_rmse_pair(
            obs_times=obs_times_batch,
            prior_mean_a=rec_a["prior_rmse_mean"], post_mean_a=rec_a["post_rmse_mean"],
            prior_mean_b=rec_b["prior_rmse_mean"], post_mean_b=rec_b["post_rmse_mean"],
            sigma_obs=sigma_obs, n_traj=B, label_a=label_a, label_b=label_b,
            title=(
                f"EnKF prior vs posterior RMSE — {label_a} vs {label_b}\n"
                f"(B={B} trajectories, N_ens={N_ens}, obs every {obs_every_n}th grid "
                f"point, σ_obs={sigma_obs})"
            ),
            save_path=os.path.join(save_dir, f"rmse_{pair_name}.pdf"),
        )

        n_comparison_pdfs += 4

    logging.info(
        f"plot_comparisons: wrote {n_comparison_pdfs} pairwise comparison "
        f"PDFs ({len(strategy_pairs)} pairs × 4 categories) to {save_dir}"
    )
    return save_dir


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
        S is large.
      * a final row with the pooled binned spread-skill scatter, shown
        in both linear and log/log axes, side by side.
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
    _tight_layout(fig, rect=[0.04, 0, 1, 1 - 0.85 / fig_h])
    _save(fig, save_path)
    logging.info(
        f"Bulk calibration plot ({S} strategies, stacked small-multiples) saved to: {save_path}"
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
    show_bands: bool | None = None,
) -> None:
    """
    ERF comparison for every strategy on ONE set of axes, written as a
    multi-page PDF:

      page 1 -- every strategy.
      page 2 -- the same axes restricted to the (at most) `TOP_K_BEST`
        strategies with the GREATEST time-mean ERF. Skipped when there are
        no more than `TOP_K_BEST` strategies.

    +/-1 sigma bands are auto-dropped once there are more than 4
    strategies, since overlapping fills stop conveying anything once they
    stack that deep -- evaluated per page, so the decluttered second page
    usually gets its bands back.
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
    """
    Prior/posterior RMSE for every strategy, split into two side-by-side
    panels (prior, posterior) rather than 2*S lines on one axes; both
    panels share a y-axis and a single strategy-color legend.

    Written as a multi-page PDF:

      page 1 -- every strategy.
      page 2 -- the same two panels restricted to the (at most)
        `TOP_K_BEST` strategies with the LOWEST time-mean POSTERIOR RMSE.
        Both panels are filtered by that single posterior ranking, so the
        prior panel still shows where those same strategies started from.
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
# Entry point -- bulk comparisons
# ─────────────────────────────────────────────────────────────────────────
def plot_comparisons_bulk(h5_path: str, workdir: str | None = None, n_bins: int = 10) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and generates:

      * one individual-trajectory PDF per (IC, strategy) -- S * P PDFs
        (unchanged from `plot_comparisons`),
      * exactly FOUR batch-comparison PDFs total, each overlaying every
        strategy at once: calibration, ERF, EnKF-vs-open-loop L2, and
        prior/posterior RMSE.

    Output layout (under ``workdir/figures/comparisons_bulk/``):

        individual_trajectories/trajectory_ic_<i>_<strategy_key>.pdf
        calibration_all.pdf
        erf_all.pdf
        l2_all.pdf
        rmse_all.pdf

    Returns the path of the ``figures/comparisons_bulk`` directory written.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(h5_path))

    save_dir = os.path.join(workdir, "figures", "comparisons_bulk")
    indiv_dir = os.path.join(save_dir, "individual_trajectories")
    os.makedirs(indiv_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        dt_window = float(meta.attrs["dt_window"])
        obs_every_n = int(meta.attrs["obs_every_n"])
        sigma_obs = float(meta.attrs["sigma_obs"])
        N_ens = int(meta.attrs["N_ens"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])
        x_grid = meta["x_grid"][:]

        strategy_keys = _decode(meta["strategy_keys"][:])
        strategy_labels = _decode(meta["strategy_labels"][:])
        strategy_propagator = _decode(meta["strategy_propagator"][:])
        label_of = dict(zip(strategy_keys, strategy_labels))
        propagator_of = dict(zip(strategy_keys, strategy_propagator))
        colors = _strategy_colors(strategy_keys)

        S = len(strategy_keys)
        if S < 2:
            logging.warning(
                "plot_comparisons_bulk: fewer than 2 strategies found in "
                f"{h5_path}; individual-trajectory plots will still be "
                "written, but no bulk comparison PDFs will be generated."
            )

        t_fine_traj = f["trajectories/t_fine"][:]

        # ── 1. Individual-trajectory PDFs: one per (IC, strategy) ───────
        n_traj_pdfs = 0
        for ic_idx in range(num_ics_traj):
            ic_grp = f[f"trajectories/ic_{ic_idx}"]
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = obs_coords_raw if obs_coords_raw.size else None

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                x_est = sg["x_est"][:]
                x_std = sg["x_std"][:]
                l2_time_avg = float(sg.attrs["l2_time_avg"])

                save_path = os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf")
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=x_est, x_std=x_std,
                    ic_idx=ic_idx, strategy_label=label_of[key],
                    l2_time_avg=l2_time_avg, save_path=save_path,
                    x_grid=x_grid, dt_window=dt_window, obs_coords=obs_coords,
                )
                n_traj_pdfs += 1

        logging.info(
            f"plot_comparisons_bulk: wrote {n_traj_pdfs} individual trajectory "
            f"PDFs ({num_ics_traj} ICs x {S} strategies) to {indiv_dir}"
        )

        # ── 2. Batch data needed for the bulk comparisons ────────────────
        batch = f["batch"]
        B = int(batch.attrs["B"])
        obs_times_batch = batch["obs_times"][:]
        window_idx = batch["window_idx"][:]
        t_dense_fine = batch["t_dense_fine"][:]

        batch_fields = (
            "prior_rmse_mean", "post_rmse_mean", "erf_mean", "erf_std",
            "rmse_window_mean", "spread_window_mean", "rmse_raw", "spread_raw",
            "l2_dense_mean",
        )
        batch_strat = {}
        for key in strategy_keys:
            sg = batch[f"strategies/{key}"]
            batch_strat[key] = {field: sg[field][:] for field in batch_fields}

        open_loop = {}
        if "open_loop" in batch:
            for prop_key in batch["open_loop"]:
                og = batch[f"open_loop/{prop_key}"]
                open_loop[prop_key] = dict(t=og["t"][:], l2_dense_mean=og["l2_dense_mean"][:])

    if S < 2:
        return save_dir

    # ── 3. Four bulk comparison PDFs, every strategy overlaid on each ───
    _plot_calibration_bulk(
        strategy_keys=strategy_keys, label_of=label_of,
        window_idx=window_idx, dt_window=dt_window,
        spread_window_of={k: batch_strat[k]["spread_window_mean"] for k in strategy_keys},
        rmse_window_of={k: batch_strat[k]["rmse_window_mean"] for k in strategy_keys},
        spread_raw_of={k: batch_strat[k]["spread_raw"] for k in strategy_keys},
        rmse_raw_of={k: batch_strat[k]["rmse_raw"] for k in strategy_keys},
        colors=colors,
        title=(
            f"Calibration: ensemble spread vs RMSE — all strategies\n"
            f"(B={B} trajectories, N_ens={N_ens})"
        ),
        save_path=os.path.join(save_dir, "calibration_all.pdf"),
        n_bins=n_bins,
    )

    _plot_erf_bulk(
        strategy_keys=strategy_keys, label_of=label_of,
        obs_times=obs_times_batch,
        erf_mean_of={k: batch_strat[k]["erf_mean"] for k in strategy_keys},
        erf_std_of={k: batch_strat[k]["erf_std"] for k in strategy_keys},
        colors=colors, n_traj=B,
        title=(
            f"EnKF Error Reduction Factor per observation time — all strategies\n"
            f"(N_ens={N_ens}, obs every {obs_every_n}th grid point, σ_obs={sigma_obs})"
        ),
        save_path=os.path.join(save_dir, "erf_all.pdf"),
    )

    curves, curve_colors = {}, {}
    used_props = sorted({propagator_of[k] for k in strategy_keys})
    ol_palette = ["#B0BEC5", "#78909C", "#546E7A"]
    for i, prop_key in enumerate(used_props):
        if prop_key in open_loop:
            lbl = f"{prop_key} open-loop"
            curves[lbl] = (open_loop[prop_key]["t"], open_loop[prop_key]["l2_dense_mean"])
            curve_colors[lbl] = ol_palette[i % len(ol_palette)]
    for key in strategy_keys:
        curves[label_of[key]] = (t_dense_fine, batch_strat[key]["l2_dense_mean"])
        curve_colors[label_of[key]] = colors[key]

    _plot_l2_per_timestep(
        curves=curves,
        title=f"EnKF vs open-loop: mean relative L2 per timestep — all strategies  (B={B})",
        save_path=os.path.join(save_dir, "l2_all.pdf"),
        colors=curve_colors,
    )

    _plot_rmse_bulk(
        strategy_keys=strategy_keys, label_of=label_of,
        obs_times=obs_times_batch,
        prior_mean_of={k: batch_strat[k]["prior_rmse_mean"] for k in strategy_keys},
        post_mean_of={k: batch_strat[k]["post_rmse_mean"] for k in strategy_keys},
        sigma_obs=sigma_obs, n_traj=B, colors=colors,
        title=(
            f"EnKF prior vs posterior RMSE — all strategies\n"
            f"(N_ens={N_ens}, obs every {obs_every_n}th grid point, σ_obs={sigma_obs})"
        ),
        save_path=os.path.join(save_dir, "rmse_all.pdf"),
    )

    logging.info(
        f"plot_comparisons_bulk: wrote 4 bulk comparison PDFs "
        f"({S} strategies each) to {save_dir}"
    )
    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Entry point -- equilibrium (climatological/attractor) variance
# ─────────────────────────────────────────────────────────────────────────
def plot_equilibrium_variance(
    h5_path: str, workdir: str | None = None, log_scale: bool = True,
) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and writes ONE PDF
    comparing, per grid point, the long-term equilibrium
    (climatological/attractor) variance of:

      * the reference/truth trajectory,
      * each propagator's open-loop, "static" (unfiltered) physics
        rollout, and
      * every filtered EnKF strategy in the file,

    overlaid on a single set of axes -- the "static physics vs filtered
    strategies" steady-state diagnostic. Works for any number of
    strategies S >= 1 (no pairwise blow-up, since every curve is compared
    against the one shared reference rather than against every other
    curve).

    Output: ``<workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf``

    Returns the path of the ``figures/comparisons_bulk`` directory written.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(h5_path))

    save_dir = os.path.join(workdir, "figures", "comparisons_bulk")
    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        N_ens = int(meta.attrs["N_ens"])
        obs_every_n = int(meta.attrs["obs_every_n"])
        sigma_obs = float(meta.attrs["sigma_obs"])
        x_grid = meta["x_grid"][:]
        obs_indices = meta["obs_indices"][:] if "obs_indices" in meta else None

        strategy_keys = _decode(meta["strategy_keys"][:])
        strategy_labels = _decode(meta["strategy_labels"][:])
        label_of = dict(zip(strategy_keys, strategy_labels))
        colors = _strategy_colors(strategy_keys)

        batch = f["batch"]
        B = int(batch.attrs["B"])

        missing_eqvar = "reference" not in batch or any(
            "eqvar_mean" not in batch[f"strategies/{k}"] for k in strategy_keys
        )
        if missing_eqvar:
            raise KeyError(
                f"{h5_path} has no equilibrium-variance data (missing "
                "'batch/reference' and/or per-strategy 'eqvar_mean'). "
                "Re-run evaluate_filters to regenerate the HDF5 file "
                "before calling plot_equilibrium_variance."
            )

        burn_in_frac = float(batch.attrs["eqvar_burn_in_frac"])
        reference_mean = batch["reference/eqvar_mean"][:]
        reference_std = batch["reference/eqvar_std"][:]

        eqvar_mean_of, eqvar_std_of = {}, {}
        for key in strategy_keys:
            sg = batch[f"strategies/{key}"]
            eqvar_mean_of[key] = sg["eqvar_mean"][:]
            eqvar_std_of[key] = sg["eqvar_std"][:]

        open_loop_eqvar = {}
        if "open_loop" in batch:
            for prop_key in batch["open_loop"]:
                og = batch[f"open_loop/{prop_key}"]
                if "eqvar_mean" in og:
                    open_loop_eqvar[prop_key] = dict(
                        mean=og["eqvar_mean"][:], std=og["eqvar_std"][:],
                    )

    save_path = os.path.join(save_dir, "equilibrium_variance_all.pdf")
    _plot_equilibrium_variance_bulk(
        strategy_keys=strategy_keys, label_of=label_of,
        eqvar_mean_of=eqvar_mean_of, eqvar_std_of=eqvar_std_of,
        reference_mean=reference_mean, reference_std=reference_std,
        open_loop_eqvar=open_loop_eqvar, colors=colors,
        title=(
            f"Equilibrium variance — static (open-loop) physics vs filtered "
            f"strategies\n(B={B} trajectories, N_ens={N_ens}, obs every "
            f"{obs_every_n}th grid point, σ_obs={sigma_obs}, "
            f"burn-in={burn_in_frac:.0%} of window discarded)"
        ),
        save_path=save_path,
        x_grid=x_grid,
        obs_indices=obs_indices,
        log_scale=log_scale,
    )

    logging.info(f"plot_equilibrium_variance: wrote {save_path}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Runner: classic 3-way comparison
# ─────────────────────────────────────────────────────────────────────────
"""
Run the classic 3-way EnKF comparison

    1. DD  propagator + multiplicative inflation
    2. PI  propagator + multiplicative inflation
    3. PI  propagator + Route B (residual-scaled additive) inflation

through the two-stage modular pipeline (`evaluate_filters` +
`plot_comparisons`).

Outputs
-------
    <workdir>/<config.wandb.name>.h5                       -- evaluate_filters
    <workdir>/figures/comparisons/individual_trajectories/  -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons/calibration_*_vs_*.pdf    -- 3 pairs
    <workdir>/figures/comparisons/erf_*_vs_*.pdf
    <workdir>/figures/comparisons/l2_*_vs_*.pdf
    <workdir>/figures/comparisons/rmse_*_vs_*.pdf
"""


def run_3way_comparison(config, workdir: str, test_h5_path: str | None = None,
                        n_bins: int = 10) -> str:
    """
    Runs the default DD-mult / PI-mult / PI-Route-B 3-way EnKF evaluation
    and writes every comparison figure.

    Returns the path of the `figures/comparisons` directory written by
    `plot_comparisons`.
    """
    os.makedirs(workdir, exist_ok=True)

    logging.info(
        "Stage 1/2: evaluate_filters — running DD+mult / PI+mult / "
        "PI+RouteB on shared data and writing the results HDF5 ..."
    )
    h5_path = evaluate_filters(
        config=config,
        workdir=workdir,
        strategies=None,      # None -> build_default_3way_strategies
        propagators=None,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/2: plot_comparisons — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Runner: 4-way inflation-scheme comparison, crossed with the propagators
# ─────────────────────────────────────────────────────────────────────────
"""
Run a 4-way EnKF covariance-inflation comparison, crossed with the
propagators that expose a PDE residual (PI and/or Hybrid; "dd" is not
eligible)

For each propagator in `config.kf.compare_propagators` (default: PI and
Hybrid, i.e. `["pi", "hy"]`):

    1. <kind> propagator + Multiplicative inflation
    2. <kind> propagator + Route B (residual-scaled additive) inflation
    3. <kind> propagator + plain Additive inflation (Route B with the
       flow-dependent term zeroed out, i.e. beta=0)
    4. <kind> propagator + Relaxation-to-Prior Perturbations (RTPP)

so differences ACROSS inflation schemes, at fixed propagator, isolate the
effect of the inflation scheme, while differences ACROSS propagators, at
fixed inflation scheme, isolate the effect of the surrogate itself. Pass
`config.kf.compare_propagators = ["pi"]` to recover the original PI-only
4-way comparison.

Knobs, all read from `config.kf`
--------------------------------
    compare_propagators -- which propagators to cross the 4 inflation
                         schemes with. Default `["pi", "hy"]`; "dd" is not
                         eligible (no PDE residual for Route B/additive).
    inflation_factor  -- window-level multiplicative inflation (scaled to
                         a per-fine-step factor internally).
    route_b_alpha     -- Route B / additive constant floor.
    route_b_beta      -- Route B flow-dependent coefficient. THIS NEEDS
                         KS-SPECIFIC CALIBRATION: the KS residual carries
                         fourth-order spatial derivatives summed over 256
                         grid points, so ||rho||^2 is nothing like L96's.
                         Run `run_route_b_inflation_sweep` first, or read
                         `route_b_scale_mean` out of the results HDF5 to
                         see the realised alpha + beta*||rho||^2.
    rtpp_alpha        -- RTPP relaxation factor, in [0, 1]; typical tuned
                         values are 0.5-0.9.
    rtpp_alpha_fine   -- multiplicative inflation inside RTPP's (shared)
                         predict step. Defaults to 1.0 so `rtpp_alpha` is
                         the only active inflation mechanism, which is what
                         makes the comparison fair across inflation schemes.
"""


def run_4way_comparison(config, workdir: str, test_h5_path: str | None = None,
                        n_bins: int = 10) -> str:
    """
    Runs the Mult / Route-B / Additive / RTPP inflation-strategy
    evaluation, crossed with the configured propagators (default PI and
    Hybrid -- "dd" has no PDE residual and is not eligible), and writes
    every comparison figure.

    Returns the path of the `figures/comparisons` directory written by
    `plot_comparisons`.
    """
    os.makedirs(workdir, exist_ok=True)

    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)

    # ── EnKF / inflation configuration (mirrors evaluate_filters' own
    #    reads of config.kf, since we build the strategy set ourselves) ──
    N_ens = config.kf.get("N_ens", 50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)
    alpha_rb = config.kf.get("route_b_alpha", 1.0)
    beta_rb = config.kf.get("route_b_beta", 1.0)
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    alpha_rtpp = config.kf.get("rtpp_alpha", 0.5)
    alpha_fine_rtpp = config.kf.get("rtpp_alpha_fine", 1.0)
    propagator_kinds = tuple(config.kf.get("compare_propagators", ["pi", "hy"]))

    DT_WINDOW = float(config.get("dt_window", 1.0))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)

    logging.info(
        f"Building 4-way strategy set (x {list(propagator_kinds)}): "
        "Mult / RouteB / Additive / RTPP ..."
    )
    strategies, propagators, N, t_star_window = build_default_4way_strategies(
        config, N_ens, alpha_fine, alpha_rb, beta_rb, n_quad_rb, alpha_rtpp,
        grid, alpha_fine_rtpp=alpha_fine_rtpp, propagator_kinds=propagator_kinds,
    )
    if len(strategies) > 12:
        logging.warning(
            f"{len(strategies)} strategies -> {len(strategies) * (len(strategies) - 1) // 2} "
            "pairwise comparison PDFs from plot_comparisons. Consider "
            "config.kf.compare_propagators = ['pi'] for the original "
            "PI-only 4-way comparison if that's too many."
        )

    logging.info(
        f"Stage 1/2: evaluate_filters — running {len(strategies)} propagator x "
        "inflation-scheme strategies on shared data and writing the results HDF5 ..."
    )
    h5_path = evaluate_filters(
        config=config,
        workdir=workdir,
        strategies=strategies,
        propagators=propagators,
        t_star_window=t_star_window,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/2: plot_comparisons — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Multiplicative-inflation-factor sweep, crossed with the propagators
# ─────────────────────────────────────────────────────────────────────────

def build_mult_sweep_strategies(config, N_ens, alpha_coarse_list, steps_per_window,
                                grid, propagator_kinds=("dd", "pi")):
    """
    Builds a multiplicative-inflation-factor sweep, crossed with each
    propagator in `propagator_kinds` ("dd", "pi" and/or "hy" -- the
    hybrid physics-informed + data-driven checkpoint named in
    `config.wandb.name_hy`, which the KS pipeline trains but L96 has no
    counterpart for):

        <kind>_mult_a<tag> -- that propagator + multiplicative inflation at
                              alpha_coarse_list[i] (converted internally to
                              the fine-step-scaled `alpha_fine`).

    `predict_fn`/`update_fn` are built ONCE per propagator: multiplicative
    inflation doesn't change the EnKF closure itself, only the `alpha_fine`
    scalar carried alongside it in the strategy dict. So the sweep is cheap
    -- no extra model calls or checkpoint loads per alpha, just extra
    strategy dict entries reusing the same closures.

    Returns (strategies, propagators, N, t_star_window).
    """
    _, _, t_star_window = _window_grid(config, grid)

    loaded, closures = {}, {}
    N = None
    for kind in propagator_kinds:
        model, params = _load_ks_model(config, kind, t_star_window, grid)
        loaded[kind] = (model, params)
        closures[kind] = model.make_enkf_fns(params, N_ens=N_ens)
        if N is None:
            N = model.N
        assert model.N == N, (
            f"'{kind}' checkpoint grid size ({model.N}) != {N}; can't share a "
            "strategy/propagator set across them."
        )

    label_of_kind = {"dd": "DD", "pi": "PI", "hy": "Hybrid"}

    strategies = []
    for alpha_coarse in alpha_coarse_list:
        alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
        tag = f"{alpha_coarse:g}".replace(".", "p")
        for kind in propagator_kinds:
            predict_fn, update_fn = closures[kind]
            strategies.append(dict(
                key=f"{kind}_mult_a{tag}",
                label=f"{label_of_kind.get(kind, kind.upper())} + Mult. Infl. "
                      f"(\u03b1={alpha_coarse:g})",
                kind="standard", propagator=kind,
                predict_fn=predict_fn, update_fn=update_fn,
                alpha_fine=alpha_fine,
            ))

    return strategies, loaded, N, t_star_window


"""
Run a multiplicative-inflation-factor calibration sweep

    For every value `alpha` in `config.kf.inflation_factor_list`, and for
    each propagator in `config.kf.sweep_propagators` (default DD, PI and
    Hybrid -- "dd", "pi", "hy"), evaluate that propagator + multiplicative
    inflation @ alpha.

Choosing `config.kf.inflation_factor_list`
------------------------------------------
    Values too close to 1.0 risk ensemble collapse / filter divergence
    over a long smoother run; values too large needlessly inflate the
    posterior and hurt RMSE. Operational EnKF practice keeps the
    window-level factor in roughly [1.00, 1.30]. A reasonable first-pass
    grid bracketing this codebase's own default of 1.05:

        config.kf.inflation_factor_list = [
            1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30,
        ]

    (1.00 is the no-inflation control.) Note the factor is applied
    per-fine-step as `alpha ** (1/steps_per_window)`, so its effect
    depends on `dt_fine` -- re-tune after changing it.

    Once the coarse sweep identifies an optimum, re-run a finer sweep
    bracketing it, e.g. `np.linspace(best - 0.03, best + 0.03, 7)`.

    Use `plot_comparisons_bulk` (as this runner does) rather than
    `plot_comparisons` for sweeps: the pairwise variant writes
    4 * C(S, 2) PDFs, which is 306 files for a 9-alpha x 2-propagator set.
"""


def run_mult_inflation_sweep(config, workdir: str, test_h5_path: str | None = None,
                             n_bins: int = 10) -> str:
    """
    Runs the multiplicative-inflation-factor sweep and writes every
    comparison figure, including the steady-state equilibrium-variance
    diagnostic.

    Returns the path of the `figures/comparisons_bulk` directory.
    """
    os.makedirs(workdir, exist_ok=True)

    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)

    N_ens = config.kf.get("N_ens", 50)
    propagator_kinds = tuple(config.kf.get("sweep_propagators", ["dd", "pi", "hy"]))
    alpha_coarse_list = list(config.kf.get(
        "inflation_factor_list",
        [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30],
    ))
    if len(alpha_coarse_list) == 0:
        raise ValueError("config.kf.inflation_factor_list is empty.")
    if not all(a > 0 for a in alpha_coarse_list):
        raise ValueError(
            f"config.kf.inflation_factor_list must be strictly positive, "
            f"got {alpha_coarse_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 1.0))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = len(propagator_kinds) * len(alpha_coarse_list)
    logging.info(
        f"Building Mult.-inflation sweep: {list(propagator_kinds)} x "
        f"{len(alpha_coarse_list)} inflation factor(s) ({alpha_coarse_list}) "
        f"-> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one bulk figure may get "
            "crowded. Consider a shorter inflation_factor_list for a first "
            "pass (see module docstring)."
        )

    strategies, propagators, N, t_star_window = build_mult_sweep_strategies(
        config, N_ens, alpha_coarse_list, steps_per_window, grid, propagator_kinds,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} propagator x "
        "alpha strategies on shared data and writing the results HDF5 ..."
    )
    h5_path = evaluate_filters(
        config=config, workdir=workdir, strategies=strategies,
        propagators=propagators, t_star_window=t_star_window,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info("Stage 3/3: plot_equilibrium_variance ...")
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Pure-additive-inflation-strength sweep: propagators exposing r_net only
# (PI and/or Hybrid -- "dd" has no PDE residual)
# ─────────────────────────────────────────────────────────────────────────

def build_add_sweep_strategies(config, N_ens, alpha_list, steps_per_window, grid,
                               propagator_kinds=("pi", "hy")):
    """
    Builds a pure-additive-inflation-strength sweep, crossed with each
    propagator in `propagator_kinds`, reusing ONE `make_route_b_enkf_fns`
    closure pair per propagator -- one strategy per (propagator, alpha)
    pair:

        <kind>_add_a<tag> -- that propagator + Route B inflation with
                             `beta` pinned to 0.0, i.e. the flow-dependent
                             `beta * ||rho||^2` term switched off, so the
                             residual-scaled Route B machinery degenerates
                             to *pure* additive inflation: a fixed-
                             covariance process-noise floor `alpha * Q0`
                             injected every fine step, with
                             `alpha_list[i]` setting the floor's strength.

    Route B needs the PDE residual, which only the physics-informed
    `KSUDON` ("pi") and the hybrid `KSUDON_Hybrid` ("hy") expose
    (`r_net`) -- `KSUDON_DD` doesn't, so "dd" is not a valid entry in
    `propagator_kinds` here.

    `predict_fn`/`update_fn` are built ONCE per propagator: pinning
    `beta=0.0` doesn't change the EnKF closure itself, only the scalars
    carried alongside it in the strategy dict. So the sweep is cheap --
    no extra model calls or checkpoint loads per alpha, just extra
    strategy dict entries reusing the same closures.

    `Q0`'s spatial structure comes from `config.kf.Q0_sigma` and
    `config.kf.Q0_corr_len` (see `kf.periodic_gaussian_cov` for why a
    correlated Q0 is often the better choice for a field).

    Returns (strategies, propagators, N, t_star_window).
    """
    _, _, t_star_window = _window_grid(config, grid)

    loaded, closures = {}, {}
    N = None
    for kind in propagator_kinds:
        if kind == "dd":
            raise ValueError(
                "'dd' has no PDE residual (r_net), so it cannot run Route "
                "B / additive inflation. Use 'pi' and/or 'hy' in "
                "propagator_kinds."
            )
        model, params = _load_ks_model(config, kind, t_star_window, grid)
        loaded[kind] = (model, params)
        closures[kind] = model.make_route_b_enkf_fns(params, N_ens=N_ens)
        if N is None:
            N = model.N
        assert model.N == N, (
            f"'{kind}' checkpoint grid size ({model.N}) != {N}; can't share a "
            "strategy/propagator set across them."
        )

    P0_sigma = config.kf.get("P0_sigma", 0.5)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    Q0_corr_len = float(config.kf.get("Q0_corr_len", config.kf.get("P0_corr_len", 0.0)))
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    Q_coarse = build_cov(N, grid["L"], Q0_sigma, Q0_corr_len)
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    label_of_kind = {"pi": "PI", "hy": "Hybrid"}

    strategies = []
    for alpha in alpha_list:
        tag = f"{alpha:g}".replace(".", "p")
        for kind in propagator_kinds:
            predict_fn, update_fn = closures[kind]
            strategies.append(dict(
                key=f"{kind}_add_a{tag}",
                label=f"{label_of_kind.get(kind, kind.upper())} + Add. Infl. "
                      f"(\u03b1={alpha:g})",
                kind="route_b", propagator=kind,
                predict_fn=predict_fn, update_fn=update_fn,
                Q0=Q_fine, alpha=float(alpha), beta=0.0, n_quad=n_quad_rb,
            ))

    return strategies, loaded, N, t_star_window


"""
Run a pure-additive-inflation-strength sweep, crossed with the propagators
that expose a PDE residual (PI and/or Hybrid; "dd" is not eligible)

Choosing `config.kf.inflation_alpha_list`
------------------------------------------
    Pure additive inflation injects a FIXED-covariance perturbation
    `alpha * Q0` at every fine step, independent of the current ensemble
    spread -- unlike multiplicative inflation, which rescales the existing
    spread and so is a no-op at 1.0. The natural "no correction" control
    for THIS sweep is therefore `alpha = 0.0`, not 1.0; include it.

    `Q0` itself is `Q0_sigma^2` times either the identity or the
    correlated kernel (`Q0_corr_len`), fine-step-scaled. A first-pass grid
    with a 0.0 control and headroom above 1.0:

        config.kf.inflation_alpha_list = [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
        ]

    Because the stored KS field is nondimensionalised to O(1), `Q0_sigma`
    around 0.05-0.2 is a more sensible starting point than the L96 configs'
    0.3 -- the sweep then scales that floor up or down.

    Once the coarse sweep identifies an optimum, re-run a finer sweep
    bracketing it, e.g. `np.linspace(best - 0.25, best + 0.25, 7)`
    (clipped at 0).
"""


def run_add_inflation_sweep(config, workdir: str, test_h5_path: str | None = None,
                            n_bins: int = 10) -> str:
    """
    Runs the pure-additive-inflation-strength sweep (Route B with beta
    pinned to 0.0), crossed with the configured propagators
    (`config.kf.sweep_propagators_additive`, default PI and Hybrid -- "dd"
    has no PDE residual and is not eligible), and writes every comparison
    figure, including the steady-state equilibrium-variance diagnostic.

    Returns the path of the `figures/comparisons_bulk` directory.
    """
    os.makedirs(workdir, exist_ok=True)

    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)

    N_ens = config.kf.get("N_ens", 50)
    propagator_kinds = tuple(config.kf.get("sweep_propagators_additive", ["pi", "hy"]))
    alpha_list = list(config.kf.get(
        "inflation_alpha_list", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    ))
    if len(alpha_list) == 0:
        raise ValueError("config.kf.inflation_alpha_list is empty.")
    if not all(a >= 0 for a in alpha_list):
        raise ValueError(
            f"config.kf.inflation_alpha_list must be non-negative, got {alpha_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 1.0))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = len(propagator_kinds) * len(alpha_list)
    logging.info(
        f"Building Additive-inflation sweep: {list(propagator_kinds)} + pure "
        f"additive inflation (Route B, beta=0) x {len(alpha_list)} alpha "
        f"value(s) ({alpha_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one bulk figure may get "
            "crowded. Consider a shorter inflation_alpha_list or fewer "
            "propagators for a first pass."
        )

    strategies, propagators, N, t_star_window = build_add_sweep_strategies(
        config, N_ens, alpha_list, steps_per_window, grid, propagator_kinds,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} propagator x "
        "additive-alpha strategies on shared data ..."
    )
    h5_path = evaluate_filters(
        config=config, workdir=workdir, strategies=strategies,
        propagators=propagators, t_star_window=t_star_window,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info("Stage 3/3: plot_equilibrium_variance ...")
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# RTPP relaxation-factor sweep, crossed with the propagators
# ─────────────────────────────────────────────────────────────────────────

def build_rtpp_sweep_strategies(config, N_ens, alpha_rtpp_list, alpha_fine_rtpp,
                                grid, propagator_kinds=("dd", "pi")):
    """
    Builds an RTPP relaxation-factor sweep crossed with each propagator in
    `propagator_kinds`:

        <kind>_rtpp_a<tag> -- that propagator + RTPP at relaxation factor
                              alpha_rtpp_list[i], with the (shared)
                              predict-step multiplicative inflation pinned
                              to `alpha_fine_rtpp` (default 1.0, i.e. off)
                              so the swept `alpha_rtpp` is the only active
                              inflation mechanism.

    RTPP needs no PDE residual -- it only reshapes the posterior ensemble
    anomalies -- so, unlike Route B, it is available on every model class
    here (`make_rtpp_enkf_fns` is defined on both `KSUDON` and
    `KSUDON_DD`, and inherited by `KSUDON_Hybrid`).

    `predict_fn`/`update_fn` are built ONCE per propagator and reused
    across every alpha_rtpp, which is a runtime scalar passed into
    `run_enkf_smoother_rtpp` per strategy dict rather than baked into the
    closure.

    Returns (strategies, propagators, N, t_star_window).
    """
    _, _, t_star_window = _window_grid(config, grid)

    loaded, closures = {}, {}
    N = None
    for kind in propagator_kinds:
        model, params = _load_ks_model(config, kind, t_star_window, grid)
        loaded[kind] = (model, params)
        closures[kind] = model.make_rtpp_enkf_fns(params, N_ens=N_ens)
        if N is None:
            N = model.N
        assert model.N == N, (
            f"'{kind}' checkpoint grid size ({model.N}) != {N}; can't share a "
            "strategy/propagator set across them."
        )

    label_of_kind = {"dd": "DD", "pi": "PI", "hy": "Hybrid"}

    strategies = []
    for alpha_rtpp in alpha_rtpp_list:
        tag = f"{alpha_rtpp:g}".replace(".", "p")
        for kind in propagator_kinds:
            predict_fn, update_fn = closures[kind]
            strategies.append(dict(
                key=f"{kind}_rtpp_a{tag}",
                label=f"{label_of_kind.get(kind, kind.upper())} + RTPP "
                      f"(\u03b1={alpha_rtpp:g})",
                kind="rtpp", propagator=kind,
                predict_fn=predict_fn, update_fn=update_fn,
                alpha_fine=alpha_fine_rtpp, alpha_rtpp=alpha_rtpp,
            ))

    return strategies, loaded, N, t_star_window


"""
Run an RTPP relaxation-factor sweep

Choosing `config.kf.rtpp_alpha_list`
------------------------------------
    RTPP relaxes each posterior ensemble perturbation partway back toward
    its (larger, pre-update) prior perturbation:
    `x'_post <- (1 - alpha_rtpp) * x'_post + alpha_rtpp * x'_prior`. So
    `alpha_rtpp = 0` is the "no correction" control (posterior spread used
    as-is -- the RTPP analogue of additive inflation's `alpha = 0.0`
    control, NOT multiplicative inflation's 1.0), and `alpha_rtpp = 1`
    discards the update's spread reduction entirely. Values are only
    meaningful in [0, 1].

    The relaxation literature (Zhang, Snyder & Sacher 2004; typical
    operational practice) usually finds tuned values in [0.5, 0.9]. A
    first-pass grid, denser in that band, with a 0.0 control:

        config.kf.rtpp_alpha_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    RTPP is a particularly good fit for a field like KS: because it
    rebuilds the posterior anomalies from the PRIOR anomalies, the restored
    spread inherits the forecast's own spatial correlation structure
    instead of the arbitrary structure of a prescribed Q0.
"""


def run_rtpp_inflation_sweep(config, workdir: str, test_h5_path: str | None = None,
                             n_bins: int = 10) -> str:
    """
    Runs the RTPP relaxation-factor sweep, crossed with the configured
    propagators and with the shared predict-step multiplicative inflation
    pinned at `config.kf.rtpp_alpha_fine` (default 1.0), and writes every
    comparison figure including the equilibrium-variance diagnostic.

    Returns the path of the `figures/comparisons_bulk` directory.
    """
    os.makedirs(workdir, exist_ok=True)

    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)

    N_ens = config.kf.get("N_ens", 50)
    propagator_kinds = tuple(config.kf.get("sweep_propagators", ["dd", "pi", "hy"]))
    alpha_fine_rtpp = config.kf.get("rtpp_alpha_fine", 1.0)
    alpha_rtpp_list = list(config.kf.get(
        "rtpp_alpha_list", [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ))
    if len(alpha_rtpp_list) == 0:
        raise ValueError("config.kf.rtpp_alpha_list is empty.")
    if not all(0.0 <= a <= 1.0 for a in alpha_rtpp_list):
        raise ValueError(
            f"config.kf.rtpp_alpha_list must lie in [0, 1], got {alpha_rtpp_list}"
        )

    n_strategies = len(propagator_kinds) * len(alpha_rtpp_list)
    logging.info(
        f"Building RTPP sweep: {list(propagator_kinds)} x "
        f"{len(alpha_rtpp_list)} relaxation factor(s) ({alpha_rtpp_list}) "
        f"-> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one bulk figure may get "
            "crowded. Consider a shorter rtpp_alpha_list for a first pass."
        )

    strategies, propagators, N, t_star_window = build_rtpp_sweep_strategies(
        config, N_ens, alpha_rtpp_list, alpha_fine_rtpp, grid, propagator_kinds,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} propagator x "
        "rtpp_alpha strategies on shared data ..."
    )
    h5_path = evaluate_filters(
        config=config, workdir=workdir, strategies=strategies,
        propagators=propagators, t_star_window=t_star_window,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info("Stage 3/3: plot_equilibrium_variance ...")
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Route B (modified additive) beta sweep: propagators exposing r_net only
# (PI and/or Hybrid -- "dd" has no PDE residual)
# ─────────────────────────────────────────────────────────────────────────

def build_route_b_sweep_strategies(config, N_ens, beta_list, alpha_rb_fixed,
                                   steps_per_window, grid,
                                   propagator_kinds=("pi", "hy")):
    """
    Builds a Route B beta-sweep, crossed with each propagator in
    `propagator_kinds`, reusing ONE `make_route_b_enkf_fns` closure pair
    per propagator -- one strategy per (propagator, beta) pair, all
    sharing the SAME fixed `alpha_rb_fixed` floor:

        <kind>_route_b_b<tag> -- that propagator + Route B inflation,
                                 scale = alpha_rb_fixed + beta * ||rho||^2.

    This is the mirror image of `build_add_sweep_strategies`, which pins
    `beta=0.0` and sweeps the constant floor `alpha`.

    Route B needs the PDE residual, which only the physics-informed
    `KSUDON` ("pi") and the hybrid `KSUDON_Hybrid` ("hy") expose
    (`r_net`) -- `KSUDON_DD` doesn't, so "dd" is not a valid entry in
    `propagator_kinds` here.

    Returns (strategies, propagators, N, t_star_window).
    """
    _, _, t_star_window = _window_grid(config, grid)

    loaded, closures = {}, {}
    N = None
    for kind in propagator_kinds:
        if kind == "dd":
            raise ValueError(
                "'dd' has no PDE residual (r_net), so it cannot run Route "
                "B inflation. Use 'pi' and/or 'hy' in propagator_kinds."
            )
        model, params = _load_ks_model(config, kind, t_star_window, grid)
        loaded[kind] = (model, params)
        closures[kind] = model.make_route_b_enkf_fns(params, N_ens=N_ens)
        if N is None:
            N = model.N
        assert model.N == N, (
            f"'{kind}' checkpoint grid size ({model.N}) != {N}; can't share a "
            "strategy/propagator set across them."
        )

    P0_sigma = config.kf.get("P0_sigma", 0.5)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    Q0_corr_len = float(config.kf.get("Q0_corr_len", config.kf.get("P0_corr_len", 0.0)))
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    Q_coarse = build_cov(N, grid["L"], Q0_sigma, Q0_corr_len)
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    label_of_kind = {"pi": "PI", "hy": "Hybrid"}

    strategies = []
    for beta in beta_list:
        tag = f"{beta:g}".replace(".", "p")
        for kind in propagator_kinds:
            predict_fn, update_fn = closures[kind]
            strategies.append(dict(
                key=f"{kind}_route_b_b{tag}",
                label=f"{label_of_kind.get(kind, kind.upper())} + Route B "
                      f"(\u03b1={alpha_rb_fixed:g}, \u03b2={beta:g})",
                kind="route_b", propagator=kind,
                predict_fn=predict_fn, update_fn=update_fn,
                Q0=Q_fine, alpha=float(alpha_rb_fixed), beta=float(beta),
                n_quad=n_quad_rb,
            ))

    return strategies, loaded, N, t_star_window


"""
Run a Route B (modified additive) beta sweep, crossed with the
propagators that expose a PDE residual (PI and/or Hybrid; "dd" is not
eligible)

Choosing `alpha_rb_fixed`
-------------------------
    Route B's scale is `alpha + beta * ||rho||^2`. Two complementary
    experiments are worth running, both supported by the same argument:

    1. `alpha_rb_fixed = 0.0` -- isolates the flow-dependent term in pure
       form: inflation comes ENTIRELY from `beta * ||rho||^2`. Useful to
       see whether residual-scaling alone can substitute for a floor, but
       it means zero inflation whenever the residual is small, which risks
       under-dispersion in exactly those windows -- a diagnostic sweep,
       not necessarily a deployable operating point.

    2. `alpha_rb_fixed = <best alpha from run_add_inflation_sweep>` --
       holds the floor at whatever already works in the beta=0 setting and
       asks whether layering the flow-dependent term on top improves
       things. This is the practically relevant question, and the reason
       Route B exists as a *modification* of plain additive inflation.

    Run this function twice, once at each, rather than folding both into a
    2-D alpha x beta grid.

Choosing `config.kf.route_b_beta_list`  (KS-specific!)
-------------------------------------------------------
    Do NOT inherit L96's beta ~ 250 default. The KS residual
    rho = v_tau - (L_op v_hat + N(v_hat)) is evaluated spectrally and
    involves u_xxxx, and `residual_l2_norm_sq` sums it over all 256 grid
    points, so ||rho||^2 lives on a completely different scale from L96's
    40-variable O(1) residual.

    Calibrate empirically instead of guessing:

      1. Run this sweep once with a wide logarithmic grid, e.g.
         `[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]`.
      2. Read `batch/strategies/<key>/route_b_scale_mean` out of the
         results HDF5. That IS the realised `alpha + beta*||rho||^2` per
         fine step, averaged over the ensemble and the ICs.
      3. Keep the beta whose realised scale sits in the same ballpark as
         the best pure-additive alpha found by `run_add_inflation_sweep`
         -- that is the point at which the flow-dependent term is
         contributing comparably to a well-tuned floor rather than
         dominating or vanishing.
      4. Re-run a finer sweep around it, e.g.
         `np.linspace(best * 0.5, best * 1.5, 7)`.

    A 0.0 entry is always worth keeping: it recovers whatever
    `alpha_rb_fixed` alone gives, directly comparable to the additive
    sweep's point at that alpha.
"""


def run_route_b_inflation_sweep(
    config, workdir: str, alpha_rb_fixed: float | None = None,
    test_h5_path: str | None = None, n_bins: int = 10,
) -> str:
    """
    Runs the Route B beta sweep -- one calibration pass per value in
    `config.kf.route_b_beta_list`, crossed with the configured propagators
    (`config.kf.sweep_propagators_route_b`, default PI and Hybrid -- "dd"
    has no PDE residual and is not eligible), with the constant floor
    pinned at `alpha_rb_fixed` (falls back to `config.kf.route_b_alpha` if
    None) -- and writes every comparison figure including the
    equilibrium-variance diagnostic.

    Returns the path of the `figures/comparisons_bulk` directory.
    """
    os.makedirs(workdir, exist_ok=True)

    if test_h5_path is None:
        test_h5_path = _default_test_path(config)
    grid = _read_test_meta(test_h5_path)

    N_ens = config.kf.get("N_ens", 50)
    propagator_kinds = tuple(config.kf.get("sweep_propagators_route_b", ["pi", "hy"]))
    if alpha_rb_fixed is None:
        # Default 0.0: pure amplified residual error, no constant floor.
        alpha_rb_fixed = config.kf.get("route_b_alpha", 0.0)
    beta_list = list(config.kf.get(
        "route_b_beta_list", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    ))
    if len(beta_list) == 0:
        raise ValueError("config.kf.route_b_beta_list is empty.")
    if not all(b >= 0 for b in beta_list):
        raise ValueError(
            f"config.kf.route_b_beta_list must be non-negative, got {beta_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 1.0))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = len(propagator_kinds) * len(beta_list)
    logging.info(
        f"Building Route B beta sweep: {list(propagator_kinds)} + Route B "
        f"(alpha fixed at {alpha_rb_fixed:g}) x {len(beta_list)} beta "
        f"value(s) ({beta_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one bulk figure may get "
            "crowded. Consider a shorter route_b_beta_list or fewer "
            "propagators for a first pass."
        )

    strategies, propagators, N, t_star_window = build_route_b_sweep_strategies(
        config, N_ens, beta_list, alpha_rb_fixed, steps_per_window, grid,
        propagator_kinds,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} propagator x "
        "Route B beta strategies on shared data ..."
    )
    h5_path = evaluate_filters(
        config=config, workdir=workdir, strategies=strategies,
        propagators=propagators, t_star_window=t_star_window,
        test_h5_path=test_h5_path,
    )
    logging.info(f"  wrote {h5_path}")

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info("Stage 3/3: plot_equilibrium_variance ...")
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


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
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "ks_test_data.h5")

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
    x_pred_list = []

    rollout_windows = min(test_windows, (num_test_pts - 1) // pts_pw)
    for w in range(rollout_windows):
        pred_window = predict_batch(u_current_batch)    # (num_ics, pts_pw+1, N)

        # Avoid duplicating the overlapping boundary states between windows
        if w == 0:
            x_pred_list.append(pred_window)
        else:
            x_pred_list.append(pred_window[:, 1:, :])

        u_current_batch = pred_window[:, -1, :]

    x_pred_full = jnp.concatenate(x_pred_list, axis=1)
    n_pts = x_pred_full.shape[1]
    u_ref = u_test[:, :n_pts, :]
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
            x_true=np.array(u_ref[ic_idx]),
            x_est=np.array(x_pred_full[ic_idx]),
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
    norm_err = jnp.linalg.norm(err, axis=-1)
    norm_ref = jnp.linalg.norm(u_ref, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    overall_mean_l2 = np.mean(np.array(l2_rel_per_traj_time), axis=0)

    batch_save_path = os.path.join(
        workdir, "figures", config.wandb.name, "batch_l2_error_analysis.pdf"
    )

    _plot_batch_l2_over_time(
        t_ax=t_ax,
        overall_mean_l2=overall_mean_l2,
        save_path=batch_save_path,
    )

    logging.info(f"Batch L2 error plot saved to: {batch_save_path}")




def _plot_trajectory_summary(
    logging.info(f"Trajectory summary for IC {ic_idx} saved to: {save_path}")

def _plot_batch_l2_over_time(
    plt.close(fig)


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # ── 1. Load Dense Test Dataset ──────────────────────────────────────────
    data_dir = config.training.get("data_dir", "data")
    test_file = os.path.join(data_dir, "ks_test_data.h5")

    logging.info(f"Loading test dataset from {test_file}...")
    with h5py.File(test_file, 'r') as f:
        u_test = jnp.array(f['u'][:])[:100, :, :]     # Shape: (num_ics, num_test_pts, 256)
        N = f.attrs['N']
        dt = f.attrs['dt']
        test_windows = f.attrs['test_windows']
        
    num_ics, num_test_pts, N_loaded = u_test.shape
    assert N_loaded == 256, f"Expected state dimension 256, got {N_loaded}"

    # Reconstruct time definitions
    # 1 time unit = 1 window
    w_dt = 1.0
    pts_pw = int(round(w_dt / dt))
    t_ax = np.arange(num_test_pts) * dt
    
    # Single-window relative time grid required by the surrogate model
    t_star_window = t_ax[:pts_pw + 1]

    # ── 2. Setup Model & Load Checkpoint ────────────────────────────────────
    if config.mode == "eval":
        model = models.KSUDON(config, t_star_window)
    else:
        model = models.KSUDON_DD(config, t_star_window)
        
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "ks_udon_model")
    
    logging.info(f"Restoring DeepONet model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # JIT-compile a vmapped batch predictor
    # KS is autonomous, so input is just u (shape: 256)
    predict_batch = jax.jit(jax.vmap(lambda u: model.x_pred_fn(params, u, t_star_window), in_axes=0))

    # ── 3. Batched Autoregressive Rollout ───────────────────────────────────
    logging.info(f"Initiating batched rollout across all {num_ics} test trajectories...")
    
    u_current_batch = u_test[:, 0, :]               # Shape: (num_ics, 256)
    x_pred_list = []
    
    for w in range(test_windows):
        # Predict the full trajectory for the current time window
        pred_window = predict_batch(u_current_batch)    # Shape: (num_ics, pts_pw+1, 256)

        # Avoid duplicating the overlapping boundary states between windows
        if w == 0:
            x_pred_list.append(pred_window)
        else:
            x_pred_list.append(pred_window[:, 1:, :])

        # Advance the initial conditions to the end of the predicted window
        u_current_batch = pred_window[:, -1, :]

    # Reconstruct the continuous dense time series
    x_pred_full = jnp.concatenate(x_pred_list, axis=1)  # Shape: (num_ics, num_test_pts, 256)

    # ── 4. Generate Individual Trajectory Plots ─────────────────────────────
    total_plots = config.saving.get("total_plots", 5)
    for ic_idx in range(min(total_plots, num_ics)):
        logging.info(f"--- Generating detailed summary for IC {ic_idx} ---")
        
        save_path = os.path.join(
            workdir, "figures", config.wandb.name, f"trajectory_summary_ic_{ic_idx}.pdf"
        )
        
        _plot_trajectory_summary(
            t_ax=t_ax,
            x_true=np.array(u_test[ic_idx]),
            x_est=np.array(x_pred_full[ic_idx]),
            ic_idx=ic_idx,
            test_windows=test_windows,
            pts_pw=pts_pw,
            save_path=save_path,
            N=N_loaded
        )

    # ── 5. Generate Batch Error Analysis ─────────────────────────
    logging.info("--- Computing Batch L2 Error Statistics ---")
    
    err = x_pred_full - u_test
    norm_err = jnp.linalg.norm(err, axis=-1)
    norm_ref = jnp.linalg.norm(u_test, axis=-1)
    l2_rel_per_traj_time = norm_err / (norm_ref + 1e-12)

    l2_rel_np = np.array(l2_rel_per_traj_time)

    # Compute overall mean across all test cases (axis 0 is batch)
    overall_mean_l2 = np.mean(l2_rel_np, axis=0)

    # Plot the aggregated analytics
    batch_save_path = os.path.join(
        workdir, "figures", config.wandb.name, "batch_l2_error_analysis.pdf"
    )
    
    _plot_batch_l2_over_time(
        t_ax=t_ax, 
        overall_mean_l2=overall_mean_l2, 
        save_path=batch_save_path
    )
    
    logging.info(f"Batch L2 error plot saved to: {batch_save_path}")