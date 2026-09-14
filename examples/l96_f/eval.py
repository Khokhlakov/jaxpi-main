"""
Modular filter evaluation for the DeepONet + EnKF Lorenz-96 pipeline.

This module splits the old "evaluate + plot in one shot" pattern
(`evaluate_enkf_3_way`) into two stages:

    1. `evaluate_filters(...)`  -- runs every requested filtering strategy
       on the SAME data (same ICs, same noisy-observation draws, same
       initial ensembles) and stores every number a downstream plotting
       script could need into a single HDF5 file, keyed by
       `config.wandb.name`.

    2. A separate plotting module (not part of this file) later reads
       that HDF5 file and reproduces every figure the old pipeline drew
       inline: individual trajectories + time-avg error, RMSE with
       spread, calibration, Error Reduction Factor (ERF), batch
       time-mean L2 error, and prior/posterior RMSE with the
       observation-noise level.

Unlike `evaluate_enkf_3_way`, which hardcodes exactly three strategies
(DD+multiplicative, PI+multiplicative, PI+Route B), `evaluate_filters`
takes an arbitrary list of strategy specifications, so new filters can
be added or removed without touching the evaluation code. A helper,
`build_default_3way_strategies`, reconstructs the original 3-way setup
for drop-in backward compatibility.

HDF5 layout written by `evaluate_filters`
------------------------------------------
    /meta                                   (attrs only)
        N, F                                 -- state dim, per-IC forcing (num_ics_traj,)
        dt_window, dt_fine, dt_obs
        sigma_obs, P0_sigma, N_ens, obs_every_n, m
        num_ics_traj, num_ics_batch, trajectory_windows, batch_windows
        obs_indices                          -- dataset, (m,) int, if static
        strategy_keys, strategy_labels        -- ordered, parallel string arrays
        strategy_propagator                   -- which propagator each strategy uses
        strategy_kind                         -- "standard" | "route_b"

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
    against by `plot_equilibrium_variance`.
"""

import os
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
#     #   (plain additive inflation is just Route B with "alpha": 0.0,
#     #    i.e. the constant floor term zeroed out, leaving only the
#     #    flow-dependent beta * ||rho||^2 term)
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
# Default 3-way strategy set (DD-mult / PI-mult / PI-RouteB), for
# backward compatibility with `evaluate_enkf_3_way`.
# ─────────────────────────────────────────────────────────────────────────

def build_default_3way_strategies(config, N_ens, alpha_fine, Q_fine, alpha_rb, beta_rb, n_quad_rb):
    """
    Loads the PI and DD checkpoints named in `config.wandb.name_pi` /
    `config.wandb.name_dd` and builds the same 3 strategies
    `evaluate_enkf_3_way` used. Returns (strategies, propagators, N).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
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

    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)
    predict_fn_cl, update_fn_cl = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)

    strategies = [
        dict(key="dd_mult", label="DD + Mult. Infl.", kind="standard",
             propagator="dd", predict_fn=predict_fn_dd, update_fn=update_fn_dd,
             alpha_fine=alpha_fine),
        dict(key="pi_mult", label="PI + Mult. Infl.", kind="standard",
             propagator="pi", predict_fn=predict_fn_cl, update_fn=update_fn_cl,
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
# Default 4-way strategy set: same PI (physics-informed) propagator
# throughout, varying only the EnKF covariance-inflation scheme, so
# differences between strategies isolate the inflation choice rather than
# conflating it with propagator quality (unlike the 3-way set above, which
# also swaps DD vs PI).
# ─────────────────────────────────────────────────────────────────────────

def build_default_4way_strategies(
    config, N_ens, alpha_fine, alpha_rb, beta_rb, n_quad_rb, alpha_rtpp,
    alpha_fine_rtpp: float = 1.0,
):
    """
    Builds the 4-way inflation-strategy comparison set, all sharing the
    single PI checkpoint named in `config.wandb.name_pi`:

        1. pi_mult     -- standard multiplicative inflation (`make_enkf`).
        2. pi_route_b  -- Route B: residual-scaled additive inflation,
                          scale = alpha_rb + beta_rb * ||rho||^2
                          (`make_route_b_enkf`).
        3. pi_additive -- plain additive inflation, obtained from the SAME
                          Route B machinery with the constant floor term
                          zeroed out (alpha=0.0), leaving only the
                          flow-dependent beta_rb * ||rho||^2 term.
        4. pi_rtpp     -- Relaxation-to-Prior Perturbations (`make_rtpp_enkf`).
                          `alpha_fine_rtpp` controls the (optional, shared)
                          multiplicative inflation in RTPP's predict step;
                          it defaults to 1.0 so `alpha_rtpp` (the relaxation
                          factor, in [0, 1]) is the only active inflation
                          mechanism for this strategy.

    Note: RTPP's plumbing through `evaluate_filters` (via `build_batched_filters`
    and `run_enkf_smoother_rtpp`) is newer than the multiplicative/Route-B
    paths, so this is the strategy most likely to need tuning of
    `alpha_rtpp` / `alpha_fine_rtpp` if results look off.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    # Route B / additive both need Q_fine. N (=40, the dynamic-variable
    # count) is fixed at model-construction time rather than only known
    # after loading the checkpoint, so -- unlike `evaluate_filters`'s
    # strategies=None fallback, which has to defer this and re-inject it
    # afterward -- it can just be built here directly.
    P0_sigma = config.kf.get("P0_sigma", 1.0)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    DT_FINE = float(config.kf.get("dt_fine", dt_window))
    steps_per_window = steps_per_window_exact(dt_window, DT_FINE)
    Q_coarse = jnp.eye(N) * Q0_sigma ** 2
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    predict_fn_mult, update_fn_mult = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)
    predict_fn_rtpp, update_fn_rtpp = model_pi.make_rtpp_enkf_fns(params_pi, N_ens=N_ens)

    strategies = [
        dict(key="pi_mult", label="PI + Mult. Infl.", kind="standard",
             propagator="pi", predict_fn=predict_fn_mult, update_fn=update_fn_mult,
             alpha_fine=alpha_fine),
        dict(key="pi_route_b", label="PI + Route B Infl.", kind="route_b",
             propagator="pi", predict_fn=predict_fn_rb, update_fn=update_fn_rb,
             Q0=Q_fine, alpha=alpha_rb, beta=beta_rb, n_quad=n_quad_rb),
        dict(key="pi_additive", label="PI + Additive Infl.", kind="route_b",
             propagator="pi", predict_fn=predict_fn_rb, update_fn=update_fn_rb,
             Q0=Q_fine, alpha=0.0, beta=beta_rb, n_quad=n_quad_rb),
        dict(key="pi_rtpp", label="PI + RTPP", kind="rtpp",
             propagator="pi", predict_fn=predict_fn_rtpp, update_fn=update_fn_rtpp,
             alpha_fine=alpha_fine_rtpp, alpha_rtpp=alpha_rtpp),
    ]
    propagators = {
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────
def evaluate_filters(
    config: ml_collections.ConfigDict,
    workdir: str,
    strategies: list[dict] | None = None,
    propagators: dict | None = None,
    t_star_window=None,
    test_h5_path: str = None,
) -> str:
    """
    Memory-chunked version of the original. Same inputs, same HDF5 output.
    """
    # ── EnKF / observation configuration ───────────────────────────────
    obs_every_n = config.kf.get("obs_every_n", 4)
    sigma_obs = config.kf.get("sigma_obs", 0.5)
    P0_sigma = config.kf.get("P0_sigma", 1.0)
    dynamic_vars = config.kf.get("dynamic_vars", False)
    N_ens = config.kf.get("N_ens", 50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)
 
    alpha_rb = config.kf.get("route_b_alpha", 1.0)
    beta_rb = config.kf.get("route_b_beta", 5.0)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
 
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
        test_h5_path = "data/l96_forcing_test.h5"
 
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
 
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
    Q_coarse = None
    Q_fine = None
 
    if strategies is None or propagators is None:
        strategies, propagators, N, t_star_window = build_default_3way_strategies(  # noqa: F821
            config, N_ens, alpha_fine, None, alpha_rb, beta_rb, n_quad_rb,
        )
        Q_coarse = jnp.eye(N) * Q0_sigma ** 2
        Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
        for spec in strategies:
            if spec["kind"] == "route_b":
                spec["Q0"] = Q_fine
    else:
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
    out_path = os.path.join(workdir, f"{config.wandb.name}.h5")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
 
    with h5py.File(out_path, "w") as f:
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
 
    logging.info(f"evaluate_filters: wrote all evaluation data to {out_path}")
    return out_path


"""
Plotting stage for the modular DeepONet + EnKF Lorenz-96 evaluation pipeline.

This module is the plotting half that `evaluate_filters` (see
`eval_modular.py`) was split off from. It reads the HDF5 file written by
`evaluate_filters` -- keyed only by strategy `key`/`label`, so it works for
an arbitrary number of strategies, not just the historical DD/PI/Route-B
3-way set -- and reproduces every figure the old inline pipeline drew,
generalized to N strategies:

    1. Individual-trajectory PDFs
       One PDF per (IC, strategy): ground truth vs that strategy's
       ensemble mean +/- spread, with assimilated-observation markers.
       If `evaluate_filters` was run with S strategies and P trajectory
       ICs, this stage writes S * P PDFs.

    2. Pairwise batch-comparison PDFs
       For every unordered pair of strategies (A, B) -- C(S, 2) pairs
       total -- four PDFs are written:

         * calibration_A_vs_B.pdf   -- ensemble spread vs RMSE
                                       (simulation-time panel) + binned
                                       spread-skill scatter (2 graphs)
         * erf_A_vs_B.pdf           -- Error Reduction Factor per
                                       observation time (1 graph)
         * l2_A_vs_B.pdf            -- EnKF vs open-loop, time-mean
                                       relative L2 error (1 graph)
         * rmse_A_vs_B.pdf          -- prior vs posterior RMSE, no
                                       spread bands (1 graph)

       So going from S=2 (1 pair) to S=3 strategies (3 pairs) adds 2 more
       of each comparison PDF (the new pairs B-vs-C and A-vs-C), exactly
       as going from S=3 to S=4 (6 pairs) would add 3 more of each, etc.

Entry point: `plot_comparisons(h5_path, workdir=None, n_bins=10)`.
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
# 2a. EnKF vs open-loop, time-mean relative L2 (one graph, arbitrary #curves)
# ─────────────────────────────────────────────────────────────────────────
def _plot_l2_per_timestep(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],  # label -> (t_axis, l2_array)
    title: str,
    save_path: str,
    colors: dict[str, str] | None = None,
) -> None:
    """Plot average L2 error continuously across fine time stamps."""
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, (t_axis, l2_arr)) in enumerate(curves.items()):
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])
        ax.plot(t_axis, l2_arr, linewidth=1.8, label=label, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Mean relative L2 error (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _save(fig, save_path)
    logging.info(f"L2-vs-open-loop comparison plot saved to: {save_path}")


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
    fig.tight_layout(rect=[0, 0, 1, 0.98])
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
# Main entry point
# ─────────────────────────────────────────────────────────────────────────
def plot_comparisons(h5_path: str, workdir: str | None = None, n_bins: int = 10) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and generates:

      * one individual-trajectory PDF per (IC, strategy) -- S * P PDFs
        for S strategies and P trajectory ICs,
      * four pairwise batch-comparison PDFs per strategy pair -- calibration
        (2 graphs), ERF (1 graph), EnKF-vs-open-loop L2 (1 graph), and
        prior/posterior RMSE (1 graph) -- 4 * C(S, 2) PDFs total.

    Output layout (under ``workdir/figures/comparisons/``):

        individual_trajectories/trajectory_ic_<i>_<strategy_key>.pdf
        calibration_<keyA>_vs_<keyB>.pdf
        erf_<keyA>_vs_<keyB>.pdf
        l2_<keyA>_vs_<keyB>.pdf
        rmse_<keyA>_vs_<keyB>.pdf

    Parameters
    ----------
    h5_path : path to the HDF5 file written by `evaluate_filters`.
    workdir : base directory for outputs; defaults to the HDF5 file's
        parent directory.
    n_bins  : number of equal-population bins for the calibration
        binned spread-skill scatter.

    Returns the path of the ``figures/comparisons`` directory written.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(h5_path))

    save_dir = os.path.join(workdir, "figures", "comparisons")
    indiv_dir = os.path.join(save_dir, "individual_trajectories")
    os.makedirs(indiv_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        N = int(meta.attrs["N"])
        dt_window = float(meta.attrs["dt_window"])
        obs_every_n = int(meta.attrs["obs_every_n"])
        sigma_obs = float(meta.attrs["sigma_obs"])
        N_ens = int(meta.attrs["N_ens"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])

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
            F_val = float(ic_grp.attrs["F"])
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = list(map(tuple, obs_coords_raw)) if obs_coords_raw.size else []

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                x_est = sg["x_est"][:]
                x_std = sg["x_std"][:]
                l2_time_avg = float(sg.attrs["l2_time_avg"])

                save_path = os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf")
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=x_est, x_std=x_std,
                    ic_idx=ic_idx, F_val=F_val, strategy_label=label_of[key],
                    l2_time_avg=l2_time_avg, save_path=save_path,
                    N=N, dt_window=dt_window, obs_coords=obs_coords,
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
                f"(B={B} trajectories, N_ens={N_ens}, obs every {obs_every_n}th var, "
                f"σ_obs={sigma_obs})"
            ),
            save_path=os.path.join(save_dir, f"erf_{pair_name}.pdf"),
        )

        # -- EnKF vs open-loop, time-mean relative L2 (1 graph) -----------
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
                f"(B={B} trajectories, N_ens={N_ens}, obs every {obs_every_n}th var, "
                f"σ_obs={sigma_obs})"
            ),
            save_path=os.path.join(save_dir, f"rmse_{pair_name}.pdf"),
        )

        n_comparison_pdfs += 4

    logging.info(
        f"plot_comparisons: wrote {n_comparison_pdfs} pairwise comparison "
        f"PDFs ({len(strategy_pairs)} pairs × 4 categories) to {save_dir}"
    )
    return save_dir





# Non pairwise plotting
"""
Bulk (all-strategies-at-once) comparison plots for the EnKF / DeepONet
covariance-inflation evaluation pipeline.

This module is the counterpart to the pairwise `plot_comparisons`: instead
of one PDF per strategy PAIR per category (4 * C(S, 2) PDFs), it produces
exactly ONE PDF per category, with every strategy overlaid on it (4 PDFs
total, regardless of how many strategies S there are).

Individual-trajectory plots are unchanged from the pairwise module -- one
PDF per (IC, strategy), still S * P PDFs.

Decluttering strategy for the bulk plots
-----------------------------------------
Naively overlaying S strategies * 2 metrics (e.g. spread AND RMSE, or
prior AND posterior RMSE) on one axes gives 2*S lines, which gets messy
fast. Instead:

  * Calibration's spread/RMSE timeseries uses small multiples: one row
    per strategy, stacked vertically, each pairing that strategy's own
    RMS ensemble spread and EnKF RMSE on a single (small) axes, rather
    than two crowded all-strategies-at-once panels. This scales cleanly
    to large S (e.g. the 16-line multiplicative-inflation sweep) without
    the per-window "simulation time" tick strip, which just overcrowds
    the axes once there are many windows/strategies.
  * The prior/posterior-RMSE panels are still split into two
    side-by-side subplots (one metric each), so each subplot only ever
    has S lines, all sharing one color-per-strategy legend.
  * Every strategy gets a single, STABLE color, assigned by
    `_strategy_colors` (backed by `_distinct_colors`, which draws from
    matplotlib's tab20/tab20b/tab20c qualitative colormaps -- 60
    distinguishable swatches before any hue repeats) and reused across
    every panel and every bulk plot, so the reader only has to learn
    the strategy -> color mapping once.
  * Legends switch to two columns once there are more than ~4-5
    strategies, and error-bar / spread-band decorations are progressively
    faded (or dropped, for ERF bands beyond 4 strategies) as S grows, so
    the trend lines stay the visually dominant element.
  * The EnKF-vs-open-loop L2 plot is curve-count-agnostic and writes a
    two-page PDF: all curves, then a second page with the pure-
    propagator open-loop curves dropped so the (usually much smaller)
    filtered-strategy errors aren't squashed against the bottom of the
    axes by the open-loop curves' larger scale.

Markers and line widths are kept small/thin throughout so S-way overlays
stay legible well beyond the ~8-10 strategies the original side-by-side
panels were tuned for.
"""


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
# 2a. EnKF vs open-loop, time-mean relative L2 (already curve-count-
#     agnostic -- called with S curves). Shared by both the pairwise
#     (`plot_comparisons`) and bulk (`plot_comparisons_bulk`) modules.
# ─────────────────────────────────────────────────────────────────────────
def _plot_l2_per_timestep(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],  # label -> (t_axis, l2_array)
    title: str,
    save_path: str,
    colors: dict[str, str] | None = None,
) -> None:
    """
    Plot average L2 error continuously across fine time stamps, as a
    two-page PDF:

      page 1 -- every curve in `curves` (matches the original behavior).
      page 2 -- the same plot with the pure-propagator open-loop curves
        (any label ending in "open-loop", e.g. "dd open-loop") omitted,
        so the filtered-strategy curves aren't dwarfed by the open-loop
        curves' much larger error scale. Skipped if there are no
        open-loop curves to drop.
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
        ax.plot(window_idx, spread_window_of[key], marker="^", markersize=2.5,
                 linewidth=1.0, linestyle="-", color=c, label="RMS ensemble σ")
        ax.plot(window_idx, rmse_window_of[key], marker="o", markersize=2.5,
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
                rmss_b, rmse_b, yerr=rmse_s, fmt="o-", markersize=2.5, capsize=1.8,
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
    """ERF comparison for every strategy on ONE set of axes. +/-1 sigma
    bands are auto-dropped once there are more than 4 strategies, since
    overlapping fills stop conveying anything once they stack that deep."""
    S = len(strategy_keys)
    if show_bands is None:
        show_bands = S <= 4
    band_alpha = 0.15 if S <= 3 else 0.08

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in strategy_keys:
        c = colors[key]
        mean, std = erf_mean_of[key], erf_std_of[key]
        ax.plot(obs_times, mean, color=c, linewidth=1.1, marker="o", markersize=2.2,
                label=label_of[key])
        if show_bands:
            ax.fill_between(obs_times, mean - std, mean + std, color=c,
                             alpha=band_alpha, linewidth=0)

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.1,
               label="ERF = 1  (no reduction)")

    ax.set_xlabel("Observation time  t", fontsize=12)
    ax.set_ylabel("Error Reduction Factor  (prior RMSE / posterior RMSE)", fontsize=11)
    ax.set_title(f"{title}  (n = {n_traj} trajectories)", fontsize=13)
    ax.legend(fontsize=8, ncol=(2 if S > 5 else 1))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if not show_bands:
        ax.text(0.99, 0.02, "±1σ bands omitted for legibility (n strategies > 4)",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                color="#666666", style="italic")

    fig.tight_layout()
    _save(fig, save_path)
    logging.info(f"Bulk ERF plot ({S} strategies) saved to: {save_path}")

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
    """Prior/posterior RMSE for every strategy, split into two side-by-side
    panels (prior, posterior) rather than 2*S lines on one axes; both
    panels share a y-axis and a single strategy-color legend. Both panels
    use small dot markers (rather than squares) and thin lines so the
    S-way overlay stays legible."""
    S = len(strategy_keys)
    fig, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for key in strategy_keys:
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

    ax_prior.legend(fontsize=8, ncol=(2 if S > 4 else 1))

    fig.suptitle(f"{title}  (n = {n_traj} trajectories)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)
    logging.info(f"Bulk prior/posterior RMSE plot ({S} strategies) saved to: {save_path}")

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
# Main entry point
# ─────────────────────────────────────────────────────────────────────────
def plot_comparisons_bulk(h5_path: str, workdir: str | None = None, n_bins: int = 10) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and generates:

      * one individual-trajectory PDF per (IC, strategy) -- S * P PDFs
        for S strategies and P trajectory ICs (unchanged from
        `plot_comparisons`),
      * exactly FOUR batch-comparison PDFs total, each overlaying every
        strategy at once: calibration (3-panel), ERF, EnKF-vs-open-loop
        L2, and prior/posterior RMSE.

    This is the "all strategies on one plot" counterpart to
    `plot_comparisons`, which instead writes 4 * C(S, 2) pairwise PDFs.

    Output layout (under ``workdir/figures/comparisons_bulk/``):

        individual_trajectories/trajectory_ic_<i>_<strategy_key>.pdf
        calibration_all.pdf
        erf_all.pdf
        l2_all.pdf
        rmse_all.pdf

    Parameters
    ----------
    h5_path : path to the HDF5 file written by `evaluate_filters`.
    workdir : base directory for outputs; defaults to the HDF5 file's
        parent directory.
    n_bins  : number of equal-population bins for the calibration
        binned spread-skill scatter.

    Returns the path of the ``figures/comparisons_bulk`` directory written.
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(h5_path))

    save_dir = os.path.join(workdir, "figures", "comparisons_bulk")
    indiv_dir = os.path.join(save_dir, "individual_trajectories")
    os.makedirs(indiv_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        N = int(meta.attrs["N"])
        dt_window = float(meta.attrs["dt_window"])
        obs_every_n = int(meta.attrs["obs_every_n"])
        sigma_obs = float(meta.attrs["sigma_obs"])
        N_ens = int(meta.attrs["N_ens"])
        num_ics_traj = int(meta.attrs["num_ics_traj"])

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
            F_val = float(ic_grp.attrs["F"])
            x_true = ic_grp["x_true"][:]
            obs_coords_raw = ic_grp["obs_coords"][:]
            obs_coords = list(map(tuple, obs_coords_raw)) if obs_coords_raw.size else []

            for key in strategy_keys:
                sg = ic_grp[f"strategies/{key}"]
                x_est = sg["x_est"][:]
                x_std = sg["x_std"][:]
                l2_time_avg = float(sg.attrs["l2_time_avg"])

                save_path = os.path.join(indiv_dir, f"trajectory_ic_{ic_idx}_{key}.pdf")
                _plot_trajectory_individual(
                    t_ax=t_fine_traj, x_true=x_true, x_est=x_est, x_std=x_std,
                    ic_idx=ic_idx, F_val=F_val, strategy_label=label_of[key],
                    l2_time_avg=l2_time_avg, save_path=save_path,
                    N=N, dt_window=dt_window, obs_coords=obs_coords,
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
            f"(N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})"
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
            f"(N_ens={N_ens}, obs every {obs_every_n}th var, σ_obs={sigma_obs})"
        ),
        save_path=os.path.join(save_dir, "rmse_all.pdf"),
    )

    logging.info(
        f"plot_comparisons_bulk: wrote 4 bulk comparison PDFs "
        f"({S} strategies each) to {save_dir}"
    )
    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Main entry point — equilibrium (climatological/attractor) variance
# ─────────────────────────────────────────────────────────────────────────
def plot_equilibrium_variance(
    h5_path: str, workdir: str | None = None, log_scale: bool = True,
) -> str:
    """
    Reads the HDF5 file written by `evaluate_filters` and writes ONE PDF
    comparing, per state variable, the long-term equilibrium
    (climatological/attractor) variance of:

      * the reference/truth trajectory (the intrinsic variability of the
        unforced, unfiltered system),
      * each propagator's open-loop, "static" (unfiltered) physics
        rollout, and
      * every filtered EnKF strategy in the file,

    overlaid on a single set of axes -- the "static physics vs filtered
    strategies" steady-state diagnostic. This is the equilibrium-variance
    counterpart to `plot_comparisons_bulk`'s other "*_all.pdf" outputs,
    and works for any number of strategies S >= 1 (no pairwise blow-up,
    since every curve is compared against the one shared reference
    rather than against every other curve).

    Output: ``<workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf``

    Parameters
    ----------
    h5_path : path to the HDF5 file written by `evaluate_filters`. Must
        have been written by a version of `evaluate_filters` that
        includes equilibrium-variance data (see the module docstring's
        "/batch/reference" and "eqvar_mean"/"eqvar_std" fields); older
        files will raise a `KeyError` telling you to re-run it.
    workdir : base directory for outputs; defaults to the HDF5 file's
        parent directory.
    log_scale : plot the variance axis on a log scale (default) so that
        variance collapse (orders-of-magnitude drops) is easy to spot.

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
            f"{obs_every_n}th var, σ_obs={sigma_obs}, "
            f"burn-in={burn_in_frac:.0%} of window discarded)"
        ),
        save_path=save_path,
        log_scale=log_scale,
    )

    logging.info(f"plot_equilibrium_variance: wrote {save_path}")
    return save_dir



"""
Run the classic 3-way EnKF comparison
 
    1. DD  propagator + multiplicative inflation
    2. PI  propagator + multiplicative inflation
    3. PI  propagator + Route B (residual-scaled additive) inflation
 
using the two-stage modular pipeline (`evaluate_filters` + `plot_comparisons`)
instead of the old single-shot `evaluate_enkf_3_way`.
 
`evaluate_filters` called with no explicit `strategies`/`propagators` falls
back to `build_default_3way_strategies`, which reconstructs exactly the
DD-mult / PI-mult / PI-Route-B set the old function hardcoded (same
checkpoints, same inflation scaling), so this reproduces the old pipeline's
numbers -- just split into a "compute" stage (writes one HDF5 file) and a
"plot" stage (reads it and writes every PDF).
 
CLI usage
---------
    python run_3way_comparison.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run
 
Programmatic usage
-------------------
    from run_3way_comparison import run_3way_comparison
    save_dir = run_3way_comparison(config, workdir)
 
Outputs
-------
    <workdir>/<config.wandb.name>.h5                      -- evaluate_filters
    <workdir>/figures/comparisons/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons/calibration_*_vs_*.pdf   -- 3 pairs: dd_mult vs
    <workdir>/figures/comparisons/erf_*_vs_*.pdf              pi_mult, pi_mult vs
    <workdir>/figures/comparisons/l2_*_vs_*.pdf                pi_route_b, dd_mult vs
    <workdir>/figures/comparisons/rmse_*_vs_*.pdf               pi_route_b
"""
 
def run_3way_comparison(config, workdir: str, test_h5_path: str | None = None, n_bins: int = 10) -> str:
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


"""
Run a 4-way EnKF covariance-inflation comparison

    1. PI propagator + Multiplicative inflation
    2. PI propagator + Route B (residual-scaled additive) inflation
    3. PI propagator + plain Additive inflation (Route B with the
       constant floor term zeroed out, i.e. alpha=0)
    4. PI propagator + Relaxation-to-Prior Perturbations (RTPP)

using the same two-stage modular pipeline (`evaluate_filters` +
`plot_comparisons`) as `run_3way_comparison`. Unlike the 3-way run, the
propagator is held fixed at the PI (physics-informed) checkpoint across
all four strategies, so the comparison isolates the effect of the
inflation scheme rather than mixing it with the DD-vs-PI propagator
choice.

`evaluate_filters`'s `strategies=None` fallback only knows how to build
the historical DD-mult / PI-mult / PI-RouteB 3-way set, so this function
builds its own strategy/propagator set via `build_default_4way_strategies`
and passes them in explicitly (along with the `t_star_window` that set
was built against, since `evaluate_filters` only computes that itself on
the None/None fallback path).

RTPP caveat
-----------
The RTPP strategy's wiring through `evaluate_filters` (`run_enkf_smoother_rtpp`,
invoked here through the new "rtpp" branch in `build_batched_filters`) is
newer and less exercised than the multiplicative and Route B paths. Two
knobs control it, both read from `config.kf`:

    rtpp_alpha       -- RTPP relaxation-to-prior factor, in [0, 1].
                         0 = no relaxation (reduces to plain multiplicative
                         inflation at `rtpp_alpha_fine`); typical tuned
                         values are in the 0.5-0.9 range.
    rtpp_alpha_fine  -- multiplicative inflation applied in RTPP's (shared)
                         predict step. Defaults to 1.0 so that `rtpp_alpha`
                         is the only active inflation mechanism for this
                         strategy, isolating RTPP's relaxation from
                         multiplicative inflation for a fair comparison
                         against the other three. Raise it only if RTPP
                         alone is not enough to prevent ensemble collapse.

If RTPP's numbers look degenerate (e.g. the ensemble collapsing or
blowing up), start by sweeping `rtpp_alpha` before suspecting the
smoother wiring itself.

CLI usage
---------
    python run_4way_comparison.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run

Programmatic usage
-------------------
    from run_4way_comparison import run_4way_comparison
    save_dir = run_4way_comparison(config, workdir)

Outputs
-------
    <workdir>/<config.wandb.name>.h5                      -- evaluate_filters
    <workdir>/figures/comparisons/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons/calibration_*_vs_*.pdf   -- 6 pairs: every
    <workdir>/figures/comparisons/erf_*_vs_*.pdf              combination of
    <workdir>/figures/comparisons/l2_*_vs_*.pdf                {mult, route_b,
    <workdir>/figures/comparisons/rmse_*_vs_*.pdf               additive, rtpp}
"""

def run_4way_comparison(config, workdir: str, test_h5_path: str | None = None, n_bins: int = 10) -> str:
    """
    Runs the PI-only Mult / Route-B / Additive / RTPP 4-way EnKF
    inflation-strategy evaluation and writes every comparison figure.

    Returns the path of the `figures/comparisons` directory written by
    `plot_comparisons`.
    """
    os.makedirs(workdir, exist_ok=True)

    # ── EnKF / inflation configuration (mirrors evaluate_filters' own
    #    reads of config.kf, since we build the strategy set ourselves) ──
    N_ens = config.kf.get("N_ens", 50)
    alpha_coarse = config.kf.get("inflation_factor", 1.05)
    alpha_rb = config.kf.get("route_b_alpha", 1.0)
    beta_rb = config.kf.get("route_b_beta", 5.0)
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    alpha_rtpp = config.kf.get("rtpp_alpha", 0.5)
    alpha_fine_rtpp = config.kf.get("rtpp_alpha_fine", 1.0)

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)

    logging.info(
        "Building 4-way strategy set: PI+Mult / PI+RouteB / PI+Additive / PI+RTPP ..."
    )
    strategies, propagators, N, t_star_window = build_default_4way_strategies(
        config, N_ens, alpha_fine, alpha_rb, beta_rb, n_quad_rb, alpha_rtpp,
        alpha_fine_rtpp=alpha_fine_rtpp,
    )

    logging.info(
        "Stage 1/2: evaluate_filters — running Mult / RouteB / Additive / "
        "RTPP on shared data and writing the results HDF5 ..."
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


"""
Run a Mult.-inflation-factor calibration sweep: DD vs PI propagator

    For every value `alpha` in `config.kf.inflation_factor_list`, and for
    each of the two propagators (data-driven "dd", physics-informed
    "pi"), evaluate:

        DD propagator + Multiplicative inflation @ alpha
        PI propagator + Multiplicative inflation @ alpha

    using the same two-stage modular pipeline (`evaluate_filters` +
    `plot_comparisons`) as `run_3way_comparison` / `run_4way_comparison`.
    Unlike those, the inflation *scheme* is held fixed (multiplicative
    only); instead we sweep the inflation *factor* itself, crossed with
    the DD/PI propagator choice, so the comparison isolates calibration
    of alpha before moving on to the other inflation schemes (Route B,
    additive, RTPP).

    `evaluate_filters`'s `strategies=None` fallback only knows how to
    build the historical DD-mult / PI-mult / PI-RouteB 3-way set at a
    single hardcoded alpha, so this function builds its own
    strategy/propagator set via `build_mult_sweep_strategies` and passes
    it in explicitly (along with the `t_star_window` that set was built
    against, since `evaluate_filters` only computes that itself on the
    None/None fallback path).

Choosing `config.kf.inflation_factor_list`
--------------------------------------------
    Multiplicative inflation factors near 1.0 correspond to (almost) no
    correction to ensemble spread; values too small risk ensemble
    collapse / filter divergence over a long smoother window, values too
    large needlessly inflate the posterior and hurt RMSE. Standard
    operational EnKF practice keeps this factor in roughly the
    [1.00, 1.30] range, with the tuned optimum depending on N_ens, model
    error, and observation density.

    Since this codebase already defaults the earlier single-value knob
    `inflation_factor` to 1.05, a reasonable first-pass coarse grid
    bracketing that default is:

        config.kf.inflation_factor_list = [
            1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30,
        ]

    (1.00 is included as a no-inflation control.) Once this coarse sweep
    identifies an approximate optimum (e.g. by RMSE / calibration on the
    held-out window), re-run a finer sweep bracketing that value, e.g.
    `np.linspace(best - 0.03, best + 0.03, 7)`.

A note on plot_comparisons and sweep size
------------------------------------------
    `plot_comparisons` (as used by the 3-way/4-way runs) plots every
    pairwise combination of strategies. That's fine for 3 or 4 strategies
    (3 or 6 pairs), but this sweep produces
    `2 * len(inflation_factor_list)` strategies -- e.g. 9 alphas x 2
    propagators = 18 strategies = C(18, 2) = 153 pairs, x4 metric types.
    That's a lot of not-very-useful pairwise PDFs. Two options, depending
    on what `plot_comparisons` actually supports internally:
      (a) keep `inflation_factor_list` short (3-5 values) for the first
          pass so the pairwise grid stays legible, and widen it only for
          a final confirmation run around the winner;
      (b) write a dedicated "metric vs alpha" summary-curve plot (one
          line per propagator) that reads the same output HDF5 -- much
          more directly useful for calibration than pairwise comparisons,
          but not included here since it needs to match
          `evaluate_filters`'s HDF5 schema, which wasn't provided. Happy
          to add this if you share that schema / `plot_comparisons`'s
          signature.

CLI usage
---------
    python run_mult_inflation_sweep.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run

Programmatic usage
-------------------
    from run_mult_inflation_sweep import run_mult_inflation_sweep
    save_dir = run_mult_inflation_sweep(config, workdir)

Outputs
-------
    <workdir>/<config.wandb.name>.h5                            -- evaluate_filters
    <workdir>/figures/comparisons_bulk/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons_bulk/calibration_all.pdf      -- plot_comparisons_bulk:
    <workdir>/figures/comparisons_bulk/erf_all.pdf                  every strategy overlaid
    <workdir>/figures/comparisons_bulk/l2_all.pdf                   on one PDF per category
    <workdir>/figures/comparisons_bulk/rmse_all.pdf                 (see note above on size
                                                                     if using plot_comparisons
                                                                     instead, which is pairwise)
    <workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf -- plot_equilibrium_variance:
                                                                     reference (unfiltered truth)
                                                                     + each propagator's static
                                                                     open-loop physics vs every
                                                                     filtered strategy's steady-
                                                                     state variance, per variable
"""


def build_mult_sweep_strategies(config, N_ens, alpha_coarse_list, steps_per_window):
    """
    Builds a DD+Mult / PI+Mult inflation-factor sweep strategy set, all
    sharing two propagators -- the DD and PI checkpoints named in
    `config.wandb.name_dd` / `config.wandb.name_pi` (mirroring the
    `name_pi` field used by `build_default_4way_strategies`; adjust the
    `name_dd` lookup below if your config uses a different key) -- with
    one strategy pair per value in `alpha_coarse_list`:

        dd_mult_a<tag> -- DD propagator + multiplicative inflation at
                          alpha_coarse_list[i] (converted internally to
                          the fine-step-scaled `alpha_fine`).
        pi_mult_a<tag> -- same, PI propagator.

    Unlike `build_default_4way_strategies` (which builds one
    predict_fn/update_fn pair per *scheme*), here `predict_fn`/`update_fn`
    are built ONCE per propagator: multiplicative inflation doesn't
    change the EnKF closure itself, only the `alpha_fine` scalar carried
    alongside it in the strategy dict. So the sweep is cheap -- no extra
    model calls or checkpoint loads per alpha, just extra strategy dict
    entries reusing the same two closures.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)

    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_model")
    if not os.path.exists(ckpt_path_dd):
        ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params

    N = model_pi.N
    assert model_dd.N == N, (
        f"DD checkpoint state dim ({model_dd.N}) != PI checkpoint state "
        f"dim ({N}); can't share a strategy/propagator set across them."
    )

    predict_fn_dd, update_fn_dd = model_dd.make_enkf_fns(params_dd, N_ens=N_ens)
    predict_fn_pi, update_fn_pi = model_pi.make_enkf_fns(params_pi, N_ens=N_ens)

    strategies = []
    for alpha_coarse in alpha_coarse_list:
        alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
        tag = f"{alpha_coarse:g}".replace(".", "p")

        strategies.append(dict(
            key=f"dd_mult_a{tag}",
            label=f"DD + Mult. Infl. (\u03b1={alpha_coarse:g})",
            kind="standard", propagator="dd",
            predict_fn=predict_fn_dd, update_fn=update_fn_dd,
            alpha_fine=alpha_fine,
        ))
        strategies.append(dict(
            key=f"pi_mult_a{tag}",
            label=f"PI + Mult. Infl. (\u03b1={alpha_coarse:g})",
            kind="standard", propagator="pi",
            predict_fn=predict_fn_pi, update_fn=update_fn_pi,
            alpha_fine=alpha_fine,
        ))

    propagators = {
        "dd": (model_dd, params_dd),
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window

def run_mult_inflation_sweep(config, workdir: str, test_h5_path: str | None = None, n_bins: int = 10) -> str:
    """
    Runs the DD-Mult / PI-Mult multiplicative-inflation-factor sweep --
    one calibration pass per value in `config.kf.inflation_factor_list`,
    crossed with the DD and PI propagators -- and writes every comparison
    figure, including the steady-state equilibrium-variance diagnostic
    (static open-loop physics + reference truth vs every filtered
    strategy; see `plot_equilibrium_variance`).

    Returns the path of the `figures/comparisons_bulk` directory written
    by `plot_comparisons_bulk` (and also used by `plot_equilibrium_variance`).
    """
    os.makedirs(workdir, exist_ok=True)

    # ── EnKF / inflation configuration ──
    N_ens = config.kf.get("N_ens", 50)
    alpha_coarse_list = list(config.kf.get(
        "inflation_factor_list",
        [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30],  # see module docstring
    ))
    if len(alpha_coarse_list) == 0:
        raise ValueError("config.kf.inflation_factor_list is empty.")
    if not all(a > 0 for a in alpha_coarse_list):
        raise ValueError(
            f"config.kf.inflation_factor_list must be strictly positive, got {alpha_coarse_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = 2 * len(alpha_coarse_list)
    logging.info(
        f"Building Mult.-inflation sweep strategy set: DD+Mult / PI+Mult "
        f"x {len(alpha_coarse_list)} inflation factor(s) "
        f"({alpha_coarse_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 8:
        logging.warning(
            f"{n_strategies} strategies -> plot_comparisons will draw "
            f"C({n_strategies}, 2) = {n_strategies * (n_strategies - 1) // 2} "
            "pairwise comparisons per metric. Consider a shorter "
            "inflation_factor_list for a first pass (see module docstring)."
        )

    strategies, propagators, N, t_star_window = build_mult_sweep_strategies(
        config, N_ens, alpha_coarse_list, steps_per_window,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} DD/PI x "
        "alpha strategies on shared data and writing the results HDF5 "
        "(including the steady-state / equilibrium-variance data) ..."
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

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info(
        f"Stage 3/3: plot_equilibrium_variance — reading {h5_path} and writing "
        "the static-physics-vs-filtered-strategies equilibrium-variance PDF ..."
    )
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir

# ─────────────────────────────────────────────────────────────────────────
# Pure-additive-inflation-strength sweep: PI propagator only
# ─────────────────────────────────────────────────────────────────────────

def build_add_sweep_strategies(config, N_ens, alpha_list, steps_per_window):
    """
    Builds a PI + pure-additive-inflation-strength sweep strategy set,
    reusing the single PI checkpoint named in `config.wandb.name_pi` and
    ONE `make_route_b_enkf_fns` closure pair -- with one strategy per
    value in `alpha_list`:

        pi_add_a<tag> -- PI propagator + Route B inflation with `beta`
                          pinned to 0.0, i.e. the flow-dependent
                          `beta * ||rho||^2` term switched off, so the
                          residual-scaled Route B machinery degenerates
                          to *pure* additive inflation: a fixed-
                          covariance process-noise floor `alpha * Q0`
                          injected every fine step, with `alpha_list[i]`
                          setting the floor's strength.

    Route B is only ever exercised against the PI checkpoint elsewhere in
    this file (`build_default_3way_strategies`, `build_default_4way_strategies`
    both only call `model_pi.make_route_b_enkf_fns`) -- there's no
    DD-model counterpart wired up here -- so, mirroring that, this sweep
    is PI-only, unlike `build_mult_sweep_strategies`, which crosses its
    alpha sweep with both the DD and PI propagators. If
    `models.L96UDON_DD` does expose `make_route_b_enkf_fns` in your
    checkout, extending this to a DD+PI 2x sweep is a straightforward
    copy of the DD-loading block in `build_mult_sweep_strategies`.

    Like `build_mult_sweep_strategies`, `predict_fn`/`update_fn` are built
    ONCE and reused across every alpha -- Route B's `alpha`/`beta` are
    runtime scalars passed into `run_enkf_smoother_route_b` per strategy
    dict, not baked into the closure -- so the sweep is cheap: no extra
    model calls or checkpoint loads per alpha.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    # Pure additive inflation is Route B with beta pinned to 0.0 (see the
    # module-level strategy-spec docs: Route B's scale is
    # `alpha + beta * ||rho||^2`; zeroing beta drops the flow-dependent
    # term and leaves only the constant `alpha * Q0` floor).
    P0_sigma = config.kf.get("P0_sigma", 1.0)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    Q_coarse = jnp.eye(N) * Q0_sigma ** 2
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)

    strategies = []
    for alpha in alpha_list:
        tag = f"{alpha:g}".replace(".", "p")
        strategies.append(dict(
            key=f"pi_add_a{tag}",
            label=f"PI + Add. Infl. (\u03b1={alpha:g})",
            kind="route_b", propagator="pi",
            predict_fn=predict_fn_rb, update_fn=update_fn_rb,
            Q0=Q_fine, alpha=float(alpha), beta=0.0, n_quad=n_quad_rb,
        ))

    propagators = {
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window


"""
Run a pure-additive-inflation-strength sweep: PI propagator only

    For every value `alpha` in `config.kf.inflation_alpha_list`, evaluate:

        PI propagator + pure additive inflation @ alpha
            (Route B with beta pinned to 0.0, i.e. a fixed-covariance
            `alpha * Q0` process-noise floor injected every fine step,
            with no flow-dependent `||rho||^2` scaling)

    using the same two-stage modular pipeline (`evaluate_filters` +
    `plot_comparisons_bulk` + `plot_equilibrium_variance`) as
    `run_mult_inflation_sweep`. Unlike that sweep, which crosses its
    alpha sweep with both the DD and PI propagators, Route B (and so
    pure additive inflation) is only ever exercised against the PI
    checkpoint elsewhere in this file, so this sweep is PI-only -- see
    `build_add_sweep_strategies`.

    `evaluate_filters`'s `strategies=None` fallback only knows how to
    build the historical DD-mult / PI-mult / PI-RouteB 3-way set, so
    this function builds its own strategy/propagator set via
    `build_add_sweep_strategies` and passes it in explicitly (along with
    the `t_star_window` that set was built against).

Choosing `config.kf.inflation_alpha_list`
--------------------------------------------
    Pure additive inflation injects a FIXED-covariance process-noise
    perturbation `alpha * Q0` at every fine step, independent of the
    current ensemble spread -- unlike multiplicative inflation, which
    rescales the existing spread and so is a no-op at `alpha_coarse =
    1.0`. That means the natural "no correction" control for THIS sweep
    is `alpha = 0.0`, not 1.0 -- make sure it's included as a baseline.

    `Q0` itself is set by `config.kf.Q0_sigma` (a per-variable noise std,
    0.3 in the config values given alongside this sweep) via
    `Q0 = alpha * diag(Q0_sigma^2)` (fine-step-scaled). So `alpha` scales
    that base floor up or down; `route_b_alpha`'s own default of 1.0
    (used, together with `route_b_beta = 250.0` and `Q0_sigma = 0.3`,
    when Route B's flow-dependent term is ALSO active -- see
    `build_default_4way_strategies` / `run_4way_comparison`) is a
    reasonable center for this sweep once beta is zeroed out. A
    first-pass grid bracketing it, with a 0.0 control and headroom above
    1.0 since a pure floor (no flow-dependent term) may need more
    strength to match Route B's own calibration:

        config.kf.inflation_alpha_list = [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
        ]

    Once this coarse sweep identifies an approximate optimum (e.g. by
    RMSE / calibration on the held-out window), re-run a finer sweep
    bracketing that value, e.g. `np.linspace(best - 0.25, best + 0.25, 7)`
    (clipped at 0).

CLI usage
---------
    python run_add_inflation_sweep.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run

Programmatic usage
-------------------
    from run_add_inflation_sweep import run_add_inflation_sweep
    save_dir = run_add_inflation_sweep(config, workdir)

Outputs
-------
    <workdir>/<config.wandb.name>.h5                            -- evaluate_filters
    <workdir>/figures/comparisons_bulk/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons_bulk/calibration_all.pdf      -- plot_comparisons_bulk:
    <workdir>/figures/comparisons_bulk/erf_all.pdf                  every strategy overlaid
    <workdir>/figures/comparisons_bulk/l2_all.pdf                   on one PDF per category
    <workdir>/figures/comparisons_bulk/rmse_all.pdf                 (see note in
                                                                     run_mult_inflation_sweep's
                                                                     docstring on pairwise size
                                                                     if using plot_comparisons
                                                                     instead)
    <workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf -- plot_equilibrium_variance:
                                                                     reference (unfiltered truth)
                                                                     + PI's static open-loop
                                                                     physics vs every additive-
                                                                     strength strategy's steady-
                                                                     state variance, per variable
"""


def run_add_inflation_sweep(config, workdir: str, test_h5_path: str | None = None, n_bins: int = 10) -> str:
    """
    Runs the PI-only pure-additive-inflation-strength sweep -- one
    calibration pass per value in `config.kf.inflation_alpha_list`, with
    Route B's beta pinned to 0.0 so the swept `alpha` scales a fixed
    `Q0`-based noise floor rather than a residual-scaled term -- and
    writes every comparison figure, including the steady-state
    equilibrium-variance diagnostic (static open-loop physics + reference
    truth vs every filtered strategy; see `plot_equilibrium_variance`).

    Returns the path of the `figures/comparisons_bulk` directory written
    by `plot_comparisons_bulk` (and also used by `plot_equilibrium_variance`).
    """
    os.makedirs(workdir, exist_ok=True)

    # ── EnKF / inflation configuration ──
    N_ens = config.kf.get("N_ens", 50)
    alpha_list = list(config.kf.get(
        "inflation_alpha_list",
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],  # see module docstring
    ))
    if len(alpha_list) == 0:
        raise ValueError("config.kf.inflation_alpha_list is empty.")
    if not all(a >= 0 for a in alpha_list):
        raise ValueError(
            f"config.kf.inflation_alpha_list must be non-negative, got {alpha_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = len(alpha_list)
    logging.info(
        f"Building Additive-inflation sweep strategy set: PI + pure "
        f"additive inflation (Route B, beta=0) x {n_strategies} alpha "
        f"value(s) ({alpha_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one plot_comparisons_bulk "
            "figure may get crowded. Consider a shorter inflation_alpha_list "
            "for a first pass (see module docstring)."
        )

    strategies, propagators, N, t_star_window = build_add_sweep_strategies(
        config, N_ens, alpha_list, steps_per_window,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} PI + "
        "additive-alpha strategies on shared data and writing the results "
        "HDF5 (including the steady-state / equilibrium-variance data) ..."
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

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info(
        f"Stage 3/3: plot_equilibrium_variance — reading {h5_path} and writing "
        "the static-physics-vs-filtered-strategies equilibrium-variance PDF ..."
    )
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir




# ─────────────────────────────────────────────────────────────────────────
# RTPP relaxation-factor sweep: DD vs PI propagator
# ─────────────────────────────────────────────────────────────────────────

def build_rtpp_sweep_strategies(config, N_ens, alpha_rtpp_list, alpha_fine_rtpp):
    """
    Builds a DD+RTPP / PI+RTPP relaxation-factor sweep strategy set, all
    sharing two propagators -- the DD and PI checkpoints named in
    `config.wandb.name_dd` / `config.wandb.name_pi` -- with one strategy
    pair per value in `alpha_rtpp_list`:

        dd_rtpp_a<tag> -- DD propagator + RTPP at relaxation factor
                           alpha_rtpp_list[i], with the (shared)
                           predict-step multiplicative inflation pinned
                           to `alpha_fine_rtpp` (default 1.0, i.e. off)
                           so the swept `alpha_rtpp` is the only active
                           inflation mechanism.
        pi_rtpp_a<tag> -- same, PI propagator.

    Note: `make_rtpp_enkf_fns` is only ever exercised against the PI
    checkpoint elsewhere in this file (`build_default_4way_strategies`
    calls `model_pi.make_rtpp_enkf_fns` only) -- unlike `make_enkf_fns`,
    which both `models.L96UDON` and `models.L96UDON_DD` already expose
    and which `build_mult_sweep_strategies` calls on both. This function
    assumes `models.L96UDON_DD` also exposes `make_rtpp_enkf_fns` (with
    the same `(params, N_ens=...)` signature as its `make_enkf_fns`), by
    direct analogy with how `make_enkf_fns` is shared across both model
    classes. If that method isn't implemented on the DD model yet in
    your checkout, add it there first (mirroring `make_rtpp_enkf_fns` on
    `L96UDON`) -- this function's DD-loading/closure-building code
    doesn't need to change once it is.

    Like `build_mult_sweep_strategies`, `predict_fn`/`update_fn` are
    built ONCE per propagator and reused across every alpha_rtpp -- RTPP's
    relaxation factor is a runtime scalar passed into
    `run_enkf_smoother_rtpp` per strategy dict, not baked into the
    closure -- so the sweep is cheap: no extra model calls or checkpoint
    loads per alpha_rtpp.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)

    logging.info("Loading DD model...")
    model_dd = models.L96UDON_DD(config, t_star_window)
    ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_model")
    if not os.path.exists(ckpt_path_dd):
        ckpt_path_dd = os.path.join(os.getcwd(), config.wandb.name_dd, "ckpt", "udon_dd_model")
    model_dd.state = restore_checkpoint(model_dd.state, ckpt_path_dd)
    params_dd = model_dd.state.params

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params

    N = model_pi.N
    assert model_dd.N == N, (
        f"DD checkpoint state dim ({model_dd.N}) != PI checkpoint state "
        f"dim ({N}); can't share a strategy/propagator set across them."
    )

    predict_fn_dd, update_fn_dd = model_dd.make_rtpp_enkf_fns(params_dd, N_ens=N_ens)
    predict_fn_pi, update_fn_pi = model_pi.make_rtpp_enkf_fns(params_pi, N_ens=N_ens)

    strategies = []
    for alpha_rtpp in alpha_rtpp_list:
        tag = f"{alpha_rtpp:g}".replace(".", "p")

        strategies.append(dict(
            key=f"dd_rtpp_a{tag}",
            label=f"DD + RTPP (\u03b1={alpha_rtpp:g})",
            kind="rtpp", propagator="dd",
            predict_fn=predict_fn_dd, update_fn=update_fn_dd,
            alpha_fine=alpha_fine_rtpp, alpha_rtpp=alpha_rtpp,
        ))
        strategies.append(dict(
            key=f"pi_rtpp_a{tag}",
            label=f"PI + RTPP (\u03b1={alpha_rtpp:g})",
            kind="rtpp", propagator="pi",
            predict_fn=predict_fn_pi, update_fn=update_fn_pi,
            alpha_fine=alpha_fine_rtpp, alpha_rtpp=alpha_rtpp,
        ))

    propagators = {
        "dd": (model_dd, params_dd),
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window


"""
Run an RTPP relaxation-factor sweep: DD vs PI propagator

    For every value `alpha_rtpp` in `config.kf.rtpp_alpha_list`, and for
    each of the two propagators (data-driven "dd", physics-informed
    "pi"), evaluate:

        DD propagator + RTPP @ alpha_rtpp
        PI propagator + RTPP @ alpha_rtpp

    using the same two-stage modular pipeline (`evaluate_filters` +
    `plot_comparisons_bulk` + `plot_equilibrium_variance`) as
    `run_mult_inflation_sweep` / `run_add_inflation_sweep`. As in those,
    the inflation *scheme* is held fixed (RTPP only, with its own
    predict-step multiplicative inflation pinned at `rtpp_alpha_fine`,
    default 1.0 -- see `build_default_4way_strategies`'s docstring);
    instead we sweep RTPP's relaxation-to-prior factor itself, crossed
    with the DD/PI propagator choice.

    `evaluate_filters`'s `strategies=None` fallback only knows how to
    build the historical DD-mult / PI-mult / PI-RouteB 3-way set, so
    this function builds its own strategy/propagator set via
    `build_rtpp_sweep_strategies` and passes it in explicitly (along
    with the `t_star_window` that set was built against).

    Note: this sweep assumes `models.L96UDON_DD` exposes
    `make_rtpp_enkf_fns` -- see the caveat in
    `build_rtpp_sweep_strategies`'s docstring.

Choosing `config.kf.rtpp_alpha_list`
--------------------------------------------
    RTPP relaxes each posterior ensemble perturbation partway back toward
    its (larger, pre-update) prior perturbation:
    `x'_post <- (1 - alpha_rtpp) * x'_post + alpha_rtpp * x'_prior`. So
    `alpha_rtpp = 0` is the "no correction" control (posterior spread
    used as-is -- the natural RTPP analogue of additive inflation's
    `alpha = 0.0` control, NOT multiplicative inflation's `alpha_coarse =
    1.0`), and `alpha_rtpp = 1` discards the update's spread reduction
    entirely, fully reverting to the prior's spread every step. Values
    are only meaningful in `[0, 1]`.

    Since this codebase already defaults the single-value knob
    `rtpp_alpha` to 0.5, and the relaxation literature (Zhang, Snyder &
    Sacher 2004; typical operational EnKF practice) usually finds tuned
    values somewhere in the `[0.5, 0.9]` band -- lower values under-
    correct spread collapse, values pushed toward 1 increasingly ignore
    the filter's own update -- a reasonable first-pass grid bracketing
    the default, denser in that typical band, with a `0.0` no-relaxation
    control:

        config.kf.rtpp_alpha_list = [
            0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]

    Once this coarse sweep identifies an approximate optimum (e.g. by
    RMSE / calibration on the held-out window), re-run a finer sweep
    bracketing that value, e.g. `np.linspace(best - 0.1, best + 0.1, 5)`
    (clipped to `[0, 1]`).

A note on plot_comparisons_bulk and sweep size
------------------------------------------------
    As in `run_mult_inflation_sweep`, this produces
    `2 * len(rtpp_alpha_list)` strategies -- e.g. 8 alphas x 2
    propagators = 16 strategies overlaid per figure. Keep
    `rtpp_alpha_list` short (4-6 values) for a first pass if the overlay
    gets too crowded to read, and widen it only for a final confirmation
    run around the winner.

CLI usage
---------
    python run_rtpp_inflation_sweep.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run

Programmatic usage
-------------------
    from run_rtpp_inflation_sweep import run_rtpp_inflation_sweep
    save_dir = run_rtpp_inflation_sweep(config, workdir)

Outputs
-------
    <workdir>/<config.wandb.name>.h5                            -- evaluate_filters
    <workdir>/figures/comparisons_bulk/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons_bulk/calibration_all.pdf      -- plot_comparisons_bulk:
    <workdir>/figures/comparisons_bulk/erf_all.pdf                  every strategy overlaid
    <workdir>/figures/comparisons_bulk/l2_all.pdf                   on one PDF per category
    <workdir>/figures/comparisons_bulk/rmse_all.pdf                 (see note above on size)
    <workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf -- plot_equilibrium_variance:
                                                                     reference (unfiltered truth)
                                                                     + each propagator's static
                                                                     open-loop physics vs every
                                                                     RTPP-alpha strategy's steady-
                                                                     state variance, per variable
"""


def run_rtpp_inflation_sweep(config, workdir: str, test_h5_path: str | None = None, n_bins: int = 10) -> str:
    """
    Runs the DD-RTPP / PI-RTPP relaxation-factor sweep -- one calibration
    pass per value in `config.kf.rtpp_alpha_list`, crossed with the DD
    and PI propagators, with the shared predict-step multiplicative
    inflation pinned at `config.kf.rtpp_alpha_fine` (default 1.0) so the
    swept relaxation factor is the only active inflation mechanism -- and
    writes every comparison figure, including the steady-state
    equilibrium-variance diagnostic (static open-loop physics + reference
    truth vs every filtered strategy; see `plot_equilibrium_variance`).

    Returns the path of the `figures/comparisons_bulk` directory written
    by `plot_comparisons_bulk` (and also used by `plot_equilibrium_variance`).
    """
    os.makedirs(workdir, exist_ok=True)

    # ── EnKF / inflation configuration ──
    N_ens = config.kf.get("N_ens", 50)
    alpha_fine_rtpp = config.kf.get("rtpp_alpha_fine", 1.0)
    alpha_rtpp_list = list(config.kf.get(
        "rtpp_alpha_list",
        [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # see module docstring
    ))
    if len(alpha_rtpp_list) == 0:
        raise ValueError("config.kf.rtpp_alpha_list is empty.")
    if not all(0.0 <= a <= 1.0 for a in alpha_rtpp_list):
        raise ValueError(
            f"config.kf.rtpp_alpha_list must lie in [0, 1], got {alpha_rtpp_list}"
        )

    n_strategies = 2 * len(alpha_rtpp_list)
    logging.info(
        f"Building RTPP sweep strategy set: DD+RTPP / PI+RTPP "
        f"x {len(alpha_rtpp_list)} relaxation factor(s) "
        f"({alpha_rtpp_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 8:
        logging.warning(
            f"{n_strategies} strategies -> the plot_comparisons_bulk overlay "
            "may get crowded. Consider a shorter rtpp_alpha_list for a "
            "first pass (see module docstring)."
        )

    strategies, propagators, N, t_star_window = build_rtpp_sweep_strategies(
        config, N_ens, alpha_rtpp_list, alpha_fine_rtpp,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} DD/PI x "
        "rtpp_alpha strategies on shared data and writing the results HDF5 "
        "(including the steady-state / equilibrium-variance data) ..."
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

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info(
        f"Stage 3/3: plot_equilibrium_variance — reading {h5_path} and writing "
        "the static-physics-vs-filtered-strategies equilibrium-variance PDF ..."
    )
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────
# Route B (modified additive) beta sweep: PI propagator only
# ─────────────────────────────────────────────────────────────────────────

def build_route_b_sweep_strategies(config, N_ens, beta_list, alpha_rb_fixed, steps_per_window):
    """
    Builds a PI + Route B beta-sweep strategy set, reusing the single PI
    checkpoint named in `config.wandb.name_pi` and ONE
    `make_route_b_enkf_fns` closure pair -- with one strategy per value
    in `beta_list`, all sharing the SAME fixed `alpha_rb_fixed` constant-
    floor term:

        pi_route_b_b<tag> -- PI propagator + Route B inflation, scale =
                              alpha_rb_fixed + beta * ||rho||^2, with
                              beta_list[i] as the flow-dependent
                              residual-scaling strength and
                              `alpha_rb_fixed` held fixed across the
                              whole sweep.

    This is the mirror image of `build_add_sweep_strategies`, which pins
    `beta=0.0` and sweeps the constant floor `alpha`: here `alpha` is
    pinned (to whatever `alpha_rb_fixed` the caller passes in) and
    `beta` -- the coefficient on the flow-dependent `||rho||^2` term --
    is swept instead. See `run_route_b_inflation_sweep`'s docstring for
    how to choose `alpha_rb_fixed` (0.0 to isolate the residual-scaled
    term alone, or the best alpha found by `run_add_inflation_sweep` to
    layer Route B's flow-dependent term on top of an already-calibrated
    floor -- both are worth running).

    As with the additive sweep, Route B is only ever exercised against
    the PI checkpoint elsewhere in this file (`build_default_3way_strategies`,
    `build_default_4way_strategies` both only call
    `model_pi.make_route_b_enkf_fns`), so this sweep is PI-only -- see
    the equivalent DD-extension caveat in `build_add_sweep_strategies`'s
    docstring.

    Like `build_add_sweep_strategies`, `predict_fn`/`update_fn` are built
    ONCE and reused across every beta -- Route B's `alpha`/`beta` are
    runtime scalars passed into `run_enkf_smoother_route_b` per strategy
    dict, not baked into the closure -- so the sweep is cheap: no extra
    model calls or checkpoint loads per beta.

    Returns (strategies, propagators, N, t_star_window).
    """
    dt_window = float(config.get("dt_window", 0.25))
    dt_integration = config.eval.get("dt_integration", 0.005)
    time_steps = int(round(dt_window / dt_integration)) + 1
    t_star_window = jnp.linspace(0.0, dt_window, time_steps)

    logging.info("Loading PI model...")
    model_pi = models.L96UDON(config, t_star_window)
    ckpt_path_pi = os.path.join(os.getcwd(), config.wandb.name_pi, "ckpt", "udon_model")
    model_pi.state = restore_checkpoint(model_pi.state, ckpt_path_pi)
    params_pi = model_pi.state.params
    N = model_pi.N

    P0_sigma = config.kf.get("P0_sigma", 1.0)
    Q0_sigma = config.kf.get("Q0_sigma", P0_sigma)
    n_quad_rb = config.kf.get("route_b_n_quad", 3)
    Q_coarse = jnp.eye(N) * Q0_sigma ** 2
    Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)

    predict_fn_rb, update_fn_rb = model_pi.make_route_b_enkf_fns(params_pi, N_ens=N_ens)

    strategies = []
    for beta in beta_list:
        tag = f"{beta:g}".replace(".", "p")
        strategies.append(dict(
            key=f"pi_route_b_b{tag}",
            label=f"PI + Route B (\u03b1={alpha_rb_fixed:g}, \u03b2={beta:g})",
            kind="route_b", propagator="pi",
            predict_fn=predict_fn_rb, update_fn=update_fn_rb,
            Q0=Q_fine, alpha=float(alpha_rb_fixed), beta=float(beta), n_quad=n_quad_rb,
        ))

    propagators = {
        "pi": (model_pi, params_pi),
    }
    return strategies, propagators, N, t_star_window


"""
Run a Route B (modified additive) beta sweep: PI propagator only

    For every value `beta` in `config.kf.route_b_beta_list`, with the
    constant floor pinned at `alpha_rb_fixed`, evaluate:

        PI propagator + Route B inflation @ (alpha_rb_fixed, beta)
            (scale = alpha_rb_fixed + beta * ||rho||^2, injected as
            process noise `scale * Q0` every fine step)

    using the same two-stage modular pipeline (`evaluate_filters` +
    `plot_comparisons_bulk` + `plot_equilibrium_variance`) as
    `run_mult_inflation_sweep` / `run_add_inflation_sweep` /
    `run_rtpp_inflation_sweep`. As with those, Route B is only ever
    exercised against the PI checkpoint elsewhere in this file, so this
    sweep is PI-only -- see `build_route_b_sweep_strategies`.

    `evaluate_filters`'s `strategies=None` fallback only knows how to
    build the historical DD-mult / PI-mult / PI-RouteB 3-way set, so
    this function builds its own strategy/propagator set via
    `build_route_b_sweep_strategies` and passes it in explicitly (along
    with the `t_star_window` that set was built against).

Choosing `alpha_rb_fixed`
--------------------------------------------
    Route B's scale is `alpha + beta * ||rho||^2`: `alpha` is a constant
    floor, `beta` scales an additional flow-dependent term that grows
    with the current residual norm. Two different, complementary
    experiments are worth running here, both supported by the same
    `alpha_rb_fixed` argument:

    1. `alpha_rb_fixed = 0.0` -- isolates the flow-dependent term in
       pure form: no constant floor at all, inflation comes ENTIRELY
       from `beta * ||rho||^2`. Useful to see whether residual-scaling
       alone can substitute for a floor, but be aware it means zero
       inflation whenever the residual is small (calm periods), which
       risks under-dispersion / eventual filter divergence in exactly
       those windows -- treat this as a diagnostic sweep, not
       necessarily a deployable operating point.

    2. `alpha_rb_fixed = <best alpha from run_add_inflation_sweep>` --
       holds the constant floor at whatever `run_add_inflation_sweep`
       already found to work well in the beta=0 (pure additive) setting,
       and asks whether layering Route B's flow-dependent term ON TOP of
       that calibrated floor further improves things. This is the more
       practically relevant question, and the reason Route B exists as a
       *modification* of plain additive inflation rather than a
       standalone scheme. If you haven't run the additive sweep yet, the
       codebase's own default `route_b_alpha = 1.0` is a reasonable
       stand-in floor to start from.

    Recommendation: run this function twice, once at each of the above,
    rather than trying to fold both into a single 2-D alpha x beta grid
    -- a joint grid multiplies strategy count fast (n_alpha x n_beta) and
    clutters the `plot_comparisons_bulk` overlays; two clean 1-D sweeps
    are much easier to read and compare against the additive-only
    baseline.

    Leave `alpha_rb_fixed=None` (the default) to fall back to whatever
    `config.kf.route_b_alpha` currently holds -- set that to 0.0 or to
    your best additive-sweep alpha before each call, or pass
    `alpha_rb_fixed` explicitly to override the config without mutating
    it.

Choosing `config.kf.route_b_beta_list`
--------------------------------------------
    The codebase's own default `route_b_beta` is 250.0 -- ||rho||^2 (a
    squared residual norm over N=40 state variables) is typically a
    small-to-moderate number, so a coefficient in the hundreds is needed
    for the flow-dependent term to compete with an O(1) alpha floor
    scaling an O(Q0_sigma^2) = O(0.09) base covariance. A first-pass grid
    bracketing that default, with a 0.0 control (recovering whatever
    `alpha_rb_fixed` alone gives -- directly comparable to the additive
    sweep's own point at that alpha) and headroom both below and above:

        config.kf.route_b_beta_list = [
            0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0,
        ]

    Once this coarse sweep identifies an approximate optimum (e.g. by
    RMSE / calibration on the held-out window), re-run a finer sweep
    bracketing that value, e.g. `np.linspace(best * 0.5, best * 1.5, 7)`
    (clipped at 0).

CLI usage
---------
    python run_route_b_inflation_sweep.py \\
        --config=configs/your_config.py \\
        --workdir=./results/my_run

Programmatic usage
-------------------
    from run_route_b_inflation_sweep import run_route_b_inflation_sweep
    # 1) isolate the flow-dependent term alone:
    save_dir_a = run_route_b_inflation_sweep(config, workdir_a, alpha_rb_fixed=0.0)
    # 2) layer it on top of the best additive-sweep alpha:
    save_dir_b = run_route_b_inflation_sweep(config, workdir_b, alpha_rb_fixed=best_add_alpha)

Outputs
-------
    <workdir>/<config.wandb.name>.h5                            -- evaluate_filters
    <workdir>/figures/comparisons_bulk/individual_trajectories/ -- per-(IC, strategy) PDFs
    <workdir>/figures/comparisons_bulk/calibration_all.pdf      -- plot_comparisons_bulk:
    <workdir>/figures/comparisons_bulk/erf_all.pdf                  every strategy overlaid
    <workdir>/figures/comparisons_bulk/l2_all.pdf                   on one PDF per category
    <workdir>/figures/comparisons_bulk/rmse_all.pdf
    <workdir>/figures/comparisons_bulk/equilibrium_variance_all.pdf -- plot_equilibrium_variance:
                                                                     reference (unfiltered truth)
                                                                     + PI's static open-loop
                                                                     physics vs every Route-B-beta
                                                                     strategy's steady-state
                                                                     variance, per variable
"""


def run_route_b_inflation_sweep(
    config, workdir: str, alpha_rb_fixed: float | None = None,
    test_h5_path: str | None = None, n_bins: int = 10,
) -> str:
    """
    Runs the PI-only Route B beta sweep -- one calibration pass per value
    in `config.kf.route_b_beta_list`, with the constant floor pinned at
    `alpha_rb_fixed` (falls back to `config.kf.route_b_alpha` if left
    `None`) -- and writes every comparison figure, including the
    steady-state equilibrium-variance diagnostic (static open-loop
    physics + reference truth vs every filtered strategy; see
    `plot_equilibrium_variance`).

    See the module docstring above for why you'll typically want to call
    this twice -- once with `alpha_rb_fixed=0.0` to isolate the flow-
    dependent term, once with `alpha_rb_fixed` set to the best alpha
    found by `run_add_inflation_sweep` to layer Route B on top of an
    already-calibrated additive floor.

    Returns the path of the `figures/comparisons_bulk` directory written
    by `plot_comparisons_bulk` (and also used by `plot_equilibrium_variance`).
    """
    os.makedirs(workdir, exist_ok=True)

    # ── EnKF / inflation configuration ──
    N_ens = config.kf.get("N_ens", 50)
    if alpha_rb_fixed is None:
        alpha_rb_fixed = config.kf.get("route_b_alpha", 0.0) # Default: pure amplified residual error
    beta_list = list(config.kf.get(
        "route_b_beta_list",
        [0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0],  # see module docstring
    ))
    if len(beta_list) == 0:
        raise ValueError("config.kf.route_b_beta_list is empty.")
    if not all(b >= 0 for b in beta_list):
        raise ValueError(
            f"config.kf.route_b_beta_list must be non-negative, got {beta_list}"
        )

    DT_WINDOW = float(config.get("dt_window", 0.25))
    DT_FINE = float(config.kf.get("dt_fine", DT_WINDOW))
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)

    n_strategies = len(beta_list)
    logging.info(
        f"Building Route B beta sweep strategy set: PI + Route B "
        f"(alpha fixed at {alpha_rb_fixed:g}) x {n_strategies} beta "
        f"value(s) ({beta_list}) -> {n_strategies} strategies ..."
    )
    if n_strategies > 12:
        logging.warning(
            f"{n_strategies} strategies overlaid on one plot_comparisons_bulk "
            "figure may get crowded. Consider a shorter route_b_beta_list "
            "for a first pass (see module docstring)."
        )

    strategies, propagators, N, t_star_window = build_route_b_sweep_strategies(
        config, N_ens, beta_list, alpha_rb_fixed, steps_per_window,
    )

    logging.info(
        f"Stage 1/3: evaluate_filters — running {len(strategies)} PI + "
        "Route B beta strategies on shared data and writing the results "
        "HDF5 (including the steady-state / equilibrium-variance data) ..."
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

    logging.info(f"Stage 2/3: plot_comparisons_bulk — reading {h5_path} and writing figures ...")
    save_dir = plot_comparisons_bulk(h5_path=h5_path, workdir=workdir, n_bins=n_bins)
    logging.info(f"  wrote figures to {save_dir}")

    logging.info(
        f"Stage 3/3: plot_equilibrium_variance — reading {h5_path} and writing "
        "the static-physics-vs-filtered-strategies equilibrium-variance PDF ..."
    )
    plot_equilibrium_variance(h5_path=h5_path, workdir=workdir)
    logging.info(f"  wrote {os.path.join(save_dir, 'equilibrium_variance_all.pdf')}")

    return save_dir


 