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

    /batch  (attrs: B, obs_every_n, sigma_obs, N_ens, dt_obs)
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
        strategies/{key}/route_b_scale_mean  (n_fine,)      -- only present for kind == "route_b"
        strategies/{key}/route_b_scale_std   (n_fine,)
        open_loop/{propagator}/t             (n_ol,)
        open_loop/{propagator}/l2_dense_mean (n_ol,)
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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
        per_device = -(-B // n_devices)
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
    Runs every strategy in `strategies` on the same data (same ICs, same
    noisy-observation draws, same initial ensembles) and stores every
    number needed to later reproduce:

      * individual trajectories + time-avg error over the 40 variables
        (title carries F),
      * RMSE with spread,
      * calibration plots (timeseries + binned spread-skill scatter),
      * Error Reduction Factor,
      * batch time-mean L2 error,
      * prior vs posterior RMSE with the observation error level,

    into a single HDF5 file at
    ``os.path.join(workdir, f"{config.wandb.name}.h5")``.

    If `strategies`/`propagators` are not supplied, falls back to the
    original DD-mult / PI-mult / PI-RouteB 3-way set (see
    `build_default_3way_strategies`), so this is a drop-in replacement
    for the evaluation half of `evaluate_enkf_3_way`.

    Returns the path of the HDF5 file written.
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

    # ── 1. Load the long test trajectories and forcing parameters ──────
    if test_h5_path is None:
        test_h5_path = "data/l96_forcing_test.h5"

    with h5py.File(test_h5_path, "r") as f:
        u_test = f["u"][:]
        t_test = f["t"][:]
        F_test = f["F"][:]

    logging.info(f"JAX sees {jax.local_device_count()} local device(s): {jax.local_devices()}")

    trajectory_windows = config.eval.get("trajectory_windows", 200)
    batch_windows = config.eval.get("windows", 200)
    num_ics_eval = config.eval.get("num_ics", u_test.shape[0])
    enkf_batch_size = config.kf.get("batch_l2_size", 200)

    # Shared inflation / process-noise scaling (used by the default
    # 3-way set; custom strategy lists already carry their own).
    steps_per_window = steps_per_window_exact(DT_WINDOW, DT_FINE)
    alpha_fine = scale_inflation_for_fine_steps(alpha_coarse, steps_per_window)
    Q_coarse = None
    Q_fine = None

    if strategies is None or propagators is None:
        Q_coarse = jnp.eye(40) * Q0_sigma ** 2  # N filled in once model loads; recomputed below
        strategies, propagators, N, t_star_window = build_default_3way_strategies(
            config, N_ens, alpha_fine, None, alpha_rb, beta_rb, n_quad_rb,
        )
        Q_coarse = jnp.eye(N) * Q0_sigma ** 2
        Q_fine = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
        # Re-inject the correctly-sized Q_fine into the route_b strategy.
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

    # ── 2. Per-IC single-trajectory data (for individual-trajectory plots) ──
    num_plots = min(config.saving.total_plots, u_test.shape[0])
    total_time_traj = trajectory_windows * DT_WINDOW

    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time_traj, dt_fine=DT_FINE, dt_obs=DT_OBS,
    )
    obs_step_indices = jnp.array(obs_step_indices)

    # PASS 1: sequential SciPy ground-truth solves — exact gen_data.py solver
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

    # PASS 2: batched, concurrent GPU execution for every strategy at once
    x_true_fine_batch = jnp.stack(x_true_fine_list)
    x_true_at_obs_batch = jnp.stack(x_true_at_obs_list)
    u0_batch_plots = jnp.array(u_test[:num_plots, 0, :])
    F_batch_plots = jnp.array(F_test[:num_plots])
    keys_batch_plots = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_plots))

    batched_traj_fn = build_batched_filters(
        strategies, N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R,
        DT_FINE, DT_WINDOW, total_fine_steps, obs_step_indices,
    )
    outputs_traj, y_obs_traj, idx_vars_traj = batched_traj_fn(
        keys_batch_plots, u0_batch_plots, F_batch_plots,
        x_true_at_obs_batch, dynamic_vars, specify_obs_idx,
    )

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

    # ── 3. Batch-averaged metrics (for RMSE/spread, calibration, ERF, L2, prior/post) ──
    B = min(num_ics_eval, enkf_batch_size, u_test.shape[0])
    u0_batch = u_test[:B, 0, :]
    dt_test = float(t_test[1] - t_test[0])

    total_time_batch = batch_windows * DT_WINDOW
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch, dt_fine=DT_FINE, dt_obs=DT_OBS,
    )
    obs_step_indices_batch = jnp.array(obs_step_indices_batch)

    T_obs = len(obs_step_indices_batch)
    obs_times_batch = np.array([(k + 1) * DT_OBS for k in range(T_obs)])

    fine_stride = int(round(DT_FINE / dt_test))
    n_fine_pts = total_fine_steps_batch * fine_stride + 1

    x_true_fine_batch2 = u_test[:B, 0:n_fine_pts:fine_stride, :]
    x_true_at_obs_batch2 = x_true_fine_batch2[:, obs_step_indices_batch + 1, :]
    window_step_indices_b = np.array(
        [round((k + 1) * DT_WINDOW / DT_FINE) - 1 for k in range(batch_windows)]
    )

    seed = config.training.get("seed", 42)
    master_key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(master_key, B)

    batched_batch_fn = build_batched_filters(
        strategies, N, m, obs_indices, P0_sigma, P0, N_ens, sigma_obs, R,
        DT_FINE, DT_WINDOW, total_fine_steps_batch, obs_step_indices_batch,
    )
    outputs_batch, _, _ = batched_batch_fn(
        keys_batch, jnp.array(u0_batch), F_test[:B], x_true_at_obs_batch2,
        dynamic_vars, specify_obs_idx,
    )

    def _rmse(a, b):
        return jnp.sqrt(jnp.mean((a - b) ** 2, axis=2))

    def _mean_std(a):
        return np.array(jnp.mean(a, axis=0)), np.array(jnp.std(a, axis=0))

    x_true_at_windows_b = x_true_fine_batch2[:, window_step_indices_b + 1, :]
    x_true_fine_tail = x_true_fine_batch2[:, 1:, :]
    den_dense = jnp.linalg.norm(x_true_fine_tail, axis=2) + 1e-12
    t_dense_fine = np.arange(1, total_fine_steps_batch + 1) * DT_FINE

    batch_strat_records = {}
    for spec in strategies:
        key = spec["key"]
        out = outputs_batch[key]

        post_means_obs = out["x_means"][:, obs_step_indices_batch, :N]
        prior_means_obs = out["prior_means"][:, :, :N]

        prior_rmse_ic = _rmse(prior_means_obs, x_true_at_obs_batch2)
        post_rmse_ic = _rmse(post_means_obs, x_true_at_obs_batch2)
        erf_ic = prior_rmse_ic / (post_rmse_ic + 1e-12)

        erf_mean, erf_std = _mean_std(erf_ic)
        prior_rmse_mean, prior_rmse_std = _mean_std(prior_rmse_ic)
        post_rmse_mean, post_rmse_std = _mean_std(post_rmse_ic)

        x_hat_windows = out["x_means"][:, window_step_indices_b, :N]
        rmse_ic = _rmse(x_hat_windows, x_true_at_windows_b)
        rmse_window_mean = np.array(jnp.mean(rmse_ic, axis=0))

        spread_ic = jnp.sqrt(jnp.mean(out["x_spreads"][:, window_step_indices_b, :N] ** 2, axis=2))
        spread_window_mean = np.array(jnp.mean(spread_ic, axis=0))

        rmse_raw = np.array(rmse_ic.flatten())
        spread_raw = np.array(spread_ic.flatten())

        l2_dense_mean = np.array(
            jnp.mean(jnp.linalg.norm(out["x_means"][:, :, :N] - x_true_fine_tail, axis=2) / den_dense, axis=0)
        )

        rec = dict(
            label=spec["label"], kind=spec["kind"], propagator=spec["propagator"],
            prior_rmse_mean=prior_rmse_mean, prior_rmse_std=prior_rmse_std,
            post_rmse_mean=post_rmse_mean, post_rmse_std=post_rmse_std,
            erf_mean=erf_mean, erf_std=erf_std,
            rmse_window_mean=rmse_window_mean, spread_window_mean=spread_window_mean,
            rmse_raw=rmse_raw, spread_raw=spread_raw,
            l2_dense_mean=l2_dense_mean,
        )

        if spec["kind"] == "route_b":
            q_scale_step_mean = jnp.mean(out["q_scale"], axis=2)  # (B, total_fine_steps_batch)
            rec["route_b_scale_mean"] = np.array(jnp.mean(q_scale_step_mean, axis=0))
            rec["route_b_scale_std"] = np.array(jnp.std(q_scale_step_mean, axis=0))
            rec["route_b_alpha"] = float(spec["alpha"])
            rec["route_b_beta"] = float(spec["beta"])

        batch_strat_records[key] = rec

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
            u_current = jnp.concatenate([jnp.array(u0_batch), F_test[:B, None]], axis=-1)
            x_pred_list = []
            for k in range(batch_windows):
                x_win = predict_full(u_current)
                x_pred_list.append(x_win if k == 0 else x_win[:, 1:, :])
                u_current = jnp.concatenate([x_win[:, -1, :], F_test[:B, None]], axis=-1)
            x_pred_dense = jnp.concatenate(x_pred_list, axis=1)
            total_steps_ol = x_pred_dense.shape[1]
            x_ref_dense_ol = jnp.array(u_test[:B, :total_steps_ol, :])
            denom_ol = jnp.linalg.norm(x_ref_dense_ol, axis=2) + 1e-12
            l2_ol = np.array(
                jnp.mean(jnp.linalg.norm(x_pred_dense - x_ref_dense_ol, axis=2) / denom_ol, axis=0)
            )
            open_loop_records[prop_key] = dict(t=np.array(t_test[:total_steps_ol]), l2_dense_mean=l2_ol)

    # ── 5. Write everything to HDF5 ─────────────────────────────────────
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
        batch_grp.create_dataset("obs_times", data=obs_times_batch)
        batch_grp.create_dataset("window_idx", data=np.arange(1, batch_windows + 1))
        batch_grp.create_dataset("t_dense_fine", data=t_dense_fine)
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
    ax_top.plot(t_ax, mean_abs_err, color="#2196F3", linewidth=1.6,
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

        ax.plot(t_ax, x_true[:, i], color=TRUTH_COLOR, linewidth=1.0, label="Truth")
        ax.plot(t_ax, x_est[:, i], color=EST_COLOR, linewidth=1.0, linestyle="--",
                label=strategy_label)
        if x_std is not None:
            ax.fill_between(
                t_ax, x_est[:, i] - x_std[:, i], x_est[:, i] + x_std[:, i],
                color=EST_BAND, alpha=0.30, linewidth=0, label=f"{strategy_label} ±1σ",
            )

        if i in obs_by_var:
            obs_times_i, obs_vals_i = zip(*obs_by_var[i])
            ax.scatter(obs_times_i, obs_vals_i, marker="x", s=25, linewidths=0.9,
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
    ax_ts.plot(window_idx, spread_a, marker="^", markersize=4, linewidth=1.8,
               linestyle="-", color="#8BC34A", label=f"{label_a} RMS ensemble σ")
    ax_ts.plot(window_idx, rmse_a, marker="s", markersize=4, linewidth=1.8,
               linestyle="--", color="#FF8A65", label=f"{label_a} EnKF RMSE")
    ax_ts.plot(window_idx, spread_b, marker="^", markersize=4, linewidth=1.8,
               linestyle="-", color="#4CAF50", label=f"{label_b} RMS ensemble σ")
    ax_ts.plot(window_idx, rmse_b, marker="s", markersize=4, linewidth=1.8,
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
    ax_bin.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.4,
                color="#37474F", label="1:1 (perfect calibration)")
    ax_bin.errorbar(rmss_a_b, rmse_a_b, yerr=rmse_a_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#FF8C00",
                     label=f"{label_a} ({n_bins}-bin)")
    ax_bin.errorbar(rmss_b_b, rmse_b_b, yerr=rmse_b_s, fmt="o", markersize=6,
                     capsize=3, linewidth=1.4, color="#2196F3",
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
        ax.plot(obs_times, mean, color=color, linewidth=2.0, marker=marker,
                markersize=4, label=f"{label}  (n = {n_traj} trajectories)")
        ax.fill_between(obs_times, mean - std, mean + std, color=color,
                         alpha=0.15, linewidth=0)

    ax.set_yscale("log")
    ax.axhline(y=1.0, color="#37474F", linestyle="--", linewidth=1.4,
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

    ax.plot(obs_times, prior_mean_a, color="#0A36C7", linewidth=2.0, marker="o",
            markersize=4, linestyle="-", label=f"{label_a} prior RMSE  (n = {n_traj})")
    ax.plot(obs_times, post_mean_a, color="#A30005", linewidth=2.0, marker="s",
            markersize=4, linestyle="-", label=f"{label_a} posterior RMSE  (n = {n_traj})")
    ax.plot(obs_times, prior_mean_b, color="#8E24AA", linewidth=2.0, marker="o",
            markersize=4, linestyle="--", label=f"{label_b} prior RMSE  (n = {n_traj})")
    ax.plot(obs_times, post_mean_b, color="#EC407A", linewidth=2.0, marker="s",
            markersize=4, linestyle="--", label=f"{label_b} posterior RMSE  (n = {n_traj})")

    ax.axhline(y=sigma_obs, color="#4CAF50", linestyle=":", linewidth=1.6,
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