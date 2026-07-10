import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from functools import partial
from typing import NamedTuple, Callable
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Kalman Filter
# ──────────────────────────────────────────────────────────────────────────────
 
class EnKFState(NamedTuple):
    """
    Holds the full EnKF state: an ensemble of N_ens state vectors together
    with the initial conditions used to anchor DeepONet queries for the
    current assimilation window.
 
    Fields
    ------
    ensemble : (N_ens, N)
        Current state estimates, including any observation updates and
        additive process noise accumulated since the last window reset.
        These are what gets assimilated when observations arrive.
 
    window_ics : (N_ens, N)
        Per-member initial conditions at the *start* of the current
        DeepONet window.  All predict calls within the same window query
        the surrogate as  x(t | window_ic)  for increasing t, rather than
        chaining  f(f(u, dt), dt) …  This exploits the full [t0, t1]
        training range and avoids error accumulation within a window.
 
        Reset to the current ensemble at every window boundary and after
        every observation update (so mid-window observations start a fresh
        pseudo-window from the assimilated state).
    """
    ensemble:   jnp.ndarray  # (N_ens, N)
    window_ics: jnp.ndarray  # (N_ens, N)
 
 
def make_enkf(propagator_fn: Callable, N: int, N_ens: int, N_dyn: int = 40):
    """
    Factory that builds JIT-compiled EnKF predict/update steps with Multiplicative Covariance Inflation.
    """

    @jit
    def predict(
        enkf_state: EnKFState,
        alpha:      float,        
        key:        jnp.ndarray,  
        t_query:    float,
    ) -> EnKFState:
        
        # 1. Propagate full augmented state (41-D)
        ensemble_pred = vmap(
            lambda u: propagator_fn(u, t_query)
        )(enkf_state.window_ics)
        
        # 2. Compute ensemble mean and anomalies
        x_mean = jnp.mean(ensemble_pred, axis=0, keepdims=True)  # (1, N)
        x_anom = ensemble_pred - x_mean                          # (N_ens, N)

        # 3. Apply multiplicative inflation strictly to dynamic variables
        # Construct mask: [alpha, alpha, ..., alpha (40 times), 1.0 (for F)]
        inflation_mask = jnp.ones((N,))
        inflation_mask = inflation_mask.at[:N_dyn].set(alpha)
        
        # Scale anomalies and reconstruct ensemble
        ensemble_inf = x_mean + x_anom * inflation_mask

        return EnKFState(
            ensemble=ensemble_inf,
            window_ics=enkf_state.window_ics,
        )
 
    @jit
    def update(
        enkf_state: EnKFState,
        y_obs:      jnp.ndarray,  # (m,)
        H:          jnp.ndarray,  # (m, N)
        R:          jnp.ndarray,  # (m, m)
        key:        jnp.ndarray,
    ) -> tuple[EnKFState, jnp.ndarray]:
        """
        EnKF update step — stochastic (perturbed-observation) formulation.
 
        Identical to the original update, but the returned EnKFState carries
        ``window_ics`` forward unchanged.  run_enkf_smoother is responsible
        for resetting ``window_ics`` to the posterior ensemble after every
        update so that the next predict call starts a fresh window from the
        assimilated state.
 
        Returns:
            Posterior EnKFState (ensemble updated, window_ics unchanged), K.
        """
        ensemble = enkf_state.ensemble   # (N_ens, N)
        m        = H.shape[0]
 
        # Ensemble anomalies
        x_mean = jnp.mean(ensemble, axis=0)
        X_anom = ensemble - x_mean                               # (N_ens, N)
 
        # Predicted observations and anomalies
        y_pred = vmap(lambda x: H @ x)(ensemble)                 # (N_ens, m)
        y_mean = jnp.mean(y_pred, axis=0)
        Y_anom = y_pred - y_mean                                 # (N_ens, m)
 
        # Ensemble-based Kalman gain
        scale = 1.0 / (N_ens - 1)
        PHT   = scale * X_anom.T @ Y_anom                          # (N, m)
        S     = scale * Y_anom.T @ Y_anom + R                      # (m, m)
        K     = jax.scipy.linalg.solve(S, PHT.T, assume_a='pos').T # (N, m)
 
        # Perturbed observations
        L_R         = jnp.linalg.cholesky(R + 1e-10 * jnp.eye(m))
        eps         = jax.random.normal(key, shape=(N_ens, m)) @ L_R.T
        y_perturbed = y_obs[None, :] + eps                      # (N_ens, m)
 
        # Per-member update
        innovations   = y_perturbed - y_pred
        ensemble_post = ensemble + innovations @ K.T             # (N_ens, N)
 
        # window_ics passed through unchanged; run_enkf_smoother will reset them.
        return EnKFState(
            ensemble=ensemble_post,
            window_ics=enkf_state.window_ics,
        ), K
 
    return predict, update
 

def run_enkf_smoother(
    predict_fn:        Callable,
    update_fn:         Callable,
    ensemble0:         jnp.ndarray,   # (N_ens, N)
    observations:      jnp.ndarray,   # (T_obs, m)
    obs_step_indices:  np.ndarray,    # (T_obs,) int — fine step of each obs
    H_seq:             jnp.ndarray,   # (T_obs, m, N)
    alpha_fine:        float,
    R:                 jnp.ndarray,   # (m, m)
    key:               jnp.ndarray,
    total_fine_steps:  int,
    dt_fine:           float,         # fine integration step
    dt_window:         float,         # DeepONet training window length
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fine-step-centric, window-aware EnKF smoother.
 
    Window-aware prediction
    -----------------------
    The DeepONet was trained on windows of length ``dt_window``.  Rather
    than chaining  f(f(u, dt_fine), dt_fine) …  across fine steps, this
    smoother tracks which fine step we are at *within the current window*
    and always queries the surrogate as
 
        x_pred = propagator_fn(window_ic, step_in_window * dt_fine)
 
    so that all in-window predictions exploit the full training range
    [t0, t1].  This avoids error accumulation that arises when the
    window is subdivided into many fine steps.
 
    Window reset policy
    -------------------
    ``window_ics`` (the per-member ICs used for DeepONet queries) are
    reset to the current ensemble in two situations:
 
    1. **Observation update**: immediately after assimilation, so that
       subsequent predictions in the same window start from the
       corrected state rather than the pre-update IC.  This effectively
       starts a fresh pseudo-window from the posterior mean.
 
    2. **Window boundary**: when ``step_in_window`` reaches
       ``steps_per_window``, even if no observation was assimilated.
 
    Both resets also zero ``step_in_window``, so the next call to
    ``predict_fn`` queries at ``t = dt_fine`` (the first fine step of
    the new window).
 
    Args:
        predict_fn:       JIT-compiled EnKF predict (window-aware).
        update_fn:        JIT-compiled EnKF update.
        ensemble0:        (N_ens, N) initial ensemble.
        observations:     (T_obs, m) observation vectors.
        obs_step_indices: (T_obs,) fine-step indices of each observation.
        H_seq:            (T_obs, m, N) observation matrices.
        alpha_fine:       multiplicative inflation
        R:                (m, m) observation noise covariance.
        key:              JAX PRNG key.
        total_fine_steps: total number of fine steps to run.
        dt_fine:          duration of one fine step (Python float).
        dt_window:        duration of one DeepONet training window (Python float).
 
    Returns:
        x_means:   (total_fine_steps, N) ensemble-mean state estimates.
        x_spreads: (total_fine_steps, N) per-variable ensemble std.
        prior_means_at_obs: (T_obs, N)            ensemble mean *before* the
                        update at each observation step — used to compute the
                        Error Reduction Factor (ERF = prior RMSE / post RMSE).
    """
    steps_per_window = round(dt_window / dt_fine)
 
    # O(1) lookup: fine step index -> observation index (-1 = no obs)
    obs_at_step = np.full(total_fine_steps, -1, dtype=int)
    for obs_idx, step_idx in enumerate(obs_step_indices):
        obs_at_step[step_idx] = obs_idx
 
    # Initialise state: ensemble and window_ics both start from ensemble0.
    state         = EnKFState(ensemble=ensemble0, window_ics=ensemble0)
    x_means:  list[jnp.ndarray] = []
    x_spreads: list[jnp.ndarray] = []
    prior_means_at_obs: list[jnp.ndarray] = []
    step_in_window = 0  # Python int — used to compute t_query
    
    for fine_t in range(total_fine_steps):
        # In-window query time: (step_in_window + 1) * dt_fine.
        # step_in_window = 0 on the first step of every window, so t_query
        # starts at dt_fine and increases by dt_fine on each subsequent step.
        t_query = (step_in_window + 1) * dt_fine  # Python float — static for JIT

        # Calculate the cumulative inflation for this exact point in the window
        cumulative_alpha = alpha_fine ** (step_in_window + 1)

        key, key_pred, key_upd = jax.random.split(key, 3)
 
        # Predict: queries propagator_fn(window_ic_i, t_query) for each member.
        state = predict_fn(state, cumulative_alpha, key_pred, t_query)
        
 
        # Conditionally update if an observation falls on this fine step.
        reset_window = False
        obs_idx = obs_at_step[fine_t]
        if obs_idx >= 0:
            # ── Capture prior mean BEFORE assimilation ──────────────────────
            prior_means_at_obs.append(jnp.mean(state.ensemble, axis=0))

            state, _ = update_fn(
                state,
                observations[obs_idx],
                H_seq[obs_idx],
                R,
                key_upd,
            )
            # After assimilation, start a fresh window from the posterior.
            reset_window = True
 
        x_means.append(jnp.mean(state.ensemble, axis=0))
        x_spreads.append(jnp.std(state.ensemble, axis=0))
 
        # Advance the in-window counter; reset at window boundaries.
        step_in_window += 1
        if step_in_window >= steps_per_window:
            reset_window = True
 
        if reset_window:
            # Set window_ics to the current ensemble so that the next predict
            # call starts a fresh window query from this state.
            state = EnKFState(
                ensemble=state.ensemble,
                window_ics=state.ensemble,
            )
            step_in_window = 0
 
    return (
        jnp.stack(x_means),
        jnp.stack(x_spreads),
        jnp.stack(prior_means_at_obs),   # (T_obs, N)
    )


def init_ensemble(
    x0_hat:  jnp.ndarray,   # (N_total,) augmented prior mean
    P0:      jnp.ndarray,   # (N_dyn, N_dyn) prior covariance (unpadded)
    N_ens:   int,
    key:     jnp.ndarray,
) -> jnp.ndarray:
    
    N_total = x0_hat.shape[0]
    N_dyn   = P0.shape[0]
    
    # Cholesky and noise generation strictly on the dynamic sub-state
    L = jnp.linalg.cholesky(P0 + 1e-10 * jnp.eye(N_dyn))
    z = jax.random.normal(key, shape=(N_ens, N_dyn))
    noise = z @ L.T
    
    # Pad the noise vector with zeros for the static parameters
    if N_dyn < N_total:
        noise = jnp.pad(noise, ((0, 0), (0, N_total - N_dyn)))
        
    return x0_hat[None, :] + noise


# ──────────────────────────────────────────────────────────────────────────────
# Route B: Residual-Scaled Covariance
# ──────────────────────────────────────────────────────────────────────────────
#
# `make_enkf.predict` inflates every ensemble member by the SAME fixed
# process-noise covariance Q every fine step. Route B instead lets each
# member's inflation reflect how badly ITS OWN surrogate trajectory is
# currently violating the governing PDE:
#
#     rho_n^i       = PDE residual of member i's surrogate trajectory,
#                     integrated over the current window
#     Q_n^i         = (alpha + beta * ||rho_n^i||^2_{L2}) * Q0
#     u_{n+1}^{f,i} = surrogate(window_ic_i, t) + xi_n^i,   xi_n^i ~ N(0, Q_n^i)
#
# alpha is a variance floor (additive inflation that never vanishes, even
# for a member whose residual is ~0). beta converts residual magnitude into
# extra spread, standing in for the stability constant of the (expensive,
# un-computed) tangent-linear propagator. Because every member shares Q0's
# spatial shape, sampling needs only ONE Cholesky factor of Q0 per step,
# rescaled per member by sqrt(alpha + beta*||rho_n^i||^2) -- no per-member
# Cholesky, no tangent-linear solve.


def _trapz_weights(t: jnp.ndarray) -> jnp.ndarray:
    """
    Trapezoidal quadrature weights for a 1-D (possibly non-uniform) time
    grid ``t``, shape (n_quad,), n_quad >= 2, such that
    ``sum(w * f(t)) ~= integral of f over [t[0], t[-1]]``.
    """
    dt = jnp.diff(t)
    w  = jnp.zeros_like(t)
    w  = w.at[0].set(dt[0] / 2)
    w  = w.at[-1].set(dt[-1] / 2)
    w  = w.at[1:-1].set((dt[:-1] + dt[1:]) / 2)
    return w

def residual_l2_norm_sq(
    residual_fn: Callable,
    u:           jnp.ndarray,   # (N_total,) single member's window IC
    t_quad:      jnp.ndarray,   # (n_quad,) quadrature times within the window
) -> jnp.ndarray:
    """
    Spatiotemporal L2 norm-squared of the PDE residual for ONE ensemble
    member, anchored at window IC ``u``:

        ||rho||^2_{L2(Omega x [t_quad[0], t_quad[-1]])}
            ~= sum_k  w_k * sum_j rho_j(t_k)^2

    i.e. integrated over time via the trapezoidal rule (``t_quad``) and
    summed over the spatial/state dimension.

    Args:
        residual_fn: closure ``(u, t) -> (N_dyn,)``, matching the calling
            convention of ``propagator_fn`` in ``make_enkf`` (window-IC-
            anchored, in-window query time). For L96 this is exactly
            ``r_net``: rho = x_t - F(x; mu), the same residual used in the
            physics loss -- reused here at inference time, taking no
            gradient w.r.t. params (only the internal jacfwd w.r.t. t).
            See ``L96UDON.make_residual_fn``.
        u:      (N_total,) window IC for this member (may be augmented with
                static parameters, e.g. the L96 forcing F).
        t_quad: (n_quad,) quadrature times spanning the integration window.

    Returns:
        Scalar ||rho||^2 for this member.
    """
    rho        = vmap(residual_fn, (None, 0))(u, t_quad)   # (n_quad, N_dyn)
    spatial_sq = jnp.sum(rho ** 2, axis=-1)                 # (n_quad,)
    w          = _trapz_weights(t_quad)                     # (n_quad,)
    return jnp.sum(w * spatial_sq)

def make_route_b_enkf(
    propagator_fn: Callable,
    residual_fn:   Callable,
    N:             int,
    N_ens:         int,
):
    """
    Factory for the Route B forecast step. Assimilation is untouched by
    Route B, so the returned ``update_fn`` is reused verbatim from
    ``make_enkf`` -- only the process-noise generation inside ``predict``
    changes.

    Args:
        propagator_fn: (u, t) -> (N_dyn,) surrogate forecast, e.g.
                       ``L96UDON.make_surrogate_propagator(params)``.
        residual_fn:   (u, t) -> (N_dyn,) PDE residual, same window-IC-
                       anchored convention as propagator_fn, e.g.
                       ``L96UDON.make_residual_fn(params)``.
        N:      augmented state dimension (kept for interface parity with
                ``make_enkf``; unused internally, same as there).
        N_ens:  ensemble size.

    Returns:
        predict_route_b_fn, update_fn -- both JIT-compiled.
    """
    _, update = make_enkf(propagator_fn, N, N_ens)

    @partial(jit, static_argnums=(6,))
    def predict_route_b(
        enkf_state: EnKFState,
        Q0:         jnp.ndarray,   # (N_dyn, N_dyn) fixed spatial shape/structure
        alpha:      jnp.ndarray,   # scalar >= 0, variance floor
        beta:       jnp.ndarray,   # scalar >= 0, residual sensitivity
        t_quad:     jnp.ndarray,   # (n_quad,) quadrature times for THIS step
        key:        jnp.ndarray,
        t_query:    float,         # static -- see make_enkf.predict docstring
    ) -> tuple[EnKFState, dict[str, jnp.ndarray]]:

        # 1. Deterministic surrogate forecast -- identical to standard predict.
        ensemble_pred = vmap(
            lambda u: propagator_fn(u, t_query)
        )(enkf_state.window_ics)

        N_total = ensemble_pred.shape[1]
        N_dyn   = Q0.shape[0]

        # 2. Per-member PDE residual, evaluated on EACH member's own window
        #    IC and integrated (trapezoidally) over this step's time span.
        #    A member whose surrogate currently violates the PDE more picks
        #    up a larger ||rho_i||^2 -> more inflation: the "physics-driven,
        #    flow-dependent additive inflation" Route B is built to produce,
        #    with no tangent-linear solve required.
        resid_sq = vmap(
            lambda u: residual_l2_norm_sq(residual_fn, u, t_quad)
        )(enkf_state.window_ics)                                    # (N_ens,)

        # 3. Route B scale factor per member: s_i = alpha + beta * ||rho_i||^2
        #      alpha -> floor / additive inflation, always active -- keeps
        #               the filter from collapsing when the surrogate looks
        #               locally physics-consistent (||rho_i|| ~ 0).
        #      beta  -> how strongly a physics-violating member gets
        #               inflated; absorbs the (otherwise expensive)
        #               tangent-linear stability constant into one scalar.
        scale = alpha + beta * resid_sq                             # (N_ens,)

        # 4. Sample xi_i ~ N(0, s_i * Q0). Q0's spatial shape is SHARED
        #    across members, so factor it ONCE and rescale the per-member
        #    draw by sqrt(s_i):
        #        s_i * Q0 = (sqrt(s_i) * L0) (sqrt(s_i) * L0)^T
        #    avoiding N_ens separate Cholesky factorizations every step.
        L_Q0  = jnp.linalg.cholesky(Q0 + 1e-10 * jnp.eye(N_dyn))
        z     = jax.random.normal(key, shape=(N_ens, N_dyn))
        noise = (z @ L_Q0.T) * jnp.sqrt(scale)[:, None]              # (N_ens, N_dyn)

        # 5. Pad for any static augmented parameters (e.g. forcing F).
        if N_dyn < N_total:
            noise = jnp.pad(noise, ((0, 0), (0, N_total - N_dyn)))

        new_state = EnKFState(
            ensemble=ensemble_pred + noise,
            window_ics=enkf_state.window_ics,
        )
        return new_state, {"scale": scale, "resid_sq": resid_sq}

    return predict_route_b, update

def run_enkf_smoother_route_b(
    predict_route_b_fn: Callable,
    update_fn:          Callable,
    ensemble0:           jnp.ndarray,   # (N_ens, N)
    observations:        jnp.ndarray,   # (T_obs, m)
    obs_step_indices:    np.ndarray,    # (T_obs,) int
    H_seq:               jnp.ndarray,   # (T_obs, m, N)
    Q0:                  jnp.ndarray,   # (N_dyn, N_dyn) fixed shape covariance
    alpha:               float,
    beta:                float,
    R:                   jnp.ndarray,   # (m, m)
    key:                 jnp.ndarray,
    total_fine_steps:    int,
    dt_fine:             float,
    dt_window:           float,
    n_quad:              int = 3,       # quadrature points per fine step
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Route B counterpart of ``run_enkf_smoother``: identical fine-step /
    window-reset / observation bookkeeping, but calls
    ``predict_route_b_fn(state, Q0, alpha, beta, t_quad, key, t_query)``
    instead of the fixed-Q ``predict_fn``.

    Residual integration window
    ----------------------------
    The residual for each fine step is integrated (trapezoidally, over
    ``n_quad`` points) across
    ``[step_in_window * dt_fine, (step_in_window + 1) * dt_fine]`` -- i.e.
    the time span THIS predict call actually advances. This mirrors Q0
    being the per-fine-step base covariance, exactly as ``Q_fine`` already
    is for the standard filter (see ``scale_Q_for_fine_steps``). To
    integrate the residual over the FULL DeepONet window instead, replace
    the ``t_a, t_b`` lines below with ``t_a, t_b = 0.0, dt_window`` (and
    optionally cache/skip recomputation once per window rather than once
    per fine step).

    Returns:
        x_means, x_spreads, prior_means_at_obs: identical in shape/meaning
            to ``run_enkf_smoother``.
        Q_scale_history: (total_fine_steps, N_ens) per-member Route B scale
            factor s_i at every fine step -- useful for tuning alpha/beta
            and for sanity-checking that inflation tracks known model error
            (e.g. larger near sharp gradients / observation gaps).
    """
    steps_per_window = round(dt_window / dt_fine)

    obs_at_step = np.full(total_fine_steps, -1, dtype=int)
    for obs_idx, step_idx in enumerate(obs_step_indices):
        obs_at_step[step_idx] = obs_idx

    state = EnKFState(ensemble=ensemble0, window_ics=ensemble0)
    x_means:  list[jnp.ndarray] = []
    x_spreads: list[jnp.ndarray] = []
    prior_means_at_obs: list[jnp.ndarray] = []
    Q_scale_history: list[jnp.ndarray] = []
    step_in_window = 0

    alpha = jnp.asarray(alpha)
    beta  = jnp.asarray(beta)

    for fine_t in range(total_fine_steps):
        t_query = (step_in_window + 1) * dt_fine

        # Quadrature times for THIS fine step's residual integral, anchored
        # at window-local time (matches propagator_fn/residual_fn convention).
        t_a = step_in_window * dt_fine
        t_b = t_query
        t_quad = jnp.linspace(t_a, t_b, n_quad)

        key, key_pred, key_upd = jax.random.split(key, 3)

        state, diag = predict_route_b_fn(state, Q0, alpha, beta, t_quad, key_pred, t_query)
        Q_scale_history.append(diag["scale"])

        reset_window = False
        obs_idx = obs_at_step[fine_t]
        if obs_idx >= 0:
            prior_means_at_obs.append(jnp.mean(state.ensemble, axis=0))

            state, _ = update_fn(
                state,
                observations[obs_idx],
                H_seq[obs_idx],
                R,
                key_upd,
            )
            reset_window = True

        x_means.append(jnp.mean(state.ensemble, axis=0))
        x_spreads.append(jnp.std(state.ensemble, axis=0))

        step_in_window += 1
        if step_in_window >= steps_per_window:
            reset_window = True

        if reset_window:
            state = EnKFState(
                ensemble=state.ensemble,
                window_ics=state.ensemble,
            )
            step_in_window = 0

    return (
        jnp.stack(x_means),
        jnp.stack(x_spreads),
        jnp.stack(prior_means_at_obs),
        jnp.stack(Q_scale_history),
    )

