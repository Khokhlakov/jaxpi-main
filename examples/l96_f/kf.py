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
        PHT   = scale * X_anom.T @ Y_anom                       # (N, m)
        S     = scale * Y_anom.T @ Y_anom + R                   # (m, m)
        K     = PHT @ jnp.linalg.inv(S)                        # (N, m)
 
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