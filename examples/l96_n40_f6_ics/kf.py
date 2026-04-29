import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from functools import partial
from typing import NamedTuple, Callable


class EKFState(NamedTuple):
    """Holds the full EKF state at a single time step."""
    x_hat: jnp.ndarray  # (N,)   — posterior state estimate
    P:     jnp.ndarray  # (N, N) — posterior error covariance
 
 
def make_ekf(propagator_fn: Callable, N: int):
    """
    Factory that builds JIT-compiled EKF predict/update steps.
 
    Args:
        propagator_fn: callable (u: (N,)) -> (N,), the DeepONet
                       evaluated at a fixed dt. Must be pure (no side effects).
        N: state dimension (40 for L96).
 
    Returns:
        predict_fn, update_fn — both JIT-compiled.
    """
 
    @jit
    def predict(ekf_state: EKFState, Q: jnp.ndarray) -> EKFState:
        """
        EKF prediction step.
 
        Propagates the state estimate and covariance forward using the
        surrogate model. The Jacobian is computed automatically via jacfwd.
 
        Args:
            ekf_state: current posterior (x_hat, P).
            Q: (N, N) process noise covariance.
 
        Returns:
            Prior EKFState (x_hat_pred, P_pred).
        """
        x_hat = ekf_state.x_hat
 
        # Propagate mean through the surrogate
        x_hat_pred = propagator_fn(x_hat)                       # (N,)
 
        # Linearise: Jacobian of surrogate w.r.t. input state
        # F[i, j] = d(output_i) / d(input_j)
        F = jacfwd(propagator_fn)(x_hat)                        # (N, N)
 
        # Propagate covariance
        P_pred = F @ ekf_state.P @ F.T + Q                      # (N, N)
 
        return EKFState(x_hat=x_hat_pred, P=P_pred)
 
    @jit
    def update(
        ekf_state: EKFState,
        y_obs:     jnp.ndarray,  # (m,)
        H:         jnp.ndarray,  # (m, N)
        R:         jnp.ndarray,  # (m, m)
    ) -> tuple[EKFState, jnp.ndarray]:
        """
        EKF update step (measurement assimilation).
 
        Args:
            ekf_state: prior (x_hat_pred, P_pred) from predict().
            y_obs: (m,) observation vector.
            H: (m, N) linear observation matrix.
            R: (m, m) observation noise covariance.
 
        Returns:
            Posterior EKFState and Kalman gain K.
        """
        x_hat_pred = ekf_state.x_hat
        P_pred     = ekf_state.P
 
        innov = y_obs - H @ x_hat_pred                          # (m,)
        S     = H @ P_pred @ H.T + R                            # (m, m)
        K     = P_pred @ H.T @ jnp.linalg.inv(S)               # (N, m)
 
        x_hat_post = x_hat_pred + K @ innov                     # (N,)
        I_KH       = jnp.eye(N) - K @ H                        # (N, N)
        P_post     = I_KH @ P_pred @ I_KH.T + K @ R @ K.T      # (N, N) Joseph form
 
        return EKFState(x_hat=x_hat_post, P=P_post), K
 
    return predict, update
 
 
def run_ekf_smoother(
    predict_fn:    Callable,
    update_fn:     Callable,
    x0_hat:        jnp.ndarray,   # (N,)
    P0:            jnp.ndarray,   # (N, N)
    observations:  jnp.ndarray,   # (T, m)
    obs_mask:      jnp.ndarray,   # (T,) bool
    H_seq:         jnp.ndarray,   # (T, m, N)
    Q:             jnp.ndarray,
    R:             jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run the full EKF over a time sequence.
 
    Returns:
        x_hats: (T, N) filtered state estimates.
        Ps:     (T, N, N) filtered covariance matrices.
    """
    state = EKFState(x_hat=x0_hat, P=P0)
    x_hats, Ps = [], []
 
    for t in range(observations.shape[0]):
        state = predict_fn(state, Q)
 
        if obs_mask[t]:
            state, _ = update_fn(state, observations[t], H_seq[t], R)
 
        x_hats.append(state.x_hat)
        Ps.append(state.P)
 
    return jnp.stack(x_hats), jnp.stack(Ps)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Kalman Filter
# ──────────────────────────────────────────────────────────────────────────────
 
class EnKFState(NamedTuple):
    """
    Holds the full EnKF state: an ensemble of N_ens state vectors.
 
    The ensemble implicitly encodes both the mean estimate and the
    sample error covariance without ever forming an (N, N) matrix:
        x_mean ≈ mean(ensemble, axis=0)               (N,)
        P      ≈ anom.T @ anom / (N_ens - 1)          (N, N)  — never materialised
    """
    ensemble: jnp.ndarray  # (N_ens, N)
 
 
def make_enkf(propagator_fn: Callable, N: int, N_ens: int):
    """
    Factory that builds JIT-compiled EnKF predict/update steps.
 
    Key differences from the EKF:
    • No Jacobian — the nonlinear surrogate is applied to *every* member
      via vmap, so the ensemble spread propagates through the true
      nonlinearity rather than its linearisation.
    • Stochastic update — each member receives a independently-perturbed
      observation drawn from N(y_obs, R), which is necessary to maintain
      the correct posterior variance in the stochastic EnKF formulation.
    • PRNG keys — predict and update both accept a JAX key so that
      randomness is explicit and reproducible.
 
    Args:
        propagator_fn: callable (u: (N,)) -> (N,), pure surrogate at fixed dt.
        N:     state dimension (40 for L96).
        N_ens: ensemble size; ≥ 20 is typical, ≥ 50 recommended for L96.
 
    Returns:
        predict_fn, update_fn — both JIT-compiled.
    """
 
    @jit
    def predict(
        enkf_state: EnKFState,
        Q:          jnp.ndarray,   # (N, N) process noise covariance
        key:        jnp.ndarray,   # JAX PRNG key for process noise
    ) -> EnKFState:
        """
        EnKF prediction step.
 
        Each ensemble member is independently propagated through the
        nonlinear surrogate, then perturbed by additive process noise
        drawn from N(0, Q).  No linearisation is performed.
 
        The Cholesky factor of Q is used to generate correlated noise
        efficiently, and a small nugget (1e-10 I) guards against
        numerical non-positive-definiteness.
 
        Args:
            enkf_state: current posterior EnKFState (N_ens, N).
            Q:   (N, N) process noise covariance.
            key: PRNG key consumed for noise sampling.
 
        Returns:
            Prior EnKFState with propagated and noise-perturbed ensemble.
        """
        # ── 1. Propagate every member in parallel ─────────────────────────
        # vmap maps propagator_fn over the leading (N_ens) axis.
        # No Jacobian is computed; the full nonlinearity is used.
        ensemble_pred = vmap(propagator_fn)(enkf_state.ensemble)   # (N_ens, N)
 
        # ── 2. Sample additive process noise from N(0, Q) ─────────────────
        # Cholesky: Q = L L^T  →  noise = z L^T,  z ~ N(0, I)
        # The nugget prevents failure when Q is near-singular (e.g. tiny sigma_proc).
        L_Q  = jnp.linalg.cholesky(Q + 1e-10 * jnp.eye(N))        # (N, N)
        z    = jax.random.normal(key, shape=(N_ens, N))            # (N_ens, N)
        noise = z @ L_Q.T                                          # (N_ens, N)
 
        return EnKFState(ensemble=ensemble_pred + noise)
 
    @jit
    def update(
        enkf_state: EnKFState,
        y_obs:      jnp.ndarray,   # (m,)
        H:          jnp.ndarray,   # (m, N)
        R:          jnp.ndarray,   # (m, m)
        key:        jnp.ndarray,   # JAX PRNG key for observation perturbations
    ) -> tuple[EnKFState, jnp.ndarray]:
        """
        EnKF update step — stochastic (perturbed-observation) formulation.
 
        The ensemble-based cross-covariance PH^T and innovation covariance S
        replace the EKF's analytically propagated P and S:
 
            PHT = (1/(N_ens-1)) * X_anom^T  Y_anom      [N, m]
            S   = (1/(N_ens-1)) * Y_anom^T  Y_anom + R  [m, m]
            K   = PHT S^{-1}                             [N, m]
 
        Each member receives an independently-perturbed observation
        y_i ~ N(y_obs, R) so that the posterior ensemble spread is
        consistent with the Kalman gain (Burgers et al., 1998).
 
        Args:
            enkf_state: prior EnKFState from predict().
            y_obs: (m,) shared observation vector.
            H:     (m, N) linear observation matrix.
            R:     (m, m) observation noise covariance.
            key:   PRNG key consumed for observation perturbations.
 
        Returns:
            Posterior EnKFState and the ensemble-mean Kalman gain K (N, m).
        """
        ensemble = enkf_state.ensemble   # (N_ens, N)
        m        = H.shape[0]
 
        # ── 1. Ensemble anomalies in state space ──────────────────────────
        x_mean = jnp.mean(ensemble, axis=0)         # (N,)
        X_anom = ensemble - x_mean                  # (N_ens, N)
 
        # ── 2. Predicted observations and their anomalies ─────────────────
        # H is linear, so H @ x_i can be vmapped cheaply.
        y_pred = vmap(lambda x: H @ x)(ensemble)    # (N_ens, m)
        y_mean = jnp.mean(y_pred, axis=0)           # (m,)
        Y_anom = y_pred - y_mean                    # (N_ens, m)
 
        # ── 3. Ensemble-based Kalman gain ─────────────────────────────────
        scale = 1.0 / (N_ens - 1)
        PHT   = scale * X_anom.T @ Y_anom           # (N, m)  cross-covariance
        S     = scale * Y_anom.T @ Y_anom + R       # (m, m)  innovation covariance
        K     = PHT @ jnp.linalg.inv(S)            # (N, m)
 
        # ── 4. Perturbed observations: y_i ~ N(y_obs, R) ──────────────────
        # Required for the stochastic EnKF to maintain correct spread.
        L_R  = jnp.linalg.cholesky(R + 1e-10 * jnp.eye(m))
        eps  = jax.random.normal(key, shape=(N_ens, m)) @ L_R.T  # (N_ens, m)
        y_perturbed = y_obs[None, :] + eps                        # (N_ens, m)
 
        # ── 5. Per-member innovation and state update ─────────────────────
        innovations   = y_perturbed - y_pred         # (N_ens, m)
        ensemble_post = ensemble + innovations @ K.T # (N_ens, N)
 
        return EnKFState(ensemble=ensemble_post), K
 
    return predict, update
 
 
def run_enkf_smoother(
    predict_fn:   Callable,
    update_fn:    Callable,
    ensemble0:    jnp.ndarray,   # (N_ens, N) — initial ensemble
    observations: jnp.ndarray,   # (T, m)
    obs_mask:     jnp.ndarray,   # (T,) bool — True when an observation is available
    H_seq:        jnp.ndarray,   # (T, m, N) — per-step observation matrices
    Q:            jnp.ndarray,   # (N, N)
    R:            jnp.ndarray,   # (m, m)
    key:          jnp.ndarray,   # JAX PRNG master key (will be split internally)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run the full EnKF over a time sequence, mirroring run_ekf_smoother.
 
    PRNG keys are derived from `key` via jax.random.split at every step so
    that predict and update noise are independent and the whole run is
    reproducible given the same initial key.
 
    The outer Python loop is intentional: predict_fn and update_fn are
    themselves JIT-compiled, so per-step overhead is minimal.  The loop
    allows heterogeneous H matrices (dynamic observation operators) and
    obs_mask branching without recompilation.
 
    Args:
        predict_fn:   JIT-compiled EnKF predict function from make_enkf().
        update_fn:    JIT-compiled EnKF update function from make_enkf().
        ensemble0:    (N_ens, N) initial ensemble — typically sampled around
                      the perturbed IC using the prior covariance P0.
        observations: (T, m) stacked observation vectors; rows at unobserved
                      steps are ignored (zeros or any value).
        obs_mask:     (T,) boolean array; True ↔ an observation is available
                      at this step.
        H_seq:        (T, m, N) per-step observation matrices; supports
                      dynamic (time-varying) observation operators.
        Q:            (N, N) process noise covariance.
        R:            (m, m) observation noise covariance.
        key:          Master JAX PRNG key.
 
    Returns:
        x_means:   (T, N) ensemble-mean state estimate at each step.
        x_spreads: (T, N) per-variable ensemble standard deviation —
                   a cheap proxy for the marginal posterior uncertainty.
    """
    state = EnKFState(ensemble=ensemble0)
    x_means:   list[jnp.ndarray] = []
    x_spreads: list[jnp.ndarray] = []
 
    for t in range(observations.shape[0]):
        # Derive two independent keys for predict and update at this step.
        key, key_pred, key_upd = jax.random.split(key, 3)
 
        # ── Predict ───────────────────────────────────────────────────────
        state = predict_fn(state, Q, key_pred)
 
        # ── Update (only when an observation is available) ─────────────────
        # obs_mask[t] is a concrete scalar here (loop is not traced by JAX),
        # so the Python if is valid and avoids tracing the update on empty steps.
        if obs_mask[t]:
            state, _ = update_fn(state, observations[t], H_seq[t], R, key_upd)
 
        # ── Collect ensemble statistics ────────────────────────────────────
        x_means.append(jnp.mean(state.ensemble, axis=0))   # (N,)
        x_spreads.append(jnp.std(state.ensemble, axis=0))  # (N,)
 
    return jnp.stack(x_means), jnp.stack(x_spreads)
 
 
def init_ensemble(
    x0_hat:  jnp.ndarray,   # (N,) prior mean
    P0:      jnp.ndarray,   # (N, N) prior covariance
    N_ens:   int,
    key:     jnp.ndarray,
) -> jnp.ndarray:
    """
    Draw the initial ensemble from the prior distribution N(x0_hat, P0).
 
    Uses the Cholesky factorisation of P0 so that the full covariance
    structure (not just the diagonal) is respected during initialisation.
 
    Args:
        x0_hat: (N,) prior mean state estimate.
        P0:     (N, N) prior error covariance.
        N_ens:  number of ensemble members.
        key:    JAX PRNG key.
 
    Returns:
        ensemble: (N_ens, N) initial ensemble.
    """
    N   = x0_hat.shape[0]
    L   = jnp.linalg.cholesky(P0 + 1e-10 * jnp.eye(N))
    z   = jax.random.normal(key, shape=(N_ens, N))
    return x0_hat[None, :] + z @ L.T              # (N_ens, N)