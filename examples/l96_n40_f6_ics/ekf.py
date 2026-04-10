import jax
import jax.numpy as jnp
from jax import jacfwd, jit
from functools import partial
from typing import NamedTuple, Callable


class EKFState(NamedTuple):
    """Holds the full EKF state at a single time step."""
    x_hat: jnp.ndarray  # (N,)  — posterior state estimate
    P: jnp.ndarray      # (N,N) — posterior error covariance


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

        # --- Propagate mean through the surrogate ---
        x_hat_pred = propagator_fn(x_hat)               # (N,)

        # --- Linearise: Jacobian of surrogate w.r.t. input state ---
        # jacfwd differentiates propagator_fn w.r.t. its argument (x_hat)
        # Result F has shape (N, N): F[i, j] = d(output_i)/d(input_j)
        F = jacfwd(propagator_fn)(x_hat)                 # (N, N)

        # --- Propagate covariance ---
        P_pred = F @ ekf_state.P @ F.T + Q              # (N, N)

        return EKFState(x_hat=x_hat_pred, P=P_pred)

    @jit
    def update(
        ekf_state: EKFState,
        y_obs: jnp.ndarray,
        H: jnp.ndarray,
        R: jnp.ndarray,
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
        P_pred = ekf_state.P

        # Innovation and its covariance
        innov = y_obs - H @ x_hat_pred                  # (m,)
        S = H @ P_pred @ H.T + R                        # (m, m)

        # Kalman gain
        K = P_pred @ H.T @ jnp.linalg.inv(S)           # (N, m)

        # Posterior state and covariance (Joseph form for numerical stability)
        x_hat_post = x_hat_pred + K @ innov             # (N,)
        I_KH = jnp.eye(N) - K @ H                      # (N, N)
        P_post = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # (N, N)

        return EKFState(x_hat=x_hat_post, P=P_post), K

    return predict, update


def run_ekf_smoother(
    predict_fn: Callable,
    update_fn: Callable,
    x0_hat: jnp.ndarray,
    P0: jnp.ndarray,
    observations: jnp.ndarray,   # (T, m) — None rows = no observation at that step
    obs_mask: jnp.ndarray,        # (T,) bool — True when observation available
    H: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run the full EKF over a time sequence.

    Returns:
        x_hats: (T, N) filtered state estimates.
        Ps: (T, N, N) filtered covariance matrices.
    """
    state = EKFState(x_hat=x0_hat, P=P0)
    x_hats, Ps = [], []

    for t in range(observations.shape[0]):
        # Predict
        state = predict_fn(state, Q)

        # Update only when an observation is available
        if obs_mask[t]:
            state, _ = update_fn(state, observations[t], H, R)

        x_hats.append(state.x_hat)
        Ps.append(state.P)

    return jnp.stack(x_hats), jnp.stack(Ps)