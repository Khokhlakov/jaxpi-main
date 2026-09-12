import scipy.io
import jax.numpy as jnp
import numpy as np
import h5py
import os


def build_obs_schedule(
    total_time: float,
    dt_fine:    float,
    dt_obs:     float,
    tol:        float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the fine-step indices at which observations occur.

    Enforces that dt_obs and total_time are both integer multiples of dt_fine,
    raising immediately if not — preventing silent time misalignment.

    Args:
        total_time: total simulation duration (e.g. num_windows * DT_WINDOW).
        dt_fine:    fine prediction step (e.g. 0.005).
        dt_obs:     observation interval (e.g. 0.1, 0.25, 0.5).
                    Must satisfy: dt_obs / dt_fine is a positive integer.
                    Does NOT need to be a multiple of DT_WINDOW.
        tol:        floating-point tolerance for divisibility checks.

    Returns:
        obs_times:        (T_obs,) float array of observation times.
        obs_step_indices: (T_obs,) int array — 0-indexed fine step at which
                          each observation occurs.  Step i covers the interval
                          (i*dt_fine, (i+1)*dt_fine], so the state after step i
                          lives at time (i+1)*dt_fine.
        total_fine_steps: total number of fine predict steps.

    Raises:
        ValueError: if any divisibility check fails.
    """
    def _check_divisible(numerator: float, denominator: float, label: str) -> int:
        ratio = numerator / denominator
        n     = int(round(ratio))
        if abs(ratio - n) > tol:
            raise ValueError(
                f"{label}: {numerator} is not evenly divisible by {denominator}. "
                f"Ratio = {ratio:.10f}, nearest integer = {n}, "
                f"residual = {abs(ratio - n):.2e} > tol={tol}. "
                f"Choose dt_fine so that both dt_obs and total_time are "
                f"exact integer multiples of dt_fine."
            )
        return n

    total_fine_steps = _check_divisible(total_time, dt_fine,  "total_time / dt_fine")
    steps_per_obs    = _check_divisible(dt_obs,     dt_fine,  "dt_obs / dt_fine")
    n_obs            = _check_divisible(total_time, dt_obs,   "total_time / dt_obs")

    # Observation times: first at dt_obs, last at total_time
    obs_times        = np.array([(k + 1) * dt_obs        for k in range(n_obs)])
    # 0-indexed: after step s the clock reads (s+1)*dt_fine
    # obs at time (k+1)*dt_obs corresponds to step (k+1)*steps_per_obs - 1
    obs_step_indices = np.array([(k + 1) * steps_per_obs - 1 for k in range(n_obs)],
                                dtype=int)

    return obs_times, obs_step_indices, total_fine_steps


def scale_Q_for_fine_steps(
    Q_coarse:      "jnp.ndarray",
    steps_per_window: int,
) -> "jnp.ndarray":
    """
    Scale a window-level process noise covariance to a per-fine-step value.

    If Q_coarse is calibrated so that one window's accumulated noise has
    covariance Q_coarse, and noise increments are independent across fine steps
    (discrete Wiener process), then each fine step must contribute Q_coarse /
    steps_per_window so that the total over the window remains Q_coarse.

    Args:
        Q_coarse:         (N, N) process noise calibrated for one full window.
        steps_per_window: integer ratio DT_WINDOW / dt_fine.

    Returns:
        Q_fine: (N, N) per-fine-step process noise covariance.
    """
    return Q_coarse / steps_per_window
