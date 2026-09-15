"""
Timing / scaling helpers shared by the KS DeepONet + EnKF evaluation
pipeline.

This is the KS counterpart of ``examples.l96_f.utils``: the existing KS
``build_obs_schedule`` / ``scale_Q_for_fine_steps`` are kept verbatim, and
the three helpers the L96 filtering code additionally depends on --
``check_divisible``, ``steps_per_window_exact`` and
``scale_inflation_for_fine_steps`` -- are added so ``kf.py`` / ``eval.py``
can import them unchanged.
"""

import scipy.io
import jax.numpy as jnp
import numpy as np
import h5py
import os


def check_divisible(numerator: float, denominator: float, label: str,
                    tol: float = 1e-9) -> int:
    """
    Assert ``numerator / denominator`` is an integer and return it.

    Every time-grid relation in this pipeline (dt_obs / dt_fine,
    dt_window / dt_fine, dt_fine / dt_solver, ...) has to be an exact
    integer ratio, otherwise observation times, window boundaries and
    reference-solver steps silently drift apart over a long assimilation
    run. Failing loudly here is much cheaper than debugging a half-step
    offset later.
    """
    ratio = numerator / denominator
    n = int(round(ratio))
    if abs(ratio - n) > tol:
        raise ValueError(
            f"{label}: {numerator} is not evenly divisible by {denominator}. "
            f"Ratio = {ratio:.10f}, nearest integer = {n}, "
            f"residual = {abs(ratio - n):.2e} > tol={tol}."
        )
    return n


def steps_per_window_exact(dt_window: float, dt_fine: float,
                           tol: float = 1e-9) -> int:
    """
    Integer number of fine steps per DeepONet training window, raising if
    dt_window is not an exact multiple of dt_fine (the same guarantee
    ``build_obs_schedule`` already gives for dt_obs / total_time).
    """
    return check_divisible(dt_window, dt_fine, "dt_window / dt_fine", tol)


def steps_per_fine_exact(dt_fine: float, dt_solver: float,
                         tol: float = 1e-9) -> int:
    """
    Integer number of reference-solver (ETDRK4) steps per fine filter step.

    KS-specific: unlike L96 -- whose ground truth comes from an adaptive
    SciPy solve_ivp that can be asked for arbitrary output times -- the KS
    reference solver is an exponential time-differencing RK4 scheme whose
    coefficients are precomputed for ONE fixed ``dt``. The fine filter step
    therefore has to be an exact integer multiple of that solver dt for the
    truth trajectory to land exactly on the filter's fine grid.
    """
    return check_divisible(dt_fine, dt_solver, "dt_fine / dt_solver", tol)


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
        dt_fine:    fine prediction step (e.g. 0.1).
        dt_obs:     observation interval (e.g. 0.2, 0.5, 1.0).
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
    total_fine_steps = check_divisible(total_time, dt_fine, "total_time / dt_fine", tol)
    steps_per_obs = check_divisible(dt_obs, dt_fine, "dt_obs / dt_fine", tol)
    n_obs = check_divisible(total_time, dt_obs, "total_time / dt_obs", tol)

    # Observation times: first at dt_obs, last at total_time
    obs_times = np.array([(k + 1) * dt_obs for k in range(n_obs)])
    # 0-indexed: after step s the clock reads (s+1)*dt_fine
    # obs at time (k+1)*dt_obs corresponds to step (k+1)*steps_per_obs - 1
    obs_step_indices = np.array([(k + 1) * steps_per_obs - 1 for k in range(n_obs)],
                                dtype=int)

    return obs_times, obs_step_indices, total_fine_steps


def scale_Q_for_fine_steps(
    Q_coarse:         "jnp.ndarray",
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


def scale_inflation_for_fine_steps(
    alpha_coarse:     float,
    steps_per_window: int,
) -> float:
    """
    Scale a window-level multiplicative inflation factor to a per-fine-step value.

    To ensure that compounding inflation across k fine steps equals the desired
    window-level inflation alpha_coarse, the per-step factor must be the k-th root:
        prod_{i=1}^k (alpha_fine) = alpha_coarse  =>  alpha_fine = alpha_coarse ** (1 / k)

    Args:
        alpha_coarse:     Desired covariance inflation over one full window (e.g., 1.05).
        steps_per_window: Integer ratio DT_WINDOW / dt_fine.

    Returns:
        alpha_fine: Per-fine-step multiplicative inflation scalar.
    """
    return float(alpha_coarse ** (1.0 / steps_per_window))