import scipy.io
import jax.numpy as jnp
import numpy as np
import h5py
import os

def get_pi_train_data(filepath="data/l96_forcing_train.h5"):
    """
    Loads the full pre-computed pool of L96 states for PI training.
    """
    with h5py.File(filepath, 'r') as f:
        # Shape: (num_samples, 40) - Note: Adjust to 41 if F is included
        u0_pool = jnp.array(f['u'][:]) 
    return u0_pool

def get_test_dataset(filepath="data/l96_forcing_test.h5", window_pts=51):
    """
    Loads dense test trajectories and extracts the first window for L2 logging.
    """
    with h5py.File(filepath, 'r') as f:
        u_test = f['u'][:]  # Shape: (num_ics, num_test_pts, 40)
        t_test = f['t'][:]  # Shape: (num_test_pts,)

    # Extract only the first window for evaluation
    x_ref = jnp.array(u_test[:, :window_pts, :])
    u0_ref = jnp.array(u_test[:, 0, :])
    t_star = jnp.array(t_test[:window_pts])
    
    return x_ref, u0_ref, t_star


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

# Data driven helper functions
def dd_get_train_data(data_dir="data/", num_files=10, windows_per_traj=31):
    """
    Reads the 10 training files and compiles them into a single tensor.
    Returns: numpy array of shape (62000, 51, 40)
    """
    all_windows = []
    
    for set_idx in range(1, num_files + 1):
        file_path = os.path.join(data_dir, f'l96_train_{set_idx}.h5')
        
        with h5py.File(file_path, 'r') as f:
            # Shape: (200, 1551, 40)
            data = f['usol_all'][:] 
            
            # Slice out the overlapping windows
            for w in range(windows_per_traj):
                start_idx = w * 50
                end_idx = start_idx + 51
                # window_slice shape: (200, 51, 40)
                window_slice = data[:, start_idx:end_idx, :]
                all_windows.append(window_slice)
                
    # Concatenate all 310 blocks (10 files * 31 windows) along the batch axis
    train_data = np.concatenate(all_windows, axis=0)
    return train_data

def dd_get_test_data_rollout(data_dir="data/", windows_per_traj=31, num_ics=200, N=40):
    """
    Reads the test file and maintains the trajectory and window hierarchy.
    Returns: numpy array of shape (200, 31, 51, 40)
    """
    file_path = os.path.join(data_dir, 'l96_test.h5')
    
    # Preallocate the target array
    rollout_data = np.zeros((num_ics, windows_per_traj, 51, N), dtype=np.float32)
    
    with h5py.File(file_path, 'r') as f:
        # Shape: (200, 1551, 40)
        data = f['usol_all'][:]
        
        for w in range(windows_per_traj):
            start_idx = w * 50
            end_idx = start_idx + 51
            
            # Assign directly into the preallocated structure
            rollout_data[:, w, :, :] = data[:, start_idx:end_idx, :]
            
    return rollout_data

def dd_get_test_data(data_dir="data/l96_test.h5", windows_per_traj=31):
    """
    Reads the test file and compiles it into a flat tensor of individual windows.
    Returns: numpy array of shape (6200, 51, 40)
    """
    # Leverage the rollout function to get the hierarchical data
    rollout_data = dd_get_test_data_rollout(data_dir, windows_per_traj)
    
    # Reshape from (200, 31, 51, 40) to (6200, 51, 40)
    # This collapses the first two dimensions (trajectories and windows) into a single batch dimension
    test_data = rollout_data.reshape(-1, 51, rollout_data.shape[-1])
    
    return test_data