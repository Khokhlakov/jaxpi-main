"""
Output layout in train_rollouts.mat:
  u0_original   : (num_initial_ics, N)            — original ICs (slot 0)
  u0_rollout_1  : (num_initial_ics, N)            — 1 window forward
  u0_rollout_2  : (num_initial_ics, N)            — 2 windows forward
  ...
  u0_rollout_K  : (num_initial_ics, N)            — K = max_additions windows
  t_window      : scalar                          — window length in time units
  max_additions : scalar
  num_initial_ics : scalar
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
 
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N               = 40
F               = 6.0
mean_ic         = 6.0
std_ic          = 2.0
num_initial_ics = 10000
max_additions   = 12
seed            = 42
 
# Window length 
t_window_start = 0.0
t_window_end   = 0.25
 
output_path = os.path.join(
    os.getcwd(), "examples", "l96_n40_f6_ics", "data", "train_rollouts_025.mat"
)
 
# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------
def lorenz_96(t, state, F=6.0):
    x_p1 = np.roll(state, -1)
    x_m1 = np.roll(state,  1)
    x_m2 = np.roll(state,  2)
    return (x_p1 - x_m2) * x_m1 - state + F
 
 
def rollout_one_window(u0_batch: np.ndarray, dt: float) -> np.ndarray:
    """
    Advance every IC in u0_batch forward by one window of length dt.
 
    Args:
        u0_batch : (num_ics, N) array of initial conditions.
        dt       : window length (time units).
 
    Returns:
        (num_ics, N) array of states at t = dt.
    """
    results = np.empty_like(u0_batch)
    for i, u0 in enumerate(u0_batch):
        sol = solve_ivp(
            lorenz_96,
            [0.0, dt],
            u0,
            method="LSODA",
            rtol=1e-10,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed for IC {i}: {sol.message}"
            )
        results[i] = sol.y[:, -1]
    return results
 
 
rng = np.random.default_rng(seed)
u0_original_np = mean_ic + std_ic * rng.standard_normal(
    size=(num_initial_ics, N)
).astype(np.float64)
 
dt = t_window_end - t_window_start
 
print(f"Generating {max_additions} rollout slots for {num_initial_ics} ICs.")
print(f"Window length: {dt}  |  F = {F}  |  N = {N}")
print(f"Output: {output_path}\n")
 
# ---------------------------------------------------------------------------
# Roll out cumulatively: slot k = u0_original advanced k windows
# ---------------------------------------------------------------------------
save_dict = {
    "u0_original":     u0_original_np,
    "t_window":        np.array([dt]),
    "max_additions":   np.array([max_additions]),
    "num_initial_ics": np.array([num_initial_ics]),
}
 
u_current = u0_original_np.copy()
 
for addition in range(1, max_additions + 1):
    t0 = time.time()
    print(f"Computing rollout slot {addition}/{max_additions}...", end=" ", flush=True)
 
    u_current = rollout_one_window(u_current, dt)
    save_dict[f"u0_rollout_{addition}"] = u_current.copy()
 
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  |  "
          f"mean={u_current.mean():.3f}  std={u_current.std():.3f}")
 
# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
savemat(output_path, save_dict)
print(f"\nSaved to {output_path}")
