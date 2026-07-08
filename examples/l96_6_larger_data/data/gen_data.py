"""
gen_data.py — Lorenz-96 data generation for a physics-informed DeepONet
              with forcing parameter F treated as a learnable network input.

Design
------
* F is sampled independently per trajectory from U[F_low, F_high].
* Each state is augmented to 41-D: [u_0, …, u_39, F].
* A burn-in period removes transient behaviour before data are recorded.
* One continuous solver call per trajectory covers burn-in → training
  boundaries → dense test output with no restarts.

Output files
------------
examples/l96_forcing/data/l96_forcing_train.h5
    u            : (num_ics·M, N+1)  float32  — states [u, F] pooled across
                                                 all trajectories and windows;
                                                 every row is an initial condition
    F            : (num_ics,)        float32  — per-trajectory F values
    t_boundaries : (M,)              float32  — window-start times relative to
                                                end of burn-in

examples/l96_forcing/data/l96_forcing_test.h5
    u  : (num_ics, num_test_pts, N)  float32  — dense test trajectories
    F  : (num_ics,)                  float32  — per-trajectory F values
    t  : (num_test_pts,)             float32  — times relative to M·window_size

Time axis (per trajectory)
--------------------------
  [0,          burn_time)              ← integrated; nothing recorded (burn-in)
  [burn_time,  burn_time+(M-1)·ws]    ← M boundary states recorded (training)
  [burn_time+M·ws,                     ← 50·L+1 dense states recorded (test)
   burn_time+(M+L)·ws]

Note: the state at burn_time + M·ws belongs exclusively to the test set;
      the training set contains exactly M time stamps per trajectory.
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
import h5py


# ── Configurable Parameters ────────────────────────────────────────────────────

N           = 40       
num_ics     = 500  
F_val       = 6.0    

M           = 300      # Training windows per trajectory
M_pi        = 1 * M    # PI training windows
window_size = 0.25     # Window duration [time units]
L           = 30       # Test trajectory length [in windows]

burn_time   = 15.0     # 80 windows of 0.25
dt          = 0.005    

SEED        = 42       

# ── Derived Quantities ─────────────────────────────────────────────────────────

pts_pw       = int(round(window_size / dt))   # Points per window (e.g. 50)
num_test_pts = L * pts_pw + 1                 # Test points per trajectory (e.g. 51)

assert abs(window_size / dt - pts_pw) < 1e-10, (
    f"window_size ({window_size}) must be exactly divisible by dt ({dt})."
)


# ── Lorenz-96 ODE ──────────────────────────────────────────────────────────────

def lorenz96(t, u, N, F):
    """
    Lorenz-96 right-hand side:
        du_k/dt = (u_{k+1} − u_{k−2}) · u_{k−1} − u_k + F
    with periodic boundary conditions.
    """
    return (np.roll(u, -1) - np.roll(u, 2)) * np.roll(u, 1) - u + F


# ── Output Paths ───────────────────────────────────────────────────────────────

data_dir   = os.getcwd()
os.makedirs(data_dir, exist_ok=True)

train_file_pi = os.path.join(data_dir, 'l96_forcing_train.h5')
train_file_dd = os.path.join(data_dir, 'l96_forcing_train_dd.h5')
test_file     = os.path.join(data_dir, 'l96_forcing_test.h5')


# ── Sample Forcing Values ──────────────────────────────────────────────────────

rng      = np.random.default_rng(SEED)
F_values = np.full((num_ics,), F_val, dtype=np.float64)


# ── Build Combined t_eval (single solver call per trajectory) ─────────────────
#
# Training: Dense integration for M windows (M * 50 + 1 points)
t_train_abs = burn_time + np.arange(M_pi * pts_pw + 1, dtype=np.float64) * dt

# Gap Phase: We skip exactly 1 window (0.25 units) to avoid train/test intersection.
# The solver will integrate through this period without saving the states.

# Test Phase: Starts at (M + 1) * ws
t_test_abs = burn_time + (M_pi + 1) * window_size + np.arange(L * pts_pw + 1, dtype=np.float64) * dt

t_combined = np.concatenate([t_train_abs, t_test_abs])   

# Relative test times for storage (t=0 ↔ physical time (M+1)·window_size)
t_test_rel = np.linspace(0.0, L * window_size, num_test_pts, dtype=np.float32)

# ── Pre-allocate Output Arrays ─────────────────────────────────────────────────

# PI: 21 boundary states per trajectory (0, ws, 2ws, ..., M*ws)
u_pi_bounds = np.zeros((num_ics, M_pi + 1, N), dtype=np.float32)

# DD: 20 windows per trajectory, 51 states per window
u_dd_windows = np.zeros((num_ics, M, pts_pw + 1, N), dtype=np.float32)

# Test: Dense test trajectories
u_test = np.zeros((num_ics, num_test_pts, N), dtype=np.float32)


# ── Generation Loop ────────────────────────────────────────────────────────────

print("=" * 64)
print("Lorenz-96 data generation  (variable forcing F as parameter)")
print("=" * 64)
print(f"  N = {N}  |  trajectories = {num_ics}  |  F = {F_val}")
print(f"  Burn-in  : {burn_time} t.u.")
print(f"  Training : M = {M} states × {num_ics} trajectories  "
      f"→  {num_ics * M} pooled samples")
print(f"  Test     : L = {L} window(s),  {num_test_pts} pts / trajectory")
print()

t_gen_start = time.time()

for i in range(num_ics):
    F_i = float(F_values[i])

    # Initial condition near the L96 attractor (mean state ≈ F)
    u0 = rng.normal(loc=F_i, scale=2.0, size=N)

    # ── Single continuous integration ────────────────────────────────────────
    #   The solver integrates from t=0; t_combined starts at burn_time, so the
    #   transient [0, burn_time) is computed internally but not saved.
    sol = solve_ivp(
        lorenz96,
        t_span=(0.0, float(t_combined[-1])),
        y0=u0,
        t_eval=t_combined,
        method='LSODA',
        args=(N, F_i),
        rtol=1e-13,
        atol=1e-14,
    )

    if not sol.success:
        print(f"  [!] Integration failed for trajectory {i}  (F = {F_i:.4f})")

    # sol.y shape: (N,  M + 50·L+1)
    #   cols 0   … M-1     → M training boundary states (t = k·ws, k=0..M-1)
    #   cols M   … M+50·L  → test states (t = M·ws to (M+L)·ws)

    # sol.y shape: (N, len(t_combined))
    len_train = len(t_train_abs)
    
    # 1. Split the continuous output back into Train and Test segments
    u_train_dense = sol.y[:, :len_train].T.astype(np.float32) # Shape: (1001, N)
    u_test_dense  = sol.y[:, len_train:].T.astype(np.float32) # Shape: (1001, N)

    # 2. Extract Sparse PI states (every pts_pw index)
    idx_bounds = np.arange(0, len_train, pts_pw)
    u_pi_bounds[i] = u_train_dense[idx_bounds]

    # 3. Extract Dense DD windows (overlapping slices of size 51)
    for k in range(M):
        u_dd_windows[i, k] = u_train_dense[k * pts_pw : k * pts_pw + pts_pw + 1]

    # 4. Store test states
    u_test[i] = u_test_dense

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_gen_start
        print(f"  {i + 1:3d} / {num_ics}  |  {elapsed:.1f} s elapsed")

print(f"\nGeneration complete: {time.time() - t_gen_start:.2f} s total\n")


# ── Assemble Pooled Training States ───────────────────────────────────────────
#
# Every boundary state across all trajectories and windows is treated as an
# independent initial condition.  F is appended as the 41st component so that
# the network receives (state, F) in a single 41-D vector.
#
#   u_pooled : (num_ics·M, N+1)   — last column is F
# F_col shape: (ics, M_pi+1, 1)
u_pooled_pi = u_pi_bounds.reshape(num_ics * (M_pi + 1), N)          # Pool of individual states (40-D)

# --- 2. Flatten DD Dataset ---
# F_col shape: (ics, M, 51, 1)
u_pooled_dd = u_dd_windows.reshape(num_ics * M, pts_pw + 1, N)   # Pool of individual windows (40-D)

# ── Save Training Archive (Physics-Informed) ──────────────────────────────────
print(f"Saving PI training data → {train_file_pi}")
with h5py.File(train_file_pi, 'w') as f:
    f.create_dataset('u', data=u_pooled_pi, dtype='float32', compression='gzip')
    f.create_dataset('F', data=F_values.astype('float32'))
    f.create_dataset('t_boundaries', data=(np.arange(M_pi + 1) * window_size).astype('float32'))
    f.attrs.update({
        'description': 'Flattened pool of individual L96 states for PI training. (40 dims)',
        'num_samples': num_ics * (M_pi + 1),
        'window_size': window_size,
    })
print(f"  u_pi : {u_pooled_pi.shape}")

# ── Save Training Archive (Data-Driven) ───────────────────────────────────────
print(f"Saving DD training data → {train_file_dd}")
with h5py.File(train_file_dd, 'w') as f:
    f.create_dataset('u', data=u_pooled_dd, dtype='float32', compression='gzip')
    f.create_dataset('F', data=F_values.astype('float32'))
    f.attrs.update({
        'description': 'Flattened pool of dense L96 windows for DD training. Each contains 51 states. (40 dims)',
        'num_samples': num_ics * M,
        'window_size': window_size,
    })
print(f"  u_dd : {u_pooled_dd.shape}")

# ── Save Test Archive ──────────────────────────────────────────────────────────
print(f"\nSaving test data      →  {test_file}")
with h5py.File(test_file, 'w') as f:
    f.create_dataset('u', data=u_test, dtype='float32', compression='gzip')
    f.create_dataset('F', data=F_values.astype('float32'))
    f.create_dataset('t', data=t_test_rel)
    f.attrs.update({
        'description': 'Dense Lorenz-96 test trajectories.',
        'start_time':  float((M_pi + 1) * window_size), # Updated to reflect new offset
    })
print(f"  u_test : {u_test.shape}")