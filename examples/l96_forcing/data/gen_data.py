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

N           = 40       # Lorenz-96 spatial dimension
num_ics     = 500      # Number of independent trajectories
F_low       = 5.0      # Minimum forcing value
F_high      = 9.0      # Maximum forcing value

M           = 20       # Training windows per trajectory
window_size = 0.25     # Window duration [time units]
L           = 20       # Test trajectory length [in windows]

burn_time   = 5.0      # Initial transient to discard [time units]
dt          = 0.005    # Dense-output time step for test trajectories

SEED        = 42       # RNG seed for reproducibility


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

data_dir   = os.path.join(os.getcwd(), 'examples', 'l96_forcing', 'data')
os.makedirs(data_dir, exist_ok=True)

train_file = os.path.join(data_dir, 'l96_forcing_train.h5')
test_file  = os.path.join(data_dir, 'l96_forcing_test.h5')


# ── Sample Forcing Values ──────────────────────────────────────────────────────

rng      = np.random.default_rng(SEED)
F_values = rng.uniform(F_low, F_high, size=num_ics).astype(np.float64)


# ── Build Combined t_eval (single solver call per trajectory) ─────────────────
#
# Absolute times:
#   t_bounds_abs[k] = burn_time + k·ws          (k = 0 … M-1) — M points
#   t_test_abs[j]   = burn_time + M·ws + j·dt   (j = 0 … 50·L) — 50·L+1 points
#
# The two ranges are disjoint: training ends at (M-1)·ws, test starts at M·ws.
#
t_bounds_abs = burn_time + np.arange(M, dtype=np.float64) * window_size
t_test_abs   = (burn_time + M * window_size
                + np.arange(0, pts_pw * L + 1, dtype=np.float64) * dt)
t_combined   = np.concatenate([t_bounds_abs, t_test_abs])   # (M + 50·L+1,)

# Relative test times for storage (t=0 ↔ physical time M·window_size)
t_test_rel = np.linspace(0.0, L * window_size, num_test_pts, dtype=np.float32)


# ── Pre-allocate Output Arrays ─────────────────────────────────────────────────

# M boundary states per trajectory (t = 0, ws, …, (M-1)·ws  relative to burn end)
u_boundaries = np.zeros((num_ics, M, N), dtype=np.float32)

# Dense test trajectories: num_test_pts = 50·L + 1 states per trajectory
u_test = np.zeros((num_ics, num_test_pts, N), dtype=np.float32)


# ── Generation Loop ────────────────────────────────────────────────────────────

print("=" * 64)
print("Lorenz-96 data generation  (variable forcing F as parameter)")
print("=" * 64)
print(f"  N = {N}  |  trajectories = {num_ics}  |  F ~ U[{F_low}, {F_high}]")
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
    u_boundaries[i] = sol.y[:, :M].T.astype(np.float32)       # (M, N)
    u_test[i]       = sol.y[:, M:].T.astype(np.float32)        # (50·L+1, N)

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
#
F_col    = np.tile(
    F_values[:, None, None].astype(np.float32),
    (1, M, 1)
)                                                               # (ics, M, 1)
u_aug    = np.concatenate([u_boundaries, F_col], axis=2)       # (ics, M, N+1)
u_pooled = u_aug.reshape(num_ics * M, N + 1)                   # (ics·M, N+1)


# ── Save Training Archive ──────────────────────────────────────────────────────

print(f"Saving training data  →  {train_file}")
with h5py.File(train_file, 'w') as f:
    f.create_dataset('u',
                     data=u_pooled,
                     dtype='float32',
                     compression='gzip')
    f.create_dataset('F',
                     data=F_values.astype('float32'))
    f.create_dataset('t_boundaries',
                     data=(np.arange(M) * window_size).astype('float32'))
    f.attrs.update({
        'description': (
            'Pooled Lorenz-96 boundary states for DeepONet training. '
            'Each row is a 41-D initial condition: u[:N] = L96 state, '
            'u[N] = F (forcing). '
            'Row i → trajectory (i // M), window start (i % M).'
        ),
        'num_samples': num_ics * M,
        'num_ics':     num_ics,
        'M':           M,
        'N':           N,
        'window_size': window_size,
        'F_low':       F_low,
        'F_high':      F_high,
        'burn_time':   burn_time,
    })

print(f"  u : {u_pooled.shape}  "
      f"(columns 0–{N-1}: L96 state,  column {N}: F)")


# ── Save Test Archive ──────────────────────────────────────────────────────────

print(f"\nSaving test data      →  {test_file}")
with h5py.File(test_file, 'w') as f:
    f.create_dataset('u',
                     data=u_test,
                     dtype='float32',
                     compression='gzip')
    f.create_dataset('F',
                     data=F_values.astype('float32'))
    f.create_dataset('t',
                     data=t_test_rel)
    f.attrs.update({
        'description': (
            f'Dense Lorenz-96 test trajectories (dt={dt}) starting at '
            f't = M * window_size = {M * window_size} '
            f'(one window beyond the training horizon).'
        ),
        'num_ics':      num_ics,
        'L':            L,
        'N':            N,
        'window_size':  window_size,
        'num_test_pts': num_test_pts,
        'start_time':   float(M * window_size),
    })

print(f"  u : {u_test.shape}  "
      f"(time range: [{M * window_size:.4f}, {(M + L) * window_size:.4f}])")