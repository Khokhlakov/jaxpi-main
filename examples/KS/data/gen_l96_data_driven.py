"""
The physics-informed l96_udon.mat uses 200 ICs × 31 windows = ~310 k
supervised pairs — enough for the PINN residual approach where each step is
expensive.  The data-driven model has a much cheaper per-step cost (no
jacfwd), so it benefits from more varied ICs and longer rollouts.
 
This script produces gen_l96_data_driven.mat with:
    • 2 000 ICs drawn from N(6, 2²) (matching training distribution)
    •    40 windows × 0.25 = 10 time units per trajectory
    •    51 points per window (0.005 spacing) → 2 001 total points per IC
    ⟹  2 000 × 40 = 80 000 effective ICs for windowed-IC sampling
 
Output layout (same convention as l96_udon.mat so get_dataset() can load it)
----------------------------------------------------------------------------
    usol_all  : (num_ics, num_points, N)   float64  full trajectories
    u0_all    : (num_ics, N)               float64  original ICs
    t         : (num_points,)              float64  time axis
 
Usage:
    python data/gen_l96_data_driven.py
"""
 
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
 
# ── Parameters ────────────────────────────────────────────────────────────────
N               = 40       # state dimension
F               = 6.0      # forcing constant
NUM_ICS         = 2_0#00    # number of independent trajectories
M_WINDOWS       = 40       # windows per trajectory (40 × 0.25 = 10 time units)
DT_WINDOW       = 0.25     # window length
STEPS_PER_WIN   = 5#0       # time steps per window  (0.005 spacing)
IC_MEAN         = 6.0      # IC sampling mean
IC_STD          = 2.0      # IC sampling std
SEED            = 41
 
OUTPUT_PATH = os.path.join(
    os.getcwd(), "examples", "l96_n40_f6_ics", "data", "gen_l96_data_driven.mat"
)
 
# ── Derived quantities ─────────────────────────────────────────────────────────
T_END      = M_WINDOWS * DT_WINDOW
NUM_POINTS = STEPS_PER_WIN * M_WINDOWS + 1   # inclusive of t=0 and t=T_END
T_EVAL     = np.linspace(0.0, T_END, NUM_POINTS)
 
print(f"Configuration")
print(f"  ICs          : {NUM_ICS}")
print(f"  Windows      : {M_WINDOWS}  (T_end = {T_END:.2f})")
print(f"  Points/traj  : {NUM_POINTS}  (dt = {T_EVAL[1] - T_EVAL[0]:.4f})")
print(f"  Eff. ICs     : {NUM_ICS * M_WINDOWS:,}  (windowed-IC sampling)")
print(f"  Output       : {OUTPUT_PATH}\n")
 
 
# ── ODE ───────────────────────────────────────────────────────────────────────
 
def lorenz96(t: float, u: np.ndarray, F: float = 6.0) -> np.ndarray:
    """Lorenz-96 RHS with periodic boundary conditions."""
    u_p1 = np.roll(u, -1)
    u_m1 = np.roll(u,  1)
    u_m2 = np.roll(u,  2)
    return (u_p1 - u_m2) * u_m1 - u + F
 
 
# ── Simulation ────────────────────────────────────────────────────────────────
rng         = np.random.default_rng(SEED)
u0_all      = IC_MEAN + IC_STD * rng.standard_normal((NUM_ICS, N)).astype(np.float64)
usol_all    = np.zeros((NUM_ICS, NUM_POINTS, N), dtype=np.float32)
wall_start  = time.time()
failed      = 0
 
for i in range(NUM_ICS):
    sol = solve_ivp(
        lorenz96,
        t_span = [0.0, T_END],
        y0     = u0_all[i],
        t_eval = T_EVAL,
        method = "LSODA",
        args   = (F,),
        rtol   = 1e-10,
        atol   = 1e-11,
    )
 
    if sol.success:
        usol_all[i] = sol.y.T.astype(np.float32)
    else:
        # On rare failure keep zeros — the sampler will still draw valid pairs
        # from the other trajectories.
        failed += 1
        print(f"  WARNING: solver failed for IC {i} — {sol.message}")
 
    if (i + 1) % 200 == 0:
        elapsed  = time.time() - wall_start
        eta      = elapsed / (i + 1) * (NUM_ICS - i - 1)
        pct_done = 100.0 * (i + 1) / NUM_ICS
        print(f"  {i+1:>5d}/{NUM_ICS}  ({pct_done:.0f}%)  "
              f"elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min")
 
total_time = time.time() - wall_start
print(f"\nDone: {NUM_ICS - failed}/{NUM_ICS} trajectories in {total_time/60:.1f} min.")
 
# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
savemat(
    OUTPUT_PATH,
    {
        "usol_all": usol_all,
        "u0_all":   u0_all.astype(np.float32),
        "t":        T_EVAL.astype(np.float64),
        "M_windows":    np.array([M_WINDOWS]),
        "dt_window":    np.array([DT_WINDOW]),
        "num_ics":      np.array([NUM_ICS]),
        "steps_per_win":np.array([STEPS_PER_WIN]),
    },
)
print(f"Saved to {OUTPUT_PATH}")
print(f"  Array shapes:  usol_all {usol_all.shape}  u0_all {u0_all.shape}  "
      f"t {T_EVAL.shape}")