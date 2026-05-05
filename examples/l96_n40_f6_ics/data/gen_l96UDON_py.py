import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
import os
import time

# --- Parameters ---
N = 40
F = 6.0
num_ics = 1000

# --- Window Configuration ---
M = 12  # Number of windows.
window_size = 0.25
t_end = M * window_size
t_span = (0.0, t_end)

# Calculate points: 51 points per 0.25 window (inclusive) maintains 0.005 step
num_points = int(50 * M + 1) 
t_eval = np.linspace(t_span[0], t_span[1], num_points)

# Preallocate arrays
usol_all = np.zeros((num_ics, num_points, N))
u0_all = np.zeros((num_ics, N))

# --- ODE Function ---
def lorenz96(t, u, N, F):
    u_plus_1  = np.roll(u, -1)
    u_minus_1 = np.roll(u, 1)
    u_minus_2 = np.roll(u, 2)
    return (u_plus_1 - u_minus_2) * u_minus_1 - u + F

# --- Simulation Loop ---
print(f'Generating {num_ics} trajectories for M={M} windows...')
print(f'Time span: {t_span} | Total points: {num_points}')
start_time = time.time()

for i in range(num_ics):
    # Randomly sample initial conditions
    u0 = np.random.normal(loc=6.0, scale=2.0, size=N)
    u0_all[i, :] = u0
    
    # Solve using high precision LSODA
    sol = solve_ivp(
        fun=lorenz96, 
        t_span=t_span, 
        y0=u0, 
        t_eval=t_eval, 
        method='LSODA', 
        args=(N, F),
        rtol=1e-13, 
        atol=1e-14
    )
    
    # Check for successful integration
    if not sol.success:
        print(f"Warning: Integration failed for IC {i}")
        
    usol_all[i, :, :] = sol.y.T
    
    if (i + 1) % 50 == 0:
        print(f'Completed {i + 1} / {num_ics}')

print(f"Total generation time: {time.time() - start_time:.2f} seconds")

# --- Save the Dataset ---
save_dict = {
    't': t_eval,
    'usol_all': usol_all,
    'u0_all': u0_all,
    'M_windows': M
}

# Filename includes M to keep data organized
file_name = f'l96_udon.mat'
data_dir = os.path.join(os.getcwd(), 'examples', 'l96_n40_f6_ics', 'data')
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, file_name)

savemat(file_path, save_dict)
print(f'Data saved to {file_path}')
