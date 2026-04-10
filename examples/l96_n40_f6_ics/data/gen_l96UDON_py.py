import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
import os
import time

# --- Parameters ---
N = 40
F = 2.0

# Narrower time domain for DeepONet/PINN window training
t_span = (0.0, 1.0)
num_points = 101
t_eval = np.linspace(t_span[0], t_span[1], num_points)

# Number of Initial Conditions to sample
num_ics = 1000

# Preallocate arrays to store the dataset
# Shape: (num_ics, num_points, N)
usol_all = np.zeros((num_ics, num_points, N))
u0_all = np.zeros((num_ics, N))

# --- ODE Function ---
def lorenz96(t, u, N, F):
    """
    Vectorized Lorenz '96 using np.roll for periodic boundary conditions
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    # np.roll(u, -1) shifts elements left (equivalent to circshift(u, -1))
    u_plus_1  = np.roll(u, -1)
    u_minus_1 = np.roll(u, 1)
    u_minus_2 = np.roll(u, 2)
    
    return (u_plus_1 - u_minus_2) * u_minus_1 - u + F

# --- Simulation Loop ---
print('Generating trajectories...')
start_time = time.time()

for i in range(num_ics):
    # Randomly sample initial conditions within a typical L96 range
    u0 = np.random.normal(loc=2.0, scale=1.0, size=N)
    
    u0_all[i, :] = u0
    
    # Solve directly on the target time points using LSODA
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
    
    # sol.y shape is (N, num_points). Transpose to (num_points, N) to match MATLAB
    usol_all[i, :, :] = sol.y.T
    
    if (i + 1) % 50 == 0:
        print(f'Completed {i + 1} / {num_ics}')

print(f"Total generation time: {time.time() - start_time:.2f} seconds")

# --- Save the Dataset ---
save_dict = {
    't': t_eval,
    'usol_all': usol_all,
    'u0_all': u0_all
}

# Define the path and save
file_path = os.path.join(
        os.getcwd(), 'examples', 'l96_n40_f6_ics', 'data', 'l96_udon.mat'
    )
savemat(file_path, save_dict)

print(f'Data saved to {file_path}')