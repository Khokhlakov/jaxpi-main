import numpy as np
from scipy.integrate import solve_ivp
import h5py
import os
import time

# --- Parameters ---
N = 40
F = 6.0
num_ics = 200

# --- Window Configuration ---
M = 31  # Number of windows
window_size = 0.25
t_end = M * window_size
t_span = (0.0, t_end)

# Calculate points: 51 points per 0.25 window maintains 0.005 step
num_points = int(50 * M + 1) 
t_eval = np.linspace(t_span[0], t_span[1], num_points)

# --- ODE Function ---
def lorenz96(t, u, N, F):
    u_plus_1  = np.roll(u, -1)
    u_minus_1 = np.roll(u, 1)
    u_minus_2 = np.roll(u, 2)
    return (u_plus_1 - u_minus_2) * u_minus_1 - u + F

# --- Directory Setup ---
data_dir = os.path.join(os.getcwd(), 'examples', 'l96_n40_f6_ics', 'data')
os.makedirs(data_dir, exist_ok=True)

# --- Initial Conditions Array ---
# Initialize the array that will hold the ICs for the current batch
current_ics = np.zeros((num_ics, N))

# Sample the very first set of random ICs for train_1
for i in range(num_ics):
    current_ics[i, :] = np.random.normal(loc=6.0, scale=2.0, size=N)

# --- Main Simulation Loop ---
total_datasets = 11

print(f'Starting generation of {total_datasets} datasets ({num_ics} trajectories, M={M} windows per set)...')
print(f'Time span per dataset: {t_span} | Total points per trajectory: {num_points}')

for set_idx in range(1, total_datasets + 1):
    
    # Determine output filename
    file_name = f'l96_train_{set_idx}.h5' if set_idx <= 10 else 'l96_test.h5'
    file_path = os.path.join(data_dir, file_name)
        
    print(f'\n--- Generating Set {set_idx}/{total_datasets}: {file_name} ---')
    start_time = time.time()
    
    # Preallocate arrays for the current dataset
    usol_all = np.zeros((num_ics, num_points, N))
    u0_all = np.copy(current_ics)
    
    for i in range(num_ics):
        u0 = current_ics[i, :]
        
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
            print(f"Warning: Integration failed for IC {i} in {file_name}")
            
        usol_all[i, :, :] = sol.y.T
        
        if (i + 1) % 50 == 0:
            print(f'  Completed {i + 1} / {num_ics}')

    print(f"Generation time for {file_name}: {time.time() - start_time:.2f} seconds")

    # --- Save the Dataset ---
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('t', data=t_eval, dtype='float32')
        f.create_dataset('usol_all', data=usol_all, dtype='float32', compression='gzip')
        f.create_dataset('u0_all', data=u0_all, dtype='float32')
        # Metadata
        f.attrs['M_windows'] = M
        f.attrs['num_ics'] = num_ics
        f.attrs['window_size'] = window_size

    print(f'Data saved to {file_path}')
    
    # --- Prepare for next iteration ---
    # Extract the states at the last timestep (index -1) across all trajectories.
    # This automatically feeds into `current_ics` to act as the `y0` for the next set.
    current_ics = usol_all[:, -1, :]

    

print("\nAll datasets generated successfully!")