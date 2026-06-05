import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the generated dataset
filename = "examples/KS/data/ks_1d_trajectories.h5"

with h5py.File(filename, 'r') as f:
    # Extract the first 3 independent trajectories: shape (3, N_STEPS, N_SPATIAL)
    u_samples = f['u'][0:3] 
    
    # Extract physical and temporal metadata for axis scaling
    L = f.attrs['L_DOMAIN']
    dt = f.attrs['DT']
    n_steps = f.attrs['N_STEPS']
    n_spatial = f.attrs['N_SPATIAL']

# 2. Reconstruct the spatial and temporal grids
x = np.linspace(0, L, n_spatial, endpoint=False)
t = np.arange(n_steps) * dt

# 3. Setup the visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Determine the global minimum and maximum across all 3 samples 
# to ensure the color scale is consistent across the subplots
vmin = np.min(u_samples)
vmax = np.max(u_samples)

# A diverging colormap (like Red-Blue) is ideal for the KS equation 
# because the solutions oscillate around a mean of zero.
cmap = 'RdBu_r'

for i in range(3):
    ax = axes[i]
    
    # We use pcolormesh to map the data accurately to the physical (x, t) grid coordinates
    im = ax.pcolormesh(x, t, u_samples[i], cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    
    ax.set_title(f"Trajectory {i+1}")
    ax.set_xlabel("Space (x)")
    
    if i == 0:
        ax.set_ylabel("Time (t)")

# Add a single colorbar for all subplots
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=40)
cbar.set_label("Amplitude u(x, t)")

plt.suptitle("Kuramoto-Sivashinsky 1D Spatiotemporal Dynamics", fontsize=16, y=0.98)
plt.show()