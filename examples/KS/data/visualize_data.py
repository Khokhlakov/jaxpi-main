import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_ks_data(filename="ks_test_data.h5", sample_idx=0):
    """
    Reads the HDF5 dataset and plots line profiles and a trajectory heatmap.
    """
    # 1. Load the dataset and metadata
    with h5py.File(filename, "r") as f:
        u_data = f["u"][:]
        L = f.attrs["L"]
        N = f.attrs["N"]
        
        # Determine total time units saved in this file
        time_steps = u_data.shape[1] 
        
    print(f"Loaded {filename}: shape {u_data.shape}")

    # 2. Reconstruct grids
    x = np.linspace(0, L, N, endpoint=False)
    t = np.arange(time_steps)

    # Extract the requested sample (e.g., the first trajectory)
    u_sample = u_data[sample_idx]

    # 3. Setup the Matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: Lineplots of a few x states ---
    # Pick a few specific time indices to plot (start, middle, and end)
    t_indices = [0, time_steps // 2, time_steps - 1]
    
    for t_idx in t_indices:
        ax1.plot(x, u_sample[t_idx, :], label=f"t = {t_idx}")
        
    ax1.set_title(f"Spatial Profiles (Sample {sample_idx})")
    ax1.set_xlabel("Spatial Domain (x)")
    ax1.set_ylabel("Amplitude (u)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot B: Trajectory Heatmap ---
    # extent=[x_min, x_max, t_min, t_max] maps the array indices to physical/time units
    im = ax2.imshow(
        u_sample, 
        aspect='auto', 
        origin='lower', 
        extent=[0, L, 0, time_steps - 1], 
        cmap='RdBu_r' # Red-Blue colormap is standard for wave dynamics
    )
    ax2.set_title(f"Spatiotemporal Heatmap (Sample {sample_idx})")
    ax2.set_xlabel("Spatial Domain (x)")
    ax2.set_ylabel("Time (t)")
    fig.colorbar(im, ax=ax2, label="Amplitude (u)")

    plt.tight_layout()
    plt.show()

# ==========================================
# Example usage block
# ==========================================
if __name__ == "__main__":
    # To visualize the 50 time-unit trajectory, use the test data
    visualize_ks_data("ks_test_data.h5", sample_idx=0)
    visualize_ks_data("ks_train_data.h5", sample_idx=0)