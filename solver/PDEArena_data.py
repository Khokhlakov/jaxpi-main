import h5py
from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt

def load_split_and_calculate_L(num_test_samples: int = 128) -> dict:
    """
    Loads the Kuramoto-Sivashinsky fixed viscosity dataset from an HDF5 file
    and splits it into training and testing sets.
    
    Args:
        num_test_samples (int): The number of samples to reserve for testing.
        
    Returns:
        dict: A dictionary containing 'train' and 'test' splits, each 
              holding the corresponding arrays ('dt', 'dx', 'pde_140-256', 't', 'x', 'L').
    """
    # Fetch repo
    repo_id     = "phlippe/Kuramoto-Sivashinsky-1D"
    file_name   = "KS_train_fixed_viscosity.h5"
    print("=" * 60)
    
    try:
        # 1. Download the file
        print(f"Downloading {file_name}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset"
        )
        print(f"Downloaded to cache. Opening file...\n")
            
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

    # Initialize the output structure
    split_data = {
        "train": {},
        "test": {}
    }
    
    # The exact keys identified from your dataset inspection
    dataset_keys = ["dt", "dx", "pde_140-256", "t", "x"]
    
    with h5py.File(file_path, "r") as f:
        for key in dataset_keys:
            full_path = f"train/{key}"
            
            # Load the entire dataset into memory as a NumPy array
            data = f[full_path][:]
            total_samples = data.shape[0]
            
            # Validate that we have enough samples to split
            if total_samples <= num_test_samples:
                raise ValueError(f"Dataset only has {total_samples} samples, cannot extract {num_test_samples} for testing.")
            
            train_size = total_samples - num_test_samples
            
            # Slice the arrays
            split_data["train"][key] = data[:train_size]
            split_data["test"][key] = data[train_size:]
            
    # Calculate the domain length L for each split
    for split_name in ["train", "test"]:
        dx_array = split_data[split_name]["dx"]
        x_array = split_data[split_name]["x"]
        
        # Number of spatial grid points (Nx = 256)
        num_spatial_points = x_array.shape[1] 
        
        # L = Nx * dx (calculated per sample trajectory)
        split_data[split_name]["L"] = dx_array * num_spatial_points
            
    return split_data

if __name__ == "__main__":
    file_path = "KS_train_fixed_viscosity.h5"  
    
    try:
        data_splits = load_split_and_calculate_L(num_test_samples=128)
        print("Data split successfully!\n")
        
        print("=" * 50)
        print(" DOMAIN LENGTH (L) ANALYSIS ")
        print("=" * 50)
        
        for split in ["train", "test"]:
            L_values = data_splits[split]["L"]
            
            print(f"--- {split.upper()} SPLIT ---")
            print(f"Number of L values: {len(L_values)}")
            print(f"Minimum L value:    {np.min(L_values):.4f}")
            print(f"Maximum L value:    {np.max(L_values):.4f}")
            print(f"Mean L value:       {np.mean(L_values):.4f}")
            print(f"Standard Dev of L:  {np.std(L_values):.4f}\n")
            
        # ==========================================
        # NEW CODE: First Sample Analysis & Plotting
        # ==========================================
        print("=" * 50)
        print(" FIRST SAMPLE ANALYSIS ")
        print("=" * 50)
        
        train_data = data_splits["train"]
        
        # Extract variables for the 0th sample
        u = train_data["pde_140-256"][0]  # Solution field: shape (Time, Space)
        x = train_data["x"][0]            # Spatial grid
        t = train_data["t"][0]            # Time steps
        L_val = train_data["L"][0]        # Domain length
        
        # Metrics
        initial_condition = u[0, :]
        eval_start = t[0]
        eval_end = t[-1]
        
        print(f"Domain Length (L): {L_val:.4f}")
        print(f"Evaluation Time:   {eval_start:.4f} to {eval_end:.4f} (Total Duration: {eval_end - eval_start:.4f})")
        print(f"Initial Condition: \n{initial_condition}\n")
        
        # Plotting the Time-Space evolution using a heatmap
        plt.figure(figsize=(10, 6))
        
        # Create a meshgrid for accurate mapping of space and time
        X, T = np.meshgrid(x, t)
        
        # pcolormesh is ideal for plotting spatiotemporal PDE arrays
        mesh = plt.pcolormesh(X, T, u, shading='auto', cmap='inferno')
        plt.colorbar(mesh, label='u(x, t)')
        
        plt.title('Kuramoto-Sivashinsky Dynamics - First Sample')
        plt.xlabel('Space (x)')
        plt.ylabel('Time (t)')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check your local path.")


# Configuration
L = 32 * np.pi
L = 100
N = 1024
dt = 0.05
t_final = 200.0

print(f"\nConfiguration:")
print(f"  Domain length: L = {L:.4f}")
print(f"  Number of modes: N = {N}")
print(f"  Spatial step: Δx = {L/N:.6f}")
print(f"  Time step: Δt = {dt}")
print(f"  Final time: T = {t_final}")
print(f"  Integration steps: {int(t_final/dt)}")
print()

# Create solver
solver = KuramotoSivashinskyAdvanced(L=L, N=N, dt=dt)

# Initial condition
u0 = np.cos(2.0 * np.pi * solver.x / L) + 0.1 * np.cos(4.0 * np.pi * solver.x / L)
solver.set_initial_condition(u0)

# Integrate
solver.integrate(t_final, save_freq=50)