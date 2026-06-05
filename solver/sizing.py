import h5py
from huggingface_hub import hf_hub_download
import numpy as np

def load_split_and_calculate_L(num_test_samples: int = 128) -> dict:
    """
    Loads the Kuramoto-Sivashinsky fixed viscosity dataset from an HDF5 file
    and splits it into training and testing sets.
    
    Args:
        num_test_samples (int): The number of samples to reserve for testing.
        
    Returns:
        dict: A dictionary containing 'train' and 'test' splits, each 
              holding the corresponding arrays ('dt', 'dx', 'pde_140-256', 't', 'x').
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
        print(f"Downloaded to cache. Opening file...")
            
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

        print(f"ASD {x_array[0:4]}")
            
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
            
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check your local path.")