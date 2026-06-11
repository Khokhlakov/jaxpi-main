import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_ks_data(filename="ks_test_data.h5", sample_idx=0, save_pdf=False):
    """
    Reads the HDF5 dataset and plots the data exactly as the neural 
    network processes it (Spatial Points 0 to 255).
    """
    # 1. Load the dataset and metadata
    with h5py.File(filename, "r") as f:
        u_data = f["u"][:]
        N = f.attrs["N"] # We only need N now, L is ignored by the NN
        
        # Determine total time units saved in this file
        time_steps = u_data.shape[1] 
        
    print(f"Loaded {filename}: shape {u_data.shape}")

    # 2. Reconstruct computational grids (NN view)
    # The neural network sees 256 discrete spatial points
    x_nodes = np.arange(N) 
    t = np.arange(time_steps)

    # Extract the requested sample 
    u_sample = u_data[sample_idx]

    # 3. Setup the Matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: Lineplots over the 256 vector nodes ---
    t_indices = [0, time_steps // 2, time_steps - 1]
    
    for t_idx in t_indices:
        ax1.plot(x_nodes, u_sample[t_idx, :], label=f"t = {t_idx}")
        
    ax1.set_title(f"Node Profiles (Sample {sample_idx})")
    ax1.set_xlabel("Spatial Points (0 to 255)")
    ax1.set_ylabel("Amplitude (u)")
    ax1.set_xlim(0, N - 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot B: Trajectory Heatmap (Matching KSUDONEvaluator) ---
    # extent is now mapped to the 256 discrete spatial points
    im = ax2.imshow(
        u_sample, 
        aspect='auto', 
        origin='lower', 
        extent=[0, N - 1, 0, time_steps - 1], 
        cmap='viridis' # Changed to viridis to match the KSUDONEvaluator script exactly
    )
    ax2.set_title(f"NN Trajectory Heatmap (Sample {sample_idx})")
    ax2.set_xlabel("Spatial Points (0 to 255)")
    ax2.set_ylabel("Time (t)")
    fig.colorbar(im, ax=ax2, label="Amplitude (u)")

    plt.tight_layout()

    # --- Save to PDF ---
    if save_pdf:
        base_name = filename.replace('.h5', '')
        pdf_filename = f"{base_name}_nn_view_sample_{sample_idx}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        print(f"Saved plot to {pdf_filename}")

    plt.show()

# ==========================================
# Example usage block
# ==========================================
if __name__ == "__main__":
    # Set save_pdf=True to generate the PDF files
    visualize_ks_data("ks_test_data.h5", sample_idx=0, save_pdf=True)
    visualize_ks_data("ks_train_data.h5", sample_idx=0, save_pdf=True)