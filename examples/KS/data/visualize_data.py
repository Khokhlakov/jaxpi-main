import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_sparse_training_lines(filename="ks_train_data.h5", sample_idx=0, save_pdf=False):
    """
    Reads the sparse training data and plots line profiles for t=0, 1.0, and 2.0.
    Since this dataset stores 1 state per time unit, these correspond to indices 0, 1, and 2.
    """
    with h5py.File(filename, "r") as f:
        u_data = f["u"][sample_idx]
        N = 128
        
    x_nodes = np.arange(N)
    
    plt.figure(figsize=(8, 5))
    
    # Indices 0, 1, 2 correspond to t=0, 1.0, 2.0
    for t_idx in [0, 1, 2]:
        plt.plot(x_nodes, u_data[t_idx, :], label=f"t = {t_idx}.0")
        
    plt.title(f"Sparse Training Data: Node Profiles (Sample {sample_idx})")
    plt.xlabel("Spatial Points (0 to 127)")
    plt.ylabel("Amplitude (u)")
    plt.xlim(0, N - 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_pdf:
        pdf_filename = f"ks_train_data_lines_sample_{sample_idx}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        print(f"Saved plot to {pdf_filename}")
        
    plt.show()

def plot_heatmap(filename, sample_idx=0, states_to_plot=None, title_prefix="", save_pdf=False):
    """
    Reads a dataset and plots a heatmap. 
    Can restrict the plot to a specific number of states (e.g., to isolate 1 window).
    """
    with h5py.File(filename, "r") as f:
        if states_to_plot:
            # Slice up to the requested number of states
            u_data = f["u"][sample_idx, :states_to_plot, :]
        else:
            u_data = f["u"][sample_idx]
            
        N = 128
        
    time_steps = u_data.shape[0]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        u_data, 
        aspect='auto', 
        origin='lower', 
        extent=[0, N - 1, 0, time_steps - 1], 
        cmap='viridis'
    )
    plt.title(f"{title_prefix} Heatmap (Sample {sample_idx})")
    plt.xlabel("Spatial Points (0 to 127)")
    plt.ylabel("Time Steps (states)")
    plt.colorbar(im, label="Amplitude (u)")
    plt.tight_layout()
    
    if save_pdf:
        base_name = filename.replace('.h5', '')
        pdf_filename = f"{base_name}_heatmap_sample_{sample_idx}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        print(f"Saved plot to {pdf_filename}")
        
    plt.show()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Plot Sparse Training Data (Line plots at t=0, 1.0, 2.0)
    plot_sparse_training_lines(
        filename="ks_train_data.h5", 
        sample_idx=0, 
        save_pdf=True
    )
    
    # 2. Plot Test Data (Heatmap of 30 windows -> 1501 states)
    plot_heatmap(
        filename="ks_test_data.h5", 
        sample_idx=0, 
        states_to_plot=1501, 
        title_prefix="Test Data (30 Windows)", 
        save_pdf=True
    )
    
    # 3. Plot Dense Training Data (Heatmap of 1 window -> 200 states + 1 initial = 201 states)
    plot_heatmap(
        filename="ks_train_data_dd.h5", 
        sample_idx=0, 
        states_to_plot=201, 
        title_prefix="Dense Train Data (1 Window)", 
        save_pdf=True
    )