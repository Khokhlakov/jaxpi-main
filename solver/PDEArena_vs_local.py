import h5py
from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt

from  ks_solver_advanced import KuramotoSivashinskyAdvanced 

def load_split_and_calculate_L(num_test_samples: int = 128) -> dict:
    """
    Loads the Kuramoto-Sivashinsky fixed viscosity dataset from an HDF5 file
    and splits it into training and testing sets.
    """
    repo_id     = "phlippe/Kuramoto-Sivashinsky-1D"
    file_name   = "KS_train_fixed_viscosity.h5"
    print("=" * 60)
    
    try:
        print(f"Downloading {file_name}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset"
        )
        print(f"Downloaded to cache. Opening file...\n")
            
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

    split_data = {
        "train": {},
        "test": {}
    }
    
    dataset_keys = ["dt", "dx", "pde_140-256", "t", "x"]
    
    with h5py.File(file_path, "r") as f:
        for key in dataset_keys:
            full_path = f"train/{key}"
            data = f[full_path][:]
            total_samples = data.shape[0]
            
            if total_samples <= num_test_samples:
                raise ValueError(f"Dataset only has {total_samples} samples, cannot extract {num_test_samples} for testing.")
            
            train_size = total_samples - num_test_samples
            
            split_data["train"][key] = data[:train_size]
            split_data["test"][key] = data[train_size:]
            
    for split_name in ["train", "test"]:
        dx_array = split_data[split_name]["dx"]
        x_array = split_data[split_name]["x"]
        
        num_spatial_points = x_array.shape[1] 
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
        # FIRST SAMPLE ANALYSIS 
        # ==========================================
        print("=" * 50)
        print(" FIRST SAMPLE ANALYSIS ")
        print("=" * 50)
        
        train_data = data_splits["train"]
        
        idx = 104
        u = train_data["pde_140-256"][idx]  # Shape: (Time, Space)
        x = train_data["x"][idx]            
        t = train_data["t"][idx]            
        L_val = train_data["L"][idx]        
        
        initial_condition = u[0, :]
        eval_start = t[0]
        eval_end = t[-1]
        
        print(f"Domain Length (L): {L_val:.4f}")
        print(f"Evaluation Time:   {eval_start:.4f} to {eval_end:.4f} (Total Duration: {eval_end - eval_start:.4f})")
        print(f"Initial Condition: \n{initial_condition}\n")
        
        # ==========================================
        # LOCAL SOLVER EXECUTION
        # ==========================================
        print("=" * 50)
        print(" LOCAL SOLVER EXECUTION ")
        print("=" * 50)
        
        N_modes = len(x)
        dt_val = train_data["dt"][0] if "dt" in train_data else 0.05 
        t_final = eval_end - eval_start
        
        dataset_frame_dt = t[1] - t[0]
        calculated_save_freq = max(1, int(dataset_frame_dt / dt_val))

        print(f"Configuration mapped from dataset:")
        print(f"  Domain length: L = {L_val:.4f}")
        print(f"  Number of modes: N = {N_modes}")
        print(f"  Spatial step: Δx = {L_val/N_modes:.6f}")
        print(f"  Time step: Δt = {dt_val}")
        print(f"  Final time: T = {t_final:.4f}")
        print(f"  Integration steps: {int(t_final/dt_val)}")
        print(f"  Save frequency: {calculated_save_freq} steps (to match dataset time grid)\n")

        solver_ran_successfully = False

        try:
            # Create solver with extracted dataset params
            solver = KuramotoSivashinskyAdvanced(L=L_val, N=N_modes, dt=dt_val)
            solver.set_initial_condition(initial_condition)
            
            print("Integrating with local solver...")
            solver.integrate(t_final, save_freq=calculated_save_freq)
            print("Integration complete!\n")
            
            solver_ran_successfully = True
            
        except NameError:
            print("--> SKIPPED: 'KuramotoSivashinskyAdvanced' is not imported or defined in this environment.")
            print("--> To run the simulation, ensure your solver module is imported at the top of the file.\n")

        # ==========================================
        # PLOTTING
        # ==========================================
        if solver_ran_successfully:
            try:
                # Convert the internal list history attributes into numpy arrays
                # FIX: Add eval_start to shift the solver's clock to match the dataset phase
                t_solver = np.array(solver.time_history) + eval_start
                u_solver = np.array(solver.solution_history)
                
                # --- Plot 1: Side-by-side Dataset vs Solver comparison ---
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                
                # Dataset Plot
                X, T = np.meshgrid(x, t)
                mesh1 = axes[0].pcolormesh(X, T, u, shading='auto', cmap='inferno')
                axes[0].set_title('Dataset')
                axes[0].set_xlabel('Space ($x$)')
                axes[0].set_ylabel('Absolute Time ($t$)')
                fig.colorbar(mesh1, ax=axes[0], label='$u(x, t)$')
                
                # Solver Plot (Now perfectly in phase)
                X_sol, T_sol = np.meshgrid(x, t_solver)
                mesh2 = axes[1].pcolormesh(X_sol, T_sol, u_solver, shading='auto', cmap='inferno')
                axes[1].set_title('Local Solver')
                axes[1].set_xlabel('Space ($x$)')
                fig.colorbar(mesh2, ax=axes[1], label='$u(x, t)$')
                
                plt.tight_layout()
                plt.show()
                
                # --- Plot 2: Native Solver Diagnostic Output ---
                print("Generating native solver diagnostic plots...")
                solver.plot_solution()
                plt.show()
                
            except AttributeError as e:
                print(f"Plotting Error: {e}")
                
        else:
            # Fallback plot (Dataset only) if the solver isn't imported/fails
            plt.figure(figsize=(10, 6))
            X, T = np.meshgrid(x, t)
            mesh = plt.pcolormesh(X, T, u, shading='auto', cmap='inferno')
            plt.colorbar(mesh, label='$u(x, t)$')
            
            plt.title('Kuramoto-Sivashinsky Dynamics - Dataset (First Sample)')
            plt.xlabel('Space ($x$)')
            plt.ylabel('Time ($t$)')
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check your local path.")