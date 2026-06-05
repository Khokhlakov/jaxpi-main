import h5py
import numpy as np
import matplotlib.pyplot as plt

from ks_solver_CPU import KuramotoSivashinskyAdvanced 

def load_local_ks_data(file_path: str = "solver\ks_test_data.h5") -> dict:
    """
    Loads the Kuramoto-Sivashinsky dataset from a local HDF5 file.
    Expects a dataset 'u' and attributes 'L', 'N', 'dt'.
    """
    print("=" * 60)
    print(f"Loading local dataset from {file_path}...")
    
    data = {}
    with h5py.File(file_path, "r") as f:
        data["u"] = f["u"][:]
        data["L"] = f.attrs["L"]
        data["N"] = f.attrs["N"]
        data["dt"] = f.attrs["dt"]
        
    return data

if __name__ == "__main__":
    file_path = "solver\ks_test_data.h5"  
    
    try:
        data = load_local_ks_data(file_path)
        print("Data loaded successfully!\n")
        
        u_dataset = data["u"]
        L_val = data["L"]
        N_modes = data["N"]
        dt_val = data["dt"]
        
        # Handle potential 3D arrays (e.g., if multiple samples were saved as a batch)
        if u_dataset.ndim == 3:
            print("Dataset has 3 dimensions (Samples, Time, Space). Selecting the first sample.")
            u = u_dataset[0]
        else:
            u = u_dataset
            
        # Reconstruct spatial and temporal grids missing from the basic .h5 structure
        x = np.linspace(0, L_val, N_modes, endpoint=False)
        t = np.arange(u.shape[0]) * dt_val
        
        # ==========================================
        # SAMPLE ANALYSIS 
        # ==========================================
        print("=" * 50)
        print(" DATASET SAMPLE ANALYSIS ")
        print("=" * 50)
        
        initial_condition = u[0, :]
        eval_start = t[0]
        eval_end = t[-1]
        t_final = eval_end - eval_start
        
        print(f"Domain Length (L): {L_val:.4f}")
        print(f"Number of Modes (N): {N_modes}")
        print(f"Evaluation Time:   {eval_start:.4f} to {eval_end:.4f} (Total Duration: {t_final:.4f})")
        print(f"Initial Condition: \n{initial_condition}\n")
        
        # ==========================================
        # LOCAL SOLVER EXECUTION
        # ==========================================
        print("=" * 50)
        print(" LOCAL SOLVER EXECUTION ")
        print("=" * 50)
        
        # Assuming the dt stored in the .h5 acts as both integration step and frame interval
        calculated_save_freq = 1 

        print(f"Configuration mapped from local dataset:")
        print(f"  Domain length: L = {L_val:.4f}")
        print(f"  Number of modes: N = {N_modes}")
        print(f"  Spatial step: Δx = {L_val/N_modes:.6f}")
        print(f"  Time step: Δt = {dt_val}")
        print(f"  Final time: T = {t_final:.4f}")
        print(f"  Integration steps: {int(t_final/dt_val)}")
        print(f"  Save frequency: {calculated_save_freq} steps (1:1 with dataset)\n")

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
                t_solver = np.array(solver.time_history) + eval_start
                u_solver = np.array(solver.solution_history)
                
                # --- Plot 1: Side-by-side Dataset vs Solver comparison ---
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                
                # Dataset Plot
                X, T = np.meshgrid(x, t)
                mesh1 = axes[0].pcolormesh(X, T, u, shading='auto', cmap='inferno')
                axes[0].set_title('Dataset (ks_test_data.h5)')
                axes[0].set_xlabel('Space ($x$)')
                axes[0].set_ylabel('Absolute Time ($t$)')
                fig.colorbar(mesh1, ax=axes[0], label='$u(x, t)$')
                
                # Solver Plot
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
            
            plt.title('Kuramoto-Sivashinsky Dynamics - Local Dataset')
            plt.xlabel('Space ($x$)')
            plt.ylabel('Time ($t$)')
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure the file is in the same directory as the script.")
    except KeyError as e:
        print(f"Error reading dataset structure: missing key {e}. Ensure the .h5 file has 'u' and attributes 'L', 'N', 'dt'.")