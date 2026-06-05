"""
Quick Start Guide for the Kuramoto-Sivashinsky Spectral Solver

Examples of basic usage patterns and common workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
from solver.ks_solver_CPU import KuramotoSivashinskyAdvanced


# ==============================================================================
# EXAMPLE 1: Basic Usage
# ==============================================================================
def example_basic():
    L_val   = 32
    N_modes = 128
    dt_val  = 0.05
    t_final = 100

    solver = KuramotoSivashinskyAdvanced(L=L_val, N=N_modes, dt=dt_val)
    u0 = np.cos(solver.x / 16) + 0.1 * np.sin(solver.x / 8)
    print(f"ASD {len(solver.x)}")
    solver.set_initial_condition(u0)

    solver.integrate(t_final, save_freq=10)
    
    t_solver = np.array(solver.time_history)
    u_solver = np.array(solver.solution_history)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Solver Plot
    X_sol, T_sol = np.meshgrid(solver.x, t_solver)
    mesh2 = ax.pcolormesh(X_sol, T_sol, u_solver, shading='auto', cmap='inferno')

    # Configuración de etiquetas y títulos
    ax.set_title(f'1D KS. L={L_val}, N_modes={N_modes}')
    ax.set_xlabel('Space ($x$)')
    ax.set_ylabel('Time ($t$)')  # Se agrega el eje Y ya que no se comparte

    # Barra de color para el gráfico único
    fig.colorbar(mesh2, ax=ax, label='$u(x, t)$')

    plt.tight_layout()
    plt.show()

    solver.plot_solution()
    plt.show()

def example_basic2():
    """Minimal working example."""
    print("\nExample 1: Basic Usage")
    print("-" * 60)
    
    # Create solver with default parameters
    solver = KuramotoSivashinskyAdvanced(
        L=32*np.pi,      # Domain length
        N=256,           # Number of Fourier modes
        dt=0.01,         # Time step
    )
    
    # Set initial condition
    u0 = np.cos(solver.x / 16) + 0.1 * np.sin(solver.x / 8)
    solver.set_initial_condition(u0)
    
    # Integrate to t = 10
    solver.integrate(t_end=10.0, save_freq=20)
    
    # Get solution at current time
    u = solver.get_solution()
    print(f"Current time: t = {solver.t}")
    print(f"Solution norm: ||u|| = {np.linalg.norm(u):.6f}")
    print(f"Energy: E = {solver.compute_energy():.6f}")
    
    return solver


# ==============================================================================
# EXAMPLE 2: Custom Initial Conditions
# ==============================================================================
def example_initial_conditions():
    """Demonstrate various initial conditions."""
    print("\nExample 2: Custom Initial Conditions")
    print("-" * 60)
    
    L = 32 * np.pi
    N = 256
    
    # Single wavenumber
    def ic_single_mode(x):
        return 2.0 * np.cos(x / 16)
    
    # Multiple modes
    def ic_multi_mode(x):
        return np.cos(x/16) + 0.5*np.cos(x/8) + 0.25*np.cos(x/4)
    
    # Random with smoothing
    def ic_random_smooth(x):
        np.random.seed(42)
        u = np.random.randn(N)
        # Smooth with spectral filtering
        from scipy.fftpack import fft, ifft
        u_hat = fft(u)
        u_hat[N//4:3*N//4] = 0  # Kill high frequencies
        return np.real(ifft(u_hat))
    
    # Perturbed traveling wave
    def ic_traveling_wave(x):
        return 2*np.cos(x/8) * (1 + 0.05*np.sin(2*x/3))
    
    initial_conditions = {
        'Single mode': ic_single_mode,
        'Multi-mode': ic_multi_mode,
        'Random smooth': ic_random_smooth,
        'Traveling wave': ic_traveling_wave,
    }
    
    x = np.linspace(0, L, N, endpoint=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, ic_func) in enumerate(initial_conditions.items()):
        u0 = ic_func(x)
        axes[idx].plot(x, u0, 'b-', linewidth=1.5)
        axes[idx].set_title(f'Initial condition: {name}')
        axes[idx].set_xlabel('$x$')
        axes[idx].set_ylabel('$u(x)$')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Saves to current working directory in Windows
    plt.savefig('initial_conditions.png', dpi=150)
    print("Figure saved: initial_conditions.png")


# ==============================================================================
# EXAMPLE 3: Monitoring Diagnostics
# ==============================================================================
def example_monitoring():
    """Track solution diagnostics during integration."""
    print("\nExample 3: Monitoring Diagnostics")
    print("-" * 60)
    
    solver = KuramotoSivashinskyAdvanced(
        L=32*np.pi, N=256, dt=0.01
    )
    
    u0 = np.cos(solver.x/16) * (1 + 0.05*np.sin(solver.x/8))
    solver.set_initial_condition(u0)
    
    # Manual integration with diagnostics
    diagnostics = {
        'time': [],
        'energy': [],
        'enstrophy': [],
        'dissipation': [],
        'l2_norm': [],
        'max': [],
        'min': [],
    }
    
    print("\n{'Time':<8} {'Energy':<12} {'Enstrophy':<12} {'||u||':<12} {'Max':<12}")
    print("-" * 56)
    
    for step in range(100):
        solver.step()
        
        u = solver.get_solution()
        
        diagnostics['time'].append(solver.t)
        diagnostics['energy'].append(solver.compute_energy())
        diagnostics['enstrophy'].append(solver.compute_enstrophy())
        diagnostics['dissipation'].append(solver.compute_dissipation())
        diagnostics['l2_norm'].append(np.linalg.norm(u))
        diagnostics['max'].append(np.max(u))
        diagnostics['min'].append(np.min(u))
        
        if (step + 1) % 25 == 0:
            print(f"{solver.t:<8.2f} {diagnostics['energy'][-1]:<12.6f} "
                  f"{diagnostics['enstrophy'][-1]:<12.6f} "
                  f"{diagnostics['l2_norm'][-1]:<12.6f} "
                  f"{diagnostics['max'][-1]:<12.6f}")
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(diagnostics['time'], diagnostics['energy'], 'b-', linewidth=2)
    axes[0,0].set_ylabel('Energy')
    axes[0,0].set_title('Energy Evolution')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(diagnostics['time'], diagnostics['enstrophy'], 'r-', linewidth=2)
    axes[0,1].set_ylabel('Enstrophy')
    axes[0,1].set_title('Enstrophy Evolution')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(diagnostics['time'], diagnostics['l2_norm'], 'g-', linewidth=2)
    axes[1,0].set_ylabel('||u||')
    axes[1,0].set_title('L² Norm Evolution')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(diagnostics['time'], diagnostics['max'], 'b-', linewidth=2, label='Max')
    axes[1,1].plot(diagnostics['time'], diagnostics['min'], 'r-', linewidth=2, label='Min')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Max/Min Values')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Time')
    
    plt.tight_layout()
    # Saves to current working directory in Windows
    plt.savefig('diagnostics.png', dpi=150)
    print("\nFigure saved: diagnostics.png")


# ==============================================================================
# EXAMPLE 4: Parameter Study
# ==============================================================================
def example_parameter_study():
    """Compare solutions with different initial amplitudes."""
    print("\nExample 4: Parameter Study")
    print("-" * 60)
    
    L = 32 * np.pi
    N = 256
    dt = 0.01
    amplitudes = [0.5, 1.0, 2.0, 3.0]
    
    fig, axes = plt.subplots(len(amplitudes), 1, figsize=(12, 10))
    
    print(f"\n{'Amplitude':<12} {'Energy':<12} {'Enstrophy':<12} {'Chaos?':<10}")
    print("-" * 46)
    
    for idx, A in enumerate(amplitudes):
        solver = KuramotoSivashinskyAdvanced(L=L, N=N, dt=dt)
        
        u0 = A * np.cos(solver.x / 16) * (1 + 0.05 * np.sin(solver.x / 8))
        solver.set_initial_condition(u0)
        solver.integrate(t_end=50.0, save_freq=50)
        
        u_final = solver.get_solution()
        energy = solver.compute_energy()
        enstrophy = solver.compute_enstrophy()
        
        # Simple chaos indicator: high enstrophy relative to energy
        chaos_ratio = enstrophy / (energy + 1e-10)
        is_chaotic = "Yes" if chaos_ratio > 5 else "No"
        
        print(f"{A:<12.1f} {energy:<12.6f} {enstrophy:<12.6f} {is_chaotic:<10}")
        
        # Plot solution
        axes[idx].plot(solver.x, u_final, 'b-', linewidth=1)
        axes[idx].set_ylabel(f'A={A}')
        axes[idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx].set_title('Final States for Different Amplitudes')
        if idx == len(amplitudes) - 1:
            axes[idx].set_xlabel('$x$')
    
    plt.tight_layout()
    # Saves to current working directory in Windows
    plt.savefig('parameter_study.png', dpi=150)
    print("\nFigure saved: parameter_study.png")


# ==============================================================================
# EXAMPLE 5: High-Resolution Run
# ==============================================================================
def example_high_resolution():
    """Run with higher resolution for better accuracy."""
    print("\nExample 5: High-Resolution Run")
    print("-" * 60)
    
    solver = KuramotoSivashinskyAdvanced(
        L=32*np.pi,
        N=512,          # High resolution
        dt=0.005,       # Smaller time step for finer grid
    )
    
    print(f"Grid points: {solver.N}")
    print(f"Spatial resolution: Δx = {solver.dx:.6f}")
    print(f"Time step: Δt = {solver.dt}")
    print(f"Dealiasing cutoff: {solver.dealiasing_cutoff} (keeps {2*solver.dealiasing_cutoff}/512 modes)")
    
    u0 = np.cos(solver.x / 16) + 0.1 * np.sin(solver.x / 8)
    solver.set_initial_condition(u0)
    
    print("\nIntegrating...")
    solver.integrate(t_end=20.0, save_freq=20)
    
    u = solver.get_solution()
    print(f"\nFinal ||u|| = {np.linalg.norm(u):.6f}")
    print(f"Spectrum extent: {np.count_nonzero(np.abs(solver.u_hat) > 1e-10)} active modes")


# ==============================================================================
# EXAMPLE 6: Animation Data Generation
# ==============================================================================
def example_animation_data():
    """Generate data for time-evolution visualization."""
    print("\nExample 6: Animation Data Generation")
    print("-" * 60)
    
    solver = KuramotoSivashinskyAdvanced(L=32*np.pi, N=256, dt=0.01)
    
    u0 = np.cos(solver.x / 16) * (1 + 0.05 * np.sin(solver.x / 8))
    solver.set_initial_condition(u0)
    
    # Integrate and collect data
    times = []
    solutions = []
    
    for i in range(50):
        solver.step()
        if (i + 1) % 10 == 0:
            times.append(solver.t)
            solutions.append(solver.get_solution().copy())
    
    # Create space-time diagram
    sol_array = np.array(solutions)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.contourf(solver.x, times, sol_array, levels=50, cmap='RdBu_r')
    ax.set_xlabel('Position $x$')
    ax.set_ylabel('Time $t$')
    ax.set_title('Space-Time Evolution of u(x,t)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$u(x,t)$')
    
    plt.tight_layout()
    # Saves to current working directory in Windows
    plt.savefig('animation_data.png', dpi=150)
    print("Space-time diagram saved: animation_data.png")
    print(f"Generated {len(times)} time steps from t=0 to t={times[-1]:.2f}")


# ==============================================================================
# MAIN: Run all examples
# ==============================================================================
def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("KURAMOTO-SIVASHINSKY SOLVER - EXAMPLES")
    print("="*70)
    
    example_basic()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
    plt.show()

