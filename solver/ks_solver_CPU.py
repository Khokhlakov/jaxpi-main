"""
Kuramoto-Sivashinsky spectral solver with robust ETDRK4 implementation.

Uses contour integrals in the complex plane to compute ETD coefficients,
avoiding the cancellation errors that occur with naive formulas for small dt.

Reference: Cox & Matthews (2002), "Exponential Time Differencing for stiff systems"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from pathlib import Path


class ETDCoefficients:
    """
    Compute ETDRK4 coefficients using contour integrals.
    
    For the system: du/dt = L*u + N(u)
    
    The ETDRK4 scheme requires coefficients that are computed using:
        φ_j(z) = (1/2πi) ∮_Γ e^(ξ*dt) / ξ^j (ξ - z)^(-j) dξ
    
    Using a circular contour avoids cancellation errors.
    """
    
    def __init__(self, L_op: np.ndarray, dt: float, num_contour_points: int = 32):
        """
        Initialize ETD coefficient calculator.
        
        Parameters
        ----------
        L_op : ndarray
            Linear operator eigenvalues (diagonal in Fourier space)
        dt : float
            Time step
        num_contour_points : int
            Number of points on integration contour
        """
        self.L_op = L_op
        self.dt = dt
        self.num_contour = num_contour_points
        self.N = len(L_op)
        
        # Compute coefficients
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        """Compute all ETD coefficients using contour integration."""
        r = 1.0
        contour = r * np.exp(2j * np.pi * np.arange(self.num_contour) / self.num_contour)
        
        # Half-step coefficients (for RK stages 1 and 2)
        self.E_half = np.zeros(self.N, dtype=complex)
        self.Q_half = np.zeros(self.N, dtype=complex) # phi_1(L*dt/2)
        
        # Full-step coefficients
        self.E = np.zeros(self.N, dtype=complex)
        self.f1 = np.zeros(self.N, dtype=complex)
        self.f2 = np.zeros(self.N, dtype=complex)
        self.f3 = np.zeros(self.N, dtype=complex)

        for i in range(self.N):
            L = self.L_op[i]
            
            # Half step contour
            z_half = L * self.dt / 2.0 + contour
            self.E_half[i] = np.mean(np.exp(z_half))
            self.Q_half[i] = np.mean((np.exp(z_half) - 1) / z_half)
            
            # Full step contour
            z = L * self.dt + contour
            self.E[i] = np.mean(np.exp(z))
            
            # KT05 ETDRK4 coefficients
            self.f1[i] = np.mean((-4 - z + np.exp(z)*(4 - 3*z + z**2)) / z**3)
            self.f2[i] = np.mean((2 + z + np.exp(z)*(-2 + z)) / z**3)
            self.f3[i] = np.mean((-4 - 3*z - z**2 + np.exp(z)*(4 - z)) / z**3)

        # Ensure real arrays
        self.E_half = np.real(self.E_half)
        self.Q_half = np.real(self.Q_half)
        self.E = np.real(self.E)
        self.f1 = np.real(self.f1)
        self.f2 = np.real(self.f2)
        self.f3 = np.real(self.f3)


class KuramotoSivashinskyAdvanced:
    """
    Advanced spectral solver for the Kuramoto-Sivashinsky equation.
    
    ∂u/∂t = -u_xxxx - u_xx - (1/2)(u²)_x
    
    Features:
    - Full spectral treatment of linear terms
    - Real-space nonlinear computation
    - Orszag 2/3 dealiasing
    - ETDRK4 time integration with robust coefficient calculation
    """
    
    def __init__(self, L: float = 32*np.pi, N: int = 256, 
                 dt: float = 0.01, dealiasing: float = 2/3):
        """
        Initialize the solver.
        
        Parameters
        ----------
        L : float
            Domain length
        N : int
            Number of Fourier modes
        dt : float
            Time step
        dealiasing : float
            Fraction of modes to keep (Orszag rule: 2/3)
        """
        self.L = L
        self.N = N
        self.dt = dt
        
        # Spatial grid and Fourier modes
        self.x = np.linspace(0, L, N, endpoint=False)
        self.dx = L / N
        self.k = np.fft.rfftfreq(N, self.dx) * 2 * np.pi
        
        # Linear operator: L = -k⁴ - k²
        self.L_op = self.k**2 - self.k**4
        
        # Dealiasing setup
        self.dealiasing_cutoff = int(N * dealiasing / 2)
        
        # Compute ETD coefficients
        self.etd = ETDCoefficients(self.L_op, dt)
        
        # Solution storage
        self.u_hat = None
        self.t = 0.0
        
        # History
        self.solution_history = []
        self.time_history = []
    
    def set_initial_condition(self, u0: np.ndarray):
        """
        Set initial condition in real space.
        
        Parameters
        ----------
        u0 : ndarray
            Initial solution on physical grid
        """
        self.u_hat = self._dealias(np.fft.rfft(u0))
        self.t = 0.0
        self.solution_history = [np.fft.irfft(self.u_hat)]
        self.time_history = [self.t]
    
    def _dealias(self, u_hat: np.ndarray) -> np.ndarray:
        """Apply Orszag's 2/3 dealiasing rule."""
        u_dealias = u_hat.copy()
        # rfft only has positive frequencies; zero out the top third.
        u_dealias[self.dealiasing_cutoff:] = 0
        return u_dealias
    
    def _nonlinear_term(self, u_hat: np.ndarray) -> np.ndarray:
        """
        Compute the nonlinear term N(u) = -(1/2)(u²)_x in Fourier space.
        
        Procedure:
        1. IFFT to real space
        2. Square the field
        3. Take spatial derivative
        4. FFT back to Fourier space
        5. Apply dealiasing
        """
        # Transform to real space
        u = np.fft.irfft(u_hat)
        u_squared_hat = np.fft.rfft(u**2)

        # Differentiate in Fourier space (ik * u_hat)
        du2dx_hat = 1j * self.k * u_squared_hat
        N_hat = 0.5 * du2dx_hat

        return -self._dealias(N_hat)
    
    def step2(self) -> float:
        """
        Perform one ETDRK4 step.
        
        Uses the four-stage ETDRK4 scheme:
        
        u_1 = a*u_n + (dt/2)*b*N_n
        u_2 = a*u_n + (dt/2)*c*N(u_1)
        u_3 = a*u_n + (dt)*c*N(u_2)
        u_{n+1} = a*u_n + (dt)*(d*N_n + 2*e*N(u_1) + 2*e*N(u_2) + f*N(u_3))
        
        Returns
        -------
        float
            Current time after step
        """
        # Stage 0: Current nonlinear term
        N_0 = self._nonlinear_term(self.u_hat)
        
        # Stage 1: Half-step prediction
        u_1_hat = (self.etd.a * self.u_hat + 
                   (self.dt/2) * self.etd.b * N_0)
        N_1 = self._nonlinear_term(u_1_hat)
        
        # Stage 2: Half-step correction
        u_2_hat = (self.etd.a * self.u_hat + 
                   (self.dt/2) * self.etd.b * N_1)
        N_2 = self._nonlinear_term(u_2_hat)
        
        # Stage 3: Full-step prediction
        u_3_hat = (self.etd.a * self.u_hat + 
                   self.dt * self.etd.c * N_2)
        N_3 = self._nonlinear_term(u_3_hat)
        
        # Final update: combine all stages
        # The exact coefficients depend on the ETDRK4 variant
        self.u_hat = (self.etd.a * self.u_hat + 
                      self.dt * (self.etd.d * N_0 + 
                                2 * self.etd.c * (N_1 + N_2) + 
                                self.etd.b * N_3))
        
        self.t += self.dt
        return self.t
    
    def step(self) -> float:
        N_0 = self._nonlinear_term(self.u_hat)
        
        # Stage 1 (Evaluated at t + dt/2)
        u_1_hat = self.etd.E_half * self.u_hat + (self.dt/2) * self.etd.Q_half * N_0
        N_1 = self._nonlinear_term(u_1_hat)
        
        # Stage 2 (Evaluated at t + dt/2)
        u_2_hat = self.etd.E_half * self.u_hat + (self.dt/2) * self.etd.Q_half * N_1
        N_2 = self._nonlinear_term(u_2_hat)
        
        # Stage 3 (Evaluated at t + dt)
        u_3_hat = self.etd.E_half * u_1_hat + (self.dt/2) * self.etd.Q_half * (2*N_2 - N_0)
        N_3 = self._nonlinear_term(u_3_hat)
        
        # Final Stage
        self.u_hat = (self.etd.E * self.u_hat + 
                    self.dt * (self.etd.f1 * N_0 + 
                                2 * self.etd.f2 * (N_1 + N_2) + 
                                self.etd.f3 * N_3))
        
        self.t += self.dt
        return self.t
    
    def integrate(self, t_end: float, save_freq: int = 1):
        """
        Integrate the system to time t_end.
        
        Parameters
        ----------
        t_end : float
            Final integration time
        save_freq : int
            Save solution every save_freq steps
        """
        n_steps = int(np.round((t_end - self.t) / self.dt))
        
        for n in range(n_steps):
            self.step()
            
            if (n + 1) % save_freq == 0:
                u_real = np.fft.irfft(self.u_hat)
                self.solution_history.append(u_real.copy())
                self.time_history.append(self.t)
                
                if (n + 1) % (10 * save_freq) == 0:
                    energy = np.sum(u_real**2) * self.dx
                    print(f"t = {self.t:8.3f}, E = {energy:12.6f}, "
                          f"||u|| = {np.linalg.norm(u_real):10.6f}")
    
    def get_solution(self) -> np.ndarray:
        """Get current solution in real space."""
        return np.fft.irfft(self.u_hat)
    
    def compute_energy(self) -> float:
        """Compute L² energy: E = (1/2) ∫ u² dx."""
        u = self.get_solution()
        return 0.5 * np.sum(u**2) * self.dx
    
    def compute_enstrophy(self) -> float:
        """Compute enstrophy: Z = (1/2) ∫ u_x² dx."""
        u = self.get_solution()
        u_x = np.fft.irfft(1j * self.k * self.u_hat)
        return 0.5 * np.sum(u_x**2) * self.dx
    
    def compute_dissipation(self) -> float:
        """Compute dissipation rate: D = -∫ u_xxxx * u dx."""
        u = self.get_solution()
        # u_xxxx via Fourier derivatives
        u_xxxx = np.fft.irfft((1j * self.k)**4 * self.u_hat)
        return -np.sum(u_xxxx * u) * self.dx
    
    def plot_solution(self, figsize=(15, 5)):
        """Visualize solution."""
        u_current = self.get_solution()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Current state
        axes[0].plot(self.x, u_current, 'b-', linewidth=1.5)
        axes[0].set_xlabel('$x$', fontsize=12)
        axes[0].set_ylabel('$u(x,t)$', fontsize=12)
        axes[0].set_title(f'Solution at $t = {self.t:.2f}$', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Spatio-temporal diagram
        if len(self.solution_history) > 1:
            sol_array = np.array(self.solution_history)
            time_array = np.array(self.time_history)
            
            im = axes[1].contourf(self.x, time_array, sol_array, levels=40, cmap='RdBu_r')
            axes[1].set_xlabel('$x$', fontsize=12)
            axes[1].set_ylabel('$t$', fontsize=12)
            axes[1].set_title('Space-time plot of $u(x,t)$', fontsize=12)
            plt.colorbar(im, ax=axes[1], label='$u$')
        
        # Energy spectrum
        spectrum = np.abs(self.u_hat[:self.N//2])
        k_pos = self.k[:self.N//2]
        axes[2].semilogy(k_pos, spectrum + 1e-15, 'b-', linewidth=1.5)
        axes[2].axvline(self.k[self.dealiasing_cutoff], color='r', linestyle='--', 
                       label='Dealiasing cutoff', linewidth=2)
        axes[2].set_xlabel('Wavenumber $k$', fontsize=12)
        axes[2].set_ylabel('$|\hat{u}(k)|$', fontsize=12)
        axes[2].set_title('Fourier spectrum', fontsize=12)
        axes[2].grid(True, alpha=0.3, which='both')
        axes[2].legend()
        
        plt.tight_layout()
        return fig


def run_benchmark():
    """Run a benchmark integration of the KS equation."""
    print("\n" + "="*80)
    print("Kuramoto-Sivashinsky Equation - Spectral Solver Benchmark")
    print("="*80)
    
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
    
    # Initial condition: slightly perturbed
    u0 = np.cos(2.0 * np.pi * solver.x / L) + 0.1 * np.cos(4.0 * np.pi * solver.x / L) #* (1.0 + 0.1 * np.sin(solver.x ))
    solver.set_initial_condition(u0)
    
    print(f"Initial condition norm: ||u₀|| = {np.linalg.norm(u0):.6f}")
    print(f"Initial energy: E₀ = {solver.compute_energy():.6f}")
    print(f"Initial enstrophy: Z₀ = {solver.compute_enstrophy():.6f}\n")
    
    # Integrate
    print("Integrating...")
    solver.integrate(t_final, save_freq=50)
    
    # Final statistics
    print(f"\nFinal statistics (t = {solver.t:.2f}):")
    print(f"  Final solution norm: ||u|| = {np.linalg.norm(solver.get_solution()):.6f}")
    print(f"  Final energy: E = {solver.compute_energy():.6f}")
    print(f"  Final enstrophy: Z = {solver.compute_enstrophy():.6f}")
    print()
    
    # Visualize
    fig = solver.plot_solution()
    output_dir = Path("./ks_solver_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'ks_solution.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_dir / 'ks_solution.png'}")
    
    return solver


if __name__ == "__main__":
    solver = run_benchmark()
    plt.show()