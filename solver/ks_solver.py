import jax
import jax.numpy as jnp
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
    
    def __init__(self, L_op: jnp.ndarray, dt: float, num_contour_points: int = 32):
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
        """Compute all ETD coefficients using vectorized contour integration."""
        r = 1.0
        contour = r * jnp.exp(2j * jnp.pi * jnp.arange(self.num_contour) / self.num_contour)
        
        # Vectorized broadcasting: L_op is (N,), contour is (num_contour,)
        # z_half and z will be of shape (N, num_contour)
        z_half = self.L_op[:, None] * self.dt / 2.0 + contour[None, :]
        z = self.L_op[:, None] * self.dt + contour[None, :]
        
        # Half-step coefficients (for RK stages 1 and 2)
        E_half = jnp.mean(jnp.exp(z_half), axis=1)
        Q_half = jnp.mean((jnp.exp(z_half) - 1) / z_half, axis=1) # phi_1(L*dt/2)
        
        # Full-step coefficients
        E = jnp.mean(jnp.exp(z), axis=1)
        f1 = jnp.mean((-4 - z + jnp.exp(z)*(4 - 3*z + z**2)) / z**3, axis=1)
        f2 = jnp.mean((2 + z + jnp.exp(z)*(-2 + z)) / z**3, axis=1)
        f3 = jnp.mean((-4 - 3*z - z**2 + jnp.exp(z)*(4 - z)) / z**3, axis=1)

        # Ensure real arrays and assign to state
        self.E_half = jnp.real(E_half)
        self.Q_half = jnp.real(Q_half)
        self.E = jnp.real(E)
        self.f1 = jnp.real(f1)
        self.f2 = jnp.real(f2)
        self.f3 = jnp.real(f3)


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
    
    def __init__(self, L: float = 32*jnp.pi, N: int = 256, 
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
        self.x = jnp.linspace(0, L, N, endpoint=False)
        self.dx = L / N
        self.k = jnp.fft.rfftfreq(N, self.dx) * 2 * jnp.pi
        
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
    
    def set_initial_condition(self, u0: jnp.ndarray):
        """
        Set initial condition in real space.
        
        Parameters
        ----------
        u0 : ndarray
            Initial solution on physical grid
        """
        self.u_hat = self._dealias(jnp.fft.rfft(u0))
        self.t = 0.0
        self.solution_history = [jnp.fft.irfft(self.u_hat)]
        self.time_history = [self.t]
    
    def _dealias(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Apply Orszag's 2/3 dealiasing rule."""
        # JAX arrays are immutable; use .at[].set() instead of slice assignment
        return u_hat.at[self.dealiasing_cutoff:].set(0)
    
    def _nonlinear_term(self, u_hat: jnp.ndarray) -> jnp.ndarray:
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
        u = jnp.fft.irfft(u_hat)
        u_squared_hat = jnp.fft.rfft(u**2)

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
        # Note: Kept exact original logic, but `self.etd.a`, `self.etd.b`, etc.
        # are not defined in the ETDCoefficients initialization. Use `step()` instead.
        N_0 = self._nonlinear_term(self.u_hat)
        
        u_1_hat = (self.etd.a * self.u_hat + 
                   (self.dt/2) * self.etd.b * N_0)
        N_1 = self._nonlinear_term(u_1_hat)
        
        u_2_hat = (self.etd.a * self.u_hat + 
                   (self.dt/2) * self.etd.b * N_1)
        N_2 = self._nonlinear_term(u_2_hat)
        
        u_3_hat = (self.etd.a * self.u_hat + 
                   self.dt * self.etd.c * N_2)
        N_3 = self._nonlinear_term(u_3_hat)
        
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
        n_steps = int(jnp.round((t_end - self.t) / self.dt))
        
        for n in range(n_steps):
            self.step()
            
            if (n + 1) % save_freq == 0:
                u_real = jnp.fft.irfft(self.u_hat)
                self.solution_history.append(u_real.copy())
                self.time_history.append(self.t)
                
                if (n + 1) % (10 * save_freq) == 0:
                    energy = jnp.sum(u_real**2) * self.dx
                    print(f"t = {self.t:8.3f}, E = {energy:12.6f}, "
                          f"||u|| = {jnp.linalg.norm(u_real):10.6f}")
    
    def get_solution(self) -> jnp.ndarray:
        """Get current solution in real space."""
        return jnp.fft.irfft(self.u_hat)
    
    def compute_energy(self) -> float:
        """Compute L² energy: E = (1/2) ∫ u² dx."""
        u = self.get_solution()
        return 0.5 * jnp.sum(u**2) * self.dx
    
    def compute_enstrophy(self) -> float:
        """Compute enstrophy: Z = (1/2) ∫ u_x² dx."""
        u = self.get_solution()
        u_x = jnp.fft.irfft(1j * self.k * self.u_hat)
        return 0.5 * jnp.sum(u_x**2) * self.dx
    
    def compute_dissipation(self) -> float:
        """Compute dissipation rate: D = -∫ u_xxxx * u dx."""
        u = self.get_solution()
        # u_xxxx via Fourier derivatives
        u_xxxx = jnp.fft.irfft((1j * self.k)**4 * self.u_hat)
        return -jnp.sum(u_xxxx * u) * self.dx