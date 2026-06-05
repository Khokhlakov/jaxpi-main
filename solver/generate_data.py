import jax
import jax.numpy as jnp
import h5py
import numpy as np
from typing import Callable, Optional, Tuple
from pathlib import Path
from functools import partial

class ETDCoefficients:
    """Compute ETDRK4 coefficients using contour integrals."""
    
    def __init__(self, L_op: jnp.ndarray, dt: float, num_contour_points: int = 32):
        self.L_op = L_op
        self.dt = dt
        self.num_contour = num_contour_points
        self.N = len(L_op)
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        r = 1.0
        contour = r * jnp.exp(2j * jnp.pi * jnp.arange(self.num_contour) / self.num_contour)
        
        z_half = self.L_op[:, None] * self.dt / 2.0 + contour[None, :]
        z = self.L_op[:, None] * self.dt + contour[None, :]
        
        E_half = jnp.mean(jnp.exp(z_half), axis=1)
        Q_half = jnp.mean((jnp.exp(z_half) - 1) / z_half, axis=1)
        
        E = jnp.mean(jnp.exp(z), axis=1)
        f1 = jnp.mean((-4 - z + jnp.exp(z)*(4 - 3*z + z**2)) / z**3, axis=1)
        f2 = jnp.mean((2 + z + jnp.exp(z)*(-2 + z)) / z**3, axis=1)
        f3 = jnp.mean((-4 - 3*z - z**2 + jnp.exp(z)*(4 - z)) / z**3, axis=1)

        self.E_half = jnp.real(E_half)
        self.Q_half = jnp.real(Q_half)
        self.E = jnp.real(E)
        self.f1 = jnp.real(f1)
        self.f2 = jnp.real(f2)
        self.f3 = jnp.real(f3)

class KuramotoSivashinskyAdvanced:
    def __init__(self, L: float = 32.0, N: int = 128, 
                 dt: float = 0.005, dealiasing: float = 2/3):
        self.L = L
        self.N = N
        self.dt = dt
        
        self.x = jnp.linspace(0, L, N, endpoint=False)
        self.dx = L / N
        self.k = jnp.fft.rfftfreq(N, self.dx) * 2 * jnp.pi
        
        self.L_op = self.k**2 - self.k**4
        self.dealiasing_cutoff = int(N * dealiasing / 2)
        self.etd = ETDCoefficients(self.L_op, dt)
    
    @partial(jax.jit, static_argnums=(0,))
    def _dealias(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        return u_hat.at[self.dealiasing_cutoff:].set(0)
    
    @partial(jax.jit, static_argnums=(0,))
    def _nonlinear_term(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        u = jnp.fft.irfft(u_hat)
        u_squared_hat = jnp.fft.rfft(u**2)
        du2dx_hat = 1j * self.k * u_squared_hat
        N_hat = 0.5 * du2dx_hat
        return -self._dealias(N_hat)
    
    @partial(jax.jit, static_argnums=(0,))
    def step_pure(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Pure JAX implementation of the ETDRK4 step."""
        N_0 = self._nonlinear_term(u_hat)
        
        u_1_hat = self.etd.E_half * u_hat + (self.dt/2) * self.etd.Q_half * N_0
        N_1 = self._nonlinear_term(u_1_hat)
        
        u_2_hat = self.etd.E_half * u_hat + (self.dt/2) * self.etd.Q_half * N_1
        N_2 = self._nonlinear_term(u_2_hat)
        
        u_3_hat = self.etd.E_half * u_1_hat + (self.dt/2) * self.etd.Q_half * (2*N_2 - N_0)
        N_3 = self._nonlinear_term(u_3_hat)
        
        u_hat_next = (self.etd.E * u_hat + 
                      self.dt * (self.etd.f1 * N_0 + 
                                 2 * self.etd.f2 * (N_1 + N_2) + 
                                 self.etd.f3 * N_3))
        return u_hat_next

# ==========================================
# Data Generation Logic
# ==========================================

def generate_datasets(
    num_samples: int = 200, 
    L: float = 32.0 * jnp.pi, 
    N: int = 128, 
    dt: float = 0.005, 
    t_burn: float = 50.0,
    max_additions: int = 4
):
    solver = KuramotoSivashinskyAdvanced(L=L, N=N, dt=dt)
    
    # 1. Generate Smooth Initial Noise
    key = jax.random.PRNGKey(42)
    k_array = solver.k
    
    def create_single_ic(k):
        # Create random phases and amplitudes, smooth with exp(-decay)
        keys = jax.random.split(k, 2)
        amp = jax.random.normal(keys[0], k_array.shape) * jnp.exp(-0.1 * k_array**2)
        phase = jax.random.uniform(keys[1], k_array.shape, minval=0, maxval=2*jnp.pi)
        u_hat = solver._dealias(amp * jnp.exp(1j * phase))
        return u_hat

    keys = jax.random.split(key, num_samples)
    u_hat_initial = jax.vmap(create_single_ic)(keys)

    # ==========================================
    # Pre-compute steps as pure Python integers
    # ==========================================
    burn_steps = int(t_burn / dt)
    interval_steps = int(0.5 / dt)

    # 2. Define Scanning Functions 
    # (No @jax.jit needed here; they inherit compilation from simulate_sample)
    def advance_time(u_hat, steps):
        """Advances the state by `steps` without saving intermediate values."""
        def body_fn(u, _):
            return solver.step_pure(u), None
        u_hat_final, _ = jax.lax.scan(body_fn, u_hat, None, length=steps)
        return u_hat_final

    def advance_and_save(u_hat, interval_steps, num_intervals):
        """Advances and saves state at the end of each interval."""
        def step_interval(u, _):
            u_next = advance_time(u, interval_steps)
            return u_next, u_next
        u_hat_final, trajectory = jax.lax.scan(step_interval, u_hat, None, length=num_intervals)
        return u_hat_final, trajectory

    def advance_save_all(u_hat, steps):
        """Advances and saves state at every single step."""
        def body_fn(u, _):
            u_next = solver.step_pure(u)
            return u_next, u_next
        u_hat_final, trajectory = jax.lax.scan(body_fn, u_hat, None, length=steps)
        return u_hat_final, trajectory

    # 3. Parallel Execution over Samples via vmap
    @jax.jit
    def simulate_sample(u_hat_0):
        # A. Burn-in Phase
        u_hat_burned = advance_time(u_hat_0, burn_steps)
        
        # B. Train Phase: Save IC, then advance max_additions times saving every 0.25
        u_train_end, train_trajectory = advance_and_save(u_hat_burned, interval_steps, max_additions)
        
        # Prepend the burned IC to the training trajectory
        train_data = jnp.concatenate([u_hat_burned[None, ...], train_trajectory], axis=0)
        
        # C. Test Phase: advance 0.25 without saving, then save every step for the last 0.25
        u_test_start = advance_time(u_train_end, interval_steps) # First 0.25 silent
        
        # We need 51 values total (start of interval + 50 steps)
        _, test_trajectory = advance_save_all(u_test_start, interval_steps)
        test_data = jnp.concatenate([u_test_start[None, ...], test_trajectory], axis=0)
        
        return train_data, test_data

    print("Simulating (JIT compiling first time)...")
    train_data_hat, test_data_hat = jax.vmap(simulate_sample)(u_hat_initial)
    
    # Convert Fourier modes back to Physical space for the Neural Network
    train_data_real = jnp.fft.irfft(train_data_hat, axis=-1)
    test_data_real = jnp.fft.irfft(test_data_hat, axis=-1)
    
    # Convert to standard numpy for HDF5 compatibility
    train_data_np = np.array(train_data_real)
    test_data_np = np.array(test_data_real)
    
    print(f"Train dataset shape: {train_data_np.shape} -> (samples, time_states, spatial_grid)")
    print(f"Test dataset shape:  {test_data_np.shape}")

    # 4. Save to HDF5
    with h5py.File("ks_train_data.h5", "w") as f:
        f.create_dataset("u", data=train_data_np)
        f.attrs["L"] = L
        f.attrs["N"] = N
        f.attrs["dt"] = dt
        f.attrs["max_additions"] = max_additions
        
    with h5py.File("ks_test_data.h5", "w") as f:
        f.create_dataset("u", data=test_data_np)
        f.attrs["L"] = L
        f.attrs["N"] = N
        f.attrs["dt"] = dt

    print("Saved 'ks_train_data.h5' and 'ks_test_data.h5'.")

if __name__ == "__main__":
    generate_datasets(num_samples=200, L=32.0, N=128, dt=0.005, t_burn=100.0, max_additions=2)