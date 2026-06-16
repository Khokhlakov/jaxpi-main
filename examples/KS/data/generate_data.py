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
    def __init__(self, L: float = 64.0, N: int = 256, 
                 dt: float = 0.02, dealiasing: float = 2/3):
        self.L = L
        self.N = N
        self.dt = dt
        
        # Normalization constants
        self.c_t = 1.0
        self.c_x = L / 2.0
        self.c_u = 3.5
        # Normalized spatial domain (xi)
        self.xi_L = 2.0
        self.xi = jnp.linspace(0, self.xi_L, N, endpoint=False)
        self.dxi = self.xi_L / N
        # Wavenumbers for the normalized domain
        self.k_xi = jnp.fft.rfftfreq(N, self.dxi) * 2 * jnp.pi
        # 4. Normalized Linear Operator
        # u_xixi -> -k_xi^2. The equation has -u_xixi, so it becomes positive.
        self.L_op = (self.k_xi**2 / self.c_x**2) - (self.k_xi**4 / self.c_x**4)

        self.dealiasing_cutoff = int(N * dealiasing / 2)
        self.etd = ETDCoefficients(self.L_op, dt)
    
    @partial(jax.jit, static_argnums=(0,))
    def _dealias(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        return u_hat.at[self.dealiasing_cutoff:].set(0)
    
    @partial(jax.jit, static_argnums=(0,))
    def _nonlinear_term(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        u = jnp.fft.irfft(u_hat)
        u_squared_hat = jnp.fft.rfft(u**2)
        # Derivative with respect to normalized xi
        du2dxi_hat = 1j * self.k_xi * u_squared_hat
        # Apply the (c_u / c_x) scaling coefficient
        N_hat = 0.5 * (self.c_u / self.c_x) * du2dxi_hat
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
    L: float = 64.0, 
    N: int = 256, 
    dt: float = 0.02, 
    t_burn: float = 50.0,
    max_additions: int = 4,
    test_windows: int = 50
):
    solver = KuramotoSivashinskyAdvanced(L=L, N=N, dt=dt)
    
    # 1. Generate Smooth Initial Noise
    key = jax.random.PRNGKey(42)
    k_array = solver.k_xi
    
    def create_single_ic(k):
        # Create random phases and amplitudes, smooth with exp(-decay)
        keys = jax.random.split(k, 2)
        # Revert back to the physical wavenumber just for smoothing 
        # to prevent the exponential decay from truncating too early.
        physical_k = k_array / solver.c_x
        amp = jax.random.normal(keys[0], k_array.shape) * jnp.exp(-0.1 * physical_k**2)
        phase = jax.random.uniform(keys[1], k_array.shape, minval=0, maxval=2*jnp.pi)
        u_hat = solver._dealias(amp * jnp.exp(1j * phase))
        return u_hat

    keys = jax.random.split(key, num_samples)
    u_hat_initial = jax.vmap(create_single_ic)(keys)

    # ==========================================
    # Pre-compute steps as pure Python integers
    # ==========================================
    burn_steps = int(t_burn / dt)
    interval_steps = int(1.0 / dt)

    # 2. Define Scanning Functions 
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
        # Burn-in Phase
        u_hat_burned = advance_time(u_hat_0, burn_steps)
        
        # Train Phase: Dense and Sparse Generation
        total_train_steps = max_additions * interval_steps

        # Advance and save ALL steps to generate the dense dataset
        u_train_end, train_trajectory_dd = advance_save_all(u_hat_burned, total_train_steps)
        
        # Dense train data: prepend the IC (yields total_train_steps + 1 states)
        train_data_dd = jnp.concatenate([u_hat_burned[None, ...], train_trajectory_dd], axis=0)

        # Sparse train data: recover original behavior by slicing every 200 steps
        train_data = train_data_dd[::interval_steps]

        # Test Phase: advance 1.0 without saving, then save every step for the last 1.0
        u_test_start = advance_time(u_train_end, interval_steps)
        
        # Compute total steps needed for test_windows windows
        total_test_steps = test_windows * interval_steps
        # Advance and save every step
        _, test_trajectory = advance_save_all(u_test_start, total_test_steps)
        test_data = jnp.concatenate([u_test_start[None, ...], test_trajectory], axis=0)
        
        return train_data, train_data_dd, test_data

    print("Simulating (JIT compiling first time)...")
    train_data_hat, train_data_dd_hat, test_data_hat = jax.vmap(simulate_sample)(u_hat_initial)
    
    # Convert Fourier modes back to Physical space for the Neural Network (256 points)
    train_data_real     = jnp.fft.irfft(train_data_hat, axis=-1)
    train_data_dd_real  = jnp.fft.irfft(train_data_dd_hat, axis=-1)
    test_data_real      = jnp.fft.irfft(test_data_hat, axis=-1)
    
    # Keep 256 points
    train_data_np       = np.array(train_data_real)
    train_data_dd_np    = np.array(train_data_dd_real)
    test_data_np        = np.array(test_data_real)

    # DOWNSAMPLE: Slice the spatial array to keep 128 points [..., ::2]
    #train_data_np = np.array(train_data_real)[..., ::2]
    #train_data_dd_np = np.array(train_data_dd_real)[..., ::2]
    #test_data_np = np.array(test_data_real)[..., ::2]
    
    print(f"PI  train data shape : {train_data_np.shape}"
          f"  →  (samples, time_states, N)  [pooled at load time in train.py]")
    
    # ── Data-Driven pool — split long trajectories into 1-unit windows ──
    starts  = np.arange(max_additions) * interval_steps          # (max_additions,)
    offsets = np.arange(interval_steps + 1)                      # (interval_steps+1,)  e.g. 51
    indices = starts[:, None] + offsets[None, :]                 # (max_additions, interval_steps+1)

    # Advanced index into time axis:
    #   train_data_dd_np[:, indices, :]  →  (num_samples, max_additions, interval_steps+1, N)
    train_data_dd_windowed = train_data_dd_np[:, indices, :]

    # Merge the first two dimensions → flat pool of windows
    #   (num_samples * max_additions, interval_steps+1, N)
    #   e.g. (100 * 500, 51, 256) = (50000, 51, 256)
    train_data_dd_windowed = train_data_dd_windowed.reshape(
        -1, interval_steps + 1, train_data_dd_np.shape[-1]
    )

    print(f"DD  train data shape : {train_data_dd_windowed.shape}"
          f"  →  (windows, time_pts_per_window, N)")
    print(f"    = {num_samples} trajectories × {max_additions} windows"
          f" × {interval_steps + 1} time pts × {train_data_dd_np.shape[-1]} spatial pts")
    print(f"Test dataset shape:  {test_data_np.shape}")

    # 4. Save to HDF5
    with h5py.File("ks_train_data.h5", "w") as f:
        f.create_dataset("u", data=train_data_np)
        f.attrs["L"] = L
        f.attrs["N"] = N
        f.attrs["dt"] = dt
        f.attrs["max_additions"] = max_additions
    
    with h5py.File("ks_train_data_dd.h5", "w") as f:
        f.create_dataset("u", data=train_data_dd_windowed)
        f.attrs["L"]              = L
        f.attrs["N"]              = N
        f.attrs["dt"]             = dt
        f.attrs["interval_steps"] = interval_steps   # pts per window (excl. IC) = 1/dt
        f.attrs["num_windows"]    = max_additions     # windows per original trajectory
        f.attrs["num_samples"]    = num_samples       # original IC count
        # total pool size = num_samples * num_windows

    with h5py.File("ks_test_data.h5", "w") as f:
        f.create_dataset("u", data=test_data_np)
        f.attrs["L"] = L
        f.attrs["N"] = N
        f.attrs["dt"] = dt
        f.attrs["test_windows"] = test_windows

    print("Saved 'ks_train_data.h5', 'ks_train_data_dd.h5', and 'ks_test_data.h5'.")

if __name__ == "__main__":
    generate_datasets(num_samples=100, 
                      L=64, 
                      N=256, 
                      dt=0.02, 
                      t_burn=100.0, 
                      max_additions=500,
                      test_windows=30)