from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, jacfwd

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
import numpy as np


class KSUDON(ForwardIVP):
    def __init__(self, config, t_star):
        super().__init__(config)
        self.t_star = t_star 

        # System parameters for normalized KS equation
        self.N = 256
        self.c_u = 3.5
        self.c_t = 1.0
        self.c_x = 32.0
        
        # Dealiasing convention (2/3 rule)
        self.dealiasing_cutoff = int(self.N * (2/3) / 2)

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid (t partition)
        self.x_pred_fn = vmap(self.x_net, (None, None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, None, 0))
        self.r_grid_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def x_net(self, params, u, t):
        t = jnp.atleast_1d(t)
        return self.state.apply_fn(params, u, t)
    
    New chat
Search chats
Images
New
Library
New notebook
Flatten Physics-Informed Training Data
Replicating GPU Server Environments
Non-Dimensionalizing Kuramoto-Sivashinsky for PINNs
DeepONet for Lorenz'96 with Parametric Forcing
Plotting PDE Dynamics and Initial Conditions
Evaluating Kuramoto-Sivashinsky Solver for ML
Kuramoto-Sivashinsky Solver Bugs
Kuramoto-Sivashinsky Solver Evaluation
PDE Data Generation Script Explained
Kuramoto-Sivashinsky Data Generation Refinements
JAX to NumPy KS Simulation
Generating Kuramoto-Sivashinsky Solutions
Adapting DeepONet Evaluation for Data-Driven Model
Adapting Evaluation for Data-Driven DeepONet
Adapting Evaluation for Data-Driven DeepONet
Data-Driven DeepONet for Lorenz'96
Kuramoto-Sivashinsky Equation Simulation
EnKF Calibration: RMSE vs. L2 Error
Kalman Filter MLP Timing Conflict
EnKF With Finer Predictions
Generate Train and Test Data
EKF with Time-Varying Observations
DeepONet Training Data Augmentation Logic
I-20 Sponsor Certification Inquiry
Sponsor Certification Form Inquiry
Separate Temporal and Spatial Parameters
Fix Missing Minecraft Mod Dependencies
Minecraft Server Latency: Bogotá-Panama
WhatsApp Chat Text Extraction and Context
DeepONet Lorenz 96 Bug Fixes
is it possible to replicate the solution of Lorenz 96 obtained in Matlab using Python? i.e., same precision.
Lorenz 96 Data Generation for DeepONet
Lorenz 96 Model DeepONet Translation
DeepONet Lorenz 96 Autoregressive Training
DeepONet Training Curriculum Expansion
Lorenz 96 Reference Solution Verification
Ensemble Kalman Filter Explained
GZIP File Identification and Decompression
Lorenz System: Symmetry, Chaos, Mixing
Causal Training Bug Fixes
Fixing Python Module Not Found Error
Adjusting L63 DeepONet Time Horizon
Activation Function Choice for L63
Tuning DeepONet for Chaotic Systems
Plot Multiple Trajectories
DeepONet Rollout and Error Analysis
DeepONet Long-Term Rollout Analysis
Lorenz 63 Rollout Evaluation
JAX Import Error and Updates
DeepONet: Understanding Parametric PDEs
Gaussian Process Distributions Over Time
Function Spaces for System Identification
Conversation with Gemini
The code attached was designed to generate the datasets for a DeepONet (Data driven and physics informed). The data driven part is ready. Indicate the changes to be made so all samples generated for the physics informed training are flattened to a pool. Currently, the ks_train_data.h5 contains data with shape similar to (100, 501, 256), we want to flatten the first dimensions to obtain a pool of ics of dimensions (50100, 256).

To flatten the physics-informed (PI) training data from (num_samples, time_states, N) to a single pool of initial conditions (num_samples * time_states, N) directly during data generation, you just need to reshape the train_data_np array right after it's converted to a NumPy array.

Here are the specific modifications you need to make to your generate_datasets function.

The Code Changes
Locate this section of your code (around line 141) where the arrays are being defined, and add the .reshape() method:

Python
    # Convert Fourier modes back to Physical space for the Neural Network (256 points)
    train_data_real     = jnp.fft.irfft(train_data_hat, axis=-1)
    train_data_dd_real  = jnp.fft.irfft(train_data_dd_hat, axis=-1)
    test_data_real      = jnp.fft.irfft(test_data_hat, axis=-1)
    
    # Keep 256 points
    train_data_np       = np.array(train_data_real)
    
    # ---------------------------------------------------------
    # NEW: Flatten the PI data to create a flat pool of ICs
    # Reshapes from (100, 501, 256) -> (50100, 256)
    # ---------------------------------------------------------
    train_data_np = train_data_np.reshape(-1, train_data_np.shape[-1])
    
    train_data_dd_np    = np.array(train_data_dd_real)
    test_data_np        = np.array(test_data_real)

    # DOWNSAMPLE: Slice the spatial array to keep 128 points [..., ::2]
    #train_data_np = np.array(train_data_real)[..., ::2]
    #train_data_dd_np = np.array(train_data_dd_real)[..., ::2]
    #test_data_np = np.array(test_data_real)[..., ::2]
    
    # Update the print statement to reflect the new pooled shape
    print(f"PI  train data shape : {train_data_np.shape}"
          f"  →  (pooled_samples, N)")
Why this works
Using .reshape(-1, train_data_np.shape[-1]) tells NumPy to keep the last dimension strictly as N (which is 256) and infer the first dimension by multiplying num_samples by time_states (e.g., 100×501=50100).

Because NumPy arrays are stored in row-major order (C-contiguous) by default, flattening the first two dimensions perfectly stacks the consecutive time states for each sample sequentially, preserving the physical coherence of the data points for your DeepONet.

Attached are the training module for two DeepONets, a data driven one and a physics informed one, and the code for generating the training datasets and the testing dataset. The data driven part of the code is done, use it as a reference for completing the physics informed one. Take into consideration that the data driven version uses data with dimensions (51, 256), there are 50000 samples like this, each sample corresponds to a simulation that last one time unit, and the states at every 0.02 (51 of them) time units are kept. For the physics informed version, the training samples are more sparse: single states used as initial conditions, 50100 of them, thus the training dataset for the physics informed has dimensions (50100, 256). Indicate the changes to be made in the code to make the following modifications:



Remove the data flattening from train_and_evaluate, since it is no longer needed.

Optimize the key spliting and batch generation so the processors spend less time idle (take train_and_evaluate_dd as the reference).

Verify that the rest of the code is correctly implemented.

To optimize your physics-informed (PI) DeepONet training loop and align its performance with the data-driven counterpart, we need to eliminate host-to-device communication overhead. The previous implementation relied on CPU-based samplers (UniformSampler and SpaceSampler) that yield Python iterators, bottlenecking the GPUs while they wait for data.

By moving the random key splitting and index generation directly onto the devices using jax.pmap, the processors can fetch and construct batches entirely from local memory.

Here are the step-by-step changes to update train_and_evaluate in train.py.

1. Remove Data Flattening
Since you have already flattened the data in generate_data.py, loading the data is now straightforward. Update the dataset loading block to read the array directly without the .reshape() step.

Change this:

Python
    with h5py.File(train_file, 'r') as f_train:
        # u_train shape: (num_samples, time_states, 256)
        u_train = jnp.array(f_train['u'][:])
        
        # Flatten the first two dimensions so EVERY recorded state acts 
        # as a valid initial condition (IC) for the physics residual sampler.
        # u_pool Shape: (num_samples * time_states, 256)
        u_pool = u_train.reshape(-1, u_train.shape[-1])
To this:

Python
    with h5py.File(train_file, 'r') as f_train:
        # Data is already flattened by the generator -> (50100, 256)
        u_pool = jnp.array(f_train['u'][:])
2. Implement On-Device Sampling
We need to replace the UniformSampler and SpaceSampler with an on-device mapped function. Because this is a physics-informed model, the time inputs t shouldn't be sampled from a discrete grid index like in the data-driven model. Instead, t must be continuously sampled from a uniform distribution over the domain t∈[0.0,1.0] to enforce the PDE residual everywhere.

Remove these lines:

Python
    # dom_t     = jnp.array([[0.0, 1.0]]) 
    # sampler_t = UniformSampler(dom_t, batch_size)
    # sampler_u   = SpaceSampler(u_pool, batch_size)
    # res_sampler = zip(sampler_u, sampler_t)
Replace them with this block:

Python
    # ── On-Device Samplers ─────────────────────────────────────────────────
    num_devices = jax.local_device_count()
    batch_size_per_device = config.training.batch_size_per_device

    # Initialize distributed PRNG keys
    device_keys = jax.pmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(num_devices))

    # Replicate the static data pool across all available GPUs/TPUs
    u_pool_repl = jax.device_put_replicated(u_pool, jax.devices())

    @jax.pmap
    def get_batch_on_device(device_key, data_pool):
        """Splits keys and samples entirely on-device. No host involvement."""
        new_key, key_u, key_t = jax.random.split(device_key, 3)

        # 1. Sample Initial Conditions (u) from the flattened pool
        idx_u = jax.random.randint(key_u, (batch_size_per_device,), 0, pool_size)
        batch_u = data_pool[idx_u]

        # 2. Sample continuous collocation points in time (t) uniformly in [0.0, 1.0]
        batch_t = jax.random.uniform(key_t, (batch_size_per_device, 1), minval=0.0, maxval=1.0)

        return new_key, (batch_u, batch_t)
3. Update the Training Loop
Finally, modify the training loop to pull batches using the new get_batch_on_device function instead of calling next() on the iterator.

Change this:

Python
        # ── Forward + gradient step ────────────────────────────────────────
        batch       = next(res_sampler)          # (batch_u, batch_t), each (devices, B, ·)
        model.state = model.step(model.state, batch)
To this:

Python
        # ── Forward + gradient step ────────────────────────────────────────
        device_keys, batch = get_batch_on_device(device_keys, u_pool_repl)
        model.state = model.step(model.state, batch)
Verification and Checks
I have reviewed the rest of the PI setup and it is correctly implemented:

Reference Evaluation (Logging): You are slicing x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]. Because x_ref_eval_all is taken from ks_test_data.h5 (which was generated continuously at dt=0.02), the time grid alignment t 
star
​
 =0,0.02,0.04... correctly lines up with the first 50 steps of the dense test set. This provides a clean L2 error calculation against the test ground-truth during training.

Adaptive Weighting: Passing batch as a tuple (batch_u, batch_t) into model.update_weights(model.state, batch) remains fully compatible with JAX tree maps.

Memory Efficiency: Dropping the host-side samplers means you can now safely remove from jaxpi.samplers import UniformSampler, SpaceSampler from your imports to keep the script clean.

Attached is the models module, where the physics informed DeepONet is implemented. Check that the loss are correctly computed, since the 4th derivative calculation can be tricky. Attached is the data generation code as well, it may be useful to take as a reference since it contains a solver for the KS 1D equation. Verify that the physics informed DeepONet is correctly implemented.

The physics-informed implementation is fundamentally well-structured, and the terms of the normalized Kuramoto-Sivashinsky residual are correctly derived. The signs for the 2nd and 4th spatial derivatives correctly match the operator defined in generate_data.py.

However, as you correctly suspected, the 4th derivative is tricky. There are three crucial bugs in models.py related to the spectral formulation, the Nyquist frequency, and matrix operations that will prevent the network from converging.

Here are the specific changes you need to make to KSUDON.

1. Fix the Wavenumber Domain Scaling (k)
In generate_data.py, your spatial domain is L=64, and you scale it by c 
x
​
 =32.0. This maps the physical x∈[0,64] to the normalized domain ξ∈[0,2].

Because the domain length for ξ is 2.0 (not 2π), the Fourier wavenumbers must be k 
n
​
 =nπ. Currently, models.py computes integer wavenumbers (k 
n
​
 =n), meaning your DeepONet is calculating spatial derivatives for a domain of length 2π. You must multiply the wavenumbers by π.

2. Zero-out the Nyquist Frequency (Gradient Blowup)
Calculating the 4th spatial derivative involves multiplying the Fourier coefficients by k 
4
 . At N=256, the highest frequency (Nyquist frequency) is k 
128
​
 =128π.

When you compute (128π) 
4
 , this applies a multiplier of roughly 2.6×10 
10
  to the 128th mode. While generate_data.py uses a 2/3 dealiasing rule to keep the ground truth smooth, the DeepONet's raw predictions (x_hat) will inherently contain high-frequency noise. Multiplying this noise by k 
4
  will immediately cause the 4th derivative (and your loss gradients) to explode.

Because the Nyquist frequency is purely real, the .real operation does not drop it like it does for odd derivatives. You must manually zero out the Nyquist frequency in the wavenumber array before calculating derivatives.

Update r_net in models.py to fix both issues:

Python
    def r_net(self, params, u, t):
        x = self.x_net(params, u, t).reshape(self.N)
        x_t = jacfwd(self.x_net, argnums=2)(params, u, t).reshape(self.N)

        # ---------------------------------------------------------
        # NEW: Multiply by pi to match the xi domain length of 2.0
        # ---------------------------------------------------------
        k = jnp.fft.fftfreq(self.N) * self.N * jnp.pi

        # ---------------------------------------------------------
        # NEW: Zero out the Nyquist frequency to prevent gradient 
        # blowup in the 2nd and 4th (even) derivatives.
        # ---------------------------------------------------------
        k = k.at[self.N // 2].set(0.0)

        # Compute Fourier transform of the spatial state
        x_hat = jnp.fft.fft(x)

        # Spectral derivatives via IFFT
        x_xi = jnp.fft.ifft(1j * k * x_hat).real
        x_xixi = jnp.fft.ifft(-k**2 * x_hat).real
        x_4xi = jnp.fft.ifft(k**4 * x_hat).real

        # Normalized Kuramoto-Sivashinsky Residual
        term_t = x_t / self.c_t
        term_nonlin = (self.c_u / self.c_x) * x * x_xi
        term_2nd = x_xixi / self.c_x**2
        term_4th = x_4xi / self.c_x**4

        r_x = term_t + term_nonlin + term_2nd + term_4th
        return r_x
3. Fix the Causal Weighting Array Sorting
If you enable causal weighting (self.config.weighting.use_causal = True), the code will fail to sort the time array correctly.

In res_and_w, you pass batch_t, which (based on standard PI sampling) has the shape (B, 1). When you call jnp.argsort(batch_t), NumPy defaults to sorting along the last axis (axis=-1). Sorting an axis of length 1 does nothing; it simply returns an array of zeros. Consequently, t_sorted will extract the wrong indices, breaking the causality timeline.

You need to flatten batch_t inside res_and_w before applying argsort.

Update res_and_w in models.py:

Python
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        batch_u, batch_t = batch

        # ---------------------------------------------------------
        # NEW: Flatten batch_t before sorting so argsort operates 
        # across the batch elements rather than the last axis.
        # ---------------------------------------------------------
        batch_t = batch_t.reshape(-1)

        # Sort time points only — ICs are not reordered
        idx = jnp.argsort(batch_t)
        t_sorted = batch_t[idx]

        # Evaluate residual on the full Cartesian grid
        r_pred = self.r_grid_fn(params, batch_u, t_sorted)

        # Transpose to (num_t, num_u, N) so chunking splits along time
        r_pred = r_pred.transpose(1, 0, 2)

        # Chunk along time axis: (num_chunks, num_t_per_chunk, num_u, N)
        r_chunks = r_pred.reshape(self.num_chunks, -1, batch_u.shape[0], self.N)

        # Chunk loss: average over time-within-chunk, ICs, and spatial points
        l = jnp.mean(r_chunks ** 2, axis=(1, 2, 3)) 

        # Causal weights: w_i = exp(-tol * sum_{j<i} L_j)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w
With these corrections, the mathematical mapping between your high-fidelity simulator and the DeepONet's physics-informed loss function is strictly aligned.

Check the correctness for this implementation. I still get the spike, although less pronounced (30000). Ignore the causal weighing code, since it is not being used currently. I also share the module jaxpi.models, which contains classes like ForwardIVP.

The 30,000 loss spike is a classic symptom of spectral leakage and fighting gradients. While your mask successfully capped the frequencies, the current implementation of r_net has a mathematical flaw: it mixes filtered and unfiltered signals in physical space, breaking the physical coherence of the residual.

Here is exactly why it is spiking and how to fix it.

The Root Cause: Inconsistent Filtering and Aliasing
There are three conflicting operations happening in your current r_net:

Inconsistent Filtering: You apply the dealiasing mask to x_hat before computing the spatial derivatives, meaning x_xi, x_xixi, and x_4xi are perfectly smooth (frequencies ≤85). However, the raw network outputs x and x_t are unfiltered and contain high-frequency noise from the random initialization (up to k=128).

Non-linear Aliasing: You compute term_nonlin = (c_u / c_x) * x * x_xi in physical space. Multiplying an unfiltered signal (x) with a filtered signal (x_xi) physically generates new frequencies that exceed your cutoff limit.

Fighting Gradients: When you sum r_x, x_t is forced to balance the aliased high frequencies generated by the non-linear term. Because term_2nd and term_4th have zero energy in the high frequencies, the neural network receives massive gradient penalties trying to solve a mathematically broken PDE in the upper third of the spectrum.

The Solution: Galerkin Projection
To perfectly match the physics of your generate_data.py simulator, you must compute the entire residual in Fourier space and apply the dealiasing mask to the final sum. This acts as a strict Galerkin projection, enforcing the physics only on the resolved modes and completely ignoring the high-frequency noise from the network's initialization.

Replace your current r_net in KS.models.py with this exact implementation:

Python
    def r_net(self, params, u, t):
        x = self.x_net(params, u, t).reshape(self.N)
        x_t = jacfwd(self.x_net, argnums=2)(params, u, t).reshape(self.N)

        # Wavenumbers for domain [0, 2*pi]
        k = jnp.fft.fftfreq(self.N) * self.N * jnp.pi

        # Transform state and time derivative to Fourier space
        x_hat = jnp.fft.fft(x)
        x_t_hat = jnp.fft.fft(x_t)

        # 1. Time derivative
        term_t_hat = x_t_hat / self.c_t

        # 2. Non-linear term: compute u^2 in physical space, then differentiate spectrally
        # u * u_x = 0.5 * (u^2)_x  -->  0.5 * 1j * k * fft(u^2)
        u_squared_hat = jnp.fft.fft(x ** 2)
        term_nonlin_hat = 0.5 * (self.c_u / self.c_x) * 1j * k * u_squared_hat

        # 3. Linear spatial derivatives
        term_2nd_hat = (-k**2 / self.c_x**2) * x_hat
        term_4th_hat = (k**4 / self.c_x**4) * x_hat

        # Combine the entire residual in Fourier space
        r_hat = term_t_hat + term_nonlin_hat + term_2nd_hat + term_4th_hat

        # Apply the exact dealiasing mask (Galerkin projection)
        mask = (jnp.abs(jnp.fft.fftfreq(self.N) * self.N) < self.dealiasing_cutoff)
        r_hat_masked = r_hat * mask

        # Return to physical space for the final MSE loss computation
        r_x = jnp.fft.ifft(r_hat_masked).real
        return r_x

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        batch_u, batch_t = batch

        # Sort time points only — ICs are not reordered
        idx = jnp.argsort(batch_t)
        t_sorted = batch_t[idx]

        # Evaluate residual on the full Cartesian grid
        # r_grid_fn: vmap over u (outer), vmap over t (inner)
        # Output shape: (num_u, num_t, N)
        r_pred = self.r_grid_fn(params, batch_u, t_sorted)

        # Transpose to (num_t, num_u, N) so chunking splits along time
        r_pred = r_pred.transpose(1, 0, 2)

        # Chunk along time axis: (num_chunks, num_t_per_chunk, num_u, N)
        r_chunks = r_pred.reshape(self.num_chunks, -1, batch_u.shape[0], self.N)

        # Chunk loss: average over time-within-chunk, ICs, and spatial points
        l = jnp.mean(r_chunks ** 2, axis=(1, 2, 3))  # shape: (num_chunks,)

        # Causal weights: w_i = exp(-tol * sum_{j<i} L_j)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # batch: (batch_u, batch_t)
        batch_u, batch_t = batch
        batch_t = batch_t.reshape(-1)

        # IC Loss
        x_pred_ic = vmap(self.x_net, (None, 0, None))(params, batch_u, self.t0)
        ics_loss = jnp.mean((batch_u - x_pred_ic) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True: 
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        elif self.config.training.use_cartesian_prod == True:
            r_pred = self.r_grid_fn(params, batch_u, batch_t)
            res_loss = jnp.mean(r_pred ** 2)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch_u, batch_t)
            res_loss = jnp.mean(r_pred ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict

    def make_surrogate_propagator(self, params) -> Callable:
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t]) 
            return self.x_net(params, u, t_vec).reshape(self.N)
 
        return propagator

    def make_ekf_fns(self, params, dt: float):
        from examples.KS.kf import make_ekf
        propagator_vt = self.make_surrogate_propagator(params) 
        propagator    = lambda u: propagator_vt(u, dt)         
        return make_ekf(propagator, self.N)
 
    def make_enkf_fns(self, params, N_ens: int = 50):
        from examples.KS.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)  
        return make_enkf(propagator, self.N, N_ens)
 
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test_batch, x_test_batch):
        x_pred_batch_fn = vmap(self.x_pred_fn, (None, 0, None))
        x_pred_batch = x_pred_batch_fn(params, u_test_batch, self.t_star)

        def single_traj_error(pred, test):
            return jnp.linalg.norm(pred - test) / jnp.linalg.norm(test)
        
        batch_errors = vmap(single_traj_error)(x_pred_batch, x_test_batch)
        return jnp.mean(batch_errors)

class KSUDONEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star

        fig, ax = plt.subplots(figsize=(10, 8))
        
        c = ax.pcolormesh(np.arange(self.model.N), t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("Spatial Points (0 to 255)")
        ax.set_ylabel("Time (t)")
        ax.set_title("KS UDON Trajectory Heatmap")
        fig.colorbar(c, ax=ax)
        
        plt.tight_layout()
        self.log_dict["x_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref_batch, x_ref_batch):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref_batch, x_ref_batch)

        if self.config.logging.log_preds:
            self.log_preds(state.params, u_ref_batch[0])

        return self.log_dict
    
# Data driven
class KSUDON_DD(ForwardIVP):
    def __init__(self, config, t_star):
        super().__init__(config)
        self.t_star = t_star 
        self.N = 256
        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.x_pred_fn = vmap(self.x_net, (None, None, 0))

    def x_net(self, params, u, t):
        t = jnp.atleast_1d(t)
        return self.state.apply_fn(params, u, t)
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        batch_u, batch_t, batch_x_true = batch
        batch_t = batch_t.reshape(-1)

        x_pred = vmap(self.x_net, (None, 0, 0))(params, batch_u, batch_t)
        data_loss = jnp.mean((x_pred - batch_x_true) ** 2)

        loss_dict = {"data_loss": data_loss}
        return loss_dict

    def make_surrogate_propagator(self, params) -> Callable:
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t])  
            return self.x_net(params, u, t_vec).reshape(self.N)
 
        return propagator

    def make_ekf_fns(self, params, dt: float):
        from examples.KS.kf import make_ekf
        propagator_vt = self.make_surrogate_propagator(params)  
        propagator    = lambda u: propagator_vt(u, dt)          
        return make_ekf(propagator, self.N)
 
    def make_enkf_fns(self, params, N_ens: int = 50):
        from examples.KS.kf import make_enkf
        propagator = self.make_surrogate_propagator(params) 
        return make_enkf(propagator, self.N, N_ens)

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test_batch, x_test_batch):
        x_pred_batch_fn = vmap(self.x_pred_fn, (None, 0, None))
        x_pred_batch = x_pred_batch_fn(params, u_test_batch, self.t_star)

        def single_traj_error(pred, test):
            return jnp.linalg.norm(pred - test) / jnp.linalg.norm(test)
        
        batch_errors = vmap(single_traj_error)(x_pred_batch, x_test_batch)
        return jnp.mean(batch_errors)

class KSUDONEvaluator_DD(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star

        fig, ax = plt.subplots(figsize=(10, 8))
        
        c = ax.pcolormesh(np.arange(self.model.N), t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("Spatial Points (0 to 256)")
        ax.set_ylabel("Time (t)")
        ax.set_title("KS UDON (Data-Driven) Trajectory Heatmap")
        fig.colorbar(c, ax=ax)
        
        plt.tight_layout()
        self.log_dict["x_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref_batch, x_ref_batch):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref_batch, x_ref_batch)

        if self.config.logging.log_preds:
            self.log_preds(state.params, u_ref_batch[0])

        return self.log_dict