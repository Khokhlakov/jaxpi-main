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
        self.c_t = 0.01
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
    
    def r_net(self, params, u, t):
        x = self.x_net(params, u, t).reshape(self.N)
        x_t = jacfwd(self.x_net, argnums=2)(params, u, t).reshape(self.N)

        # Wavenumbers for domain [0, 2*pi]
        k = jnp.fft.fftfreq(self.N) * self.N * jnp.pi

        # Compute Fourier transform of the spatial state
        x_hat = jnp.fft.fft(x)

        mask = (jnp.abs(jnp.fft.fftfreq(self.N) * self.N) < self.dealiasing_cutoff)
        x_hat = x_hat * mask

        # Spectral derivatives via IFFT (dropping negligible imaginary artifacts)
        x_xi = jnp.fft.ifft(1j * k * x_hat).real
        x_xixi = jnp.fft.ifft(-k**2 * x_hat).real
        x_4xi = jnp.fft.ifft(k**4 * x_hat).real

        # Normalized Kuramoto-Sivashinsky Residual
        term_t = x_t / self.c_t

        x_filtered = jnp.fft.ifft(x_hat).real

        term_nonlin = (self.c_u / self.c_x) * x_filtered * x_xi
        term_2nd = x_xixi / self.c_x**2
        term_4th = x_4xi / self.c_x**4

        r_x = term_t + term_nonlin + term_2nd + term_4th
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