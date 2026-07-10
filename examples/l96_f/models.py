from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, jacfwd

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
import numpy as np

class L96UDON(ForwardIVP):
    def __init__(self, config, t_star):
        super().__init__(config)
        self.t_star = t_star 

        # System parameters
        self.N = 40
        
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

        # Extract F from the 41st dimension of the input vector
        F = u[-1] 

        x_plus_1 = jnp.roll(x, -1)
        x_minus_1 = jnp.roll(x, 1)
        x_minus_2 = jnp.roll(x, 2)

        r_x = x_t - ((x_plus_1 - x_minus_2) * x_minus_1 - x + F)
        return r_x

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        batch_u, batch_t = batch
        
        # Flatten to a 1D array of shape (batch_size,) before sorting
        batch_t_flat = batch_t.flatten()

        # Now argsort correctly sorts the time values across the batch
        idx = jnp.argsort(batch_t_flat)
        t_sorted = batch_t_flat[idx]

        # Evaluate residual on the full Cartesian grid
        # r_grid_fn: vmap over u (outer), vmap over t (inner)
        r_pred = self.r_grid_fn(params, batch_u, t_sorted)

        # Transpose to (num_t, num_u, N) so chunking splits along time
        r_pred = r_pred.transpose(1, 0, 2)

        # Chunk along time axis: (num_chunks, num_t_per_chunk, num_u, N)
        r_chunks = r_pred.reshape(self.num_chunks, -1, batch_u.shape[0], self.N)

        # Chunk loss: average over time-within-chunk, ICs, and variables
        l = jnp.mean(r_chunks ** 2, axis=(1, 2, 3)) 

        # Causal weights: w_i = exp(-tol * sum_{j<i} L_j)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        batch_u, batch_t = batch
        batch_t = batch_t.reshape(-1)

        # IC Loss: Compare prediction only against the 40 state variables
        x_pred_ic = vmap(self.x_net, (None, 0, None))(params, batch_u, self.t0)
        ics_loss = jnp.mean((batch_u[:, :self.N] - x_pred_ic) ** 2)

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
            # Predict the 40 dynamic variables using the 41-D input u
            x_pred = self.x_net(params, u, t_vec).reshape(self.N)
            # Re-append the constant forcing parameter F located at u[self.N:] (or u[-1:])
            return jnp.concatenate([x_pred, u[self.N:]])
        return propagator
 
    def make_enkf_fns(self, params, N_ens: int = 50):
        from examples.l96_f.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)
        # Pass the augmented state dimension (self.N + 1 = 41) to the EnKF factory
        return make_enkf(propagator, self.N + 1, N_ens, N_dyn=self.N)
    
    def make_residual_fn(self, params) -> Callable:
        """
        Closure over trained ``params`` exposing the PDE residual with the
        SAME (u, t) calling convention as ``make_surrogate_propagator``: u
        is the (N+1,) augmented window IC, t is the in-window query time.

        This is exactly ``r_net`` -- rho = x_t - F(x; mu), the same residual
        already used in the physics loss -- reused here at inference time.
        No gradient is taken w.r.t. params; only the jacfwd already inside
        ``r_net`` (w.r.t. t) is exercised.

        Used by Route B (``kf.make_route_b_enkf``) to turn the surrogate's
        own physics-consistency into flow-dependent process-noise inflation.
        """
        def residual(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t])
            return self.r_net(params, u, t_vec)
        return residual

    def make_route_b_enkf_fns(self, params, N_ens: int = 50):
        """
        Route B (residual-scaled covariance) EnKF predict/update pair.
        See ``kf.make_route_b_enkf`` for the forecast-step math.

        Only defined here (on the physics-informed model): ``L96UDON_DD``
        has no PDE residual (no ``r_net``, since it's a purely data-driven
        fit) and so cannot drive Route B's physics-based inflation -- use
        the fixed-Q ``make_enkf_fns`` for it instead.
        """
        from examples.l96_f.kf import make_route_b_enkf
        propagator = self.make_surrogate_propagator(params)
        residual   = self.make_residual_fn(params)
        # Pass the augmented state dimension (self.N + 1 = 41) to the EnKF factory
        return make_route_b_enkf(
            propagator_fn=propagator,
            residual_fn=residual,
            N=self.N + 1,
            N_ens=N_ens,
        )

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test_batch, x_test_batch):
        # 1. Vectorize x_pred_fn to handle a batch of initial conditions (axis 0 of u_test_batch)
        x_pred_batch_fn = vmap(self.x_pred_fn, (None, 0, None))
        x_pred_batch = x_pred_batch_fn(params, u_test_batch, self.t_star)

        # 2. Relative L2 error of a single trajectory
        def single_traj_error(pred, test):
            return jnp.linalg.norm(pred - test) / jnp.linalg.norm(test)
        
        # 3. Vectorize the error calculation across the batch and take the mean
        batch_errors = vmap(single_traj_error)(x_pred_batch, x_test_batch)
        return jnp.mean(batch_errors)

class L96UDONEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Construct heatmap: variables on x-axis, time on y-axis
        c = ax.pcolormesh(np.arange(self.model.N), t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("Variables (0 to 39)")
        ax.set_ylabel("Time (t)")
        ax.set_title("L96 UDON Trajectory Heatmap")
        fig.colorbar(c, ax=ax)
        
        plt.tight_layout()
        self.log_dict["x_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref_batch, x_ref_batch):
        self.log_dict = super().__call__(state, batch)

        # Causal weights now need the full batch (batch_u, batch_t)
        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref_batch, x_ref_batch)

        if self.config.logging.log_preds:
            self.log_preds(state.params, u_ref_batch[0])

        return self.log_dict
    
# Data driven
class L96UDON_DD(ForwardIVP):
    def __init__(self, config, t_star):
        super().__init__(config)
        self.t_star = t_star 
        self.N = 40
        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid (t partition)
        self.x_pred_fn = vmap(self.x_net, (None, None, 0))

    def x_net(self, params, u, t):
        t = jnp.atleast_1d(t)
        return self.state.apply_fn(params, u, t)
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # batch now expects: (batch_u, batch_t, batch_x_true)
        batch_u, batch_t, batch_x_true = batch
        batch_t = batch_t.reshape(-1)

        # Predict state from initial condition and time
        x_pred = vmap(self.x_net, (None, 0, 0))(params, batch_u, batch_t)
        
        # Supervised data loss (MSE)
        data_loss = jnp.mean((x_pred - batch_x_true) ** 2)

        loss_dict = {"data_loss": data_loss}
        return loss_dict

    def make_surrogate_propagator(self, params) -> Callable:
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t])
            # Predict the 40 dynamic variables using the 41-D input u
            x_pred = self.x_net(params, u, t_vec).reshape(self.N)
            # Re-append the constant forcing parameter F located at u[self.N:] (or u[-1:])
            return jnp.concatenate([x_pred, u[self.N:]])
        return propagator
    
    def make_enkf_fns(self, params, N_ens: int = 50):
        from examples.l96_f.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)
        # Pass the augmented state dimension (self.N + 1 = 41) to the EnKF factory
        return make_enkf(propagator, self.N + 1, N_ens, N_dyn=self.N)

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test_batch, x_test_batch):
        # 1. Vectorize x_pred_fn to handle a batch of initial conditions 
        x_pred_batch_fn = vmap(self.x_pred_fn, (None, 0, None))
        x_pred_batch = x_pred_batch_fn(params, u_test_batch, self.t_star)

        # 2. Relative L2 error of a single trajectory
        def single_traj_error(pred, test):
            return jnp.linalg.norm(pred - test) / jnp.linalg.norm(test)
        
        # 3. Vectorize the error calculation across the batch and take the mean
        batch_errors = vmap(single_traj_error)(x_pred_batch, x_test_batch)
        return jnp.mean(batch_errors)

class L96UDONEvaluator_DD(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Construct heatmap: variables on x-axis, time on y-axis
        c = ax.pcolormesh(np.arange(self.model.N), t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("Variables (0 to 39)")
        ax.set_ylabel("Time (t)")
        ax.set_title("L96 UDON (Data-Driven) Trajectory Heatmap")
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