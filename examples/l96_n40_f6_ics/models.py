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
        self.F = 6.0
        
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

        # Lorenz 96 ODE: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # Using jnp.roll for periodic boundary conditions
        x_plus_1 = jnp.roll(x, -1)
        x_minus_1 = jnp.roll(x, 1)
        x_minus_2 = jnp.roll(x, 2)

        r_x = x_t - ((x_plus_1 - x_minus_2) * x_minus_1 - x + self.F)
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

        # Chunk loss: average over time-within-chunk, ICs, and variables
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

    def make_surrogate_propagator(self, params, dt: float) -> Callable:
        """
        Returns a pure function  u: (N,) -> x(t+dt): (N,)
        suitable for use as the EKF propagator.

        The DeepONet is queried at t=dt with u as the initial condition.
        params are frozen (closed over), so this is a plain array -> array fn.

        Args:
            params: frozen network parameters (unreplicated).
            dt: integration window in the same units as t_star.

        Returns:
            propagator: Callable[(N,) -> (N,)]
        """
        t_vec = jnp.array([dt])  # trunk expects a 1-D vector

        def propagator(u: jnp.ndarray) -> jnp.ndarray:
            return self.x_net(params, u, t_vec).reshape(self.N)

        return propagator

    def make_ekf_fns(self, params, dt: float):
        """
        Convenience method: build and return EKF predict/update functions
        bound to this model's surrogate propagator.

        Usage:
            predict, update = model.make_ekf_fns(params, dt=0.05)
            ekf_state = EKFState(x_hat=x0, P=P0)
            ekf_state = predict(ekf_state, Q)
            ekf_state, K = update(ekf_state, y_obs, H, R)
        """
        from kf import make_ekf
        propagator = self.make_surrogate_propagator(params, dt)
        return make_ekf(propagator, self.N)

    def make_enkf_fns(self, params, dt: float, N_ens: int = 50):
        """
        Convenience method: build and return EnKF predict/update functions
        bound to this model's surrogate propagator.
 
        The propagator is identical to the one used by make_ekf_fns, but the
        EnKF factory wraps it with vmap (one forward pass per ensemble member)
        rather than jacfwd.  This removes the linearisation assumption and lets
        the ensemble spread capture the true non-Gaussian uncertainty of L96.
 
        Args:
            params: frozen (unreplicated) network parameters.
            dt:     assimilation window length, same units as t_star.
            N_ens:  ensemble size.
                    • ≥ 20  — minimum for stable covariance estimation on L96-N40
                    • 50    — good balance of cost vs accuracy (default)
                    • ≥ 100 — recommended when sigma_proc is large or obs are sparse
 
        Returns:
            predict_fn, update_fn — both JIT-compiled EnKF functions.
 
        Usage:
            predict, update = model.make_enkf_fns(params, dt=0.05, N_ens=50)
            key  = jax.random.PRNGKey(0)
            key, k0 = jax.random.split(key)
            ensemble0 = init_ensemble(x0_hat, P0, N_ens, k0)      # from enkf.py
            enkf_state = EnKFState(ensemble=ensemble0)
            key, kp, ku = jax.random.split(key, 3)
            enkf_state        = predict(enkf_state, Q, kp)
            enkf_state, K_bar = update(enkf_state, y_obs, H, R, ku)
        """
        from kf import make_enkf
        propagator = self.make_surrogate_propagator(params, dt)
        return make_enkf(propagator, self.N, N_ens)
 
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test, x_test):
        # Predict the trajectory under initial condition 
        x_pred = self.x_pred_fn(params, u_test, self.t_star)
        error = jnp.linalg.norm(x_pred - x_test) / jnp.linalg.norm(x_test)
        return error

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

    def __call__(self, state, batch, u_ref, x_ref):
        self.log_dict = super().__call__(state, batch)

        # Causal weights now need the full batch (batch_u, batch_t)
        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, x_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, u_ref)

        return self.log_dict