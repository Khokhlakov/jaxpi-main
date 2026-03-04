from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, jacfwd

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class L63(ForwardIVP):
    def __init__(self, config, xyz0, t_star):
        super().__init__(config)

        self.xyz0 = xyz0
        self.t_star = t_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]
        
        # System parameters
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0

        # Predictions over a grid (t partition)
        self.xyz_pred_fn = vmap(self.xyz_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    def xyz_net(self, params, t):
        z = jnp.array([t])
        xyz = self.state.apply_fn(params, z)
        print("Lazlo")
        print(xyz)
        print("\n")
        print(xyz[0])
        return xyz

    def grad_net(self, params, t):
        xyz_t = jacfwd(self.xyz_net, argnums=1)(params, t)
        return xyz_t

    def r_net(self, params, t):
        xyz = self.xyz_net(params, t)
        x, y, z = xyz[0], xyz[1], xyz[2]

        xyz_t = jacfwd(self.xyz_net, argnums=1)(params, t)
        x_t, y_t, z_t = xyz_t[0], xyz_t[1], xyz_t[2]

        r_x = x_t - self.sigma * (y - x)
        r_y = y_t - (x * (self.rho - z) - y)
        r_z = z_t - (x * y - self.beta * z)
        return jnp.array([r_x, r_y, r_z])

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch_t):
        "Compute residuals and weights for causal training"
        # Sort time coordinates
        t_sorted = jnp.sort(batch_t)
        r_pred = vmap(self.r_net, (None, 0))(params, t_sorted)
        # Split residuals into chunks
        r_pred = r_pred.reshape(self.num_chunks, -1, 3)
        # Err
        l = jnp.mean(r_pred**2, axis=(1,2))
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        batch_t = batch.flatten()
        xyz_pred = self.xyz_net(params, self.t0)
        ics_loss = jnp.mean((self.xyz0 - xyz_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch_t)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0))(params, batch_t)
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        batch_t = batch.flatten()
        ics_ntk = ntk_fn, (self.xyz_net, params, self.t0)

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            t_sorted = jnp.sort(batch_t)
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, t_sorted
            )
            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, t_sorted)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, batch_t
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, xyz_test):
        xyz_pred = self.xyz_pred_fn(params, self.t_star)
        error = jnp.linalg.norm(xyz_pred - xyz_test) / jnp.linalg.norm(xyz_test)
        return error



class L63Evaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, xyz_ref):
        l2_error = self.model.compute_l2_error(params, xyz_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        xyz_pred = self.model.xyz_pred_fn(params, self.model.t_star)
        t = self.model.t_star

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        labels = ['x(t)', 'y(t)', 'z(t)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(3):
            axes[i].plot(t, xyz_pred[:, i], label=f"Pred {labels[i]}", color=colors[i], lw=1.5)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
        axes[2].set_xlabel("Time (t)")
        plt.tight_layout()
        self.log_dict["xyz_pred"] = fig
        plt.close()

    def __call__(self, state, batch, xyz_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            batch_t = batch.flatten()
            _, causal_weight = self.model.res_and_w(state.params, batch_t)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, xyz_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
