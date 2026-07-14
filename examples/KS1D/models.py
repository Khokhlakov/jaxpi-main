from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, jacfwd

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
import numpy as np

# Reference numerical solver (see generate_data.py). We reuse it here ONLY
# for its spectral *operators* (L_op, the dealiasing mask, and the
# nonlinear-term evaluator) -- never its ETDRK4 time-stepping coefficients.
# The PINN residual needs the continuous-time right-hand side of
#
#       dv/dtau = L_op * v_hat + N(v_hat)                      (*)
#
# i.e. the semi-discrete ODE the reference solver *advances*, not a
# discrete update rule for a specific dt.
from data.generate_data import KuramotoSivashinskyAdvanced


# =============================================================================
# Why the residual is computed the way it is (read this before editing r_net)
# =============================================================================
# L96 is a 40-dimensional ODE system: its "spatial" coupling between
# neighbouring variables is just an algebraic `jnp.roll` shift, so the only
# derivative the physics loss ever needs is d/dt, taken with a single
# `jacfwd` over the scalar trunk input.
#
# The 1D KS equation is a genuine PDE: its right-hand side needs spatial
# derivatives up to 4th order (u_xx and u_xxxx). Two ways to get those from
# a branch/trunk network:
#
#   (a) Add a continuous spatial coordinate `xi` to the trunk net and take
#       `jacfwd`/`jacrev` of the (now scalar-in, scalar-out) network four
#       times over. This is exact in principle but numerically fragile in
#       practice (four nested forward-mode passes through an MLP amplify
#       roundoff and are also comparatively expensive), and it requires
#       widening the trunk net to accept `(t, xi)` instead of just `t` --
#       i.e. changing the architecture, not just the loss.
#
#   (b) Keep the branch/trunk(t)-only architecture from the L96 code
#       completely intact (only widen the output from N=40 state variables
#       to N=grid points), and differentiate the network's predicted
#       *spatial profile* the same way the reference solver differentiates
#       it: spectrally, via FFT. Because the network's output lives on the
#       same uniform periodic grid the solver uses, this gives exact (to
#       float precision) derivatives of *any* order in a single FFT/iFFT
#       pair, with no repeated autodiff and no architecture change.
#
# We use (b). `r_net` below evaluates the network at (u, t), takes d/dt via
# one `jacfwd`, and gets the full spatial right-hand side of (*) by pushing
# the network's output through the reference solver's own `L_op` and
# `_nonlinear_term`.
# =============================================================================


class KSUDON(ForwardIVP):
    """Physics-informed DeepONet for the nondimensionalized 1D KS equation.

    Branch input  u : (N,)  initial spatial profile v(xi, tau=0) on the N
                            uniformly-spaced grid points used by the
                            reference solver (real space, not Fourier modes)
    Trunk  input  t : scalar tau in [t0, t1] (one training "window")
    Output x_net    : (N,)  predicted profile v(xi, tau) on the same N grid
                            points

    `t0`/`t1` span a single window (the same normalized-time window the
    reference solver advances between saved snapshots), matching the
    windowed/autoregressive training strategy used for L96.
    """

    def __init__(self, config, t_star, L: float = 64.0, N: int = 256, dt: float = 0.02):
        super().__init__(config)
        self.t_star = t_star

        # Spatial grid size == branch/trunk output width.
        self.N = N

        # Reference solver instance: only its spectral operators are used
        # below (L_op, _dealias, _nonlinear_term, k_xi, c_x, c_u). `dt` is
        # only needed because the constructor also builds ETDRK4
        # coefficients we never call -- it has no bearing on the residual.
        self.solver = KuramotoSivashinskyAdvanced(L=L, N=N, dt=dt)

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

        # Continuous-time RHS of dv/dtau = L_op * v_hat + N(v_hat), evaluated
        # spectrally on the network's predicted profile -- the exact same
        # operator the reference ETDRK4 solver advances in time (just
        # without the RK4 stage combination, since we want the instantaneous
        # RHS, not a discrete step).
        x_hat = jnp.fft.rfft(x)
        rhs_hat = self.solver.L_op * x_hat + self.solver._nonlinear_term(x_hat)
        rhs = jnp.fft.irfft(rhs_hat, n=self.N)

        r_x = x_t - rhs
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

        # Chunk loss: average over time-within-chunk, ICs, and grid points
        l = jnp.mean(r_chunks ** 2, axis=(1, 2, 3))

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
        """
        Return a variable-time surrogate propagator suitable for both the EKF
        (when wrapped with a fixed ``t``) and the window-aware EnKF.

        The returned callable has the signature::

            propagator(u: (N,), t: float) -> (N,)

        where ``t`` is the query time *within the current assimilation window*
        (0 < t <= DT_WINDOW). The DeepONet is queried as x(t | u), exploiting
        the full [t0, t1] training range rather than always stepping by dt_fine.

        Because ``t`` is a plain Python float (treated as static inside
        ``make_enkf``'s ``@partial(jit, static_argnums=(3,))`` predict function),
        ``jnp.array([t])`` resolves to a compile-time constant array -- no
        retracing overhead beyond the first ``steps_per_window`` distinct values.

        Args:
            params: frozen (unreplicated) network parameters.

        Returns:
            propagator: Callable[(N,), float -> (N,)]
        """
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t])  # t is a static Python float -> constant array
            return self.x_net(params, u, t_vec).reshape(self.N)

        return propagator

    def make_ekf_fns(self, params, dt: float):
        """
        Convenience method: build JIT-compiled EKF predict/update functions.

        Wraps the variable-time surrogate propagator with a fixed ``dt`` so
        that the EKF's ``jacfwd``-based linearisation always evaluates the
        Jacobian at the same integration length.

        Usage:
            predict, update = model.make_ekf_fns(params, dt=0.05)
            ekf_state = EKFState(x_hat=x0, P=P0)
            ekf_state = predict(ekf_state, Q)
            ekf_state, K = update(ekf_state, y_obs, H, R)
        """
        from examples.KS.kf import make_ekf
        propagator_vt = self.make_surrogate_propagator(params)  # (u, t) -> u
        # Fix t=dt so the EKF always linearises over exactly one fine step.
        propagator    = lambda u: propagator_vt(u, dt)          # (u,) -> (N,)
        return make_ekf(propagator, self.N)

    def make_enkf_fns(self, params, N_ens: int = 50):
        """
        Convenience method: build JIT-compiled EnKF predict/update functions
        using the variable-time surrogate propagator.

        Unlike ``make_ekf_fns``, no fixed ``dt`` is baked in. The returned
        ``predict_fn`` accepts ``t_query`` as a *static* argument at each call
        site, allowing ``run_enkf_smoother`` to vary the in-window query time
        step-by-step:

            predict_fn(enkf_state, Q, key, t_query=k * dt_fine)

        This means step ``k`` within a window evaluates
        ``x(window_ic, k * dt_fine)`` directly instead of chaining
        ``f(f(...f(u, dt_fine)...), dt_fine)``, which would accumulate surrogate
        error across fine steps.

        The ``dt_fine`` and ``dt_window`` needed for the window-reset logic live
        in ``run_enkf_smoother``, not here.

        Args:
            params: frozen (unreplicated) network parameters.
            N_ens:  ensemble size (grid is 1D and periodic; N_ens on the
                    order of a few dozen to ~100 is typically sufficient,
                    but tune against your own observation operator).

        Returns:
            predict_fn, update_fn -- both JIT-compiled EnKF functions.

        Usage:
            predict, update = model.make_enkf_fns(params, N_ens=50)
            # Then call run_enkf_smoother with dt_fine and dt_window.
        """
        from examples.KS.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)  # (u, t) -> u
        return make_enkf(propagator, self.N, N_ens)

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


class KSUDONEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star

        # Physical spatial axis [0, L) instead of a bare variable index --
        # meaningful now that the "variables" are grid points of a 1D field.
        x_phys = np.arange(self.model.N) * (self.model.solver.L / self.model.N)

        fig, ax = plt.subplots(figsize=(10, 8))

        c = ax.pcolormesh(x_phys, t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("x")
        ax.set_ylabel("Time (t)")
        ax.set_title("KS UDON Trajectory Heatmap")
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
class KSUDON_DD(ForwardIVP):
    def __init__(self, config, t_star, N: int = 256, L: float = 64.0):
        super().__init__(config)
        self.t_star = t_star
        self.N = N
        self.L = L  # kept only so the evaluator can plot a physical x-axis
        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid (t partition)
        self.x_pred_fn = vmap(self.x_net, (None, None, 0))

    def x_net(self, params, u, t):
        t = jnp.atleast_1d(t)
        return self.state.apply_fn(params, u, t)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # batch expects: (batch_u, batch_t, batch_x_true)
        batch_u, batch_t, batch_x_true = batch
        batch_t = batch_t.reshape(-1)

        # Predict state from initial condition and time
        x_pred = vmap(self.x_net, (None, 0, 0))(params, batch_u, batch_t)

        # Supervised data loss (MSE)
        data_loss = jnp.mean((x_pred - batch_x_true) ** 2)

        loss_dict = {"data_loss": data_loss}
        return loss_dict

    def make_surrogate_propagator(self, params) -> Callable:
        """
        Return a variable-time surrogate propagator suitable for both the EKF
        (when wrapped with a fixed ``t``) and the window-aware EnKF.

        The returned callable has the signature::

            propagator(u: (N,), t: float) -> (N,)

        where ``t`` is the query time *within the current assimilation window*
        (0 < t <= DT_WINDOW). The DeepONet is queried as x(t | u), exploiting
        the full [t0, t1] training range rather than always stepping by dt_fine.

        Because ``t`` is a plain Python float (treated as static inside
        ``make_enkf``'s ``@partial(jit, static_argnums=(3,))`` predict function),
        ``jnp.array([t])`` resolves to a compile-time constant array -- no
        retracing overhead beyond the first ``steps_per_window`` distinct values.

        Args:
            params: frozen (unreplicated) network parameters.

        Returns:
            propagator: Callable[(N,), float -> (N,)]
        """
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:
            t_vec = jnp.array([t])  # t is a static Python float -> constant array
            return self.x_net(params, u, t_vec).reshape(self.N)

        return propagator

    def make_ekf_fns(self, params, dt: float):
        """
        Convenience method: build JIT-compiled EKF predict/update functions.

        Wraps the variable-time surrogate propagator with a fixed ``dt`` so
        that the EKF's ``jacfwd``-based linearisation always evaluates the
        Jacobian at the same integration length.

        Usage:
            predict, update = model.make_ekf_fns(params, dt=0.05)
            ekf_state = EKFState(x_hat=x0, P=P0)
            ekf_state = predict(ekf_state, Q)
            ekf_state, K = update(ekf_state, y_obs, H, R)
        """
        from examples.KS.kf import make_ekf
        propagator_vt = self.make_surrogate_propagator(params)  # (u, t) -> u
        # Fix t=dt so the EKF always linearises over exactly one fine step.
        propagator    = lambda u: propagator_vt(u, dt)          # (u,) -> (N,)
        return make_ekf(propagator, self.N)

    def make_enkf_fns(self, params, N_ens: int = 50):
        """
        Convenience method: build JIT-compiled EnKF predict/update functions
        using the variable-time surrogate propagator.

        Unlike ``make_ekf_fns``, no fixed ``dt`` is baked in. The returned
        ``predict_fn`` accepts ``t_query`` as a *static* argument at each call
        site, allowing ``run_enkf_smoother`` to vary the in-window query time
        step-by-step:

            predict_fn(enkf_state, Q, key, t_query=k * dt_fine)

        This means step ``k`` within a window evaluates
        ``x(window_ic, k * dt_fine)`` directly instead of chaining
        ``f(f(...f(u, dt_fine)...), dt_fine)``, which would accumulate surrogate
        error across fine steps.

        The ``dt_fine`` and ``dt_window`` needed for the window-reset logic live
        in ``run_enkf_smoother``, not here.

        Args:
            params: frozen (unreplicated) network parameters.
            N_ens:  ensemble size.

        Returns:
            predict_fn, update_fn -- both JIT-compiled EnKF functions.

        Usage:
            predict, update = model.make_enkf_fns(params, N_ens=50)
            # Then call run_enkf_smoother with dt_fine and dt_window.
        """
        from examples.KS.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)  # (u, t) -> u
        return make_enkf(propagator, self.N, N_ens)

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


class KSUDONEvaluator_DD(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, x_ref):
        l2_error = self.model.compute_l2_error(params, u_ref, x_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, u_ref):
        x_pred = self.model.x_pred_fn(params, u_ref, self.model.t_star)
        t = self.model.t_star
        x_phys = np.arange(self.model.N) * (self.model.L / self.model.N)

        fig, ax = plt.subplots(figsize=(10, 8))

        c = ax.pcolormesh(x_phys, t, x_pred, cmap='viridis', shading='auto')
        ax.set_xlabel("x")
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