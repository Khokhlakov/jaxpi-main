import os
import time
import logging
from functools import partial
from typing import Callable
 
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from flax.training import train_state, checkpoints
from scipy.io import loadmat
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import wandb
 
from jaxpi import archs
from examples.l96_6_3.utils import build_obs_schedule, scale_Q_for_fine_steps, get_dataset
from examples.l96_6_3.eval import (
    _load_l2_eval_pool,
    _plot_l2_per_window,
    _plot_trajectory_summary,
    _plot_erf,
)
 
 
# ── Checkpoint helpers (thin wrappers around Flax / Orbax) ───────────────────
 
def _save_checkpoint(state, ckpt_dir: str, keep: int = 3) -> None:
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir, state, step=int(state.step), keep=keep, overwrite=True
    )
 
 
def _restore_checkpoint(state, ckpt_dir: str):
    ckpt_dir = os.path.abspath(ckpt_dir)
    return checkpoints.restore_checkpoint(ckpt_dir, state)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# L96MLP model class
# ─────────────────────────────────────────────────────────────────────────────
 
class L96MLP:
    """
    Data-driven MLP surrogate:  x(T) → x(T + dt_window).
 
    Architecture
    ------------
    A plain feed-forward MLP (or its modified variant) taken directly from
    jaxpi.archs so that activation functions, weight reparameterisation, and
    Fourier embeddings are all available via the config.
 
    The network maps  R^N → R^N  using a standard supervised MSE loss on
    consecutive window-boundary pairs.
 
    Parameters
    ----------
    config : ml_collections.ConfigDict
        Must contain at minimum:
          config.arch.{arch_name, num_layers, hidden_dim, activation}
          config.optim.{learning_rate, decay_steps, decay_rate, beta1, beta2}
          config.seed
    dt_window : float
        Duration of one prediction step (same units as t_star).  Defaults
        to 0.25 to match the DeepONet training window.
    """
 
    N: int = 40  # L96 state dimension
 
    def __init__(self, config, dt_window: float = 0.25):
        self.config = config
        self.dt_window = dt_window
        self.N = 40
 
        # ── Architecture ──────────────────────────────────────────────────
        arch_cfg = config.arch
        arch_name = arch_cfg.arch_name
 
        arch_kwargs = dict(
            num_layers=arch_cfg.num_layers,
            hidden_dim=arch_cfg.hidden_dim,
            out_dim=self.N,
            activation=arch_cfg.activation,
        )
        # Optional extras shared with jaxpi.archs
        for opt_key in ("fourier_emb", "reparam", "periodicity"):
            val = arch_cfg.get(opt_key, None)
            if val is not None:
                arch_kwargs[opt_key] = val
 
        if arch_name == "Mlp":
            self.arch = archs.Mlp(**arch_kwargs)
        elif arch_name == "ModifiedMlp":
            self.arch = archs.ModifiedMlp(**arch_kwargs)
        else:
            raise NotImplementedError(
                f"arch_name '{arch_name}' not supported for L96MLP. "
                f"Choose 'Mlp' or 'ModifiedMlp'."
            )
 
        # ── Parameter initialisation ──────────────────────────────────────
        key = jax.random.PRNGKey(config.seed)
        params = self.arch.init(key, jnp.ones(self.N))
 
        # ── Optimiser ─────────────────────────────────────────────────────
        lr_schedule = optax.exponential_decay(
            init_value=config.optim.learning_rate,
            transition_steps=config.optim.decay_steps,
            decay_rate=config.optim.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr_schedule,
            b1=config.optim.beta1,
            b2=config.optim.beta2,
        )
 
        self.state = train_state.TrainState.create(
            apply_fn=self.arch.apply,
            params=params,
            tx=tx,
        )
 
    # ── Forward pass ──────────────────────────────────────────────────────
 
    @partial(jit, static_argnums=(0,))
    def predict(self, params, u: jnp.ndarray) -> jnp.ndarray:
        """Single-step prediction.
 
        Args:
            params: frozen network parameters.
            u:      (N,) state vector at time T.
 
        Returns:
            (N,) predicted state at time T + dt_window.
        """
        return self.arch.apply(params, u)
 
    # ── Loss and gradient step ────────────────────────────────────────────
 
    @partial(jit, static_argnums=(0,))
    def step(
        self,
        state: train_state.TrainState,
        u_batch:      jnp.ndarray,   # (B, N) inputs
        x_next_batch: jnp.ndarray,   # (B, N) targets
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """One gradient descent step.
 
        Returns updated state and the scalar MSE loss value.
        """
        def loss_fn(params):
            x_pred = vmap(state.apply_fn, (None, 0))(params, u_batch)
            return jnp.mean((x_pred - x_next_batch) ** 2)
 
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_val
 
    # ── Evaluation ───────────────────────────────────────────────────────
 
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(
        self,
        params,
        u_batch:      jnp.ndarray,   # (B, N)
        x_next_batch: jnp.ndarray,   # (B, N)
    ) -> jnp.ndarray:
        """Batch-mean relative L2 error for one-step prediction."""
        x_pred = vmap(self.arch.apply, (None, 0))(params, u_batch)
 
        def single_err(pred, true):
            return jnp.linalg.norm(pred - true) / (jnp.linalg.norm(true) + 1e-12)
 
        return jnp.mean(vmap(single_err)(x_pred, x_next_batch))
 
    # ── KF propagator interface ───────────────────────────────────────────
 
    def make_surrogate_propagator(self, params) -> Callable:
        """
        Return a propagator with signature  (u: (N,), t: float) -> (N,).
 
        The MLP is a fixed-dt map.  The ``t`` argument is accepted for
        interface compatibility with make_ekf / make_enkf in kf.py but is
        intentionally ignored — the network always maps x(T) to x(T + dt_window).
 
        When using this propagator inside the window-aware KF smoother, set
        ``dt_fine = dt_window`` in the smoother call so that  t_query  is
        always equal to  dt_window  and the fixed-dt assumption is satisfied.
 
        Args:
            params: frozen (unreplicated) network parameters.
 
        Returns:
            propagator: Callable[(N,), float -> (N,)]
        """
        arch_apply = self.arch.apply  # capture once — avoids re-tracing
 
        def propagator(u: jnp.ndarray, t: float) -> jnp.ndarray:  # noqa: ARG001
            return arch_apply(params, u)
 
        return propagator
 
    def make_ekf_fns(self, params):
        """
        Build JIT-compiled EKF predict/update functions using the MLP
        as the surrogate propagator.
 
        The EKF linearises the MLP at each step via  jacfwd, giving the
        first-order covariance update  F P F^T + Q  where F is the MLP
        Jacobian.
 
        Returns:
            predict_fn, update_fn — as produced by kf.make_ekf.
        """
        from examples.l96_6_3.kf import make_ekf
        propagator = self.make_surrogate_propagator(params)
        return make_ekf(propagator, self.N)
 
    def make_enkf_fns(self, params, N_ens: int = 50):
        """
        Build JIT-compiled EnKF predict/update functions using the MLP
        as the ensemble propagator.
 
        Args:
            params: frozen network parameters.
            N_ens:  ensemble size (≥ 50 recommended for L96-N40).
 
        Returns:
            predict_fn, update_fn — as produced by kf.make_enkf.
        """
        from examples.l96_6_3.kf import make_enkf
        propagator = self.make_surrogate_propagator(params)
        return make_enkf(propagator, self.N, N_ens)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def load_mlp_training_pairs(
    mat_path:      str,
    max_additions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (input, target) pairs from the rollout pool file.
 
    Consecutive slots in the file represent states separated by exactly
    one window of length dt_window:
 
        input   = u0_rollout_{k}     (or u0_original for k=0)
        target  = u0_rollout_{k+1}
 
    yielding  max_additions × num_initial_ics  pairs in total.
 
    Args:
        mat_path:      path to train_rollouts_025.mat (or equivalent).
        max_additions: number of rollout slots to read (= config.training.max_additions).
 
    Returns:
        inputs:  (max_additions * num_initial_ics, N) float32 array.
        targets: (max_additions * num_initial_ics, N) float32 array.
    """
    data = loadmat(mat_path)
 
    u0_original = data["u0_original"].astype(np.float32)
 
    slots: list[np.ndarray] = [u0_original]
    for k in range(1, max_additions + 1):
        key_name = f"u0_rollout_{k}"
        if key_name not in data:
            raise KeyError(
                f"Key '{key_name}' not found in {mat_path}. "
                f"Regenerate with max_additions >= {k}."
            )
        slots.append(data[key_name].astype(np.float32))
 
    # Pair every slot with the next: slot[k] → slot[k+1]
    inputs  = np.concatenate(slots[:-1], axis=0)   # (max_additions * B, N)
    targets = np.concatenate(slots[1:],  axis=0)   # (max_additions * B, N)
 
    return inputs, targets
 
 
def _build_eval_pairs_from_dataset(
    time_steps: int = 50,
    max_windows: int = 5,
    trajs: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one-step evaluation pairs from l96_udon.mat.
 
    Window boundaries in the dataset live at indices  k * time_steps
    (k = 0, 1, …, max_windows).  Pairs  (x[k*T], x[(k+1)*T])  are the
    held-out evaluation analogues of the training pairs from the rollout
    pool, drawn from a completely separate set of ICs.
 
    Returns:
        u_eval:  (max_windows * trajs, N)
        x_eval:  (max_windows * trajs, N)
    """
    x_ref_all, _, _ = get_dataset()   # (num_ics, num_points, N)
 
    u_list, x_list = [], []
    for k in range(max_windows):
        t_start = k * time_steps
        t_end   = (k + 1) * time_steps
        u_list.append(np.array(x_ref_all[:trajs, t_start,  :]))   # state at start
        x_list.append(np.array(x_ref_all[:trajs, t_end,    :]))   # state one window later
 
    return (
        np.concatenate(u_list, axis=0).astype(np.float32),
        np.concatenate(x_list, axis=0).astype(np.float32),
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
 
def train_mlp(config, workdir: str) -> "L96MLP":
    """
    Train the MLP surrogate on consecutive window-boundary pairs.
 
    Training data
    -------------
    All  max_additions × num_initial_ics  pairs from the rollout pool file
    are loaded once at startup and randomly mini-batched each step.
 
    Evaluation
    ----------
    One-step L2 error is computed on held-out pairs from l96_udon.mat
    (completely separate ICs from the training pool) at every
    config.logging.log_every_steps.
 
    Checkpointing
    -------------
    Model state is saved to  workdir/<config.wandb.name>/ckpt/mlp_model
    every config.saving.save_every_steps steps.
 
    Args:
        config:  ml_collections.ConfigDict.  Relevant sub-trees:
                   config.arch, config.optim, config.training,
                   config.logging, config.saving, config.wandb.
        workdir: root directory for checkpoints and figures.
 
    Returns:
        Trained L96MLP instance with updated state.
    """
    wandb.init(project=config.wandb.project, name=config.wandb.name)
 
    dt_window     = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 10)
    batch_size    = config.training.batch_size_per_device
    max_steps     = config.training.max_steps
    seed          = config.training.get("seed", 42)
 
    mat_path = os.path.join(
        "data",
        config.training.get("augmentation_file_name", "train_rollouts_025.mat"),
    )
 
    # ── Load training data ────────────────────────────────────────────────
    logging.info(f"Loading MLP training pairs from {mat_path} …")
    inputs_np, targets_np = load_mlp_training_pairs(mat_path, max_additions)
    num_pairs = inputs_np.shape[0]
    logging.info(
        f"Training pairs: {num_pairs}  "
        f"({max_additions} slots × {num_pairs // max_additions} ICs)"
    )
 
    inputs  = jnp.array(inputs_np)
    targets = jnp.array(targets_np)
 
    # ── Load evaluation data (held-out from l96_udon.mat) ─────────────────
    time_steps  = config.training.get("time_steps_eval", 50)
    trajs_eval  = config.training.get("trajs_eval",      100)
    max_win_eval = min(max_additions, config.training.get("num_time_windows", 5))
 
    u_eval_np, x_eval_np = _build_eval_pairs_from_dataset(
        time_steps=time_steps,
        max_windows=max_win_eval,
        trajs=trajs_eval,
    )
    u_eval = jnp.array(u_eval_np)
    x_eval = jnp.array(x_eval_np)
    logging.info(f"Evaluation pairs: {u_eval.shape[0]}")
 
    # ── Build model ───────────────────────────────────────────────────────
    model = L96MLP(config, dt_window=dt_window)
 
    # ── Training loop ─────────────────────────────────────────────────────
    key        = jax.random.PRNGKey(seed)
    start_time = time.time()
    logging.info("Starting MLP training …")
 
    for step in range(max_steps):
 
        # Random mini-batch
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=num_pairs)
        u_batch = inputs[idx]
        x_batch = targets[idx]
 
        model.state, train_loss = model.step(model.state, u_batch, x_batch)
 
        # ── Logging ───────────────────────────────────────────────────────
        if step % config.logging.log_every_steps == 0:
            eval_l2 = model.compute_l2_error(model.state.params, u_eval, x_eval)
 
            end_time = time.time()
            elapsed  = end_time - start_time
 
            log_dict = {
                "train/mse_loss": float(train_loss),
                "eval/l2_error":  float(eval_l2),
            }
            wandb.log(log_dict, step=step)
            logging.info(
                f"Step {step:>7d} | "
                f"train MSE: {float(train_loss):.3e} | "
                f"eval L2:   {float(eval_l2):.3e} | "
                f"{elapsed:.1f}s"
            )
            start_time = end_time
 
        # ── Checkpointing ─────────────────────────────────────────────────
        if config.saving.save_every_steps is not None:
            if (
                (step + 1) % config.saving.save_every_steps == 0
                or (step + 1) == max_steps
            ):
                ckpt_path = os.path.join(
                    workdir, config.wandb.name, "ckpt", "mlp_model"
                )
                _save_checkpoint(
                    model.state, ckpt_path,
                    keep=config.saving.num_keep_ckpts,
                )
                logging.info(f"Checkpoint saved to {ckpt_path}")
 
    logging.info("MLP training complete.")
    return model
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Open-loop evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _evaluate_batch_l2_openloop_mlp(
    model:   "L96MLP",
    params,
    config,
    workdir: str,
) -> np.ndarray:
    """
    Compute and plot the batch-averaged open-loop L2 error per window for
    the MLP.  Mirrors _evaluate_batch_l2_openloop from eval.py.
 
    The MLP is autoregressively applied k times starting from u0_original
    and compared to the ground-truth u0_rollout_k at each window boundary.
 
    Returns:
        l2_per_window: (max_additions,) array of mean relative L2 errors.
    """
    dt_window     = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
 
    logging.info("Computing MLP batch L2 per window (open-loop) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, model.N)
    B = u0_original.shape[0]
 
    # Vectorised single-step predictor over the batch dimension
    predict_one_step = jax.jit(
        jax.vmap(lambda u: model.arch.apply(params, u), in_axes=0)
    )
 
    l2_per_window: list[float] = []
    u_current = u0_original   # (B, N)
 
    for k in range(max_additions):
        u_current = predict_one_step(u_current)          # (B, N)
 
        ref_k = rollout_states[k]                        # (B, N)
        numer = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom = jnp.linalg.norm(ref_k,              axis=1)
        l2_mean = float(jnp.mean(numer / (denom + 1e-12)))
        l2_per_window.append(l2_mean)
        logging.info(f"  Window {k + 1:>3d} | MLP mean L2: {l2_mean:.3e}")
 
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_mlp_openloop.pdf")
    _plot_l2_per_window(
        curves    = {"Open-loop (MLP)": np.array(l2_per_window)},
        dt        = dt_window,
        title     = f"Open-loop MLP: batch-average L2 per window  (B={B})",
        save_path = save_path,
        colors    = {"Open-loop (MLP)": "#9C27B0"},
    )
    return np.array(l2_per_window)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Per-IC open-loop evaluation
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_mlp(config, workdir: str) -> None:
    """
    Open-loop evaluation of the MLP: per-IC trajectory summaries and
    batch-averaged L2 per window.
 
    For each IC the MLP is rolled out autoregressively for
    config.training.num_time_windows steps of length dt_window.  Because
    the MLP produces states only at window boundaries, the trajectory
    plot shows one point per window rather than the dense time series
    generated by the DeepONet.  The ground truth is solved via scipy at
    those same boundary times so that L2 values are directly comparable.
 
    Args:
        config:  ml_collections.ConfigDict (same structure as for L96UDON).
        workdir: root directory containing the checkpoint and figures.
    """
    dt_window   = float(config.get("dt_window", 0.25))
    num_windows = config.training.num_time_windows
 
    # ── Load checkpoint ───────────────────────────────────────────────────
    model     = L96MLP(config, dt_window=dt_window)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt", "mlp_model")
    model.state = _restore_checkpoint(model.state, ckpt_path)
    params    = model.state.params
    logging.info("Restored MLP checkpoint for evaluation.")
 
    # ── Reference ICs ─────────────────────────────────────────────────────
    _, u0_ref_all, _ = get_dataset()
 
    def lorenz_96(t, state, F: float = 6.0):
        xp1 = np.roll(state, -1)
        xm1 = np.roll(state,  1)
        xm2 = np.roll(state,  2)
        return (xp1 - xm2) * xm1 - state + F
 
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
 
    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- MLP Open-loop Evaluation for IC {ic_idx} ---")
 
        # ── Autoregressive rollout ─────────────────────────────────────────
        # States only at window boundaries:  t = 0, dt, 2*dt, …, K*dt
        u_current = jnp.array(u0_ref_all[ic_idx])
        states = [np.array(u_current)]
 
        for _ in range(num_windows):
            u_current = model.predict(params, u_current)
            states.append(np.array(u_current))
 
        x_pred = np.stack(states, axis=0)              # (num_windows+1, N)
        t_axis = np.array([k * dt_window for k in range(num_windows + 1)])
 
        # ── Ground truth at the same boundary times ────────────────────────
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, t_axis[-1]],
            y0=np.array(u0_ref_all[ic_idx]),
            t_eval=t_axis,
            rtol=1e-9, atol=1e-11,
        )
        x_true = sol.y.T   # (num_windows+1, N)
 
        l2 = (
            np.linalg.norm(x_pred - x_true)
            / (np.linalg.norm(x_true) + 1e-12)
        )
        print(f"IC {ic_idx} | MLP full-rollout L2: {l2:.3e}")
 
        _plot_trajectory_summary(
            t_ax       = t_axis,
            x_true     = x_true,
            x_est      = x_pred,
            x_std      = None,
            ic_idx     = ic_idx,
            est_label  = "MLP",
            save_path  = os.path.join(
                save_dir, f"trajectory_summary_mlp_ic_{ic_idx}.pdf"
            ),
            N          = model.N,
            dt_window  = dt_window,
            obs_coords = None,
        )
 
        # ── Heatmap ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
 
        im0 = axes[0].pcolormesh(
            np.arange(model.N), t_axis, x_true, cmap="viridis", shading="auto"
        )
        axes[0].set_title(f"Exact L96 Reference (IC {ic_idx})", fontsize=14)
        axes[0].set_ylabel("Time (t)", fontsize=14)
        axes[0].set_xlabel("Variables (0–39)", fontsize=14)
        fig.colorbar(im0, ax=axes[0])
 
        im1 = axes[1].pcolormesh(
            np.arange(model.N), t_axis, x_pred, cmap="viridis", shading="auto"
        )
        axes[1].set_title(f"MLP Rollout (IC {ic_idx})", fontsize=14)
        axes[1].set_xlabel("Variables (0–39)", fontsize=14)
        fig.colorbar(im1, ax=axes[1])
 
        abs_err = np.abs(x_true - x_pred)
        im2 = axes[2].pcolormesh(
            np.arange(model.N), t_axis, abs_err, cmap="magma", shading="auto"
        )
        axes[2].set_title(f"Absolute Error (IC {ic_idx})", fontsize=14)
        axes[2].set_xlabel("Variables (0–39)", fontsize=14)
        fig.colorbar(im2, ax=axes[2])
 
        fig.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"mlp_rollout_analysis_ic_{ic_idx}.pdf"),
            bbox_inches="tight", dpi=300,
        )
        plt.close(fig)
 
    # ── Batch L2 ──────────────────────────────────────────────────────────
    _evaluate_batch_l2_openloop_mlp(model, params, config, workdir)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EKF evaluation with MLP surrogate
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_mlp_with_ekf(config, workdir: str) -> None:
    """
    Per-IC EKF evaluation using the MLP as surrogate propagator.
 
    Important timing constraint
    ---------------------------
    The MLP is a fixed-dt map, so  dt_fine must equal dt_window.  The
    EKF therefore operates at coarse resolution: one predict-update cycle
    per window.  Observations must also satisfy  dt_obs >= dt_window.
 
    The linearisation used inside the EKF is the analytic MLP Jacobian
    computed via  jax.jacfwd, which is exact (no finite-difference
    approximation).
 
    Config keys read (under config.ekf):
        obs_every_n, sigma_obs, sigma_proc, P0_sigma, dynamic_vars,
        dt_obs (defaults to dt_window).
    Config keys read (under config.kf):
        specify_obs_idx, obs_idx_list.
 
    Args:
        config:  ml_collections.ConfigDict.
        workdir: root directory.
    """
    from examples.l96_6_3.kf import run_ekf_smoother
 
    obs_every_n  = config.ekf.get("obs_every_n",  4)
    sigma_obs    = config.ekf.get("sigma_obs",    0.5)
    sigma_proc   = config.ekf.get("sigma_proc",   0.1)
    P0_sigma     = config.ekf.get("P0_sigma",     1.0)
    dynamic_vars = config.ekf.get("dynamic_vars", False)
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
 
    DT_WINDOW = float(config.get("dt_window", 0.25))
    # MLP is fixed-dt: dt_fine MUST equal dt_window
    DT_FINE   = DT_WINDOW
    DT_OBS    = float(config.ekf.get("dt_obs", DT_WINDOW))
 
    # ── Load model ────────────────────────────────────────────────────────
    model     = L96MLP(config, dt_window=DT_WINDOW)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt", "mlp_model")
    model.state = _restore_checkpoint(model.state, ckpt_path)
    params    = model.state.params
    N         = model.N
 
    predict_fn, update_fn = model.make_ekf_fns(params)
 
    # ── Covariances ───────────────────────────────────────────────────────
    # steps_per_window = 1 when dt_fine = dt_window, so Q_fine = Q_coarse.
    steps_per_window = round(DT_WINDOW / DT_FINE)   # = 1
    Q_coarse = jnp.eye(N) * sigma_proc ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
 
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
 
    num_windows = config.training.num_time_windows
    total_time  = num_windows * DT_WINDOW
 
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )
 
    def lorenz_96(t, state, F: float = 6.0):
        xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F
 
    _, u0_ref_all, _ = get_dataset()
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
 
    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- MLP EKF Evaluation for IC {ic_idx} ---")
        u_true = u0_ref_all[ic_idx]
 
        # Ground truth at fine (= window) resolution
        t_eval = np.linspace(0.0, total_time, total_fine_steps + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time],
            y0=np.array(u_true),
            t_eval=t_eval,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = jnp.array(sol.y.T)                       # (total_fine_steps+1, N)
        x_true_at_obs = x_true_fine[obs_step_indices + 1]        # (T_obs, N)
 
        # ── Build observations ─────────────────────────────────────────────
        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []
 
        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]
 
            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey  = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m,), replace=False)
            else:
                obs_idx_vars = obs_indices
 
            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise
 
            H_list.append(H_t)
            y_obs_list.append(y_t)
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
 
        # ── Perturbed IC ───────────────────────────────────────────────────
        key, key_ic = jax.random.split(key)
        x0_hat = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
 
        # ── Run EKF ────────────────────────────────────────────────────────
        x_hats, Ps, _ = run_ekf_smoother(
            predict_fn, update_fn,
            x0_hat, P0,
            y_obs_seq,
            obs_step_indices,
            H_seq,
            Q_fine,
            R,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
 
        ekf_std_fine = np.sqrt(
            np.clip(np.diagonal(np.array(Ps), axis1=1, axis2=2), 0, None)
        )   # (total_fine_steps, N)
 
        # ── Window-boundary L2 ─────────────────────────────────────────────
        window_step_indices = np.array([
            round((w + 1) * DT_WINDOW / DT_FINE) - 1
            for w in range(num_windows)
        ])
        x_hats_at_windows = x_hats[window_step_indices]
        x_true_at_windows = x_true_fine[window_step_indices + 1]
 
        l2_ekf = jnp.linalg.norm(x_hats_at_windows - x_true_at_windows) \
               / jnp.linalg.norm(x_true_at_windows)
        print(f"IC {ic_idx} | MLP EKF L2 (window boundaries): {l2_ekf:.3e}")
 
        t_fine_axis = t_eval[1:]   # exclude t=0 to align with filter output
        _plot_trajectory_summary(
            t_ax       = t_fine_axis,
            x_true     = np.array(x_true_fine[1:]),
            x_est      = np.array(x_hats),
            x_std      = ekf_std_fine,
            ic_idx     = ic_idx,
            est_label  = "MLP EKF estimate",
            save_path  = os.path.join(
                save_dir, f"trajectory_summary_mlp_ekf_ic_{ic_idx}.pdf"
            ),
            N          = model.N,
            dt_window  = DT_WINDOW,
            obs_coords = obs_coords,
        )
 
    # ── Batch L2: MLP open-loop vs MLP EKF ───────────────────────────────
    _evaluate_batch_l2_ekf_mlp(
        model, params,
        predict_fn, update_fn,
        Q_fine, R, P0,
        obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config, workdir,
    )
 
 
def _evaluate_batch_l2_ekf_mlp(
    model,
    params,
    predict_fn,
    update_fn,
    Q_fine,
    R,
    P0,
    obs_every_n:  int,
    sigma_obs:    float,
    P0_sigma:     float,
    dynamic_vars: bool,
    dt_fine:      float,
    dt_obs:       float,
    config,
    workdir:      str,
) -> None:
    """
    Batch-averaged L2 per window: MLP open-loop vs MLP EKF.
    Mirrors _evaluate_batch_l2_ekf from eval.py.
    """
    from examples.l96_6_3.kf import run_ekf_smoother
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
 
    dt_window     = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    N             = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    batch_size = config.ekf.get("batch_l2_size", 200)
 
    logging.info("Computing batch L2 per window (MLP open-loop vs MLP EKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)
 
    B = min(u0_original.shape[0], batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  Using {B} ICs for batch evaluation.")
 
    predict_one_step = jax.jit(
        jax.vmap(lambda u: model.arch.apply(params, u), in_axes=0)
    )
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
    m = len(obs_indices)
 
    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch,
        dt_fine=dt_fine,
        dt_obs=dt_obs,
    )
    T_obs = len(obs_step_indices_batch)
 
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])
 
    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(max_additions)
    ])
 
    ekf_l2_sum = np.zeros(max_additions)
    erf_sum    = np.zeros(T_obs)
    erf_sq_sum = np.zeros(T_obs)
 
    def lorenz_96(t, state, F: float = 6.0):
        xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F
 
    for ic in range(B):
        key    = jax.random.PRNGKey(ic + 88888)
        u_true = u0_original[ic]
 
        t_eval_fine = np.linspace(0.0, total_time_batch, total_fine_steps_batch + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time_batch],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = sol.y.T
        x_true_at_obs = x_true_fine[obs_step_indices_batch + 1]
 
        H_list, y_obs_list = [], []
        for obs_idx in range(T_obs):
            x_true_t = x_true_at_obs[obs_idx]
 
            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey  = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m,), replace=False)
            else:
                obs_idx_vars = obs_indices
 
            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise
            H_list.append(H_t)
            y_obs_list.append(y_t)
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
 
        key, key_ic = jax.random.split(key)
        x0_hat = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
 
        x_hats, _, prior_means_at_obs = run_ekf_smoother(
            predict_fn, update_fn,
            x0_hat, P0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,
            R,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )
 
        post_means_at_obs = x_hats[obs_step_indices_batch]
 
        prior_rmse = np.sqrt(np.mean(
            (np.array(prior_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))
        post_rmse = np.sqrt(np.mean(
            (np.array(post_means_at_obs)  - x_true_at_obs) ** 2, axis=1
        ))
        erf_ic      = prior_rmse / (post_rmse + 1e-12)
        erf_sum    += erf_ic
        erf_sq_sum += erf_ic ** 2
 
        for k in range(max_additions):
            ref_k   = rollout_states[k][ic]
            step_k  = window_step_indices[k]
            x_hat_k = x_hats[step_k]
            ekf_l2_sum[k] += float(
                jnp.linalg.norm(x_hat_k - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
 
    # Open-loop MLP
    ol_l2     = np.zeros(max_additions)
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_step(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k,              axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))
 
    l2_ekf   = ekf_l2_sum / B
    erf_mean = erf_sum    / B
    erf_std  = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))
 
    save_dir  = os.path.join(workdir, "figures", config.wandb.name)
    save_path = os.path.join(save_dir, "batch_l2_per_window_mlp_ekf.pdf")
    _plot_l2_per_window(
        curves={
            "Open-loop (MLP)": ol_l2,
            "MLP EKF":         l2_ekf,
        },
        dt        = dt_window,
        title     = f"MLP EKF vs open-loop: batch-average L2 per window  (B={B})",
        save_path = save_path,
        colors    = {"Open-loop (MLP)": "#9C27B0", "MLP EKF": "#FF5722"},
    )
 
    erf_save_path = os.path.join(save_dir, "batch_erf_mlp_ekf.pdf")
    _plot_erf(
        obs_times  = obs_times_batch,
        erf_mean   = erf_mean,
        erf_std    = erf_std,
        n_traj     = B,
        title      = (
            f"MLP EKF Error Reduction Factor per observation time\n"
            f"(B={B} trajectories, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path  = erf_save_path,
    )
    logging.info(f"MLP EKF batch evaluation saved to {save_dir}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EnKF evaluation with MLP surrogate
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_mlp_with_enkf(config, workdir: str) -> None:
    """
    Per-IC EnKF evaluation using the MLP as ensemble propagator, followed
    by batch-averaged L2 and Error Reduction Factor plots.
 
    Mirrors evaluate_with_enkf from eval.py; see that function for a
    detailed description of the filter setup.
 
    Args:
        config:  ml_collections.ConfigDict.
        workdir: root directory.
    """
    from examples.l96_6_3.kf import run_enkf_smoother, init_ensemble
 
    obs_every_n  = config.ekf.get("obs_every_n",   4)
    sigma_obs    = config.ekf.get("sigma_obs",      0.5)
    P0_sigma     = config.ekf.get("P0_sigma",       1.0)
    dynamic_vars = config.ekf.get("dynamic_vars",   False)
    N_ens        = config.enkf.get("N_ens",         50)
    sigma_model  = config.enkf.get("sigma_model",   0.1)
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
 
    DT_WINDOW = float(config.get("dt_window", 0.25))
    # MLP is fixed-dt: dt_fine MUST equal dt_window
    DT_FINE   = DT_WINDOW
    DT_OBS    = float(config.ekf.get("dt_obs", DT_WINDOW))
 
    # ── Load model ────────────────────────────────────────────────────────
    model     = L96MLP(config, dt_window=DT_WINDOW)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt", "mlp_model")
    model.state = _restore_checkpoint(model.state, ckpt_path)
    params    = model.state.params
    N         = model.N
 
    predict_fn, update_fn = model.make_enkf_fns(params, N_ens=N_ens)
 
    # ── Covariances ───────────────────────────────────────────────────────
    steps_per_window = round(DT_WINDOW / DT_FINE)   # = 1
    Q_coarse = jnp.eye(N) * sigma_model ** 2
    Q_fine   = scale_Q_for_fine_steps(Q_coarse, steps_per_window)
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
 
    m  = len(obs_indices)
    R  = jnp.eye(m) * sigma_obs ** 2
    P0 = jnp.eye(N) * P0_sigma ** 2
 
    num_windows = config.training.num_time_windows
    total_time  = num_windows * DT_WINDOW
 
    obs_times, obs_step_indices, total_fine_steps = build_obs_schedule(
        total_time=total_time,
        dt_fine=DT_FINE,
        dt_obs=DT_OBS,
    )
 
    def lorenz_96(t, state, F: float = 6.0):
        xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F
 
    _, u0_ref_all, _ = get_dataset()
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
 
    for ic_idx in range(config.saving.total_plots):
        logging.info(f"--- MLP EnKF Evaluation for IC {ic_idx} (N_ens={N_ens}) ---")
        u_true = u0_ref_all[ic_idx]
 
        t_eval_fine = np.linspace(0.0, total_time, total_fine_steps + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = jnp.array(sol.y.T)                    # (total_fine_steps+1, N)
        x_true_at_obs = x_true_fine[obs_step_indices + 1]     # (T_obs, N)
 
        key = jax.random.PRNGKey(ic_idx)
        H_list, y_obs_list, obs_coords = [], [], []
 
        for obs_idx in range(len(obs_times)):
            x_true_t = x_true_at_obs[obs_idx]
 
            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey  = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m,), replace=False)
            else:
                obs_idx_vars = obs_indices
 
            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise
 
            H_list.append(H_t)
            y_obs_list.append(y_t)
            for j, vi in enumerate(obs_idx_vars):
                obs_coords.append((int(vi), obs_times[obs_idx], float(y_t[j])))
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
 
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
 
        x_means, x_spreads, _ = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices,
            H_seq,
            Q_fine,
            R,
            key,
            total_fine_steps,
            dt_fine=DT_FINE,
            dt_window=DT_WINDOW,
        )
 
        t_fine_axis = t_eval_fine[1:]
 
        _plot_trajectory_summary(
            t_ax       = t_fine_axis,
            x_true     = np.array(x_true_fine[1:]),
            x_est      = np.array(x_means),
            x_std      = np.array(x_spreads),
            ic_idx     = ic_idx,
            est_label  = "MLP EnKF mean",
            save_path  = os.path.join(
                save_dir, f"trajectory_summary_mlp_enkf_ic_{ic_idx}.pdf"
            ),
            N          = model.N,
            dt_window  = DT_WINDOW,
            obs_coords = obs_coords,
        )
 
        window_step_indices = np.array([
            round((w + 1) * DT_WINDOW / DT_FINE) - 1
            for w in range(num_windows)
        ])
        x_means_at_windows = x_means[window_step_indices]
        x_true_at_windows  = x_true_fine[window_step_indices + 1]
 
        l2_enkf     = jnp.linalg.norm(x_means_at_windows - x_true_at_windows) \
                    / jnp.linalg.norm(x_true_at_windows)
        mean_spread = float(jnp.mean(x_spreads))
        print(
            f"IC {ic_idx} | MLP EnKF L2: {l2_enkf:.3e} "
            f"| Mean σ: {mean_spread:.3e}"
        )
 
    # ── Batch evaluation ──────────────────────────────────────────────────
    _evaluate_batch_l2_enkf_mlp(
        model, params,
        predict_fn, update_fn,
        Q_fine, P0,
        N_ens, obs_every_n, sigma_obs, P0_sigma,
        dynamic_vars,
        DT_FINE, DT_OBS,
        config, workdir,
    )
 
 
def _evaluate_batch_l2_enkf_mlp(
    model,
    params,
    predict_fn,
    update_fn,
    Q_fine,
    P0,
    N_ens:        int,
    obs_every_n:  int,
    sigma_obs:    float,
    P0_sigma:     float,
    dynamic_vars: bool,
    dt_fine:      float,
    dt_obs:       float,
    config,
    workdir:      str,
) -> None:
    """
    Batch-averaged L2 per window + calibration plot + ERF plot for the
    MLP EnKF.  Mirrors _evaluate_batch_l2_enkf from eval.py.
    """
    from examples.l96_6_3.kf import run_enkf_smoother, init_ensemble
 
    specify_obs_idx = config.kf.get("specify_obs_idx", False)
    obs_idx_list    = config.kf.get("obs_idx_list",    None)
 
    dt_window     = float(config.get("dt_window", 0.25))
    max_additions = config.training.get("max_additions", 5)
    N             = model.N
    mat_path      = os.path.join(
        "data",
        config.training.get("augmentation_file_name_eval", "train_rollouts_025.mat"),
    )
    batch_size = config.ekf.get("batch_l2_size", 100)
 
    logging.info("Computing batch L2 per window (MLP open-loop vs MLP EnKF) …")
    u0_original, rollout_states = _load_l2_eval_pool(mat_path, max_additions, N)
 
    B = min(u0_original.shape[0], batch_size)
    u0_original   = u0_original[:B]
    rollout_states = [r[:B] for r in rollout_states]
    logging.info(f"  Using {B} ICs (N_ens={N_ens}) for batch L2 / ERF evaluation.")
 
    if specify_obs_idx and obs_idx_list:
        obs_indices = jnp.array(obs_idx_list)
    else:
        obs_indices = jnp.arange(0, N, obs_every_n)
    m       = len(obs_indices)
    R_fixed = jnp.eye(m) * sigma_obs ** 2
 
    predict_one_step = jax.jit(
        jax.vmap(lambda u: model.arch.apply(params, u), in_axes=0)
    )
 
    total_time_batch = max_additions * dt_window
    _, obs_step_indices_batch, total_fine_steps_batch = build_obs_schedule(
        total_time=total_time_batch,
        dt_fine=dt_fine,
        dt_obs=dt_obs,
    )
    T_obs = len(obs_step_indices_batch)
 
    obs_times_batch = np.array([(k + 1) * dt_obs for k in range(T_obs)])
 
    window_step_indices = np.array([
        round((k + 1) * dt_window / dt_fine) - 1
        for k in range(max_additions)
    ])
 
    enkf_l2_sum     = np.zeros(max_additions)
    enkf_rmse_sum   = np.zeros(max_additions)
    enkf_spread_sum = np.zeros(max_additions)
    erf_sum         = np.zeros(T_obs)
    erf_sq_sum      = np.zeros(T_obs)
 
    def lorenz_96(t, state, F: float = 6.0):
        xp1 = np.roll(state, -1); xm1 = np.roll(state, 1); xm2 = np.roll(state, 2)
        return (xp1 - xm2) * xm1 - state + F
 
    for ic in range(B):
        key    = jax.random.PRNGKey(ic + 55555)
        u_true = u0_original[ic]
 
        t_eval_fine = np.linspace(0.0, total_time_batch, total_fine_steps_batch + 1)
        sol = solve_ivp(
            lorenz_96,
            t_span=[0.0, total_time_batch],
            y0=np.array(u_true),
            t_eval=t_eval_fine,
            rtol=1e-9, atol=1e-11,
        )
        x_true_fine   = sol.y.T
        x_true_at_obs = x_true_fine[obs_step_indices_batch + 1]
 
        H_list, y_obs_list = [], []
        for obs_idx in range(T_obs):
            x_true_t = x_true_at_obs[obs_idx]
 
            if not (specify_obs_idx and obs_idx_list) and dynamic_vars:
                key, subkey  = jax.random.split(key)
                obs_idx_vars = jax.random.choice(subkey, N, shape=(m,), replace=False)
            else:
                obs_idx_vars = obs_indices
 
            m_t = len(obs_idx_vars)
            H_t = jnp.zeros((m_t, N)).at[jnp.arange(m_t), obs_idx_vars].set(1.0)
            key, subkey = jax.random.split(key)
            noise = sigma_obs * jax.random.normal(subkey, shape=(m_t,))
            y_t   = x_true_t[obs_idx_vars] + noise
            H_list.append(H_t)
            y_obs_list.append(y_t)
 
        H_seq     = jnp.stack(H_list)
        y_obs_seq = jnp.stack(y_obs_list)
 
        key, key_ic, key_ens = jax.random.split(key, 3)
        x0_hat    = u_true + P0_sigma * jax.random.normal(key_ic, shape=(N,))
        ensemble0 = init_ensemble(x0_hat, P0, N_ens, key_ens)
 
        x_means, x_spreads, prior_means_at_obs = run_enkf_smoother(
            predict_fn, update_fn,
            ensemble0,
            y_obs_seq,
            obs_step_indices_batch,
            H_seq,
            Q_fine,
            R_fixed,
            key,
            total_fine_steps_batch,
            dt_fine=dt_fine,
            dt_window=dt_window,
        )
 
        post_means_at_obs = x_means[obs_step_indices_batch]
 
        prior_rmse = np.sqrt(np.mean(
            (np.array(prior_means_at_obs) - x_true_at_obs) ** 2, axis=1
        ))
        post_rmse = np.sqrt(np.mean(
            (np.array(post_means_at_obs)  - x_true_at_obs) ** 2, axis=1
        ))
        erf_ic      = prior_rmse / (post_rmse + 1e-12)
        erf_sum    += erf_ic
        erf_sq_sum += erf_ic ** 2
 
        for k in range(max_additions):
            ref_k   = rollout_states[k][ic]
            step_k  = window_step_indices[k]
            x_hat_k = x_means[step_k]
            enkf_l2_sum[k] += float(
                jnp.linalg.norm(x_hat_k - ref_k)
                / (jnp.linalg.norm(ref_k) + 1e-12)
            )
            enkf_rmse_sum[k] += float(jnp.sqrt(jnp.mean((x_hat_k - ref_k) ** 2)))
            enkf_spread_sum[k] += float(jnp.sqrt(jnp.mean(x_spreads[step_k] ** 2)))
 
    # Open-loop
    ol_l2     = np.zeros(max_additions)
    u_current = u0_original
    for k in range(max_additions):
        u_current = predict_one_step(u_current)
        ref_k     = rollout_states[k]
        numer     = jnp.linalg.norm(u_current - ref_k, axis=1)
        denom     = jnp.linalg.norm(ref_k,              axis=1)
        ol_l2[k]  = float(jnp.mean(numer / (denom + 1e-12)))
 
    l2_enkf     = enkf_l2_sum     / B
    rmse_enkf   = enkf_rmse_sum   / B
    spread_mean = enkf_spread_sum / B
    erf_mean    = erf_sum         / B
    erf_std     = np.sqrt(np.maximum(erf_sq_sum / B - erf_mean ** 2, 0.0))
 
    save_dir   = os.path.join(workdir, "figures", config.wandb.name)
    window_idx = np.arange(1, max_additions + 1)
 
    # ── L2 + calibration panel ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    ax = axes[0]
    ax.plot(window_idx, ol_l2,   marker="o", markersize=4, linewidth=1.8,
            label="Open-loop (MLP)", color="#9C27B0")
    ax.plot(window_idx, l2_enkf, marker="s", markersize=4, linewidth=1.8,
            label=f"MLP EnKF (N_ens={N_ens})", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xlabel(f"Window index", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (log scale)", fontsize=12)
    ax.set_title("MLP EnKF vs open-loop: L2 per window", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(window_idx)
    ax_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
 
    ax2 = axes[1]
    ax2.plot(window_idx, spread_mean, marker="^", markersize=4, linewidth=1.8,
             label="RMS ensemble σ", color="#4CAF50")
    ax2.plot(window_idx, l2_enkf,     marker="s", markersize=4, linewidth=1.8,
             linestyle="--", label="EnKF RMSE", color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Window index", fontsize=12)
    ax2.set_ylabel("Log scale", fontsize=12)
    ax2.set_title("Calibration: ensemble spread vs RMSE", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
 
    ax2_time = ax2.twiny()
    ax2_time.set_xlim(ax2.get_xlim())
    ax2_time.set_xticks(window_idx)
    ax2_time.set_xticklabels(
        [f"{k * dt_window:.3g}" for k in window_idx],
        fontsize=8, rotation=45, ha="left",
    )
    ax2_time.set_xlabel("Simulation time  (window × dt)", fontsize=10)
 
    fig.suptitle(
        f"MLP EnKF batch evaluation  (B={B}, N_ens={N_ens}, "
        f"obs every {obs_every_n}th var, σ_obs={sigma_obs})",
        fontsize=13,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "batch_l2_per_window_mlp_enkf.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info(f"MLP EnKF batch L2 plot saved to {save_path}")
 
    # ── ERF ───────────────────────────────────────────────────────────────
    erf_save_path = os.path.join(save_dir, "batch_erf_mlp_enkf.pdf")
    _plot_erf(
        obs_times  = obs_times_batch,
        erf_mean   = erf_mean,
        erf_std    = erf_std,
        n_traj     = B,
        title      = (
            f"MLP EnKF Error Reduction Factor per observation time\n"
            f"(B={B} trajectories, N_ens={N_ens}, "
            f"obs every {obs_every_n}th var, σ_obs={sigma_obs}, dt_obs={dt_obs:.3g})"
        ),
        save_path  = erf_save_path,
    )
    logging.info(f"MLP EnKF ERF plot saved to {erf_save_path}")
