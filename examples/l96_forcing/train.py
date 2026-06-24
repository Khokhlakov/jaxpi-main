import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import UniformSampler, SpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from flax.jax_utils import replicate

import models
from utils import get_dataset, dd_get_train_data
import h5py


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_and_evaluate(config, workdir: str):

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load Dataset from HDF5 ─────────────────────────────────────────────
    train_file = "data/l96_forcing_train.h5"
    test_file  = "data/l96_forcing_test.h5"

    with h5py.File(train_file, 'r') as f_train:
        # u_pool contains states pooled across all trajectories and windows. 
        # Shape: (num_ics * M, 41) -> [40 state variables + 1 F value]
        u_pool_np = np.array(f_train['u'][:])
        
    with h5py.File(test_file, 'r') as f_test:
        # Dense test trajectories. Shape: (num_ics, num_test_pts, 40)
        x_ref_eval_all = jnp.array(f_test['u'][:])
        # Per-trajectory F values. Shape: (num_ics,)
        F_test_all = jnp.array(f_test['F'][:])
        # Times relative to M·window_size. Shape: (num_test_pts,)
        t_star_all = jnp.array(f_test['t'][:])

    # ── Reference data (used only for eval logging during training) ────────
    trajs_per_window = 100
    time_steps = 50

    # L2 evaluation dataset setup (First window only)
    t_star = t_star_all[0:time_steps]
    
    # Extract the first 100 trajectories for the first time_steps (40 variables)
    x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]  # Shape: (100, 50, 40)
    
    # Construct the branch network input [u(t0), F] for evaluation
    u_init_eval = x_ref_eval_all[0:trajs_per_window, 0, :]            # Shape: (100, 40)
    F_eval = F_test_all[0:trajs_per_window].reshape(-1, 1)            # Shape: (100, 1)
    u_ref_eval = jnp.concatenate([u_init_eval, F_eval], axis=1)       # Shape: (100, 41)

    logging.info("L2 dataset imported:")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Samplers ───────────────────────────────────────────────────────────
    # 1. Define bounds and device counts
    num_devices = jax.local_device_count()
    batch_size_per_device = config.training.batch_size_per_device
    pool_size = u_pool_np.shape[0]
    
    # t is sampled uniformly over the training window [t0, t1].
    dom_t_np = np.array([[t_star[0], t_star[-1]]])

    # 2. Securely push exact copies of the dataset to every local device
    u_pool_repl = jax.device_put_replicated(u_pool_np, jax.local_devices())
    dom_t_repl  = jax.device_put_replicated(dom_t_np, jax.local_devices())

    # 3. Define the PMAP On-Device Sampler
    @jax.pmap
    def sample_on_device(key, local_u_pool, local_dom_t):
        """Samples initial conditions and time points locally on the accelerator."""
        key_u, key_t = jax.random.split(key)
        
        # Sample u (SpaceSampler equivalent)
        idx_u   = jax.random.randint(key_u, (batch_size_per_device,), 0, pool_size)
        batch_u = local_u_pool[idx_u, :]
        
        # Sample t (UniformSampler equivalent)
        t_min   = local_dom_t[0, 0]
        t_max   = local_dom_t[0, 1]
        batch_t = jax.random.uniform(key_t, (batch_size_per_device, 1), minval=t_min, maxval=t_max)
        
        return batch_u, batch_t

    # 4. Initialize reproducible PRNG Keys for all devices
    seed = config.training.get("seed", 42)
    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, num_devices)

    # ── Build model ────────────────────────────────────────────────────────
    model     = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)

    if config.saving.get("restore_checkpoint", False):
        ckpt_path   = os.path.join(os.getcwd(), config.saving.restore_checkpoint_path)
        model.state = restore_checkpoint(model.state, ckpt_path)
        model.state = replicate(model.state)
        logging.info(f"Restored and re-replicated checkpoint from: {ckpt_path}")

    # ── Training loop ──────────────────────────────────────────────────────
    logger     = Logger()
    start_time = time.time()
    logging.info("Waiting for JIT compilation…")

    for step in range(config.training.max_steps):

        # ── Roll keys and Sample on-device ─────────────────────────────────
        # vmap returns a single array of shape (num_devices, 2, 2)
        split_keys = jax.vmap(jax.random.split)(keys)
        
        # Slice along axis 1 to separate the new keys and the current step's keys
        keys      = split_keys[:, 0]  # Shape: (num_devices, 2) - save for next step
        step_keys = split_keys[:, 1]  # Shape: (num_devices, 2) - use for this step
        
        # Generates a batch of shape (num_devices, batch_size_per_device, ...)
        batch = sample_on_device(step_keys, u_pool_repl, dom_t_repl)

        # ── Forward + gradient step ────────────────────────────────────────
        model.state = model.step(model.state, batch)

        # ── Adaptive loss weighting (optional) ────────────────────────────
        if config.weighting.scheme in ("grad_norm", "ntk"):
            if step % config.weighting.update_every_steps == 0 and step >= 500:#config.weighting.warmup_steps::
                model.state = model.update_weights(model.state, batch)

        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))

                # Evaluator expects state, train batch, branch eval input, and ground truth trajectory
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)

                # Track pool parameters (Fixed values now, since augmentation is dropped)
                log_dict["pool/active_ics"] = pool_size

                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
        
        # ── Checkpointing ──────────────────────────────────────────────────
        if config.saving.save_every_steps is not None:
            if ((step + 1) % config.saving.save_every_steps == 0
                    or (step + 1) == config.training.max_steps):
                ckpt_path = os.path.join(
                    os.getcwd(), config.wandb.name, "ckpt", "udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model

# Data driven
def train_and_evaluate_dd(config, workdir: str):
    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load Dataset from HDF5 ───────────────────────────────────────────────
    # Pooled dense windows produced by gen_data.py:
    #   u : (num_ics * M, pts_pw + 1, N + 1)
    #       - last column of the trailing axis is F, broadcast across every
    #         timestep of a window
    #       - remaining N columns are the dense Lorenz-96 trajectory for
    #         that window (state at t = 0, ws, 2·dt, ... up to window_size)
    train_file = "data/l96_forcing_train_dd.h5"

    with h5py.File(train_file, 'r') as f_train:
        train_data_np = np.array(f_train['u'][:])              # (num_windows, num_t, N+1)
        window_size   = float(f_train.attrs.get('window_size', 0.25))

    train_data = jnp.array(train_data_np)

    num_windows, num_t, state_dim = train_data.shape
    N = state_dim - 1   # number of physical state variables (F occupies the last column)

    # ── Reference data & Time Grid ──────────────────────────────────────────
    # Window-local time grid: 0 -> window_size over num_t points
    dt     = window_size / (num_t - 1)
    t_star = jnp.arange(num_t, dtype=jnp.float32) * dt

    logging.info(f"Loaded Supervised Data: {train_data.shape}")

    # Parameters for validation
    trajs_per_window = min(100, num_windows)

    # Branch input: [u(t0), F] for each evaluation window
    u_ref_eval = train_data[0:trajs_per_window, 0, :]                # (100, N+1)
    # Trunk target: dense state-only trajectory across the window (F dropped)
    x_ref_eval = train_data[0:trajs_per_window, :, :N]               # (100, num_t, N)

    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Model & Evaluator ──────────────────────────────────────────────────
    model     = models.L96UDON_DD(config, t_star)
    evaluator = models.L96UDONEvaluator_DD(config, model)

    if config.saving.get("restore_checkpoint", False):
        ckpt_path   = os.path.join(os.getcwd(), config.saving.restore_checkpoint_path)
        model.state = restore_checkpoint(model.state, ckpt_path)
        model.state = replicate(model.state)
        logging.info(f"Restored and re-replicated checkpoint from: {ckpt_path}")

    # ── On-device Sampler (fully parallel, no host round-trips per step) ───
    num_devices           = jax.local_device_count()
    batch_size_per_device = config.training.batch_size_per_device
    seed                  = config.training.get("seed", 42)

    init_key    = jax.random.PRNGKey(seed)
    device_keys = jax.pmap(lambda i: jax.random.fold_in(init_key, i))(jnp.arange(num_devices))

    # Push exact copies of the window pool + time grid onto every local device
    train_data_repl = jax.device_put_replicated(train_data, jax.devices())
    t_star_repl     = jax.device_put_replicated(t_star, jax.devices())

    @jax.pmap
    def get_batch_on_device(device_key, data, t_array):
        """Splits keys and samples (window, timestep) pairs entirely on-device."""
        new_key, sample_key = jax.random.split(device_key)
        key_win, key_t       = jax.random.split(sample_key)

        idx_win = jax.random.randint(key_win, (batch_size_per_device,), 0, num_windows)
        idx_t   = jax.random.randint(key_t,   (batch_size_per_device,), 0, num_t)

        # Branch input: [u(t0), F] for the sampled window
        batch_u = data[idx_win, 0, :]

        # Trunk query time
        batch_t = t_array[idx_t].reshape(batch_size_per_device, 1)

        # Trunk target: state only at the sampled time (drop trailing F column)
        batch_x = data[idx_win, idx_t, :N]

        return new_key, (batch_u, batch_t, batch_x)

    # ── Training loop ──────────────────────────────────────────────────────
    logger     = Logger()
    start_time = time.time()
    logging.info("Waiting for JIT compilation…")
 
    for step in range(config.training.max_steps):
 
        # ── Data Sampling + Gradient Step ──────────────────────────────────
        device_keys, batch = get_batch_on_device(device_keys, train_data_repl, t_star_repl)
        model.state         = model.step(model.state, batch)
 
        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
 
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)
                log_dict["pool/active_windows"] = num_windows
                wandb.log(log_dict, step)
 
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
 
        # ── Checkpointing ──────────────────────────────────────────────────
        if config.saving.save_every_steps is not None:
            if ((step + 1) % config.saving.save_every_steps == 0
                    or (step + 1) == config.training.max_steps):
                ckpt_path = os.path.join(
                    os.getcwd(), config.wandb.name, "ckpt", "udon_dd_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )
 
    return model