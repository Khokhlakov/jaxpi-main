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
        u_pool_np = np.array(f_train['u'][:])
        F_train = np.array(f_train['F'][:])  # Load F
        
        # Expand F to match pooled states (501 states per IC)
        states_per_ic = u_pool_np.shape[0] // len(F_train)
        F_expanded = np.repeat(F_train, states_per_ic)[:, None]
        
        # Concatenate to form a 41-D input
        u_pool_np = np.concatenate([u_pool_np, F_expanded], axis=-1)
        
    with h5py.File(test_file, 'r') as f_test:
        x_ref_eval_all = jnp.array(f_test['u'][:])
        t_star_all = jnp.array(f_test['t'][:])
        F_test = jnp.array(f_test['F'][:])   # Load test F

    # ── Reference data ─────────────────────────────────────────────────────
    trajs_per_window = 100
    time_steps = 51

    t_star = t_star_all[0:time_steps]
    x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]
    
    # Construct the branch network input [u(t0), F] for evaluation
    u_init_eval = x_ref_eval_all[0:trajs_per_window, 0, :] 
    F_eval = F_test[0:trajs_per_window, None]
    
    # Concatenate F for the evaluation branch input
    u_ref_eval = jnp.concatenate([u_init_eval, F_eval], axis=-1)

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
            if step % config.weighting.update_every_steps == 0:#config.weighting.warmup_steps::
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
    train_file = "data/l96_forcing_train_dd.h5"

    with h5py.File(train_file, 'r') as f_train:
        train_data_np = np.array(f_train['u'][:])  
        window_size   = float(f_train.attrs.get('window_size', 0.25))
        F_train       = np.array(f_train['F'][:])

    # Expand F to match the windows and timesteps (500 windows per IC)
    windows_per_ic = train_data_np.shape[0] // len(F_train)
    F_win = np.repeat(F_train, windows_per_ic)[:, None, None]
    
    # Broadcast F across all 51 timesteps per window
    F_broadcast = np.broadcast_to(F_win, (train_data_np.shape[0], train_data_np.shape[1], 1))
    
    # Append F to form a 41-D state 
    train_data = jnp.array(np.concatenate([train_data_np, F_broadcast], axis=-1))

    num_windows, num_t, state_dim = train_data.shape
    N = state_dim

    # ── Reference data & Time Grid ──────────────────────────────────────────
    # Window-local time grid: 0 -> window_size over num_t points
    dt     = window_size / (num_t - 1)
    t_star = jnp.arange(num_t, dtype=jnp.float32) * dt

    # ── Load Test Dataset from HDF5 ─────────────────────────────────────────────
    test_file  = "data/l96_forcing_test.h5"

    with h5py.File(test_file, 'r') as f_test:
        x_ref_eval_all = jnp.array(f_test['u'][:])
        F_test = jnp.array(f_test['F'][:])
        
    trajs_per_window = 100
    time_steps = 51 
    num_windows_to_eval = 10 

    u_refs = []
    x_refs = []
    pts_pw = 50 
    
    F_eval = F_test[0:trajs_per_window, None]
    
    for w in range(num_windows_to_eval):
        start_idx = w * pts_pw
        end_idx = start_idx + time_steps
        
        # Append F to the reference branch input
        u_init = x_ref_eval_all[0:trajs_per_window, start_idx, :]
        u_init_F = jnp.concatenate([u_init, F_eval], axis=-1)
        u_refs.append(u_init_F)
        
        x_refs.append(x_ref_eval_all[0:trajs_per_window, start_idx:end_idx, :])
        
    u_ref_eval = jnp.concatenate(u_refs, axis=0) 
    x_ref_eval = jnp.concatenate(x_refs, axis=0)

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

        # Branch input: [u(t0), F] for the sampled window (41 Dimensions)
        batch_u = data[idx_win, 0, :]

        # Trunk query time
        batch_t = t_array[idx_t].reshape(batch_size_per_device, 1)

        # Trunk target: state only at the sampled time (Slice to 40 Dimensions)
        batch_x = data[idx_win, idx_t, :40]

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