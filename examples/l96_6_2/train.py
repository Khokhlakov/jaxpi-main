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
# Curriculum data-unlocking schedule
# ---------------------------------------------------------------------------
def _curriculum_pool_size(step: int, total_size: int, num_initial: int,
                           update_interval: int, max_additions: int) -> int:
    """
    Progressive dataset-unlocking schedule.

    Only `num_initial` samples are visible to the sampler at step 0. Every
    `update_interval` steps another chunk of `num_initial` samples is
    unlocked, up to `max_additions` times (so at most `max_additions + 1`
    chunks are ever exposed in total). The result is clamped to
    `total_size` so it never asks for more samples than actually exist in
    the pool, even if the config values would otherwise overshoot it.
    """
    num_unlocks = min(step // update_interval, max_additions)
    return int(min(total_size, num_initial * (1 + num_unlocks)))


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
        # Shape: (num_ics * M, 40) -> [40 state variables]
        u_pool_np = np.array(f_train['u'][:])
        
    with h5py.File(test_file, 'r') as f_test:
        # Dense test trajectories. Shape: (num_ics, num_test_pts, 40)
        x_ref_eval_all = jnp.array(f_test['u'][:])
        # Per-trajectory F values. Shape: (num_ics,)
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
    u_ref_eval = u_init_eval

    logging.info("L2 dataset imported:")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Samplers ───────────────────────────────────────────────────────────
    # 1. Define bounds and device counts
    num_devices = jax.local_device_count()
    batch_size_per_device = config.training.batch_size_per_device
    pool_size = u_pool_np.shape[0]

    # Curriculum schedule config (defaults match the requested scheme)
    num_initial_ics = config.training.get("num_initial_ics", 1000)
    update_interval = config.training.get("update_interval", 250)
    max_additions   = config.training.get("max_additions", 17)

    # t is sampled uniformly over the training window [t0, t1].
    dom_t_np = np.array([[t_star[0], t_star[-1]]])

    # 2. Securely push exact copies of the dataset to every local device.
    #    The FULL pool always lives on-device; the curriculum below only
    #    restricts the index range the sampler is allowed to draw from, so
    #    unlocking a new chunk never requires re-transferring data.
    u_pool_repl = jax.device_put_replicated(u_pool_np, jax.local_devices())
    dom_t_repl  = jax.device_put_replicated(dom_t_np, jax.local_devices())

    # 3. Define the PMAP On-Device Sampler
    @jax.pmap
    def sample_on_device(key, local_u_pool, local_dom_t, active_size):
        """Samples initial conditions and time points locally on the accelerator."""
        key_u, key_t = jax.random.split(key)
        
        # Sample u (SpaceSampler equivalent) — restricted to the unlocked range
        idx_u   = jax.random.randint(key_u, (batch_size_per_device,), 0, active_size)
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

    # 5. Curriculum state: how many ICs are currently "unlocked". This is a
    #    tiny per-device scalar, replicated once and only re-sent to device
    #    when it actually changes (every `update_interval` steps) — passing
    #    it as a traced argument (rather than a Python constant baked into
    #    the pmap trace) means changing its value never triggers a
    #    recompilation of `sample_on_device`.
    current_active_size = _curriculum_pool_size(
        0, pool_size, num_initial_ics, update_interval, max_additions
    )
    active_size_repl = jax.device_put_replicated(
        jnp.int32(current_active_size), jax.local_devices()
    )
    logging.info(
        f"[curriculum] starting with {current_active_size}/{pool_size} ICs unlocked"
    )

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

        # ── Curriculum: unlock another chunk every update_interval steps ───
        target_active_size = _curriculum_pool_size(
            step, pool_size, num_initial_ics, update_interval, max_additions
        )
        if target_active_size != current_active_size:
            current_active_size = target_active_size
            active_size_repl = jax.device_put_replicated(
                jnp.int32(current_active_size), jax.local_devices()
            )
            logging.info(
                f"[curriculum] step {step}: unlocked pool -> "
                f"{current_active_size}/{pool_size} ICs"
            )

        # ── Roll keys and Sample on-device ─────────────────────────────────
        # vmap returns a single array of shape (num_devices, 2, 2)
        split_keys = jax.vmap(jax.random.split)(keys)
        
        # Slice along axis 1 to separate the new keys and the current step's keys
        keys      = split_keys[:, 0]  # Shape: (num_devices, 2) - save for next step
        step_keys = split_keys[:, 1]  # Shape: (num_devices, 2) - use for this step
        
        # Generates a batch of shape (num_devices, batch_size_per_device, ...)
        batch = sample_on_device(step_keys, u_pool_repl, dom_t_repl, active_size_repl)

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

                # Track pool parameters (reflects the curriculum schedule)
                log_dict["pool/active_ics"] = current_active_size

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
 
    # ── Reference data & Time Grid ─────────────────────────────────────────
    # The dataset provides 51 steps of 0.005 from 0 to 0.25 units of time
    t_star = jnp.linspace(0.0, 0.25, 51)
    
    # Import full explicit dataset
    data_dir = config.training.get("data_dir", "data/")
    train_data_np = dd_get_train_data(data_dir) # Shape (62000, 51, 40)
    train_data = jnp.array(train_data_np)
    
    logging.info(f"Loaded Supervised Data: {train_data.shape}")
    
    # Parameters for validation
    trajs_per_window = 100
    
    # U_ref represents the initial condition (t=0) for each trajectory 
    u_ref_eval = train_data[0:trajs_per_window, 0, :]               
    x_ref_eval = train_data[0:trajs_per_window, :, :] 
  
    # ── Model & Evaluator ──────────────────────────────────────────────────
    model     = models.L96UDON_DD(config, t_star)
    evaluator = models.L96UDONEvaluator_DD(config, model)
 
    if config.saving.get("restore_checkpoint", False):
        ckpt_path   = os.path.join(os.getcwd(), config.saving.restore_checkpoint_path)
        model.state = restore_checkpoint(model.state, ckpt_path)
        model.state = replicate(model.state)
        logging.info(f"Restored and re-replicated checkpoint from: {ckpt_path}")

    # ── Data Sampler ───────────────────────────────────────────────────────
    num_devices           = jax.local_device_count()
    batch_size_per_device = config.training.batch_size_per_device
    global_batch_size     = num_devices * batch_size_per_device

    num_trajs  = train_data.shape[0]
    num_t      = train_data.shape[1]
    
    key = jax.random.PRNGKey(config.training.get("seed", 42))

    @jax.jit
    def get_batch(key):
        """Samples random pairs and reshapes for pmap (num_devices, batch_size, ...)."""
        key_traj, key_t = jax.random.split(key)
        
        # 1. Sample globally (for all devices at once)
        idx_traj = jax.random.randint(key_traj, (global_batch_size,), 0, num_trajs)
        idx_t    = jax.random.randint(key_t, (global_batch_size,), 0, num_t)
        
        batch_u = train_data[idx_traj, 0, :]      
        batch_t = t_star[idx_t]                   
        batch_x = train_data[idx_traj, idx_t, :]  
        
        # 2. Reshape to explicitly add the device dimension
        # Resulting shapes: (3, 100, 40), (3, 100, 1), (3, 100, 40)
        batch_u = batch_u.reshape(num_devices, batch_size_per_device, -1)
        batch_t = batch_t.reshape(num_devices, batch_size_per_device, 1)
        batch_x = batch_x.reshape(num_devices, batch_size_per_device, -1)
        
        return (batch_u, batch_t, batch_x)

    # ── Training loop ──────────────────────────────────────────────────────
    logger     = Logger()
    start_time = time.time()
    logging.info("Waiting for JIT compilation…")
 
    for step in range(config.training.max_steps):
 
        # ── Data Sampling + Gradient Step ──────────────────────────────────
        key, subkey = jax.random.split(key)
        batch       = get_batch(subkey)
        model.state = model.step(model.state, batch)
 
        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
 
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)
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
