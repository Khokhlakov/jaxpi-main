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
        u_pool = jnp.array(f_train['u'][:])
        
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

    # ── Hyper-parameters ───────────────────────────────────────────────────
    num_vars   = 40
    batch_size = config.training.batch_size_per_device
    seed       = config.training.get("seed", 42)
    pool_size  = u_pool.shape[0]

    logging.info(f"Static Training Pool Ready: {pool_size} samples.")

    # ── Samplers ───────────────────────────────────────────────────────────
    # t is sampled uniformly over the training window [t0, t1].
    dom_t     = jnp.array([[t_star[0], t_star[-1]]])
    sampler_t = UniformSampler(dom_t, batch_size)

    # IC sampler: samples rows uniformly at random from the full pre-computed pool.
    sampler_u   = SpaceSampler(u_pool, batch_size)
    res_sampler = zip(sampler_u, sampler_t)

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

        # ── Forward + gradient step ────────────────────────────────────────
        batch       = next(res_sampler)          # (batch_u, batch_t), each (devices, B, ·)
        model.state = model.step(model.state, batch)

        # ── Adaptive loss weighting (optional) ────────────────────────────
        if config.weighting.scheme in ("grad_norm", "ntk"):
            if step % config.weighting.update_every_steps == 0:
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
                logging.info({k: getattr(v, "shape", type(v)) for k, v in log_dict.items()})

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
