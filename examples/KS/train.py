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
from utils import get_dataset
import h5py

# ==========================================
# Physics-Informed Training Pipeline
# ==========================================
def train_and_evaluate(config, workdir: str):

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load Dataset from HDF5 ─────────────────────────────────────────────
    # Assuming the script runs from jaxpi-main/examples/KS/
    train_file = "data/ks_train_data.h5"
    test_file  = "data/ks_test_data.h5"

    with h5py.File(train_file, 'r') as f_train:
        # u_train shape: (num_samples, time_states, 256)
        u_train = jnp.array(f_train['u'][:])
        
        # Flatten the first two dimensions so EVERY recorded state acts 
        # as a valid initial condition (IC) for the physics residual sampler.
        # u_pool Shape: (num_samples * time_states, 256)
        u_pool = u_train.reshape(-1, u_train.shape[-1])
        
    with h5py.File(test_file, 'r') as f_test:
        # Dense test trajectories. Shape: (num_ics, num_test_pts, 256)
        x_ref_eval_all = jnp.array(f_test['u'][:])
        dt = f_test.attrs.get("dt", 0.02)

    # ── Reference data (used only for eval logging during training) ────────
    # Evaluate against the first 50 fine-grained integration steps of the test set
    trajs_per_window = min(100, x_ref_eval_all.shape[0])
    time_steps = 50 

    # t_star represents the physical time for these 50 evaluation steps
    t_star = jnp.arange(time_steps) * dt
    
    # Extract the trajectories and the initial branch inputs [u(t0)]
    x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]  # Shape: (100, 50, 256)
    u_ref_eval = x_ref_eval_all[0:trajs_per_window, 0, :]             # Shape: (100, 256)

    logging.info("L2 dataset imported:")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Hyper-parameters ───────────────────────────────────────────────────
    batch_size = config.training.batch_size_per_device
    seed       = config.training.get("seed", 42)
    pool_size  = u_pool.shape[0]

    logging.info(f"Static Training Pool Ready: {pool_size} samples.")

    # ── Samplers ───────────────────────────────────────────────────────────
    # t is sampled uniformly over the training window [0, 1.0].
    # Adjust dom_t if your DeepONet is learning a different temporal window length
    dom_t     = jnp.array([[0.0, 1.0]]) 
    sampler_t = UniformSampler(dom_t, batch_size)

    # IC sampler: samples rows uniformly at random from the full pre-computed pool.
    sampler_u   = SpaceSampler(u_pool, batch_size)
    res_sampler = zip(sampler_u, sampler_t)

    # ── Build model ────────────────────────────────────────────────────────
    model     = models.KSUDON(config, t_star)
    evaluator = models.KSUDONEvaluator(config, model)

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


# ==========================================
# Data-Driven Training Pipeline
# ==========================================
def train_and_evaluate_dd(config, workdir: str):
    
    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)
 
    # ── Reference data & Time Grid ─────────────────────────────────────────
    train_file = "data/ks_train_data_dd.h5"
    
    with h5py.File(train_file, 'r') as f_train:
        # train_data shape: (num_samples, time_states, 128)
        train_data = jnp.array(f_train['u'][:])
        
        # Extract the integration step to map physical time correctly
        dt = f_train.attrs.get("dt", 0.02)
        
    logging.info(f"Loaded Dense Supervised Data: {train_data.shape}")
    
    num_t  = train_data.shape[1]
    
    # Map the time grid using the dense dataset's dt
    t_star = jnp.arange(num_t, dtype=float) * dt
    
    # Parameters for validation
    trajs_per_window = min(100, train_data.shape[0])
    
    # U_ref represents the initial condition (t=0) for each trajectory 
    u_ref_eval = train_data[0:trajs_per_window, 0, :]               
    x_ref_eval = train_data[0:trajs_per_window, :, :] 
  
    # ── Model & Evaluator ──────────────────────────────────────────────────
    model     = models.KSUDON_DD(config, t_star)
    evaluator = models.KSUDONEvaluator_DD(config, model)
 
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
    
    key = jax.random.PRNGKey(config.training.get("seed", 42))

    # copy the dataset onto every GPU
    train_data_repl = jax.device_put_replicated(train_data, jax.devices())
    t_star_repl     = jax.device_put_replicated(t_star, jax.devices())

    def sample_fn(device_key, data_array, t_array):
        """Samples random pairs natively ON each GPU to avoid PCIe transfers."""
        key_traj, key_t = jax.random.split(device_key)
        
        # Sample locally
        idx_traj = jax.random.randint(key_traj, (batch_size_per_device,), 0, num_trajs)
        idx_t    = jax.random.randint(key_t, (batch_size_per_device,), 0, num_t)
        
        batch_u = data_array[idx_traj, 0, :]  

        # reshape for local device    
        batch_t = t_array[idx_t].reshape(batch_size_per_device, 1)                   
        batch_x = data_array[idx_traj, idx_t, :]  
        
        return (batch_u, batch_t, batch_x)

    # Compile the mapped function.
    # in_axes: (0 -> split across GPUs, None -> broadcast to GPUs)
    get_batch = jax.pmap(sample_fn, in_axes=(0, 0, 0))

    # ── Training loop ──────────────────────────────────────────────────────
    logger     = Logger()
    start_time = time.time()
    logging.info("Waiting for JIT compilation…")
 
    for step in range(config.training.max_steps):
 
        # ── Data Sampling + Gradient Step ──────────────────────────────────
        key, subkey_master = jax.random.split(key)
        subkeys = jax.random.split(subkey_master, num_devices)

        batch       = get_batch(subkeys, train_data_repl, t_star_repl)
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
                    os.getcwd(), config.wandb.name, "ckpt", "udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )
 
    return model