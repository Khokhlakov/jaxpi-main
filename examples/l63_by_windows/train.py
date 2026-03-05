import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

def train_one_window(config, workdir, model, res_sampler, xyz_ref, idx):
    # Calculate step offset so W&B logs appear sequentially across windows
    step_offset = idx * config.training.max_steps
    logger = Logger()
    
    # Initialize evaluator for the current window
    evaluator = models.L63Evaluator(config, model)

    print(f"Waiting for JIT (Window {idx + 1})...")
    start_time = time.time()
    
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                
                # Evaluate against the reference data for this specific window
                log_dict = evaluator(state, batch, xyz_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving checkpoint per window
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", f"time_window_{idx + 1}")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Get dataset
    xyz_full_ref, t_star = get_dataset()
    
    # Initial condition for the very first window
    xyz0 = xyz_full_ref[0, :]

    # Calculate windowing parameters
    num_windows = config.training.num_time_windows
    steps_per_window = len(t_star) // num_windows
    
    for idx in range(num_windows):
        logging.info(f"Training time window {idx + 1}/{num_windows}")

        # Slice the time and reference data for the current window
        start_idx = idx * steps_per_window
        end_idx = (idx + 1) * steps_per_window
        
        # Ensure we include the last point for the time domain
        t_window = t_star[start_idx:end_idx]
        xyz_window_ref = xyz_full_ref[start_idx:end_idx, :]

        # Define domain for the sampler [t_start, t_end]
        t0, t1 = t_window[0], t_window[-1]
        dom = jnp.array([[t0, t1]])

        # Initialize residual sampler for this window's domain
        res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

        # Initialize model with the current initial condition (xyz0)
        model = models.L63(config, xyz0, t_window)

        # Train the current window
        model = train_one_window(
            config, workdir, model, res_sampler, xyz_window_ref, idx
        )

        # Update initial condition for the next window
        if idx < num_windows - 1:
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params
            
            # Predict the state at the beginning of the next window
            # Using the time point t_star[end_idx]
            t_next_start = t_star[end_idx]
            xyz0 = model.sol_fn(params, t_next_start).flatten()
            
            # Clean up to manage memory
            del model, state, params
            
    return None