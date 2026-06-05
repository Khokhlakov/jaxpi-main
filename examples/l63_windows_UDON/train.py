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

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # NEW: Unpack the 40 variables from the batch dataset
    x_train_batch, u0_train_batch, t_star = get_dataset()
    
    # We'll use the first IC/trajectory in the batch as our evaluation reference during training
    u_ref_eval = u0_train_batch[0, :]
    x_ref_eval = x_train_batch[0, :, :] 

    logging.info(f"Initializing L96 DeepONet with {u0_train_batch.shape[0]} trajectories...")

    model = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)

    # Samplers: We still sample t and u independently to enforce the Physics-Informed part
    dom_t = jnp.array([[t_star[0], t_star[-1]]])
    
    # Dynamically use the min/max of our actual 40-dim training data ICs to define the sampling domain
    dom_u = jnp.stack([
        jnp.min(u0_train_batch, axis=0),
        jnp.max(u0_train_batch, axis=0)
    ], axis=-1)

    batch_size = config.training.batch_size_per_device
    sampler_t = UniformSampler(dom_t, batch_size)
    sampler_u = UniformSampler(dom_u, batch_size)
    res_sampler = zip(sampler_u, sampler_t)

    logger = Logger()
    print("Waiting for JIT compilation...")
    start_time = time.time()
    
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
                
                # Pass the reference IC and trajectory to the evaluator
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)
                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model