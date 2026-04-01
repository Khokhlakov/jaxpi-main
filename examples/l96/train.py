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


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # 1. Fetch original data ONLY for the time horizon and evaluation reference
    x_ref_data, u0_ref_data, t_star = get_dataset()
    t_end = t_star[-1]
    
    # Use the first IC/trajectory from the REAL dataset as evaluation reference
    u_ref_eval = u0_ref_data[0, :]
    x_ref_eval = x_ref_data[0, :, :] 

    # 2. NEW REGIME: Generate an initial set of training ICs following a Gaussian distribution
    num_initial_ics = getattr(config.training, 'num_initial_ics', 1000)
    l96_dim = 40 
    
    # Generating N(0, 1).
    u0_train_batch = jnp.random.normal(loc=0.0, scale=1.0, size=(num_initial_ics, l96_dim))
    
    # Track the "latest" wave of ICs to roll them forward to t, 2t, 3t, etc.
    latest_u_batch = u0_train_batch.copy()

    logging.info(f"Initializing L96 DeepONet with {u0_train_batch.shape[0]} Gaussian trajectories...")

    model = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)

    dom_t = jnp.array([[t_star[0], t_star[-1]]])

    # 3. Helper function to update samplers dynamically when new data is added
    def update_samplers(current_u_data):
        dom_u_new = jnp.stack([
            jnp.min(current_u_data, axis=0),
            jnp.max(current_u_data, axis=0)
        ], axis=-1)
        batch_size = config.training.batch_size_per_device
        sampler_t_new = UniformSampler(dom_t, batch_size)
        sampler_u_new = UniformSampler(dom_u_new, batch_size)
        return zip(sampler_u_new, sampler_t_new)

    res_sampler = update_samplers(u0_train_batch)

    logger = Logger()
    print("Waiting for JIT compilation...")
    start_time = time.time()
    
    # Training regime parameters
    addition_interval = 10000
    max_additions = getattr(config.training, 'max_data_additions', 5) # Set how many times to add
    additions_made = 0
    
    for step in range(config.training.max_steps):
        
        # --- NEW: AUTO-REGRESSIVE DATA ADDITION ---
        if step > 0 and step % addition_interval == 0 and additions_made < max_additions:
            logging.info(f"Step {step}: Predicting and adding initial conditions for time {(additions_made + 1)}*t...")
            
            # Get unwrapped state for inference (handling multi-device replication if any)
            state_dev = jax.device_get(tree_map(lambda x: x[0] if x.ndim > 1 else x, model.state))
            
            # Create a trunk input array for the time horizon t_end
            t_input = jnp.full((latest_u_batch.shape[0], 1), t_end)
            
            # Predict the state at time 't' given the 'latest' batch of initial conditions
            # NOTE: You may need to adapt `model.apply` based on how your specific Flax/JAX 
            # model expects inference calls. (e.g., separating branch and trunk inputs)
            new_predictions = model.apply({'params': state_dev.params}, latest_u_batch, t_input)
            new_predictions = np.array(jax.device_get(new_predictions))
            
            # Append the new predictions to the full training pool
            u0_train_batch = np.vstack([u0_train_batch, new_predictions])
            
            # The new predictions become the "latest" batch so that at the next 
            # 10,000 steps, we evaluate Net(new_predictions, t) to effectively get 2t, 3t, etc.
            latest_u_batch = new_predictions
            
            # Rebuild the samplers with the newly expanded data bounds
            res_sampler = update_samplers(u0_train_batch)
            additions_made += 1
            
            logging.info(f"Dataset updated! Total ICs: {u0_train_batch.shape[0]}. Rebuilt samplers.")
        # ------------------------------------------

        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
                
                # Evaluator still uses the true physics dataset reference we grabbed at the top
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