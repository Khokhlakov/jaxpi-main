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

    # Fetch original dataset (kept for evaluation reference)
    x_train_batch, u0_train_batch_orig, t_star = get_dataset()
    
    # We'll use the first IC/trajectory in the batch as our evaluation reference during training
    u_ref_eval = u0_train_batch_orig[0, :]
    x_ref_eval = x_train_batch[0, :, :] 

    # -------------------------------------------------------------------------
    # NEW: 1. Generate an initial set of ICs from a Gaussian distribution
    # -------------------------------------------------------------------------
    key = jax.random.PRNGKey(config.training.get("seed", 42))
    num_initial_ics = config.training.get("num_initial_ics", 1000)
    num_vars = 40 # Standard for Lorenz 96
    
    # Adjust mean and std_dev to match the typical L96 scale (e.g., mean=8, std=1)
    mean, std_dev = 8.0, 1.0 
    u0_train_batch = mean + std_dev * jax.random.normal(key, shape=(num_initial_ics, num_vars))

    logging.info(f"Initializing L96 DeepONet with {u0_train_batch.shape[0]} Gaussian trajectories...")

    model = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)

    # Samplers: We still sample t and u independently to enforce the Physics-Informed part
    dom_t = jnp.array([[t_star[0], t_star[-1]]])
    
    # Dynamically use the min/max of our actual 40-dim training data ICs to define the sampling domain
    def get_dom_u(u0_batch):
        return jnp.stack([
            jnp.min(u0_batch, axis=0),
            jnp.max(u0_batch, axis=0)
        ], axis=-1)

    dom_u = get_dom_u(u0_train_batch)
    batch_size = config.training.batch_size_per_device
    
    sampler_t = UniformSampler(dom_t, batch_size)
    sampler_u = UniformSampler(dom_u, batch_size)
    res_sampler = zip(sampler_u, sampler_t)

    logger = Logger()
    print("Waiting for JIT compilation...")
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # NEW: 2. Setup variables for the curriculum/dataset expansion regime
    # -------------------------------------------------------------------------
    update_interval = 10000
    max_additions = config.training.get("max_additions", 5) # Set your desired limit here
    additions_done = 0
    t_end = t_star[-1] # The time 't' we project forward to
    
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # -------------------------------------------------------------------------
        # NEW: 3. Periodically expand the IC dataset using network predictions
        # -------------------------------------------------------------------------
        if step > 0 and step % update_interval == 0 and additions_done < max_additions:
            # 1. Properly extract the UN-REPLICATED state and params
            # We use jax.device_get to bring it to CPU and tree_map to grab the first device's copy
            single_state = jax.device_get(tree_map(lambda x: x[0], model.state))
            
            # 2. Ensure t_end has a 'feature' dimension (Trunk expects a vector)
            t_end_vec = jnp.array([t_end]) 
            
            # 3. Define the prediction function using the model's logic
            # Note: We pass single_state.params, NOT model.state.params
            predict_batch_fn = jax.vmap(model.x_net, in_axes=(None, 0, None))
            
            # 4. Generate the new initial conditions
            # We use single_state.params here to avoid the shape error
            new_u0_predictions = predict_batch_fn(single_state.params, u0_train_batch, t_end_vec)
            
            # 5. Update the dataset
            u0_train_batch = jnp.concatenate([u0_train_batch, new_u0_predictions], axis=0)
            
            # Update the domain bounds and re-instantiate the samplers
            dom_u = get_dom_u(u0_train_batch)
            sampler_u = UniformSampler(dom_u, batch_size)
            
            # Re-bundle the sampler (sampler_t remains the same)
            res_sampler = zip(sampler_u, sampler_t)
            
            additions_done += 1
            logging.info(f"Step {step}: Success! New IC dataset size: {u0_train_batch.shape[0]}")

        # Logging and Evaluation
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

        # Checkpointing
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "udon_model")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model