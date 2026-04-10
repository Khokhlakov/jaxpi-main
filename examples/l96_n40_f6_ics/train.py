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

 
# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_and_evaluate(config, workdir: str):
 
    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)
 
    # ── Reference data (used only for eval logging during training) ────────
    x_train_batch, u0_ref_orig, t_star = get_dataset()
    u_ref_eval = u0_ref_orig[0, :]      # shape (N,)
    x_ref_eval = x_train_batch[0, :, :] # shape (num_t, N)
 
    # ── Hyper-parameters ───────────────────────────────────────────────────
    num_vars        = 40
    num_initial_ics = config.training.get("num_initial_ics", 10000)
    max_additions   = config.training.get("max_additions",   5)
    update_interval = config.training.get("update_interval", 10000)
    batch_size      = config.training.batch_size_per_device
    seed            = config.training.get("seed", 42)
 
    # ── Sample the fixed original IC set from N(2, 1) ──────────────────────
    key = jax.random.PRNGKey(seed)
    mean_ic, std_ic = 6.0, 2.0
    u0_original = mean_ic + std_ic * jax.random.normal(
        key, shape=(num_initial_ics, num_vars)
    )  # shape (num_initial_ics, N) — kept fixed for all rollout augmentations
 
    logging.info(
        f"Sampled {num_initial_ics} original ICs from N({mean_ic}, {std_ic}²)."
    )
 
    # ── Pre-allocate the full IC pool ──────────────────────────────────────
    # Total size is known: num_initial_ics rows per addition, plus the original.
    total_pool_size = num_initial_ics * (max_additions + 1)
    u0_pool = jnp.zeros((total_pool_size, num_vars))
    # Write original ICs into slot 0
    u0_pool = u0_pool.at[:num_initial_ics].set(u0_original)
    active_size   = num_initial_ics          # number of valid rows right now
    additions_done = 0
 
    logging.info(
        f"Pre-allocated IC pool: {total_pool_size} rows "
        f"({max_additions + 1} slots × {num_initial_ics} ICs). "
        f"Active: {active_size}."
    )
 
    # ── Build model ────────────────────────────────────────────────────────
    model     = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)
 
    if config.saving.get("restore_checkpoint", False):
        ckpt_path   = os.path.join(os.getcwd(), config.saving.restore_checkpoint_path)
        model.state = restore_checkpoint(model.state, ckpt_path)
        model.state = replicate(model.state)
        logging.info(f"Restored and re-replicated checkpoint from: {ckpt_path}")
 
    # ── Samplers ───────────────────────────────────────────────────────────
    # t is sampled uniformly over the training window [t0, t1].
    dom_t     = jnp.array([[t_star[0], t_star[-1]]])
    sampler_t = UniformSampler(dom_t, batch_size)
 
    # IC sampler: samples rows uniformly at random from the active pool slice.
    # Rebuilt whenever active_size grows.
    sampler_u = SpaceSampler(u0_pool[:active_size], batch_size)
    res_sampler = zip(sampler_u, sampler_t)
 
    # ── JIT-compiled batch prediction for pool expansion ───────────────────
    # vmap over ICs (axis 1), fix params and t.
    predict_batch = jax.jit(jax.vmap(model.x_net, in_axes=(None, 0, None)))
    # Trunk expects a 1-D time vector: shape (1,)
    t_end_vec = jnp.array([t_star[-1]])
 
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
 
        # ── IC pool expansion ──────────────────────────────────────────────
        # Trigger: every update_interval steps, as long as budget remains.
        if (step > 0
                and step % update_interval == 0
                and additions_done < max_additions):
 
            rollout_steps = additions_done + 1   # 1, 2, 3, … max_additions
 
            # Extract a single-device, CPU-side copy of the parameters.
            single_params = jax.device_get(
                tree_map(lambda x: x[0], model.state)
            ).params
 
            # Roll out `rollout_steps` windows starting from u0_original.
            u_rolled = u0_original               # shape (num_initial_ics, N)
            for _ in range(rollout_steps):
                # predict_batch: (params, (num_ics, N), (1,)) → (num_ics, 1, N)
                u_rolled = predict_batch(single_params, u_rolled, t_end_vec)
                u_rolled = u_rolled.reshape(num_initial_ics, num_vars)
 
            # Write the new ICs into the next free slot of the pre-allocated pool.
            slot_start = num_initial_ics * (additions_done + 1)
            slot_end   = slot_start + num_initial_ics
            u0_pool    = u0_pool.at[slot_start:slot_end].set(u_rolled)
            active_size = slot_end
 
            # Rebuild sampler over the enlarged active region.
            sampler_u = SpaceSampler(u0_pool[:active_size], batch_size)
            res_sampler = zip(sampler_u, sampler_t)
            additions_done += 1
 
            logging.info(
                f"Step {step:>7d} | Pool expansion #{additions_done}: "
                f"rollout ×{rollout_steps} window(s) → "
                f"active ICs {active_size}/{total_pool_size}"
            )
 
        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state    = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
 
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)
 
                # Track pool coverage for visibility
                log_dict["pool/active_ics"]  = active_size
                log_dict["pool/additions"]   = additions_done
 
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