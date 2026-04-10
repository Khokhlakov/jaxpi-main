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
from jaxpi.utils import save_checkpoint, restore_checkpoint

from flax.jax_utils import replicate

import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)
 
    # Fetch dataset — used only for the evaluation reference trajectory
    x_train_batch, u0_train_batch_orig, t_star = get_dataset()
    u_ref_eval = u0_train_batch_orig[0, :]
    x_ref_eval = x_train_batch[0, :, :]
 
    # -------------------------------------------------------------------------
    # 1. Generate the original ICs and pre-allocate the full pool buffer.
    #
    #    The pool will eventually hold:
    #      slot 0          : u0_original          (rows [0,          N_ic) )
    #      slot 1          : rollout depth-1 ICs  (rows [N_ic,     2*N_ic) )
    #      ...
    #      slot max_add    : rollout depth-k ICs  (rows [k*N_ic, (k+1)*N_ic) )
    #
    #    Total rows = N_ic * (max_additions + 1)  — known at startup.
    #    A plain NumPy array is used so slice-assignment is a zero-copy write.
    # -------------------------------------------------------------------------
    key = jax.random.PRNGKey(config.training.get("seed", 42))
    num_initial_ics = config.training.get("num_initial_ics", 10000)
    max_additions   = config.training.get("max_additions", 5)
    num_vars        = 40  # L96 dimension
 
    mean, std_dev = 2.0, 1.0
    key, ic_key = jax.random.split(key)
    u0_original = np.array(
        mean + std_dev * jax.random.normal(ic_key, shape=(num_initial_ics, num_vars))
    )
 
    total_pool_rows = num_initial_ics * (max_additions + 1)
    u0_pool = np.empty((total_pool_rows, num_vars), dtype=np.float32)
    u0_pool[:num_initial_ics] = u0_original   # write slot 0
    current_pool_size = num_initial_ics        # live rows visible to the sampler
 
    logging.info(
        f"Pre-allocated IC pool: {total_pool_rows} rows x {num_vars} cols "
        f"({u0_pool.nbytes / 1e6:.1f} MB). "
        f"Initialised slot 0 with {num_initial_ics} Gaussian ICs."
    )
 
    model = models.L96UDON(config, t_star)
    evaluator = models.L96UDONEvaluator(config, model)
 
    if config.saving.get("restore_checkpoint", False):
        ckpt_path = os.path.join(os.getcwd(), config.saving.restore_checkpoint_path)
        model.state = restore_checkpoint(model.state, ckpt_path)
        model.state = replicate(model.state)
        logging.info(f"Restored and re-replicated checkpoint from: {ckpt_path}")
 
    # -------------------------------------------------------------------------
    # 2. Samplers.
    #
    #    sampler_t  — uniform over [t0, t1], unchanged.
    #    make_ic_sampler — draws random rows from the *live* prefix of u0_pool.
    #                      Rebuilt whenever current_pool_size changes.
    # -------------------------------------------------------------------------
    dom_t      = jnp.array([[t_star[0], t_star[-1]]])
    batch_size = config.training.batch_size_per_device
    sampler_t  = UniformSampler(dom_t, batch_size)
 
    def make_ic_sampler(pool: np.ndarray, live_rows: int, batch_sz: int, rng):
        """Yields (batch_sz, num_vars) rows sampled uniformly from pool[:live_rows]."""
        while True:
            rng, subkey = jax.random.split(rng)
            idx = np.array(
                jax.random.randint(subkey, shape=(batch_sz,), minval=0, maxval=live_rows)
            )
            yield jnp.array(pool[idx])
 
    key, sampler_key = jax.random.split(key)
    sampler_u   = make_ic_sampler(u0_pool, current_pool_size, batch_size, sampler_key)
    res_sampler = zip(sampler_u, sampler_t)
 
    # -------------------------------------------------------------------------
    # 3. JIT-compiled batch propagator — shared across all rollout additions.
    # -------------------------------------------------------------------------
    t_end_vec = jnp.array([t_star[-1]])
 
    @jax.jit
    def predict_batch(params, u_batch):
        """One time-horizon forward pass for a batch of ICs.
 
        Args:
            params:  unreplicated network parameters.
            u_batch: (B, N) array of initial conditions.
 
        Returns:
            (B, N) predicted states at t_end.
        """
        return jax.vmap(model.x_net, in_axes=(None, 0, None))(
            params, u_batch, t_end_vec
        ).reshape(u_batch.shape[0], num_vars)
 
    # -------------------------------------------------------------------------
    # 4. Training loop.
    # -------------------------------------------------------------------------
    update_interval = config.training.get("update_interval", 10000)
    additions_done  = 0
 
    logger     = Logger()
    print("Waiting for JIT compilation...")
    start_time = time.time()
 
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)
 
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)
 
        # ---------------------------------------------------------------------
        # 5. Periodic dataset expansion.
        #
        #    addition k  ->  roll u0_original forward k horizons, write the
        #                    resulting states into slot k of u0_pool.
        #
        #    current_pool_size is advanced by num_initial_ics after each write
        #    so the sampler immediately starts drawing from the new rows.
        # ---------------------------------------------------------------------
        if (
            step > 0
            and step % update_interval == 0
            and additions_done < max_additions
        ):
            rollout_depth = additions_done + 1  # 1, 2, 3, ...
 
            single_params = jax.device_get(
                tree_map(lambda x: x[0], model.state)
            ).params
 
            # Chain rollout_depth forward passes starting from the frozen originals.
            u_current = jnp.array(u0_original)
            for _ in range(rollout_depth):
                u_current = predict_batch(single_params, u_current)
 
            # Write into the pre-allocated slot (no allocation, no copy of existing rows).
            slot_start = (additions_done + 1) * num_initial_ics
            slot_end   = slot_start + num_initial_ics
            u0_pool[slot_start:slot_end] = np.array(u_current)
 
            # Advance the live-row cursor and rebuild the sampler.
            current_pool_size = slot_end
            key, sampler_key  = jax.random.split(key)
            sampler_u   = make_ic_sampler(u0_pool, current_pool_size, batch_size, sampler_key)
            res_sampler = zip(sampler_u, sampler_t)
 
            additions_done += 1
            logging.info(
                f"Step {step}: addition {additions_done}/{max_additions} — "
                f"depth {rollout_depth} horizon(s). "
                f"Pool rows in use: {current_pool_size}/{total_pool_rows}."
            )
 
        # Logging and evaluation
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))
 
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)
                wandb.log(log_dict, step)
 
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
 
        # Checkpointing
        if config.saving.save_every_steps is not None:
            if (
                (step + 1) % config.saving.save_every_steps == 0
                or (step + 1) == config.training.max_steps
            ):
                ckpt_path = os.path.join(
                    os.getcwd(), config.wandb.name, "ckpt", "udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )
 
    return model