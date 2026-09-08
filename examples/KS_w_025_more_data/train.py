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
# Main training loop  --  Physics-Informed (PI) branch
# ---------------------------------------------------------------------------
def train_and_evaluate(config, workdir: str):

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load Dataset from HDF5 ─────────────────────────────────────────────
    train_file = "data/ks_train_data.h5"
    test_file  = "data/ks_test_data.h5"

    with h5py.File(train_file, 'r') as f_train:
        # u_pool contains states pooled across all trajectories and windows.
        # Shape: (num_samples * (max_additions + 1), N) -> [N spatial grid pts]
        u_pool_np = np.array(f_train['u'][:])
        L  = float(f_train.attrs["L"])
        N  = int(f_train.attrs["N"])
        dt = float(f_train.attrs["dt"])

    with h5py.File(test_file, 'r') as f_test:
        # Dense test trajectories. Shape: (num_samples, test_windows*interval_steps + 1, N)
        x_ref_eval_all = jnp.array(f_test['u'][:])

    # gen_data.py always defines a single training "window" as 1.0
    # (normalized) time unit -> interval_steps = round(1.0 / dt) fine steps.
    # (ks_test_data.h5 does not store its own time axis, unlike the pooled
    # windows file, so we rebuild it from dt -- test data was saved every
    # single fine step, starting at t=0.)

    t_dt           = 0.25
    interval_steps = int(round(t_dt / dt))
    time_steps     = interval_steps + 1  # points per window, including the IC

    # ── Reference data (used only for eval logging during training) ────────
    trajs_per_window = min(100, x_ref_eval_all.shape[0])

    # L2 evaluation dataset setup (first window only)
    t_star = jnp.arange(time_steps, dtype=jnp.float32) * dt

    # Extract the first `trajs_per_window` trajectories for the first `time_steps`
    x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]  # (trajs, time_steps, N)

    # Construct the branch network input u(t0) for evaluation
    u_ref_eval = x_ref_eval_all[0:trajs_per_window, 0, :]             # (trajs, N)

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
    #    Key advancement is fused into the same pmapped call that draws the
    #    batch, so each step is a single fully-parallel dispatch with no
    #    host round-trip and no cross-device data movement (mirrors the
    #    fold_in / single-pmap pattern used by the DD sampler below).
    @jax.pmap
    def get_batch_on_device(device_key, local_u_pool, local_dom_t):
        """Splits keys and samples entirely on-device. No host involvement."""
        new_key, sample_key = jax.random.split(device_key)
        key_u, key_t = jax.random.split(sample_key)

        # Sample u (SpaceSampler equivalent)
        idx_u   = jax.random.randint(key_u, (batch_size_per_device,), 0, pool_size)
        batch_u = local_u_pool[idx_u, :]

        # Sample t (UniformSampler equivalent)
        t_min   = local_dom_t[0, 0]
        t_max   = local_dom_t[0, 1]
        batch_t = jax.random.uniform(key_t, (batch_size_per_device, 1), minval=t_min, maxval=t_max)

        return new_key, (batch_u, batch_t)

    # 4. Initialize reproducible PRNG Keys, seeded independently per device
    #    via a device-parallel fold_in (no host-side split + scatter).
    seed        = config.training.get("seed", 42)
    init_key    = jax.random.PRNGKey(seed)
    device_keys = jax.pmap(lambda i: jax.random.fold_in(init_key, i))(jnp.arange(num_devices))

    # ── Build model ────────────────────────────────────────────────────────
    model     = models.KSUDON(config, t_star, L=L, N=N, dt=dt)
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

        # ── Sample on-device (key advance + batch draw in one dispatch) ─────
        device_keys, batch = get_batch_on_device(device_keys, u_pool_repl, dom_t_repl)

        # ── Forward + gradient step ────────────────────────────────────────
        model.state = model.step(model.state, batch)

        ## ── Log LR at EVERY step ───────────────────────────────────────────
        if jax.process_index() == 0:
            # Note: Replace `lr_schedule_fn` with your actual Optax schedule variable
            # (e.g., config.training.lr_schedule or model.lr_fn)
            current_lr = float(model.lr_schedule(step))
            wandb.log({"train/learning_rate": current_lr}, step=step)##

        # ── Adaptive loss weighting (optional) ────────────────────────────
        if config.weighting.scheme in ("grad_norm", "ntk"):
            if step % config.weighting.update_every_steps == 0 and step >= config.weighting.warmup_steps:
                model.state = model.update_weights(model.state, batch)

        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))

                # Evaluator expects state, train batch, branch eval input, and ground truth trajectory
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)

                # Track pool parameters (fixed values now, since augmentation is dropped)
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
                    os.getcwd(), config.wandb.name, "ckpt", "ks_udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model


# ---------------------------------------------------------------------------
# Data-Driven (DD) branch
# ---------------------------------------------------------------------------
def train_and_evaluate_dd(config, workdir: str):
    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load Dataset from HDF5 ───────────────────────────────────────────────
    # Pooled dense windows produced by gen_data.py:
    #   u : (num_windows, interval_steps + 1, N)
    #       - num_windows = num_samples * max_additions
    #       - each window is a dense, 1-normalized-time-unit-long trajectory
    #         segment starting from a different point along the burned-in
    #         trajectory (state at tau = 0, dt, 2·dt, ... up to window_size)
    train_file = "data/ks_train_data_dd.h5"

    with h5py.File(train_file, 'r') as f_train:
        train_data_np  = np.array(f_train['u'][:])              # (num_windows, num_t, N)
        L              = float(f_train.attrs["L"])
        N              = int(f_train.attrs["N"])
        dt             = float(f_train.attrs["dt"])
        interval_steps = int(f_train.attrs["interval_steps"])   # pts per window (excl. IC)

    train_data = jnp.array(train_data_np)

    num_windows, num_t, state_dim = train_data.shape
    assert state_dim == N, (
        f"ks_train_data_dd.h5 state_dim ({state_dim}) does not match its own "
        f"'N' attribute ({N})."
    )

    # ── Reference data & Time Grid ──────────────────────────────────────────
    # Window-local time grid: 0 -> window_size over num_t points
    window_size = interval_steps * dt
    t_star = jnp.arange(num_t, dtype=jnp.float32) * dt

    # ── Load Dataset from HDF5 ─────────────────────────────────────────────
    test_file = "data/ks_test_data.h5"

    with h5py.File(test_file, 'r') as f_test:
        x_ref_eval_all = jnp.array(f_test['u'][:])

    trajs_per_window = min(100, x_ref_eval_all.shape[0])
    time_steps       = interval_steps + 1  # matches num_t above by construction

    # Step through the dense test trajectory in non-overlapping-except-for-
    # the-shared-boundary-point chunks of `interval_steps`, exactly as the
    # DD windows themselves are laid out (0-50, 50-100, ... for interval_steps=50).
    pts_pw = interval_steps
    total_test_steps = x_ref_eval_all.shape[1] - 1
    num_windows_to_eval = min(10, max(1, total_test_steps // pts_pw))

    u_refs = []
    x_refs = []

    for w in range(num_windows_to_eval):
        start_idx = w * pts_pw
        end_idx   = start_idx + time_steps

        u_refs.append(x_ref_eval_all[0:trajs_per_window, start_idx, :])
        x_refs.append(x_ref_eval_all[0:trajs_per_window, start_idx:end_idx, :])

    u_ref_eval = jnp.concatenate(u_refs, axis=0)
    x_ref_eval = jnp.concatenate(x_refs, axis=0)

    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Model & Evaluator ──────────────────────────────────────────────────
    model     = models.KSUDON_DD(config, t_star, N=N, L=L)
    evaluator = models.KSUDONEvaluator_DD(config, model)

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

        # Branch input: u(t0) for the sampled window
        batch_u = data[idx_win, 0, :]

        # Trunk query time
        batch_t = t_array[idx_t].reshape(batch_size_per_device, 1)

        # Trunk target: state at the sampled time
        batch_x = data[idx_win, idx_t, :]

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
                    os.getcwd(), config.wandb.name, "ckpt", "ks_udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model


def train_and_evaluate_hybrid(config, workdir: str):
    """Trains KSUDON_Hybrid on a loss with THREE terms, sampled every step:

        "ics"  -- IC loss,           batch_u        vs x_net(batch_u, t0)   (PI)
        "res"  -- PDE residual loss, r_net(batch_u, batch_t)                (PI)
        "data" -- supervised MSE,    x_net(data_u, data_t) vs data_x        (DD)

    The physics terms ("ics"/"res") are sampled exactly as in
    `train_and_evaluate`, from the sparse PI pool (`ks_train_data.h5`).

    The data term ("data") is sampled from a SMALL, FIXED subsample of the
    dense DD window pool (`ks_train_data_dd.h5`) -- only a
    `config.training.dd_subsample_fraction` fraction of its windows
    (default 1/100) is ever read off disk or kept resident on device. This
    mirrors the memory-bounding rationale in gen_data.py's chunked writer:
    the full DD pool is `num_samples * max_additions` dense windows, which
    can be tens of GB, so we read only the sampled rows directly out of the
    HDF5 file via fancy indexing rather than materializing the whole array.

    `config.weighting.init_weights` must supply all three keys ("ics",
    "res", "data") since jaxpi's grad-norm/fixed weighting combines loss
    terms via `tree_map` over matching dict keys (see jaxpi.models.PINN).
    If "data" is missing (e.g. an unmodified conf_base_pi.py, which only
    defines "ics"/"res") it is defaulted to 1.0 with a logged note --
    override `config.weighting.init_weights.data` explicitly for anything
    but a quick smoke test.
    """

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(project=config.wandb.project, name=config.wandb.name)

    # ── Load the PI (sparse) pool -- identical to train_and_evaluate ───────
    train_file    = "data/ks_train_data.h5"
    train_dd_file = "data/ks_train_data_dd.h5"
    test_file     = "data/ks_test_data.h5"

    with h5py.File(train_file, 'r') as f_train:
        # u_pool contains states pooled across all trajectories and windows.
        u_pool_np = np.array(f_train['u'][:])
        L  = float(f_train.attrs["L"])
        N  = int(f_train.attrs["N"])
        dt = float(f_train.attrs["dt"])

    # gen_data.py defines a single training window as `w_dt` = 0.25
    # (normalized) time units -> interval_steps = round(0.25 / dt) fine
    # steps. This window also defines the trunk's [t0, t1] domain.
    t_dt           = 0.25
    interval_steps = int(round(t_dt / dt))
    time_steps     = interval_steps + 1  # points per window, including the IC
    t_star         = jnp.arange(time_steps, dtype=jnp.float32) * dt

    # ── Load a SMALL subsample of the dense DD pool straight off disk ──────
    seed        = config.training.get("seed", 42)
    dd_fraction = config.training.get("dd_subsample_fraction", 1.0 / 100.0)

    with h5py.File(train_dd_file, 'r') as f_dd:
        num_windows_full = f_dd['u'].shape[0]
        num_t_dd         = f_dd['u'].shape[1]
        L_dd             = float(f_dd.attrs["L"])
        N_dd             = int(f_dd.attrs["N"])
        dt_dd            = float(f_dd.attrs["dt"])

        # ks_train_data.h5 and ks_train_data_dd.h5 are produced together by
        # a single generate_datasets(...) call and must agree on grid/time
        # discretization for the two loss terms to live on the same window.
        assert N_dd == N and abs(L_dd - L) < 1e-8 and abs(dt_dd - dt) < 1e-8, (
            "ks_train_data.h5 and ks_train_data_dd.h5 disagree on L/N/dt -- "
            "regenerate both from the same gen_data.generate_datasets(...) call."
        )
        assert num_t_dd == time_steps, (
            f"DD window length ({num_t_dd} pts) does not match the PI training "
            f"window ({time_steps} pts implied by t_dt={t_dt}, dt={dt})."
        )

        num_windows_sub = max(1, int(round(num_windows_full * dd_fraction)))
        rng     = np.random.default_rng(seed)
        sub_idx = np.sort(rng.choice(num_windows_full, size=num_windows_sub, replace=False))

        # Fancy-indexed HDF5 read: pulls only the sampled windows off disk --
        # never the full (num_windows_full, time_steps, N) dense array.
        train_data_sub_np = f_dd['u'][sub_idx]

    logging.info(
        f"DD data-loss pool: {num_windows_sub} / {num_windows_full} windows "
        f"({dd_fraction:.4%}) loaded for the 'data' loss term."
    )

    with h5py.File(test_file, 'r') as f_test:
        # Dense test trajectories. Shape: (num_samples, test_windows*interval_steps + 1, N)
        x_ref_eval_all = jnp.array(f_test['u'][:])

    # ── Reference data (used only for eval logging during training) ────────
    trajs_per_window = min(100, x_ref_eval_all.shape[0])

    # Extract the first `trajs_per_window` trajectories for the first `time_steps`
    x_ref_eval = x_ref_eval_all[0:trajs_per_window, 0:time_steps, :]  # (trajs, time_steps, N)

    # Construct the branch network input u(t0) for evaluation
    u_ref_eval = x_ref_eval_all[0:trajs_per_window, 0, :]             # (trajs, N)

    logging.info("L2 dataset imported:")
    logging.info(f"x_ref_eval: {x_ref_eval.shape}")
    logging.info(f"u_ref_eval (Branch Input): {u_ref_eval.shape}")
    logging.info(f"t_star: {t_star.shape}")

    # ── Samplers ───────────────────────────────────────────────────────────
    num_devices                = jax.local_device_count()
    batch_size_per_device      = config.training.batch_size_per_device
    data_batch_size_per_device = config.training.get(
        "data_batch_size_per_device", batch_size_per_device
    )
    pool_size = u_pool_np.shape[0]

    # t is sampled uniformly over the training window [t0, t1].
    dom_t_np = np.array([[t_star[0], t_star[-1]]])

    # Push exact copies of both pools + the window time grid onto every
    # local device.
    u_pool_repl  = jax.device_put_replicated(u_pool_np, jax.local_devices())
    dom_t_repl   = jax.device_put_replicated(dom_t_np, jax.local_devices())
    dd_pool_repl = jax.device_put_replicated(train_data_sub_np, jax.local_devices())
    t_dd_repl    = jax.device_put_replicated(np.asarray(t_star), jax.local_devices())

    @jax.pmap
    def get_batch_on_device(device_key, local_u_pool, local_dom_t, local_dd_pool, local_t_dd):
        """Draws BOTH the physics batch and the data batch on-device, fused
        into a single pmapped dispatch -- same pattern as the PI-only and
        DD-only samplers in train_and_evaluate / train_and_evaluate_dd, just
        combined into one call so each step is still a single dispatch with
        no host round-trip."""
        new_key, sample_key = jax.random.split(device_key)
        key_u, key_t, key_win, key_dt = jax.random.split(sample_key, 4)

        # -- Physics batch (SpaceSampler / UniformSampler equivalent) --
        idx_u   = jax.random.randint(key_u, (batch_size_per_device,), 0, pool_size)
        batch_u = local_u_pool[idx_u, :]

        t_min   = local_dom_t[0, 0]
        t_max   = local_dom_t[0, 1]
        batch_t = jax.random.uniform(key_t, (batch_size_per_device, 1), minval=t_min, maxval=t_max)

        # -- Data batch: (window, timestep) pairs from the subsampled DD pool --
        idx_win = jax.random.randint(key_win, (data_batch_size_per_device,), 0, num_windows_sub)
        idx_t   = jax.random.randint(key_dt,  (data_batch_size_per_device,), 0, num_t_dd)

        data_u = local_dd_pool[idx_win, 0, :]                      # branch input: window IC
        data_t = local_t_dd[idx_t].reshape(data_batch_size_per_device, 1)
        data_x = local_dd_pool[idx_win, idx_t, :]                  # ground-truth state at data_t

        return new_key, (batch_u, batch_t, data_u, data_t, data_x)

    # Initialize reproducible PRNG Keys, seeded independently per device.
    init_key    = jax.random.PRNGKey(seed)
    device_keys = jax.pmap(lambda i: jax.random.fold_in(init_key, i))(jnp.arange(num_devices))

    # ── Build model ────────────────────────────────────────────────────────
    # The hybrid loss has 3 terms, so config.weighting.init_weights must
    # define matching "ics"/"res"/"data" keys (see jaxpi.models.PINN.loss,
    # which combines loss terms via tree_map over these dicts). Default a
    # missing "data" weight rather than fail on an unmodified PI-only config.
    if "data" not in config.weighting.init_weights:
        logging.info(
            "config.weighting.init_weights has no 'data' entry (needed for the "
            "3-term hybrid loss) -- defaulting it to 1.0. Set "
            "config.weighting.init_weights.data explicitly to control it, "
            "e.g. in conf_base_pi.py: "
            'weighting.init_weights = ml_collections.ConfigDict('
            '{"ics": 100.0, "res": 1.0, "data": 1.0})'
        )
        config.weighting.init_weights.data = 1.0

    model     = models.KSUDON_Hybrid(config, t_star, L=L, N=N, dt=dt)
    evaluator = models.KSUDONEvaluator_Hybrid(config, model)

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

        # ── Sample on-device (key advance + both batches in one dispatch) ──
        device_keys, batch = get_batch_on_device(
            device_keys, u_pool_repl, dom_t_repl, dd_pool_repl, t_dd_repl
        )

        # ── Forward + gradient step ────────────────────────────────────────
        model.state = model.step(model.state, batch)

        # ── Adaptive loss weighting (optional) ────────────────────────────
        if config.weighting.scheme in ("grad_norm", "ntk"):
            if step % config.weighting.update_every_steps == 0 and step >= config.weighting.warmup_steps:
                model.state = model.update_weights(model.state, batch)

        # ── Logging ────────────────────────────────────────────────────────
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                state     = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_dev = jax.device_get(tree_map(lambda x: x[0], batch))

                # Evaluator expects state, train batch, branch eval input, and ground truth trajectory
                log_dict = evaluator(state, batch_dev, u_ref_eval, x_ref_eval)

                # Track pool sizes actually in use (fixed values, no augmentation)
                log_dict["pool/active_ics"]        = pool_size
                log_dict["pool/active_dd_windows"] = num_windows_sub

                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # ── Checkpointing ──────────────────────────────────────────────────
        if config.saving.save_every_steps is not None:
            if ((step + 1) % config.saving.save_every_steps == 0
                    or (step + 1) == config.training.max_steps):
                ckpt_path = os.path.join(
                    os.getcwd(), config.wandb.name, "ckpt", "ks_udon_model"
                )
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model

