import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"

    # Weights & Biases
    # rerun of conf 2 8 with the modified l2 computation
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project       = "PI-UDON-KS-1S"
    wandb.name          = "test_base" 
    wandb.ckpt_name     = "test_base" 
    wandb.tag = None

    # Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet"
    arch.num_branch_layers = 5
    arch.num_trunk_layers = 5
    arch.hidden_dim = 1024
    arch.branch_input_dim = 256
    # trunk_input_dim = config.input_dim - branch_input_dim
    arch.out_dim = 256
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2, "embed_dim": 1024})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps = 0
    optim.optimizer = "Soap"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2_500 
    optim.decay_schedule = "Exponential"

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 50
    training.use_cartesian_prod = True

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 100.0, "res": 1.0})#ml_collections.ConfigDict({"ics": 100.0, "res": 1.0}) 
    weighting.momentum = 0.9
    weighting.update_every_steps = 500
    weighting.max_weight = 100_000.0
    weighting.warmup_steps = 0

    # Causal Weighting
    weighting.use_causal = False
    weighting.causal_tol = 0.02
    weighting.num_chunks = 10

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 3
    saving.restore_checkpoint = False
    saving.restore_checkpoint_path = "sep_test_15/ckpt/udon_model"
    saving.total_plots = 1

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.windows            = 200
    eval.trajectory_windows = 200
    eval.num_ics            = 500
    eval.dt_integration     = 0.005

    # Input shape (t is the only input)
    config.input_dim = 256 + 1

    # Training window size
    config.dt_window = 1.0

    # Integer for PRNG random seed.s
    config.seed = 42

    return config