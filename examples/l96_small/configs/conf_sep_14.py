import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"

    # Weights & Biases
    # config 13 with fixed weights and no causality
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PI-UDON-L96-small-separated"
    wandb.name = "sep_test_14" 
    wandb.tag = None

    # Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet"
    arch.num_branch_layers = 4
    arch.num_trunk_layers = 4
    arch.hidden_dim = 256
    arch.branch_input_dim = 40
    arch.out_dim = 40
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10, "embed_dim": 256})
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
    optim.decay_steps = 50000 

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 500000
    training.batch_size_per_device = 8192#16384
    training.num_time_windows = 40
    training.use_cartesian_prod = False
    training.update_interval = 50000
    training.num_initial_ics = 1500
    training.max_additions = 0

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "none"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0}) 
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Causal Weighting
    weighting.use_causal = False
    weighting.causal_tol = 1e-3
    weighting.num_chunks = 16

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
    # Save at the end of each window automatically (managed in training script)
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 3

    # Input shape (t is the only input)
    config.input_dim = 41

    # Integer for PRNG random seed.
    config.seed = 42

    return config