import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "eval_enkf"

    # Weights & Biases
    # enk varying P0
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PI-UDON-L96-n40-f6-ics-2"
    wandb.name = "test2_5_enkf_P0_5"
    wandb.ckpt_name = "test2_5"
    wandb.tag = None

    # Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet"
    arch.num_branch_layers = 5
    arch.num_trunk_layers = 5
    arch.hidden_dim = 1024
    arch.branch_input_dim = 40
    arch.out_dim = 40
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10, "embed_dim": 1024})
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
    optim.decay_steps = 5000 

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 400000
    training.batch_size_per_device = 700#16384
    training.use_cartesian_prod = True
    training.update_interval = 1
    training.num_initial_ics = 10000
    training.max_additions = 7
    training.augmentation_scheme = "file" #"model" 
    training.augmentation_file_name = "train_rollouts_025.mat"

    training.num_time_windows = 20

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 100.0, "res": 1.0}) 
    weighting.momentum = 0.9
    weighting.update_every_steps = 500

    # Causal Weighting
    weighting.use_causal = False
    weighting.causal_tol = 0.02
    weighting.num_chunks = 8

    # EKF settings
    config.ekf = ml_collections.ConfigDict()
    config.ekf.obs_every_n = 4
    config.ekf.obs_interval = 0.25

    config.ekf.sigma_obs = 0.2
    config.ekf.sigma_proc = 0.1
    config.ekf.P0_sigma = 1.0 # <-
    config.ekf.dynamic_vars = False # True -> randpick vars

    # EnKF settings
    config.enkf = ml_collections.ConfigDict()
    config.enkf.sigma_model = 0.05
    config.enkf.N_ens = 70


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
    saving.total_plots = 10

    # Input shape (t is the only input)
    config.input_dim = 41

    # Integer for PRNG random seed.s
    config.seed = 42

    return config