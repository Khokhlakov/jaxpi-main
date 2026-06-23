import ml_collections
import jax.numpy as jnp

def get_config():
    # Config 1 but init weights 10:1 and using causal training
    config = ml_collections.ConfigDict()
    config.mode = "train_dd"

    # Weights & Biases
    # rerun of conf 2 8 with the modified l2 computation
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project       = "PI-UDON-L96-F"
    wandb.name          = "test_forcing_dd_1" 
    wandb.ckpt_name     = "test_forcing_dd_1" 
    wandb.tag = None

    # Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet"
    arch.num_branch_layers = 5
    arch.num_trunk_layers = 5
    arch.hidden_dim = 1024
    arch.branch_input_dim = 41
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
    optim.decay_steps = 2_300 
    optim.decay_schedule = "Exponential"

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 70#16384
    training.use_cartesian_prod = True
    training.update_interval = 1000
    training.num_initial_ics = 8000
    training.max_additions = 30

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"data_loss": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 500

    # Causal Weighting
    weighting.use_causal = True
    weighting.causal_tol = 0.01
    weighting.num_chunks = 10

    # KF settings
    config.kf = kf = ml_collections.ConfigDict()
    kf.specify_obs_idx  = False
    kf.obs_idx_list     = [0,2,4,8,12,14,16,20,24,26,28,32,36]

    # EKF settings
    config.ekf = ekf = ml_collections.ConfigDict()
    ekf.obs_every_n  = 4

    ekf.sigma_obs       = 0.2
    ekf.P0_sigma        = 0.3
    ekf.dynamic_vars    = False 
    ekf.batch_l2_size   = 200

    ekf.dt_fine = 0.005
    ekf.dt_obs  = 0.25
    # dt_fine must divide dt_obs and dt_window

    # EnKF settings
    config.enkf = ml_collections.ConfigDict()
    config.enkf.sigma_model = 1.0
    config.enkf.N_ens       = 100

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
    saving.restore_checkpoint_path = "test_1/ckpt/udon_model"
    saving.total_plots = 2

    # Input shape (t is the only input)
    config.input_dim = 42

    # Training window size
    config.dt_window = 0.25

    # Integer for PRNG random seed.s
    config.seed = 42

    return config