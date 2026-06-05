import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train_dd"
    config.model_type = "data"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PI-UDON-L96-n40-f6-ics-2"
    wandb.name = "test8_dd"
    wandb.ckpt_name = "test8_dd"
    wandb.tag = None

    # Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet"
    arch.num_branch_layers  = 4#5
    arch.num_trunk_layers   = 4#5
    arch.hidden_dim         = 256#1024
    arch.branch_input_dim   = 40
    arch.out_dim            = 40
    arch.activation         = "tanh"
    arch.periodicity        = None
    arch.fourier_emb        = None
    arch.reparam            = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps  = 1
    optim.optimizer         = "Adam"
    optim.beta1             = 0.9
    optim.beta2             = 0.999
    optim.eps               = 1e-8
    optim.learning_rate     = 1e-3
    optim.decay_rate        = 0.9
    optim.decay_steps       = 10_000 

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps              = 200_000
    training.batch_size_per_device  = 700#16384
    training.steps_per_window       = 50

    # batch_size must be divisible by num. of devices
    # Used in Data-Driven DeepONet
    training.batch_size             = 4098
    training.use_cartesian_prod     = False
    training.update_interval        = 1
    training.num_initial_ics        = 10000
    training.max_additions          = 30
    training.augmentation_scheme    = "file" #"model" 
    training.augmentation_file_name = "train_rollouts_025.mat"
    training.augmentation_file_name_eval = "train_rollouts_025.mat"

    training.num_time_windows = 30

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = {"data": 1.0}
    weighting.momentum  = 0.9
    weighting.update_every_steps = 1_000

    # Causal Weighting
    weighting.use_causal = False
    weighting.causal_tol = 0.02
    weighting.num_chunks = 8

    # KF settings
    config.kf = kf = ml_collections.ConfigDict()
    kf.specify_obs_idx  = False
    kf.obs_idx_list     = [0,2,4,8,12,14,16,20,24,26,28,32,36]

    # EKF settings
    config.ekf = ekf = ml_collections.ConfigDict()
    ekf.obs_every_n     = 4
    ekf.sigma_obs       = 0.5
    ekf.sigma_proc      = 0.1 # Unused in EnKF
    ekf.P0_sigma        = 0.3
    ekf.dynamic_vars    = False # True -> randpick vars
    ekf.batch_l2_size   = 200

    ekf.dt_fine = 0.05
    ekf.dt_obs  = 0.25
    # dt_fine must divide dt_obs and dt_window

    # EnKF settings
    config.enkf = enkf = ml_collections.ConfigDict()
    enkf.sigma_model = 1.0
    enkf.N_ens       = 100
    enkf.rk4_substeps = 10

    enkf.l96_forcing  = 6.0

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_errors      = True
    logging.log_losses      = True
    logging.log_weights     = True
    logging.log_preds       = False
    logging.log_grads       = False
    logging.log_ntk         = False

    # Saving
    config.saving = saving  = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts   = 3
    saving.restore_checkpoint = False
    saving.restore_checkpoint_path = "sep_test_15/ckpt/udon_model"
    saving.total_plots = 5

    saving.val_ics = 20

    # Input shape (t is the only input)
    config.input_dim = 41

    # Training window size
    config.dt_window = 0.25

    # Integer for PRNG random seed.s
    config.seed = 42

    return config