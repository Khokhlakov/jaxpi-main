import ml_collections
import jax.numpy as jnp

def get_config():
    # Config 1 but init weights 10:1 and using causal training
    config = ml_collections.ConfigDict()
    config.mode = "run_comparison"

    # Weights & Biases
    # Base for inflation tunning
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project       = "L96-F"
    wandb.name_pi       = "test_pi_1" 
    wandb.ckpt_name_pi  = "test_pi_1"
    wandb.name_dd       = "test_dd_1"
    wandb.ckpt_name_dd  = "test_dd_1"
    wandb.name          = "compare_89"
    wandb.ckpt_name     = "compare_89"
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
    optim.decay_steps = 2_500 
    optim.decay_schedule = "Exponential"

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 100
    training.use_cartesian_prod = True

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

    # KF settings
    config.kf = kf = ml_collections.ConfigDict()
    kf.specify_obs_idx  = False
    kf.obs_idx_list     = [0,2,4,8,12,14,16,20,24,26,28,32,36]

    kf.obs_every_n  = 4

    kf.sigma_obs       = 0.2
    kf.P0_sigma        = 0.2
    kf.dynamic_vars    = False 
    kf.batch_l2_size   = 100

    kf.dt_fine = 0.005
    kf.dt_obs  = 0.25
    # dt_fine must divide dt_obs and dt_window

    # Multiplicative Inflation
    kf.sigma_model           = 1.0 # window-level 
    kf.inflation_factor      = 1.05 # window-level 
    kf.N_ens                 = 200
    kf.inflation_factor_list = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30]

    # Route B & Additive Inflation
    kf.route_b_alpha        = 1.0
    kf.route_b_beta         = 50.0
    kf.Q0_sigma             = 0.3
    kf.route_b_n_quad       = 3
    kf.inflation_alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    kf.route_b_beta_list = [0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0]

    # RTPP
    kf.rtpp_alpha      = 0.35
    kf.rtpp_alpha_fine = 1.0
    kf.rtpp_alpha_list = [0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45]

    config.kf.inflation_factor_dd = 1.075
    config.kf.inflation_factor_pi = 1.08
    config.kf.route_b_alpha = 1.0        # add + residual
    config.kf.route_b_beta = 50.0        # both add + residual & just residual
    config.kf.route_b_additive_alpha = 3.0 
    config.kf.rtpp_alpha_dd = 0.35
    config.kf.rtpp_alpha_pi = 0.325

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
    saving.total_plots = 5

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.windows            = 500
    eval.trajectory_windows = 500
    eval.num_ics            = 200
    eval.dt_integration     = 0.005

    eval.test_data_name = "l96_forcing_test_89"
    eval.strategies = {"dd_mult_1075": dict(surrogate="DD", inflation="multiplicative", params=dict(inflation_factor=1.075)),
                       "pi_mult_108": dict(surrogate="PI", inflation="multiplicative", params=dict(inflation_factor=1.08)),
                       "pi_RB_full_1_50": dict(surrogate="PI", inflation="route_b", params=dict(alpha=1.0, beta=50.0)),
                       "pi_RB_res_50": dict(surrogate="PI", inflation="route_b", params=dict(alpha=0.0, beta=50.0)),
                       "pi_add_3": dict(surrogate="PI", inflation="additive", params=dict(alpha=3.0)),
                       "pi_rtpp_0325": dict(surrogate="PI", inflation="rtpp", params=dict(alpha_rtpp=0.325)),
                       "dd_rtpp_035": dict(surrogate="DD", inflation="rtpp", params=dict(alpha_rtpp=0.35)),
                       }

    eval.plot_groups = [["dd_mult_1075", "pi_mult_108"],
                        ["pi_rtpp_0325", "dd_rtpp_035"],
                        ["pi_mult_108", "pi_RB_full_1_50", "pi_RB_res_50", "pi_add_3", "pi_rtpp_0325"]
                        ]

    #Inflation	        Parameters
    #multiplicative	    inflation_factor
    #additive	        alpha
    #route_b	        alpha(=0 for residual), beta(=0 for additive)
    #rtpp	a           lpha_rtpp, optional alpha_fine

    # Input shape (t is the only input)
    config.input_dim = 41

    # Training window size
    config.dt_window = 0.25

    # Integer for PRNG random seed.s
    config.seed = 42

    return config