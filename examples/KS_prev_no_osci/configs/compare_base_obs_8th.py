import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "run_comparison"

    # Weights & Biases
    # 
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project       = "KS-W1"
    wandb.name_pi       = "base_pi_2"
    wandb.ckpt_name_pi  = "base_pi_2" 
    wandb.name_dd       = "base_dd_1"
    wandb.ckpt_name_dd  = "base_dd_1" 
    wandb.name_hy       = "base_hybrid_050"
    wandb.ckpt_name_hy  = "base_hybrid_050" 

    
    wandb.name          = "compare_base_obs_8th"
    wandb.ckpt_name     = "compare_base_obs_8th"
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
    optim.learning_rate = 1e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 5_000 
    optim.decay_schedule = "Exponential"

    # Training (Windowed Logic)
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 100
    training.use_cartesian_prod = True
    training.dd_data_percentage = 0.01

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 100.0, "res": 1.0})#ml_collections.ConfigDict({"ics": 100.0, "res": 1.0}) 
    weighting.momentum = 0.9
    weighting.update_every_steps = 500
    
    weighting.max_weight = 100.0#2_000_000.0
    weighting.warmup_steps = 500

    # Causal Weighting
    weighting.use_causal = False
    weighting.causal_tol = 0.02
    weighting.num_chunks = 10

    # KF settings
    config.kf = kf = ml_collections.ConfigDict()
    kf.specify_obs_idx  = False
    kf.obs_idx_list     = [0,2,4,8,12,14,16,20,24,26,28,32,36]

    kf.obs_every_n  = 8

    kf.sigma_obs       = 0.2
    kf.P0_sigma        = 0.2
    kf.dynamic_vars    = False 
    kf.batch_l2_size   = 200

    kf.dt_fine = 0.02
    kf.dt_obs  = 1.0
    # dt_fine must divide dt_obs and dt_window

    # Multiplicative Inflation
    kf.sigma_model           = 1.0 # window-level 
    kf.inflation_factor      = 1.05 # window-level 
    kf.N_ens                 = 500
    kf.inflation_factor_list = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30]

    # Route B & Additive Inflation
    kf.route_b_alpha        = 0.0
    kf.route_b_beta         = 0.1
    kf.Q0_sigma             = 0.3
    kf.route_b_n_quad       = 3
    kf.inflation_alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    kf.route_b_beta_list = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]

    # RTPP
    kf.rtpp_alpha      = 0.5
    kf.rtpp_alpha_fine = 1.0
    kf.rtpp_alpha_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
    eval.windows            = 300
    eval.trajectory_windows = 200
    eval.num_ics            = 500
    eval.dt_integration     = 0.02

    eval.test_data_name = "ks_test_data"
    eval.strategies = {"dd_mult_1075": dict(surrogate="DD", inflation="multiplicative", params=dict(inflation_factor=1.075)),
                        "pi_mult_108": dict(surrogate="PI", inflation="multiplicative", params=dict(inflation_factor=1.08)),
                        "pi_RB_full_1_50": dict(surrogate="PI", inflation="route_b", params=dict(alpha=1.0, beta=50.0)),
                        "pi_RB_res_50": dict(surrogate="PI", inflation="route_b", params=dict(alpha=0.0, beta=50.0)),
                        "pi_add_3": dict(surrogate="PI", inflation="additive", params=dict(alpha=3.0)),
                        "pi_rtpp_0325": dict(surrogate="PI", inflation="rtpp", params=dict(alpha_rtpp=0.325)),
                        "dd_rtpp_035": dict(surrogate="DD", inflation="rtpp", params=dict(alpha_rtpp=0.35)),
                        }

    eval.plot_groups = []

    #Inflation	        Parameters
    #multiplicative	    inflation_factor
    #additive	        alpha
    #route_b	        alpha(=0 for residual), beta(=0 for additive)
    #rtpp	a           lpha_rtpp, optional alpha_fine

    eval.strategies = {
        "pi_add_1":   dict(surrogate="PI", inflation="additive",
                                        params=dict(alpha=[0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])),

        "hy_add_1":   dict(surrogate="HY", inflation="additive",
                                        params=dict(alpha=[0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])),

        "dd_mult_sweep_1": dict(surrogate="DD", inflation="multiplicative",
                        params=dict(inflation_factor=[1.0, 1.005, 1.03, 1.04, 1.05, 1.055])),

        "pi_mult_sweep_1": dict(surrogate="PI", inflation="multiplicative",
                                params=dict(inflation_factor=[1.0, 1.005, 1.03, 1.04, 1.05, 1.055])),

        "hy_mult_sweep_1": dict(surrogate="HY", inflation="multiplicative",
                        params=dict(inflation_factor=[1.0, 1.005, 1.03, 1.04, 1.05, 1.055])),

        "pi_rb_res_sweep_1":   dict(surrogate="PI", inflation="route_b",
                        params=dict(alpha=0.0, beta=[1e-1, 0.03, 0.04, 0.045, 0.05, 0.055])),

        "hy_rb_res_sweep_1":   dict(surrogate="HY", inflation="route_b",
                                params=dict(alpha=0.0, beta=[1e-1, 0.03, 0.04, 0.045, 0.05, 0.055])),

        "dd_rtpp_sweep_1": dict(surrogate="DD", inflation="rtpp",
                                params=dict(alpha_rtpp=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9])),

        "pi_rtpp_sweep_1": dict(surrogate="PI", inflation="rtpp",
                                params=dict(alpha_rtpp=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9])),

        "hy_rtpp_sweep_1": dict(surrogate="HY", inflation="rtpp",
                        params=dict(alpha_rtpp=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9])),
    }

    # Input shape (t is the only input)
    config.input_dim = 256 + 1

    # Training window size
    config.dt_window = 1.0

    # Integer for PRNG random seed.s
    config.seed = 42

    return config