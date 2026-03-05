import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    xyz_ref, t_star = get_dataset()
    xyz0 = xyz_ref[0, :]

    # Restore model
    model = models.L63(config, xyz0, t_star)
    #ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")

    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error  #P#
    l2_error = model.compute_l2_error(params, xyz_ref)
    print("L2 error: {:.3e}".format(l2_error))

    xyz_pred = model.xyz_pred_fn(params, model.t_star)
    TT = t_star.reshape(-1, 1) #

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    components = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        # Exact vs pred
        axes[i, 0].plot(t_star, xyz_ref[:, i], label='Exact', color='black', linewidth=1.5)
        axes[i, 0].plot(t_star, xyz_pred[:, i], label='Predicted', color=colors[i], linewidth=1)
        axes[i, 0].set_ylabel(f"{components[i]}(t)", fontsize=14)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="upper right")
        
        if i == 0:
            axes[i, 0].set_title("Exact vs. Predicted Trajectory", fontsize=16)
        if i == 2:
            axes[i, 0].set_xlabel("Time (t)", fontsize=14)

        # Abs error
        abs_error = jnp.abs(xyz_ref[:, i] - xyz_pred[:, i])
        axes[i, 1].plot(t_star, abs_error, color=colors[i], linewidth=1.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_yscale('log') 
        
        if i == 0:
            axes[i, 1].set_title("Absolute Error", fontsize=16)
        if i == 2:
            axes[i, 1].set_xlabel("Time (t)", fontsize=14)

    fig.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "l63.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
