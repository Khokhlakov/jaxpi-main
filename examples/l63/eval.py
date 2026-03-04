import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    x_ref, y_ref, z_ref, t_star = get_dataset()
    x0 = x_ref[0, :]
    y0 = y_ref[0, :]
    z0 = z_ref[0, :]

    # Restore model
    model = models.Burgers(config, u0, t_star, x_star) #P#
    #ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")

    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error  #P#
    l2_error = model.compute_l2_error(params, u_ref) #P#
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)  #P#
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")  #P#

    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "burger.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
