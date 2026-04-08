import scipy.io
import jax.numpy as jnp

def get_dataset(filepath="data/l96_udon.mat"):
    """
    Loads and truncates the L96 dataset to a specific time horizon.
    """
    data = scipy.io.loadmat(filepath)
    
    # 1. Extract raw data
    x_ref = jnp.array(data["usol_all"])  # (num_ics, num_points, 40)
    u0_ref = jnp.array(data["u0_all"])   # (num_ics, 40)
    t_star = jnp.array(data["t"]).flatten() # (num_points,)

    # Mask desired horizon
    mask = t_star <= 0.5
    t_star = t_star[mask]
    x_ref = x_ref[:, mask, :] # Slicing the 'num_points' dimension

    return x_ref, u0_ref, t_star