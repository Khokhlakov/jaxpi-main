import scipy.io
import jax.numpy as jnp

def get_dataset(filepath="data/l63_udon.mat"):
    """
    Loads and truncates the L63 dataset to a specific time horizon.
    """
    data = scipy.io.loadmat(filepath)
    
    # 1. Extract raw data
    xyz_ref = jnp.array(data["usol_all"])  # (num_ics, num_points, 3)
    u0_ref = jnp.array(data["u0_all"])     # (num_ics, 3)
    t_star = jnp.array(data["t"]).flatten() # (num_points,)

    # Mask desired horizon
    mask = t_star <= 0.5
    t_star = t_star[mask]
    xyz_ref = xyz_ref[:, mask, :] # Slicing the 'num_points' dimension

    return xyz_ref, u0_ref, t_star