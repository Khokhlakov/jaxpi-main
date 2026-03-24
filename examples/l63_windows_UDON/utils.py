import scipy.io
import jax.numpy as jnp

def get_dataset(filepath="data/l63_udon.mat"):
    """
    Loads the multi-trajectory Lorenz 63 dataset for DeepONet training.
    """
    data = scipy.io.loadmat(filepath)
    
    # Extract data and convert to JAX arrays
    # xyz_ref shape: (num_ics, num_points, 3)
    xyz_ref = jnp.array(data["usol_all"])
    
    # u0_ref shape: (num_ics, 3)
    u0_ref = jnp.array(data["u0_all"])
    
    # t_star shape: (num_points,)
    t_star = jnp.array(data["t"]).flatten()

    return xyz_ref, u0_ref, t_star