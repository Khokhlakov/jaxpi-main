import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/l63_t5.mat")
    xyz_ref = data["usol"]
    t_star = data["t"].flatten()

    return xyz_ref, t_star
