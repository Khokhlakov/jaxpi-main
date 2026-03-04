import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/l63.mat")
    x_ref = data["xsol"]
    y_ref = data["ysol"]
    z_ref = data["zsol"]
    t_star = data["t"].flatten()

    return x_ref, y_ref, z_ref, t_star
