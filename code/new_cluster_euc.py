import numpy as np
from cluster import Cluster


def get_inv_cov(x, frac):
    # identity
    return np.array([[1, 0], [0, 1]])


def new_cluster_euc(x, frac, fac, p, m):
    k = 1
    inv_cov = get_inv_cov(x, frac)
    S = [x.tolist()]
    return Cluster(centroid=x, inv_cov=inv_cov, k=k, S=S)
