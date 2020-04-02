import numpy as np
from cluster import Cluster


def get_inv_cov(x, frac):
    # initial value only
    range = np.ones(len(x))
    return np.diag(1/frac*range)


def new_cluster(x, frac, fac, p, m):
    k = 1
    inv_cov = get_inv_cov(x, frac)
    S = [x.tolist()]
    return Cluster(centroid=x, inv_cov=inv_cov, k=k, S=S)
