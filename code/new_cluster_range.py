import numpy as np
from cluster import Cluster


def get_inv_cov(x, frac, range_):
    # initial value only
    return np.diag(1/frac*range_)


def new_cluster(x, frac, fac, p, m, range_):
    k = 1
    inv_cov = get_inv_cov(x, frac, range_)
    S = [x.tolist()]
    return Cluster(centroid=x, inv_cov=inv_cov, k=k, S=S)
