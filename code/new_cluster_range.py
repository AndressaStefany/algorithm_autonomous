import numpy as np
from cluster import Cluster
import functools


def get_inv_cov(x, frac, range_):
    # initial value only
    return np.diag(1/frac*range_)


def get_mean_inv_cov(clusters):
    list_ = [c.inv_cov for c in clusters]
    return sum(list_)/len(clusters)


def new_cluster(x, frac, fac, p, m, range_, clusters):
    k = 1
    inv_cov = get_inv_cov(x, frac, range_) if len(
        clusters) < 4 else get_mean_inv_cov(clusters)
    S = [x.tolist()]
    return Cluster(centroid=x, inv_cov=inv_cov, k=k, S=S)
