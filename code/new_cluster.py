import numpy as np
from cluster import Cluster
from get_radius import get_radius


def get_inv_cov(x, frac):
    range = np.ones(len(x))
    return np.diag(1/frac*range)


def new_cluster(x, frac, fac, p, m):
    k = 1
    radius = get_radius(fac, p, m, k)
    inv_cov = get_inv_cov(x, frac)
    S = np.array([x])
    return Cluster(centroid=x, inv_cov=inv_cov, radius=radius, k=k, S=S)
