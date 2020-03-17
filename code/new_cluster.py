import numpy as np
from cluster import Cluster
from get_radius import get_radius


def get_inv_cov(x, frac):
    range = np.ones(len(x))
    return np.diag(1/frac*range)


def new_cluster(x, frac, fac, p, m):
    radius = get_radius(fac, p, m, 1)
    inv_cov = get_inv_cov(x, frac)
    return Cluster(centroid=x, inv_cov=inv_cov, radius=radius)
