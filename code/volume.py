import functools
import numpy as np
from numpy.linalg import inv, eig
from scipy.special import gamma, factorial
from radius import get_radius


def get_eigenvalue(inv_cov):
    cov = inv(inv_cov)
    evals, evecs = eig(cov)
    return evals


def get_gamma(p):
    if (p >= 2) & (p % 2 == 0):
        return factorial(p/2)
    else:
        return gamma(p/2)


def get_volume(cluster, fac, p, m):
    lambda_ = get_eigenvalue(cluster.inv_cov)
    r = get_radius(fac, p, m, cluster.k)
    list_ = [r/i for i in lambda_]
    mult = functools.reduce(lambda x, y: x*y, list_)
    gamma_ = get_gamma(p)

    return (2 * mult * np.pi ** (p/2))/gamma_
