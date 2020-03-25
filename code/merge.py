import numpy as np
from cluster import Cluster
from utils import transpose


def get_new_centroid(win_cluster, cluster):
    return (win_cluster.k * win_cluster.centroid + cluster.k * cluster.centroid)/(win_cluster.k + cluster.k)


def get_inv_cov(win_cluster, cluster, c):
    a = win_cluster.k * cluster.k / (win_cluster.k + cluster.k)
    c_old = win_cluster.c if win_cluster.k < cluster.k else cluster.c
    diag = np.diag(1/(transpose(c - c_old) * (c - c_old)))

    return win_cluster.k * win_cluster.inv_cov + cluster.k * cluster.inv_cov + a * diag


def merge(win_cluster, cluster):
    c = get_new_centroid(win_cluster, cluster)
    inv_cov = get_inv_cov(win_cluster, cluster, c)
    k = win_cluster.k + cluster.k
    S = win_cluster.S
    S.extend(cluster.S)

    new_cluster = Cluster(centroid=c,
                          inv_cov=inv_cov,
                          radius=radius,
                          k=k,
                          S=S)
    return new_cluster
