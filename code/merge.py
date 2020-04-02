import numpy as np
from cluster import Cluster


def get_inv_cov(win_cluster, cluster, c):
    a = win_cluster.k * cluster.k / (win_cluster.k + cluster.k)
    c_old = win_cluster.centroid if win_cluster.k < cluster.k else cluster.centroid
    diag = np.diag(
        1/np.matmul((c - c_old).reshape(-1, 1), np.array([c - c_old])))
    I = np.identity(len(win_cluster.centroid))

    result = win_cluster.k * win_cluster.inv_cov + \
        cluster.k * cluster.inv_cov + a * diag * I

    return (1/a) * result


def merge(win_cluster, cluster):
    c = (win_cluster.k * win_cluster.centroid + cluster.k *
         cluster.centroid)/(win_cluster.k + cluster.k)
    inv_cov = get_inv_cov(win_cluster, cluster, c)
    k = win_cluster.k + cluster.k
    S = win_cluster.S
    S.extend(cluster.S)

    new_cluster = Cluster(centroid=c,
                          inv_cov=inv_cov,
                          k=k,
                          S=S)
    return new_cluster
