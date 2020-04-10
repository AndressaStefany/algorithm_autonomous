import numpy as np
from utils import min_dist


def update_winner_cluster(x, cluster):
    aux = (x - cluster.centroid).reshape(-1, 1)
    alpha = 1/(cluster.k+1)
    n = 1/cluster.k
    transp = np.matmul(cluster.inv_cov, aux).T
    deno = 1 + alpha * np.matmul(np.matmul(aux.T, cluster.inv_cov), aux)

    inv_cov = cluster.inv_cov/(1-alpha) - alpha / (1-alpha) * \
        np.matmul(np.matmul(cluster.inv_cov, aux), transp) / deno

    cluster.inv_cov = inv_cov
    cluster.centroid = cluster.centroid + n * (x - cluster.centroid)
    cluster.k += 1
    cluster.S.append(list(x))


def update_nearest_cluster(x, winner_cluster, clusters):
    near_cluster = min_dist(winner_cluster.centroid, clusters)
    clusters.remove(near_cluster)

    n = 1/near_cluster.k

    near_cluster.centroid = near_cluster.centroid - \
        n * (x - near_cluster.centroid)
    clusters.append(near_cluster)
