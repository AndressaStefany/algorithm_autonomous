import numpy as np
from utils import min_dist


def update_winner_cluster(x, cluster):
    aux = x - cluster.centroid
    alpha = 1/(cluster.k+1)
    n = 1/cluster.k
    transp = (cluster.inv_cov * aux).transpose()
    deno = 1 + alpha * (aux.transpose() * cluster.inv_cov * aux)

    inv_cov = cluster.inv_cov/(1-alpha) - alpha / \
        (1-alpha) * (cluster.inv_cov * aux * transp)/deno

    cluster.inv_cov = inv_cov
    cluster.centroid = cluster.centroid + n * aux
    cluster.k += 1
    cluster.S.append(x)


def update_nearest_cluster(x, winner_cluster, clusters):
    near_cluster = min_dist(winner_cluster.centroid, clusters)
    clusters.remove(near_cluster)

    aux_centroid = near_cluster.centroid
    n = 1/near_cluster.k

    near_cluster.centroid = near_cluster.centroid - \
        n * (x - near_cluster.centroid)
    # diff_centroid = aux_centroid - near_cluster.centroid

    # for points in near_cluster.S:
    #     new_points = [c + diff_centroid for c in points]
    # near_cluster.S = new_points
    clusters.append(near_cluster)
