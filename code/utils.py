import pandas as pd
import numpy as np
from scipy.spatial import distance


def dist(x, cluster):
    return distance.mahalanobis(x, cluster.centroid, cluster.inv_cov)


def min_dist(x, clusters):
    cluster_min_dist = float('inf')
    min_cluster = None
    for cluster in clusters:
        dist_to_cluster = distance.mahalanobis(
            x, cluster.centroid, cluster.inv_cov)
        if dist_to_cluster < cluster_min_dist:
            cluster_min_dist = dist_to_cluster
            min_cluster = cluster

    return min_cluster
