import pandas as pd
import numpy as np
from scipy.spatial import distance


def get_cluster_min_dist_euc(x, clusters):
    cluster_min_dist = float('inf')
    min_cluster = None
    for cluster in clusters:
        dist_to_cluster = np.sqrt(((x - cluster.centroid) ** 2).sum())
        if dist_to_cluster < cluster_min_dist:
            cluster_min_dist = dist_to_cluster
            min_cluster = cluster
    return min_cluster
