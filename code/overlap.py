import pandas as pd
import numpy as np
import math
from utils import covariance, transpose, inverse


def olap(i, k):
    # Bhattacharyya distance
    trans = transpose(i.centroid - k.centroid)
    cov = covariance(i.centroid, k.centroid)
    inv_cov = inverse(cov)
    ln = math.log(math.e)
    det = 0

    olap = 1/8*trans*inv_cov+1/2*ln*det
    return olap


def overlap(win_cluster, clusters):
    max_olap = float("-inf")
    max_cluster = None

    for cluster in clusters:
        olap = olap(win_cluster, cluster)
        if olap > max_olap:
            max_olap = olap
            max_cluster = cluster
    return max_olap, max_cluster
