import pandas as pd
import numpy as np
import math
from utils import transpose, inverse, determinant


def olap(i, k):
    # Bhattacharyya distance
    trans = transpose(i.centroid - k.centroid)
    inv_cov = (i.inv_cov + k.inv_cov)/2
    ln = math.log(math.e)
    det_aux = determinant(i.inv_cov) * determinant(k.inv_cov)
    det = determinant(inv_cov)/math.sqrt(det_aux)

    m1 = np.matmul([i.centroid - k.centroid], inv_cov)
    m2 = np.matmul(m1, trans)

    olap = 1/8*m2+1/2*ln*det
    return olap


def overlap(win_cluster, clusters):
    max_olap = float("-inf")
    max_cluster = None

    for cluster in clusters:
        result_olap = olap(win_cluster, cluster)
        if result_olap > max_olap:
            max_olap = result_olap
            max_cluster = cluster
    return max_olap, max_cluster
