import pandas as pd
import numpy as np


def overlap(win_cluster, clusters):
    max_olap = float("-inf")
    max_cluster = None

    for cluster in clusters:
        # Bhattacharyya distance
        i, k = win_cluster, cluster
        trans = (i.centroid - k.centroid).reshape(-1, 1)
        inv_cov = (i.inv_cov + k.inv_cov)/2
        det_aux = np.linalg.det(i.inv_cov) * np.linalg.det(k.inv_cov)
        det = np.linalg.det(inv_cov)/np.sqrt(det_aux)

        m1 = np.matmul([i.centroid - k.centroid], inv_cov)
        m2 = np.matmul(m1, trans)

        result_olap = 1/8*m2+1/2*np.log(det)
        result_olap = result_olap[0][0]

        if result_olap > max_olap:
            max_olap = result_olap
            max_cluster = cluster
    return max_olap, max_cluster
