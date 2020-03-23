def olap(i, k):
    # Bhattacharyya distance
    return 0


def overlap(win_cluster, clusters):
    max_olap = float("-inf")
    max_cluster = None

    for cluster in clusters:
        olap = olap(win_cluster, cluster)
        if olap > max_olap:
            max_olap = olap
            max_cluster = cluster
    return max_olap, max_cluster
