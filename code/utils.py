from scipy.spatial import distance


def is_empty_list(list):
    return len(list) == 0


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


def covariance(x, y):
    # input = np.array
    data = np.array([x, y])
    df = pd.DataFrame(data=data)
    return df.cov()


def transpose(x):
    # input = np.array
    data = np.array([x])
    df = pd.DataFrame(data=data)
    return df.T


def inverse(x):
    # input = np.array
    return np.linalg.inv(x)
