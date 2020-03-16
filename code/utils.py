from scipy.spatial import distance

def is_empty_list(list):
    return len(list) == 0

def dist(x, cluster):
    iv = 0
    return distance.mahalanobis(x, cluster.centroid, iv)

def min_dist(x, clusters):
    cluster_min_dist = float('inf')
    min_cluster = None
    iv = 0 #?
    for cluster in clusters:
        dist_to_cluster = distance.mahalanobis(x, cluster.centroid, iv)
        if dist_to_cluster < cluster_min_dist:
            cluster_min_dist = dist_to_cluster
            min_cluster = cluster
    
    return min_cluster