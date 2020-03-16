from cluster import Cluster
from get_radius import get_radius

def new_cluster(x, fac, p, m):
    radius = get_radius(fac, p, m, 1)
    return Cluster(centroid = x, radius = radius)