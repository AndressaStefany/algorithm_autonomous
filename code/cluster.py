class Cluster:
    def __init__(self, centroid, inv_cov = 0, radius = 0, k = 1):
        self.centroid = centroid
        self.inv_cov = inv_cov
        self.radius = radius
        self.k = k
    pass