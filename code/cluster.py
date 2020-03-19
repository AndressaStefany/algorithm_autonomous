import numpy as np


class Cluster:
    def __init__(self, centroid, inv_cov, radius=0, k=1, S=[]):
        self.centroid = centroid
        self.inv_cov = inv_cov
        self.radius = radius
        self.k = k
        self.S = S
    pass
