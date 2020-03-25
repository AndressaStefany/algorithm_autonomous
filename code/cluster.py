import numpy as np


class Cluster:
    def __init__(self, centroid, inv_cov, k=1, S=[]):
        self.centroid = centroid
        self.inv_cov = inv_cov
        self.k = k
        self.S = S
    pass
