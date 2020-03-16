import pandas as pd
import numpy as np
from utils import min_dist, is_empty_list, dist
from new_cluster import new_cluster

class Autonomous:
    def __init__(self, fac, m = 4):
        self.fac = fac
        self.m = m
        self.clusters = []
        self. p = 0 # como é setado?

    pass

    # x = np.array
    def process(self, x):
        if is_empty_list:
            clusters.append(new_cluster(x, self.fac, self.p, self.m))
        else:
            win_cluster = min_dist(x, self.clusters)
            if dist(x, win_cluster) > win_cluster.radius:
                clusters.append(new_cluster(x, self.fac, self.p, self.m))
            else:
                #update cluster
                pass
        pass