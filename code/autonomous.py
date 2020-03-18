import pandas as pd
import numpy as np
from utils import min_dist, is_empty_list, dist
from new_cluster import new_cluster
from update_clusters import update_winner_cluster, update_nearest_cluster


class Autonomous:
    def __init__(self, fac, frac=100, m=4):  # default of article
        self.fac = fac
        self.frac = frac
        self.m = m
        self.clusters = []

    pass

    # x = np.array
    def process(self, x):
        p = len(x)
        if is_empty_list:
            self.clusters.append(new_cluster(
                x, self.frac, self.fac, p, self.m))
        else:
            win_cluster = min_dist(x, self.clusters)
            if dist(x, win_cluster) > win_cluster.radius:
                self.clusters.append(new_cluster(
                    x, self.frac, self.fac, p, self.m))
            else:
                self.clusters.remove(win_cluster)
                update_winner_cluster(x, win_cluster)
                update_nearest_cluster(x, win_cluster, self.clusters)

                self.clusters.append(win_cluster)
                pass
        pass
