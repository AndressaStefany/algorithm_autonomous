import pandas as pd
import numpy as np
from utils import min_dist, is_empty_list, dist
from new_cluster import new_cluster
from update_clusters import update_winner_cluster, update_nearest_cluster
from overlap import overlap
from merge import merge


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
        if is_empty_list(self.clusters):
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

                if len(self.clusters) != 0:
                    update_nearest_cluster(x, win_cluster, self.clusters)

                (max_olap, max_cluster) = overlap(
                    win_cluster, self.clusters)
                if (max_olap > 0):
                    self.clusters.remove(win_cluster)
                    self.clusters.remove(max_cluster)

                    new_cluster = merge(win_cluster, max_cluster)
                    self.clusters.append(new_cluster)
                else:
                    self.clusters.append(win_cluster)
        pass
