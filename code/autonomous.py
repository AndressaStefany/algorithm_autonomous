import pandas as pd
import numpy as np
from utils import min_dist, is_empty_list, dist
from new_cluster import new_cluster
from update_clusters import update_winner_cluster, update_nearest_cluster
from overlap import overlap
from merge import merge
from radius import get_radius
from volume import get_volume


class Autonomous:
    def __init__(self, fac, frac=100, m=4):  # default of article
        self.fac = fac
        self.frac = frac
        self.m = m
        self.clusters = []

    pass

    # input x = np.array
    def process(self, x):
        p = len(x)

        # the first cluster
        if is_empty_list(self.clusters):
            self.clusters.append(new_cluster(
                x, self.frac, self.fac, p, self.m))

        else:
            # elect winning cluster
            win_cluster = min_dist(x, self.clusters)
            win_cluster_radius = get_radius(
                self.fac, p, self.m, win_cluster.k)

            # point is not contained in the winning cluster
            if dist(x, win_cluster) > win_cluster_radius:
                self.clusters.append(new_cluster(
                    x, self.frac, self.fac, p, self.m))

            else:
                self.clusters.remove(win_cluster)
                # update winning cluster
                update_winner_cluster(x, win_cluster)

                if len(self.clusters) != 0:
                    # update nearest neighbor of winning cluster
                    update_nearest_cluster(x, win_cluster, self.clusters)

                # check overlap
                (max_olap, max_cluster) = overlap(
                    win_cluster, self.clusters)
                if (max_olap > 0):
                    cluster_merged = merge(win_cluster, max_cluster)

                    V_m = get_volume(cluster_merged, self.fac, p, self.m)
                    V_w = get_volume(win_cluster, self.fac, p, self.m)
                    V_c = get_volume(max_cluster, self.fac, p, self.m)
                    print(V_m, V_w, V_c)
                    # Check volume, for the final decision
                    if V_m <= p*(V_w + V_c):
                        self.clusters.remove(max_cluster)
                        self.clusters.append(cluster_merged)
                    else:
                        self.clusters.append(win_cluster)
        pass
