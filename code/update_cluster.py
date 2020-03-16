import numpy as np

def update_cluster(x, cluster):
    aux = x - cluster.centroid
    alpha = 1/(cluster.k+1)
    n = 1/cluster.k    
    transp = (cluster.inv_cov * aux).transpose()
    deno = 1 + alpha * (aux.transpose() * cluster.inv_cov * aux)

    inv_cov = cluster.inv_cov/(1-alpha) - alpha/(1-alpha) * (cluster.inv_cov * aux * transp)/deno

    cluster.inv_cov = inv_cov
    cluster.centroid = cluster.centroid + n * aux
    cluster.k += 1

    return cluster