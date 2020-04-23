import numpy as np
from sklearn import mixture
from scipy import stats
from cluster import Cluster
from scipy import linalg
import cmath


def split(clusters, win_cluster):
    data = np.array(win_cluster.S.copy())

    gmm_1 = mixture.GaussianMixture(
        n_components=1, covariance_type='full').fit(data[:, 0].reshape(-1, 1))
    pi_1 = gmm_1.weights_[0]
    gmm_2 = mixture.GaussianMixture(
        n_components=1, covariance_type='full').fit(data[:, 1].reshape(-1, 1))
    pi_2 = gmm_2.weights_[0]

    I = (min(pi_1, pi_2)/max(pi_1, pi_2)) * stats.ttest_ind(data[:, 0], data[:, 1], equal_var=False)[
        0]

    if (True) & (gmm_1.bic(data[:, 0].reshape(-1, 1)) > gmm_2.bic(data[:, 1].reshape(-1, 1))) & (I > 2.57):
        mu_1 = gmm_1.means_[0][0]
        cov_1 = gmm_1.covariances_[0][0][0]
        mu_2 = gmm_2.means_[0][0]
        cov_2 = gmm_2.covariances_[0][0][0]

        if cov_1 == cov_2:
            cut = (mu_1 + mu_2)/2 + (cov_1**2 *
                                     (np.log(pi_1) - np.log(pi_2)))/(mu_1-mu_2)
            S_1 = data[(data[:, 0] <= cut) & (data[:, 1] <= cut)]
            S_2 = data[(data[:, 0] > cut) & (data[:, 1] > cut)]
        else:
            # It's considered that cut_1  to belong to gmm_1, consequently, cut_2 belongs to gmm_2
            d = 2 * cov_1**2 * cov_2**2 * \
                (np.log(pi_1) - np.log(pi_2)) + \
                mu_2**2 * cov_1**2 - mu_1**2 * cov_2**2
            d_aux = (mu_1 * cov_2**2 - mu_2 *
                     cov_1**2) - (cov_1**2 - cov_2**2) * d
            root = abs(cmath.sqrt(d_aux))

            cut_1 = (mu_2 * cov_1**2 + mu_1 * cov_2 **
                     2 + root) / (cov_1**2 - cov_2**2)
            cut_2 = (mu_2 * cov_1**2 + mu_1 * cov_2 **
                     2 - root) / (cov_1**2 - cov_2**2)

            S_1 = data[(data[:, 0] <= cut_1) & (data[:, 1] <= cut_2)]
            S_2 = data[(data[:, 0] > cut_1) & (data[:, 1] > cut_2)]
        if len(S_2) != 0:
            k_1 = win_cluster.k * (len(S_1)/(len(S_1) + len(S_2)))
            k_2 = win_cluster.k * (len(S_2)/(len(S_1) + len(S_2)))

            if (k_1 != 0) & (k_2 != 0):
                cluster_1 = Cluster(centroid=np.array([S_1[:, 0].mean(), S_1[:, 1].mean()]),
                                    inv_cov=linalg.inv(np.cov(S_1.T)),
                                    k=k_1,
                                    S=S_1.tolist())

                cluster_2 = Cluster(centroid=np.array([S_2[:, 0].mean(), S_2[:, 1].mean()]),
                                    inv_cov=linalg.inv(np.cov(S_2.T)),
                                    k=k_2,
                                    S=S_2.tolist())

                clusters.append(cluster_1)
                clusters.append(cluster_2)
        else:
            clusters.append(win_cluster)
    else:
        clusters.append(win_cluster)
    pass
