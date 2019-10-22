#!/usr/bin/env python
# author: Florian Kreyssig flk24@cam.ac.uk
# Some of this code was taken from https://stackoverflow.com/a/5551499
"""
    Class CustKMeans performs KMeans clustering
    Any distance measure from scipy.spatial.distance can be used
"""
import numpy as np
from scipy.spatial.distance import cdist


def k_means(X, n_clusters, init=None, tol=.001,
            max_iter=10, custom_dist="euclidean", p=2):
    """
    X : array-like, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form
    init: ndarray, shaped (n_clusters, n_features), gives initial centroids
    max_iter : int, optional, default 10
        Maximum number of iterations of the k-means algorithm to run.
    tol : float, optional
        The relative increment in the results before declaring convergence.
    custom_dist: : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    p : scalar, optional
        The p-norm to apply (for Minkowski, weighted and unweighted)
    """
    def init_centroids():
        """Compute the initial centroids"""
        n_samples = X.shape[0]
        init_n_samples = max(2*np.sqrt(n_samples), 10*n_clusters)
        _X = np.random.choice(np.arange(X.shape[0]),
                              size=init_n_samples, replace=False)
        _init = np.random.choice(np.arange(X.shape[0]),
                                 size=n_clusters, replace=False)
        return k_means(
            _X, _init, max_iter=max_iter, custom_dist=custom_dist)[0]

    n_samples, n_features = X.shape
    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)
    if n_samples < n_clusters:
        raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            n_samples, n_clusters))
    if init is None:
        centres = init_centroids()
    else:
        n_centres, c_n_features = init.shape
        if n_centres != n_clusters:
            raise ValueError('The shape of the initial centers (%s)'
                             'does not match the number of clusters %d'
                             % (str(init.shape), n_clusters))
        if n_features != c_n_features:
            raise ValueError(
                "The number of features of the initial centers %d"
                "does not match the number of features of the data %d."
                % (c_n_features, n_features))
        centres = init.copy()

    sample_ids = np.arange(n_samples)
    prev_mean_dist = 0
    for iter_idx in range(1, max_iter+1):
        dist_to_all_centres = cdist(X, centres, metric=custom_dist, p=p)
        labels = dist_to_all_centres.argmin(axis=1)
        distances = dist_to_all_centres[sample_ids, labels]
        mean_distance = np.mean(distances)
        if (1 - tol) * prev_mean_dist <= mean_distance <= prev_mean_dist \
                or iter_idx == max_iter:
            break
        prev_mean_dist = mean_distance
        for each_center in range(n_centres):
            each_center_samples = np.where(labels == each_center)[0]
            if each_center_samples.any():
                centres[each_center] = np.mean(X[each_center_samples], axis=0)
    return centres, labels, distances


class CustKmeans:
    """
        Class to perform KMeans clustering.
        Can be used similar to sklearn.cluster Kmeans
    """

    def __init__(self, n_clusters=0, init=None, max_iter=10,
                 custom_dist='euclidean'):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.custom_dist = custom_dist
        self.centres = None

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        return k_means(X, self.n_clusters, init=self.init,
                       max_iter=self.max_iter, custom_dist=self.custom_dist)[1]
