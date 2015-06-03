# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module to compute stability estimates of clustering solutions using a method
inspired by Yeo et al. (2011). The procedure cross-validates across
subjects in the following way: the datasets are divided into training and
test set; clustering is performed on the training set, and the solution is
predicted on the test set. Then, the clustering solution is computed on the
test set, and this is compared to the predicted one using the Adjusted Rand
Index or the Adjusted Mutual Information as metrics. Clustering solutions
are swept from k=2 to a maximum k defined by the user.

At the moment the following clustering algorithms are implemented:
    - k-means
    - Gaussian Mixture Models
    - Ward Clustering (structured and unstructured)
    - complete linkage with correlation distance

References:
-----------
Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., et al. (2011). The organization of the human cerebral cortex
estimated by intrinsic functional connectivity.
Journal of Neurophysiology, 106(3), 1125–1165. doi:10.1152/jn.00338.2011
"""
from joblib import Parallel, delayed

import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import complete

from sklearn.metrics.cluster import (adjusted_rand_score,
                                     adjusted_mutual_info_score)
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans


AVAILABLE_METHODS = ['ward', 'complete', 'gmm', 'kmeans']


def cut_tree_scipy(Y, k):
    """ Given the output Y of a hierarchical clustering solution from scipy
    and a number k, cuts the tree and returns the labels.
    """
    children = Y[:, 0:2].astype(int)
    # convert children to correct values for _hc_cut
    return _hc_cut(k, children, len(children)+1)


def compute_stability_fold(samples, train, test, method='ward',
                           max_k=None, stack=False, cv_likelihood=False,
                           ground_truth=None, n_neighbors=1, **kwargs):
    """
    General function to compute the stability on a cross-validation fold.
    
    Parameters:
    -----------
        samples : list of arrays
            List of arrays containing the samples to cluster, each
            array has shape (n_samples, n_features) in PyMVPA terminology.
            We are clustering the features, i.e., the nodes.
        train : list or array
            Indices for the training set.
        test : list or array
            Indices for the test set.
        method : {'complete', 'gmm', 'kmeans', 'ward'}
            Clustering method to use. Default is 'ward'.
        max_k : int or None
            Maximum k to compute the stability testing, starting from 2. By
            default it will compute up to the maximum possible k, i.e.,
            the number of points.
        stack : bool
            Whether to stack or average the datasets. Default is False,
            meaning that the datasets are averaged by default.
        cv_likelihood : bool
            Whether to compute the cross-validated likelihood for mixture
            model; only valid if 'gmm' method is used. Default is False.
        ground_truth : array or None
            Array containing the ground truth of the clustering of the data,
            useful to compare stability against ground truth for simulations.
        n_neighbors : int
            Number of neighbors to use to predict clustering solution on
            test set using K-nearest neighbors. Currently used only for
            methods `complete` and `ward`. Default is 1.
        kwargs : optional
            Keyword arguments being passed to the clustering method (only for
            'ward' and 'gmm').
    
    Returns:
    --------
        ks : array
            A (max_k-1,) array, where ks[i] is the `k` of the clustering
            solution for iteration `i`.
        ari : array
            A (max_k-1,) array, where ari[i] is the Adjusted Rand Index of the
            predicted clustering solution on the test set and the actual
            clustering solution of the test set for `k` of ks[i].
        ami : array
            A (max_k-1,) array, where ari[i] is the Adjusted Mutual
            Information of the predicted clustering solution on the test set
            and the actual clustering solution of the test set for
            `k` of ks[i].
        likelihood : array or None
            If method is 'gmm' and cv_likelihood is True, a
            (max_k-1,) array, where likelihood[i] is the cross-validated
            likelihood of the GMM clustering solution for `k` of ks[i].
            Otherwise returns None.
        ari_gt : array or None
            If ground_truth is not None, a (max_k-1,) array, where ari_gt[i]
            is the Adjusted Rand Index of the predicted clustering solution on
            the test set for `k` of ks[i] and the ground truth clusters of the
            data.
            Otherwise returns None.
        ami_gt : array or None
            If ground_truth is not None, a (max_k-1,) array, where ami_gt[i]
            is the Adjusted Mutual Information of the predicted clustering
            solution on the test set for `k` of ks[i] and the ground truth
            clusters of the data.
            Otherwise returns None.
    """
    if method not in AVAILABLE_METHODS:
        raise ValueError('Method {0} not implemented'.format(method))

    if cv_likelihood and method != 'gmm':
        raise ValueError(
            "Cross-validated likelihood is only available for 'gmm' method")

    # if max_k is None, set max_k to maximum value
    if not max_k:
        max_k = samples[0].shape[1]

    # preallocate arrays for results
    ks = np.zeros(max_k-1, dtype=int)
    ari = np.zeros(max_k-1)
    ami = np.zeros(max_k-1)
    if cv_likelihood:
        likelihood = np.zeros(max_k-1)
    if ground_truth is not None:
        ari_gt = ami_gt = np.zeros(max_k-1)

    # get training and test
    train_set = [samples[x] for x in train]
    test_set = [samples[x] for x in test]
    
    if stack:
        train_ds = np.vstack(train_set)
        test_ds = np.vstack(test_set)
    else:
        train_ds = np.mean(np.dstack(train_set), axis=2)
        test_ds = np.mean(np.dstack(test_set), axis=2)

    # compute clustering on training set
    if method == 'complete':
        train_ds_dist = pdist(train_ds.T, metric='correlation')
        test_ds_dist = pdist(test_ds.T, metric='correlation')
        # I'm computing the full tree and then cutting
        # afterwards to speed computation
        Y_train = complete(train_ds_dist)
        # same on testing set
        Y_test = complete(test_ds_dist)
    elif method == 'ward':
        (children_train, n_comp_train, 
         n_leaves_train, parents_train) = ward_tree(train_ds.T, **kwargs)
        # same on testing set
        (children_test, n_comp_test, 
         n_leaves_test, parents_test) = ward_tree(test_ds.T, **kwargs)
    elif method == 'gmm' or method == 'kmeans':
        pass  # we'll have to run it for each k
    else:
        raise ValueError("We shouldn't get here")

    for i_k, k in enumerate(range(2, max_k+1)):
        if method == 'complete':
            # cut the tree with right K for both train and test
            train_label = cut_tree_scipy(Y_train, k)
            test_label = cut_tree_scipy(Y_test, k)
            # train a classifier on this clustering
            knn = KNeighborsClassifier(#algorithm='brute',
            # metric='correlation',
                                       n_neighbors=n_neighbors)
            knn.fit(train_ds.T, train_label)
            # predict the clusters in the test set
            prediction_label = knn.predict(test_ds.T)
        elif method == 'ward':
            # cut the tree with right K for both train and test
            train_label = _hc_cut(k, children_train, n_leaves_train)
            test_label = _hc_cut(k, children_test, n_leaves_test)
            # train a classifier on this clustering
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(train_ds.T, train_label)
            # predict the clusters in the test set
            prediction_label = knn.predict(test_ds.T)
        elif method == 'gmm':
            gmm = GMM(n_components=k, **kwargs)
            # fit on train and predict test
            gmm.fit(train_ds.T)
            prediction_label = gmm.predict(test_ds.T)
            if cv_likelihood:
                log_prob = np.sum(gmm.score(test_ds.T))
            # fit on test and get labels
            gmm.fit(test_ds.T)
            test_label = gmm.predict(test_ds.T)
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=k)
            # fit on train and predict test
            kmeans.fit(train_ds.T)
            prediction_label = kmeans.predict(test_ds.T)
            # fit on test and get labels
            kmeans.fit(test_ds.T)
            test_label = kmeans.predict(test_ds.T)
        else:
            raise ValueError("We shouldn't get here")
            
        # append results
        ks[i_k] = k
        ari[i_k] = adjusted_rand_score(prediction_label, test_label)
        ami[i_k] = adjusted_mutual_info_score(prediction_label, test_label)
        if cv_likelihood:
            likelihood[i_k] = log_prob
        if ground_truth is not None:
            ari_gt[i_k] = adjusted_rand_score(prediction_label, ground_truth)
            ami_gt[i_k] = adjusted_mutual_info_score(prediction_label,
                                                     ground_truth)

    results = [ks, ari, ami]
    if cv_likelihood:
        results.append(likelihood)
    else:
        results.append(None)

    if ground_truth is not None:
        results += [ari_gt, ami_gt]
    else:
        results += [None, None]

    return results


def compute_stability(splitter, samples, method='ward', max_k=None,
                      stack=False, cv_likelihood=False, ground_truth=None,
                      n_neighbors=1, n_jobs=1, verbose=51, **kwargs):
    """
    General function to compute the stability of clustering on a list of
    datasets.

    Parameters:
    -----------
        splitter
            A generator of training and test set, usually from
            sklearn.cross_validation.
        samples : list of arrays
            List of arrays containing the samples to cluster, each
            array has shape (n_samples, n_features) in PyMVPA terminology.
            We are clustering the features, i.e., the nodes.
        method : {'complete', 'gmm', 'kmeans', 'ward'}
            Clustering method to use. Default is 'ward'.
        max_k : int or None
            Maximum k to compute the stability testing, starting from 2. By
            default it will compute up to the maximum possible k, i.e.,
            the number of points.
        stack : bool
            Whether to stack or average the datasets. Default is False,
            meaning that the datasets are averaged by default.
        cv_likelihood : bool
            Whether to compute the cross-validated likelihood for mixture
            model; only valid if 'gmm' method is used. Default is False.
        ground_truth : array or None
            Array containing the ground truth of the clustering of the data,
            useful to compare stability against ground truth for simulations.
        n_neighbors : int
            Number of neighbors to use to predict clustering solution on
            test set using K-nearest neighbors. Currently used only for
            methods `complete` and `ward`. Default is 1.
        n_jobs : int
            Number of jobs (cores) to run the algorithm on. Default is 1.
        verbose : int
            Level of verbosity to be passed to Parallel. Default is 51,
            i.e. maximum verbosity.
        kwargs : optional
            Keyword arguments being passed to the clustering method (only for
            'ward' and 'gmm').

    Returns:
    --------
        ks : array
            A (max_k-1,) array, where ks[i] is the `k` of the clustering
            solution for iteration `i`.
        ari : array
            A (max_k-1,) array, where ari[i] is the Adjusted Rand Index of the
            predicted clustering solution on the test set and the actual
            clustering solution of the test set for `k` of ks[i].
        ami : array
            A (max_k-1,) array, where ari[i] is the Adjusted Mutual
            Information of the predicted clustering solution on the test set
            and the actual clustering solution of the test set for
            `k` of ks[i].
        likelihood : array or None
            If method is 'gmm' and cv_likelihood is True, a
            (max_k-1,) array, where likelihood[i] is the cross-validated
            likelihood of the GMM clustering solution for `k` of ks[i].
            Otherwise returns None.
        ari_gt : array or None
            If ground_truth is not None, a (max_k-1,) array, where ari_gt[i]
            is the Adjusted Rand Index of the predicted clustering solution on
            the test set for `k` of ks[i] and the ground truth clusters of the
            data.
            Otherwise returns None.
        ami_gt : array or None
            If ground_truth is not None, a (max_k-1,) array, where ami_gt[i]
            is the Adjusted Mutual Information of the predicted clustering
            solution on the test set for `k` of ks[i] and the ground truth
            clusters of the data.
            Otherwise returns None.
    """

    result = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
        compute_stability_fold)(samples, train, test, method=method,
                                max_k=max_k, stack=stack,
                                cv_likelihood=cv_likelihood,
                                ground_truth=ground_truth,
                                n_neighbors=n_neighbors,
                                **kwargs) for train, test in splitter)
    ks = []
    ari = []
    ami = []
    likelihood = []
    ari_gt = []
    ami_gt = []

    for r in result:
        ks.append(r[0])
        ari.append(r[1])
        ami.append(r[2])

        if r[3] is not None:
            likelihood.append(r[3])
        else:
            likelihood = None

        if r[4] is not None:
            ari_gt.append(r[4])
        else:
            ari_gt = None

        if r[5] is not None:
            ami_gt.append(r[5])
        else:
            ami_gt = None

    ks = np.array(ks).ravel()
    ari = np.array(ari).ravel()
    ami = np.array(ami).ravel()
    if likelihood is not None:
        likelihood = np.array(likelihood).ravel()
    if ari_gt is not None:
        ari_gt = np.array(likelihood).ravel()
    if ami_gt is not None:
        ami_gt = np.array(likelihood).ravel()

    return ks, ari, ami, likelihood, ari_gt, ami_gt
