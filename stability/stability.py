# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module to compute stability estimates of clustering solutions using a method
inspired by Yeo et al. (2011). At the moment the following clustering
algorithms are implemented:
    - kmeans
    - Gaussian Mixture Models
    - Ward Clustering (structured and unstructured)
    - complete with correlation distance

References:
-----------
Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., et al. (2011). The organization of the human cerebral cortex
estimated by intrinsic functional connectivity.
Journal of Neurophysiology, 106(3), 1125–1165. doi:10.1152/jn.00338.2011
"""
import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import complete

from sklearn.metrics.cluster import (adjusted_rand_score,
                                     adjusted_mutual_info_score)
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans

from joblib import Parallel, delayed

AVAILABLE_METHODS = ['ward', 'complete', 'gmm', 'kmeans']


def cut_tree_scipy(Y, k):
    """ Given the output Y of a hierarchical clustering solution from scipy
    and a number k, cuts the tree and returns the labels.
    """
    children = Y[:, 0:2].astype(int)
    # convert children to correct values for _hc_cut
    return _hc_cut(k, children, len(children)+1)


def compute_stability_fold(samples, train, test, method='ward',
                           max_k=300, stack=False, cv_likelihood=False,
                           **kwargs):
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
        max_k : int
            Maximum k to compute the stability testing, starting from 2.
        stack : bool
            Whether to stack or average the datasets. Default is False,
            meaning that the datasets are averaged by default.
        cv_likelihood : bool
            Whether to compute the cross-validated likelihood for mixture
            model; only valid if 'gmm' method is used. Default is False.
        kwargs : optional
            Keyword arguments being passed to the clustering method (only for
            'ward' and 'gmm').
    
    Returns:
    --------
        result: array
            A (max_k-1, 3) array, where result[:, 0] is the Adjusted Rand
            Index, result[:, 1] is the Adjusted Mutual Information, and
            result[:,  2] is the corresponding k.
            If method is 'gmm' and cv_likelihood is True, it returns a
            (max_k-1, 4) array, where result[:, 3] is the cross-validated
            likelihood.
        
    """    
    if method not in AVAILABLE_METHODS:
        raise ValueError('Method {0} not implemented'.format(method))

    if cv_likelihood and method != 'gmm':
        raise ValueError(
            "Cross-validated likelihood is only available for 'gmm' method")
    
    # preallocate matrix for results
    if cv_likelihood:
        result = np.zeros((max_k-1, 4))
    else:
        result = np.zeros((max_k-1, 3))

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
            knn = KNeighborsClassifier(algorithm='brute', metric='correlation')
            knn.fit(train_ds.T, train_label)
            # predict the clusters in the test set
            prediction_label = knn.predict(test_ds.T)
        elif method == 'ward':
            # cut the tree with right K for both train and test
            train_label = _hc_cut(k, children_train, n_leaves_train)
            test_label = _hc_cut(k, children_test, n_leaves_test)
            # train a classifier on this clustering
            knn = KNeighborsClassifier()
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
            kmeans = KMeans(n_clusters=k, **kwargs)
            # fit on train and predict test
            kmeans.fit(train_ds.T)
            prediction_label = kmeans.predict(test_ds.T)
            # fit on test and get labels
            kmeans.fit(test_ds.T)
            test_label = kmeans.predict(test_ds.T)
        else:
            raise ValueError("We shouldn't get here")
            
        # append results
        result[i_k, 0] = adjusted_rand_score(prediction_label, test_label)
        result[i_k, 1] = adjusted_mutual_info_score(prediction_label,
                                                    test_label)
        if cv_likelihood:
            result[i_k, -1] = log_prob

    result[:, 2] = range(2, max_k+1)
    return result


def compute_stability(splitter, samples, method='ward', stack=False,
                      cv_likelihood=False, max_k=300, n_jobs=1,
                      verbose=51, **kwargs):
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
        train : list or array
            Indices for the training set.
        test : list or array
            Indices for the test set.
        method : {'complete', 'gmm', 'kmeans', 'ward'}
            Clustering method to use. Default is 'ward'.
        max_k : int
            Maximum k to compute the stability testing, starting from 2.
        stack : bool
            Whether to stack or average the datasets. Default is False,
            meaning that the datasets are averaged by default.
        cv_likelihood : bool
            Whether to compute the cross-validated likelihood for mixture
            model; only valid if 'gmm' method is used. Default is False.
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
        result: array
            A (max_k-1, 3) array, where result[:, 0] is the Adjusted Rand
            Index, result[:, 1] is the Adjusted Mutual Information, and
            result[:,  2] is the corresponding k.
            If method is 'gmm' and cv_likelihood is True, it returns a
            (max_k-1, 4) array, where result[:, 3] is the cross-validated
            likelihood.

    """

    result = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
        compute_stability_fold)(samples, train, test, method=method,
                                max_k=max_k, stack=stack,
                                cv_likelihood=cv_likelihood,
                                **kwargs) for train, test in splitter)
    result = np.vstack(result)
    return result
