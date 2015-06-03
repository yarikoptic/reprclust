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
Hollinshead, M., et al. (2011).
"The organization of the human cerebral cortex estimated by intrinsic
functional connectivity."
Journal of Neurophysiology, 106(3), 1125–1165. doi:10.1152/jn.00338.2011

Lange, T., Roth, V., Braun, M. and Buhmann J. (2004)
"Stability-based validation of clustering solutions."
Neural computation 16, no. 6 (2004): 1299-1323.
"""
from joblib import Parallel, delayed

import numpy as np

from scipy.spatial.distance import pdist, hamming
from scipy.cluster.hierarchy import complete

from sklearn.cluster import KMeans
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.metrics.cluster import (adjusted_rand_score,
                                     adjusted_mutual_info_score,
                                     contingency_matrix)
from sklearn.mixture import GMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.linear_assignment_ import linear_assignment


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
                           ground_truth=None, **kwargs):
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


def compute_stability(splitter, samples, method='ward', stack=False,
                      cv_likelihood=False, max_k=None, n_jobs=1,
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
            Maximum k to compute the stability testing, starting from 2. By
            default it will compute up to the maximum possible k, i.e.,
            the number of points.
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


def get_optimal_permutation(a, b):
    """Finds an optimal permutation of the elements in b to match a,
    using the Hungarian algorithm.
    """
    w = contingency_matrix(a, b)
    # make it a cost matrix
    w = np.max(w) - w
    # minimize the cost
    permutation = linear_assignment(w)
    return permutation[:, 1]


def permute(b, permutation):
    """Permutes the element in b using the mapping defined in `permutation`.
    Value `i` in b is mapped to `permutation[i]`.
    """
    out = b.copy()
    for i, el in enumerate(out):
        out[i] = permutation[el]
    return out


def stability_score(predicted_label, test_label):
    """Computes the stability score (see Lange) for `predicted_label` and
    `test_label`.

    Note: order of the inputs matters: S(predicted, test) != S(test, predicted)
    """
    # find optimal permutation of labels between predicted and test
    test_label_ = permute(test_label,
                          get_optimal_permutation(predicted_label, test_label))
    # return hamming distance
    return hamming(predicted_label, test_label_)


def generate_random_labeling(k, size):
    """Generate a valid random labeling"""
    if size < k:
        raise ValueError('To have a labeling size cannot be lower than high')
    while True:
        a = np.random.randint(0, k, size)
        if len(np.unique(a)) == k:
            break
    return a


def norm_stability_score(predicted_label, test_label, s):
    """Computes the normalized stability score (see Lange) for
    `predicted_label` and  `test_label`. The stability score is normalized
    using a random labeling algorithm `R_k` (see original paper). `s` in the
    number of iterations of random labels to achieve an estimate of the random
    labeling.

    Note: order of the inputs matters: S(predicted, test) != S(test, predicted)
    """
    # get s random labelings and compute their score
    rand_score = 0
    for i in xrange(s):
        rand_label = generate_random_labeling(len(np.unique(test_label)),
                                              len(test_label))
        rand_score += stability_score(rand_label, test_label)
    rand_score /= s
    # normalize score
    score = stability_score(predicted_label, test_label)/rand_score
    return score
