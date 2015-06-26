# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module to compute stability estimates of clustering solutions using a method
inspired by Lange et al. (2004), and Yeo et al. (2011). The procedure
cross-validates across subjects in the following way: the datasets are divided
into training and test set; clustering is performed on the training set, and
the solution is predicted on the test set. Then, the clustering solution is
computed on the test set, and this is compared to the predicted one using the
Adjusted Rand Index, the Adjusted Mutual Information, and the Instability Score
(Lange et. al, 2004) as metrics. Clustering solutions are swept from k=2 to a
maximum k defined by the user.

At the moment the following clustering algorithms are implemented:
    - k-means
    - Gaussian Mixture Models
    - Ward Clustering (structured and unstructured)
    - complete linkage with correlation distance

References:
-----------
Lange, T., Roth, V., Braun, M. and Buhmann J. (2004)
"Stability-based validation of clustering solutions."
Neural computation 16, no. 6 (2004): 1299-1323.

Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., et al. (2011).
"The organization of the human cerebral cortex estimated by intrinsic
functional connectivity."
Journal of Neurophysiology, 106(3), 1125–1165. doi:10.1152/jn.00338.2011
"""
from joblib import Parallel, delayed

import numpy as np

from reprclust.cluster_methods import ClusterMethod
from reprclust.cluster_metrics import ari, ami

# this must be outside the class to allow parallelization
def _run_fold(self, train, test):
    """Run reproducibility algorithm on one fold for all the ks"""
    # initialize methods
    cm_train = self._cluster_method()
    cm_test = self._cluster_method()

    # XXX: this should change depending on the type of data
    # link clustering method to data
    data_train = [self._data[tr_idx] for tr_idx in train]
    data_test = [self._data[te_idx] for te_idx in test]

    if self._stack:
        data_train = np.vstack(data_train)
        data_test = np.vstack(data_train)
    else:
        data_train = np.mean(np.dstack(data_train), axis=-1)
        data_test = np.mean(np.dstack(data_test), axis=-1)

    cm_train(data_train)
    cm_test(data_test)

    # allocate storing dictionary
    result_fold = {}
    for metric in self._cluster_metrics:
        result_fold[metric.__name__] = \
            np.vstack((self._ks, np.zeros(len(self._ks))))
        if self._ground_truth is not None:
            result_fold[metric.__name__ + '_gt'] = \
                np.vstack((self._ks, np.zeros(len(self._ks))))

    # Step 1. Clustering on training/test set and prediction
    for k in self._ks:
        # cluster on training set
        cm_train.cluster(k)
        # use this clustering to predict test set
        cm_train.predict(data_test, k)
        # cluster on test set
        cm_test.cluster(k)

    # Step 2. Compute scores and store them
    for i_k, k in enumerate(self._ks):
        predicted_label = cm_train.get_predicted_clusters(k)
        test_label = cm_test.get_clusters(k)
        for metric in self._cluster_metrics:
            result_fold[metric.__name__][1, i_k] = \
                metric(predicted_label, test_label)
            if self._ground_truth is not None:
                result_fold[metric.__name__ + '_gt'][1, i_k] = \
                    metric(predicted_label, self._ground_truth)
    return result_fold


class Reproducibility(object):
    """Class to compute reproducibility on a dataset

    Attributes
    ----------
    data : list of arrays or PyMVPA Dataset (not yet implemented)
        List of arrays containing the samples to cluster, each
        array has shape (n_samples, n_features) in PyMVPA terminology.
        Alternatively, a PyMVPA Dataset with subjects as additional sample
        attribute.
        We are clustering the features, i.e., the nodes.
    splitter
        A generator of training and test set, usually from
        sklearn.cross_validation.
    cluster_methods : ClusterMethod
        Clustering method that will be run on the data.
    ks : list of int or ndarray or int
        List or array of ks to compute cluster solution of. If ks is an int,
        cluster solution will be computed from 2 to ks.
    stack : bool
        Whether to stack or average the datasets. Default is False,
        meaning that the datasets are averaged by default.
    ground_truth : array or None
        Array containing the ground truth of the clustering of the data,
        useful to compare stability against ground truth for simulations.
    cluster_metrics : list of callables
        List of functions to be applied on the clustering solutions to
        compare stability. Default are ARI (Adjusted Rand Score), and
        AMI (Adjusted Mutual Information).
    """
    def __init__(self, data, splitter, cluster_method, ks, stack=False,
                 ground_truth=None, cluster_metrics=(ari, ami)):
        # XXX: this should be decided how to allow already instatiated objects
        if not isinstance(cluster_method(), ClusterMethod):
            raise ValueError('cluster_method must be an instance of '
                             'ClusterMethod')
        if np.max(ks) > data[0].shape[0]:
            raise ValueError('Cannot find more cluster than number of '
                             'things to cluster. Got max k = {0} with number'
                             ' of features {1}'.format(np.max(ks),
                                                      data[0].shape[0]))
        self._data = data
        self._splitter = splitter
        self._cluster_method = cluster_method
        if isinstance(ks, int):
            self._ks = range(2, ks + 1)
        else:
            self._ks = ks
        self._stack = stack
        self._ground_truth = ground_truth
        self._cluster_metrics = cluster_metrics

        # we are going to store everything in a dictionary where keys are
        # the names of the cluster_metrics
        self.scores = {}
        for metric in cluster_metrics:
            self.scores[metric.__name__] = None
            # also add keys with ground truth if we have it
            if ground_truth is not None:
                self.scores[metric.__name__ + '_gt'] = None

    def run(self, n_jobs=1, verbose=52):
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        fold = delayed(_run_fold)
        result = parallel(fold(self, train, test)
                          for train, test in self._splitter)
        # store everything together now
        for metrics in result[0]:
            self.scores[metrics] = np.hstack((res[metrics] for res in result))
