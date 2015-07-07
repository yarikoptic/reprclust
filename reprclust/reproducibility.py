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
import copy

from joblib import Parallel, delayed

import numpy as np

from reprclust.cluster_methods import ClusterMethod
from reprclust.cluster_metrics import ari, ami

# this must be outside to allow parallelization
def _run_fold(data, train, test, cluster_method, ks, stack=False,
              ground_truth=None, cluster_metrics=(ari, ami)):
    """Run reproducibility algorithm on one fold for all the ks"""
    # initialize methods
    cm_train = cluster_method
    cm_test = copy.deepcopy(cm_train)

    # XXX: this should change depending on the type of data
    data_train = [data[tr_idx] for tr_idx in train]
    data_test = [data[te_idx] for te_idx in test]

    if stack:
        data_train = np.vstack(data_train)
        data_test = np.vstack(data_train)
    else:
        data_train = np.mean(np.dstack(data_train), axis=-1)
        data_test = np.mean(np.dstack(data_test), axis=-1)

    # allocate storing dictionary
    result_fold = {}
    for metric in cluster_metrics:
        result_fold[metric.__name__] = np.vstack((ks, np.zeros(len(ks))))
        if ground_truth is not None:
            result_fold[metric.__name__ + '_gt'] = \
                np.vstack((ks, np.zeros(len(ks))))

    for i_k, k in enumerate(ks):
        # Step 1. Clustering on training/test set and prediction
        # cluster on training set
        cm_train.train(data_train, k, compute_full=True)
        # cluster on test set
        cm_test.train(data_test, k, compute_full=True)

        # predict
        predicted_label = cm_train.predict(data_test, k)
        test_label = cm_test.predict(data_test, k)

        # Step 2. Compute scores and store them
        for metric in cluster_metrics:
            result_fold[metric.__name__][1, i_k] = \
                metric(predicted_label, test_label)
            if ground_truth is not None:
                result_fold[metric.__name__ + '_gt'][1, i_k] = \
                    metric(predicted_label, ground_truth)
    return result_fold


def reproducibility(data, splitter, cluster_method, ks, ground_truth=None,
                    stack=False, cluster_metrics=(ari, ami),
                    n_jobs=1, verbose=51):
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    fold = delayed(_run_fold)
    results = parallel(fold(data, train, test, cluster_method, ks,
                            ground_truth=ground_truth,
                            stack=stack,
                            cluster_metrics=cluster_metrics)
                       for train, test in splitter)

    scores = {}
    # store everything together now
    for metric in results[0]:
        scores[metric] = np.hstack((res[metric] for res in results))

    return scores

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
        if not isinstance(cluster_method, ClusterMethod):
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

    def __repr__(self):
        args = [repr(arg) for arg in [self._data, self._splitter,
                                      self._cluster_method,
                                      self._ks]]
        args += ['{0}={1}'.format(key, repr(value))
                 for key, value in [('stack', self._stack),
                                    ('ground_truth', self._ground_truth)]]
        args += ['cluster_metrics=[' +
                 ', '.join([m.__name__ for m in self._cluster_metrics]) + ']']
        args = ", ".join(args)

        return self.__class__.__name__ + '({0})'.format(args)

    def run(self, n_jobs=1, verbose=52):
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        fold = delayed(_run_fold)
        result = parallel(fold(self, train, test)
                          for train, test in self._splitter)
        # store everything together now
        for metrics in result[0]:
            self.scores[metrics] = np.hstack((res[metrics] for res in result))

    @property
    def ks(self):
        return self._ks

    def get_ks_run(self):
        """Return the ks for the run of the clustering algorithm"""
        metric = self.get_metric_names()[0]
        return self.scores[metric][0].astype(int)

    def get_metric_score(self, metric):
        return self.scores[metric][1]

    def get_metric_names(self):
        return self.scores.keys()

    def get_header_scores(self):
        return ['k'] + self.get_metric_names()

    def get_array_scores(self):
        """Return an array containing the scores in a nice formatting"""
        metric_names = self.get_metric_names()

        scores = []
        for metric in metric_names:
            scores.append(self.get_metric_score(metric).reshape(-1, 1))

        ks = self.get_ks_run().reshape(-1, 1)
        metric_scores = np.hstack(scores)
        d = np.hstack((ks, metric_scores))
        return d
