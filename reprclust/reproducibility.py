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

from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fx import mean_group_sample

import numpy as np

from reprclust.cluster_metrics import ARI, AMI

# this must be outside to allow parallelization
def _run_fold(data, train, test, cluster_method, ks, fold_fx=None,
              ground_truth=None, cluster_metrics=(ARI(), AMI()),
              space='sa.subjects'):
    """Run reproducibility algorithm on one fold for all the ks"""
    if not isinstance(data, Dataset):
        raise TypeError('Input must be a PyMVPA Dataset')

    attr, attr_space = space.split('.')
    if attr_space not in getattr(data, attr).keys():
        raise KeyError('{0} is not present in data.{1}: {2}'.format(attr_space, attr,
                                                                    getattr(data, attr).keys()))
    # initialize methods
    cm_train = cluster_method
    cm_test = copy.deepcopy(cm_train)

    if attr == 'sa':
        data_train = data[np.in1d(data.sa[attr_space], train)]
        data_test = data[np.in1d(data.sa[attr_space], test)]
    elif attr == 'fa':
        data_train = data[:, np.in1d(data.fa[attr_space], train)]
        data_test = data[:, np.in1d(data.fa[attr_space], test)]
    else:
        raise ValueError('We should not get here')

    if fold_fx is None:
        fold_fx = lambda x, y: (x.samples, y.samples)

    # apply fold_fx and transpose because clustering methods cluster rows
    # while we want to cluster columns (features)
    samples_train, samples_test = fold_fx(data_train, data_test)
    samples_train = samples_train.T
    samples_test = samples_test.T

    # allocate storing dictionary
    result_fold = {}
    for metric in cluster_metrics:
        result_fold[str(metric)] = np.vstack((ks, np.zeros(len(ks))))
        if ground_truth is not None:
            result_fold[str(metric) + '_gt'] = \
                np.vstack((ks, np.zeros(len(ks))))

    for i_k, k in enumerate(ks):
        # Step 1. Clustering on training/test set and prediction
        # cluster on training set
        cm_train.train(samples_train, k, compute_full=True)
        # cluster on test set
        cm_test.train(samples_test, k, compute_full=True)

        # predict
        predicted_label = cm_train.predict(samples_test, k)
        test_label = cm_test.predict(samples_test, k)

        # Step 2. Compute scores and store them
        for metric in cluster_metrics:
            result_fold[str(metric)][1, i_k] = \
                metric(predicted_label, test_label, data=samples_test, k=k)
            if ground_truth is not None:
                result_fold[str(metric) + '_gt'][1, i_k] = \
                    metric(predicted_label, ground_truth, data=samples_test, k=k)
    return result_fold


def reproducibility(data, splitter, cluster_method, ks, ground_truth=None,
                    fold_fx=None, cluster_metrics=(ARI(), AMI()),
                    space='sa.subjects',
                    n_jobs=1, verbose=51):
    """
    Runs the reproducibility algorithm on the data.

    Arguments
    ---------
    data : mvpa2 Dataset
    splitter : generator or equivalent
    cluster_method : list of ClusterMethod from reprclust.cluster_methods
    ks : list or np.ndarray
    ground_truth : list or np.ndarray
    fold_fx : callable applied to (data_train, data_test) that returns a
        tuple of np.ndarray corresponding to the modified input
    cluster_metrics : list of ClusterMetric from reprclust.cluster_metrics
    space : str
        In the format of 'attr.attr_space' (e.g., 'sa.subjects'), where to apply
        the splitter.
    n_jobs : int
    verbose : int
    """
    if not isinstance(ks, (list, np.ndarray)):
        raise ValueError('ks must be a list or numpy array')
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    fold = delayed(_run_fold)
    results = parallel(fold(data, train, test, cluster_method, ks,
                            ground_truth=ground_truth,
                            fold_fx=fold_fx,
                            cluster_metrics=cluster_metrics,
                            space=space)
                       for train, test in splitter)

    scores = {}
    # store everything together now
    for metric in results[0]:
        scores[metric] = np.hstack((res[metric] for res in results))

    return scores
