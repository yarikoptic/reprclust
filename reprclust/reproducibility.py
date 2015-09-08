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
              sa_space='chunks', fa_space=None):
    """Run reproducibility algorithm on one fold for all the ks"""
    # initialize methods
    if not isinstance(data, Dataset):
        raise TypeError('Input must be a PyMVPA Dataset')
    if sa_space and fa_space:
        raise ValueError('At the moment only one of sa_space or fa_space can be specified, not together')

    cm_train = cluster_method
    cm_test = copy.deepcopy(cm_train)

    if sa_space:
        if sa_space not in data.sa.keys():
            raise KeyError('{0} is not present in data.sa: {1}'.format(sa_space, data.sa.keys()))
        data_train = data[np.in1d(data.sa[sa_space], train)]
        data_test = data[np.in1d(data.sa[sa_space], test)]
    else:
        if fa_space not in data.fa.keys():
            raise KeyError('{0} is not present in data.fa: {1}'.format(fa_space, data.fa.keys()))
        data_train = data[:, np.in1d(data.fa[fa_space], train)]
        data_test = data[:, np.in1d(data.fa[fa_space], test)]

    if fold_fx:
        data_train = fold_fx(data_train)
        data_test = fold_fx(data_test)
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
        cm_train.train(data_train.samples, k, compute_full=True)
        # cluster on test set
        cm_test.train(data_test.samples, k, compute_full=True)

        # predict
        predicted_label = cm_train.predict(data_test.samples, k)
        test_label = cm_test.predict(data_test.samples, k)

        # Step 2. Compute scores and store them
        for metric in cluster_metrics:
            result_fold[str(metric)][1, i_k] = \
                metric(predicted_label, test_label, data=data_test.samples, k=k)
            if ground_truth is not None:
                result_fold[str(metric) + '_gt'][1, i_k] = \
                    metric(predicted_label, ground_truth, data=data_test.samples, k=k)
    return result_fold


def reproducibility(data, splitter, cluster_method, ks, ground_truth=None,
                    fold_fx=None, cluster_metrics=(ARI(), AMI()),
                    sa_space='chunks', fa_space=None,
                    n_jobs=1, verbose=51):
    if not isinstance(ks, (list, np.ndarray)):
        raise ValueError('ks must be a list or numpy array')
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    fold = delayed(_run_fold)
    results = parallel(fold(data, train, test, cluster_method, ks,
                            ground_truth=ground_truth,
                            fold_fx=fold_fx,
                            cluster_metrics=cluster_metrics,
                            sa_space=sa_space, fa_space=fa_space)
                       for train, test in splitter)

    scores = {}
    # store everything together now
    for metric in results[0]:
        scores[metric] = np.hstack((res[metric] for res in results))

    return scores
