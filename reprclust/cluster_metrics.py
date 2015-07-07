# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module containing metrics to compare cluster solutions
"""
import numpy as np

from scipy.spatial.distance import hamming
from scipy.stats import rankdata

from sklearn.metrics.cluster import (adjusted_rand_score,
                                     adjusted_mutual_info_score)
from sklearn.utils.linear_assignment_ import linear_assignment

# functions
def _get_optimal_permutation(a, b, k):
    """Finds an optimal permutation `p` of the elements in b to match a,
    using the Hungarian algorithm, such that `a ~ p(b)`. `k` specifies the
    number of possible labels, in case they're not all present in `a` or `b`.
    """
    assert(a.shape == b.shape)
    # if we're testing against ground truth, then we need to store a bigger
    # matrix
    n = max(len(np.unique(a)), len(np.unique(b)), k)
    w = np.zeros((n, n), dtype=int)
    for i, j in zip(a, b):
        w[i, j] += 1
    # make it a cost matrix
    w = np.max(w) - w
    # minimize the cost -- transpose because we want a map from `b` to `a`
    permutation = linear_assignment(w.T)
    return permutation[:, 1]


def _permute(b, permutation):
    """Permutes the element in b using the mapping defined in `permutation`.
    Value `i` in b is mapped to `permutation[i]`.
    """
    out = b.copy()
    for i, el in enumerate(out):
        out[i] = permutation[el]
    return out


class ClusterMetric(object):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, test_label, predicted_label, data=None, k=None):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ARI(ClusterMetric):
    def __init__(self):
        super(ARI, self).__init__()

    def __call__(self, test_label, predicted_label, **kwargs):
        return adjusted_rand_score(test_label, predicted_label)


class AMI(ClusterMetric):
    def __init__(self):
        super(AMI, self).__init__()

    def __call__(self, test_label, predicted_label, **kwargs):
        return adjusted_mutual_info_score(test_label, predicted_label)


class CorrelationScore(ClusterMetric):
    def __init__(self, corr_type='pearson'):
        super(CorrelationScore, self).__init__()
        self._corr_type = corr_type

    def __call__(self, test_label, predicted_label, data=None, *args,
                 **kwargs):
        if data is None:
            raise ValueError('CorrelationScore needs data to compute '
                             'correlation of')
        return self._correlation_score(predicted_label, test_label, data,
                                       corr_type=self._corr_type)

    @staticmethod
    def _correlation(x, y):
        """Compute correlation of vectors x and y"""
        c1 = x - x.mean()
        c2 = y - y.mean()
        return np.dot(c1, c2)/np.sqrt(((c1**2).sum() * (c2**2).sum()))

    def _correlation_score(self, predicted_label, test_label, data,
                           corr_type='pearson'):
        """Computes the correlation between the average RDMs in each
        corresponding cluster.
        """
        unique_test = np.unique(test_label)
        unique_pred = np.unique(predicted_label)
        # if we have completely different labels, complain
        if not set(unique_test).intersection(unique_pred):
            raise ValueError("predicted_label and test_label have completely "
                         "different labellings, I don't know what to do with "
                         "this.")
        # get permutation to match predicted_label to test_label
        k = max(len(np.unique(predicted_label)), len(np.unique(test_label)),
                np.max(predicted_label) + 1, np.max(test_label) + 1)
        perm = _get_optimal_permutation(test_label, predicted_label, k)
        # permute
        predicted_label = _permute(predicted_label, perm)
        # compute correlation across corresponding clusters
        corr = 0
        # to account for the case in which I'm testing against ground truth,
        # I need to cycle through the intersection of unique labels
        # recompute the unique labels after permutation
        unique_test = np.unique(test_label)
        unique_pred = np.unique(predicted_label)
        labels = np.intersect1d(unique_test, unique_pred, assume_unique=True)
        for i in labels:
            assert(len(np.unique(data[:, test_label == i])) > 0)
            assert(len(np.unique(data[:, predicted_label == i])) > 0)
            c1 = np.mean(data[:, test_label == i], axis=-1)
            c2 = np.mean(data[:, predicted_label == i], axis=-1)
            if corr_type == 'pearson':
                corr += self._correlation(rankdata(c1), rankdata(c2))
            elif corr_type == 'spearman':
                corr += self._correlation(c1, c2)
            else:
                raise ValueError("We shouldn't get here.")
        corr /= len(labels)

        return corr


class InstabilityScore(ClusterMetric):
    def __init__(self, normalization_scores):
        """Initialize with a dictionary of normalization scores"""
        super(InstabilityScore, self).__init__()
        self._normalization_scores = normalization_scores

    def __call__(self, test_label, predicted_label, k=None, **kwargs):
        if k is None:
            raise ValueError('Need to provide a k for instability score')
        if k not in self._normalization_scores:
            raise ValueError('Provided normalization scores do not have '
                             'selected value of k {0}'.format(k))

        return _instability_score(predicted_label, test_label,
                                  k)/self._normalization_scores[k]


def _instability_score(predicted_label, test_label, k):
    """Computes the stability score (see Lange) for `predicted_label` and
    `test_label` assuming `k` possible labels.
    """
    # find optimal permutation of labels between predicted and test
    test_label_ = _permute(test_label,
                           _get_optimal_permutation(predicted_label,
                                                    test_label, k))
    # return hamming distance
    return hamming(predicted_label, test_label_)


def rand_instability_score(ks, n, s):
    """Generates a score for random k-labellings of length n based on s
    iterations. Used as denominator to normalize the stability score.
    ks can be a numpy array of ks, and a dictionary is returned.
    """
    # get s random labellings and compute their score
    rand_scores = {}

    for k in ks:
        rand_score = 0
        for i in xrange(s):
            rand_label1 = np.random.randint(0, k, n)
            rand_label2 = np.random.randint(0, k, n)
            rand_score += _instability_score(rand_label1, rand_label2, k)
        rand_score /= s
        rand_scores[k] = rand_score
    return rand_scores
