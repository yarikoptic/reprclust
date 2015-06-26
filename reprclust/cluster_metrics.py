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

# alias
ari = adjusted_rand_score
ami = adjusted_mutual_info_score

def get_optimal_permutation(a, b, k):
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


def permute(b, permutation):
    """Permutes the element in b using the mapping defined in `permutation`.
    Value `i` in b is mapped to `permutation[i]`.
    """
    out = b.copy()
    for i, el in enumerate(out):
        out[i] = permutation[el]
    return out


def stability_score(predicted_label, test_label, k):
    """Computes the stability score (see Lange) for `predicted_label` and
    `test_label` assuming `k` possible labels.
    """
    # find optimal permutation of labels between predicted and test
    test_label_ = permute(test_label,
                          get_optimal_permutation(predicted_label,
                                                  test_label, k))
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


def rand_stability_score(k, n, s):
    """Generates a score for random k-labellings of length n based on s
    iterations. Used as denominator to normalize the stability score.
    """
    # get s random labellings and compute their score
    rand_score = 0
    for i in xrange(s):
        # TODO: optimize this to generate fewer random labellings
        #rand_label1 = generate_random_labeling(k, n)
        #rand_label2 = generate_random_labeling(k, n)
        rand_label1 = np.random.randint(0, k, n)
        rand_label2 = np.random.randint(0, k, n)
        rand_score += stability_score(rand_label1, rand_label2, k)
    rand_score /= s
    return rand_score


def norm_stability_score(predicted_label, test_label, rand_score, k):
    """Computes the normalized stability score (see Lange) for
    `predicted_label` and  `test_label`. The stability score is normalized
    using a random labeling algorithm `R_k` (see original paper).
    """
    assert 0. <= rand_score <= 1., 'rand_score of {0} is not a valid ' \
                                   'distance'.format(rand_score)
    # compute score and normalize it
    score = stability_score(predicted_label, test_label, k)/rand_score
    return score


def correlation(x, y):
    """Compute correlation of vectors x and y"""
    c1 = x - x.mean()
    c2 = y - y.mean()
    return np.dot(c1, c2)/np.sqrt(((c1**2).sum() * (c2**2).sum()))


def correlation_score(predicted_label, test_label, data, corr_score='pearson'):
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
    perm = get_optimal_permutation(test_label, predicted_label, k)
    # permute
    predicted_label = permute(predicted_label, perm)
    # compute correlation across corresponding clusters
    corr = 0
    # to account for the case in which I'm testing against ground truth,
    # I need to cycle through the intersection of unique labels
    # recompute the unique labels after permutation
    unique_test = np.unique(test_label)
    unique_pred = np.unique(predicted_label)
    #    if len(unique_test) <= len(unique_pred):
    #        labels = unique_test
    #    else:
    #        labels = unique_pred
    labels = np.intersect1d(unique_test, unique_pred, assume_unique=True)
    for i in labels:
        assert(len(np.unique(data[:, test_label == i])) > 0)
        assert(len(np.unique(data[:, predicted_label == i])) > 0)
        c1 = np.mean(data[:, test_label == i], axis=-1)
        c2 = np.mean(data[:, predicted_label == i], axis=-1)
        if corr_score == 'pearson':
            corr += correlation(rankdata(c1), rankdata(c2))
        elif corr_score == 'spearman':
            corr += correlation(c1, c2)
        else:
            raise ValueError("We shouldn't get here.")
    corr /= len(labels)

    return corr
