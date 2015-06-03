from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_less_equal, assert_greater_equal
from numpy.testing import assert_array_equal
import numpy as np
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
from stability.stability import (cut_tree_scipy, compute_stability_fold,
                                 get_optimal_permutation, permute,
                                 generate_random_labeling,
                                 stability_score, norm_stability_score)

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
data = np.vstack((blob1, blob2))


def test_cut_tree_scipy():
    y = pdist(data, metric='euclidean')
    z = complete(y)
    assert_array_equal(np.sort(cut_tree_scipy(z, 2)),
                       np.hstack((np.zeros(10), np.ones(10))))
    assert_equal(len(np.unique(cut_tree_scipy(z, 10))), 10)


def test_compute_stability_fold():
    # add some noise and transpose
    dss = [(data + np.random.randn(*data.shape)).T for i in xrange(10)]
    # fake test, we should get a value of 1 for k=2
    idx_train = idx_test = range(10)
    result = compute_stability_fold(dss, idx_train, idx_test, max_k=20)
    assert_array_equal(result[1][0], [1.])

    ground_truth = np.hstack((np.zeros(10), np.ones(10)))
    # smoke test with all possible parameters
    for method in ['ward', 'complete', 'kmeans', 'gmm']:
        for stack in [True, False]:
            for cv_likelihood in [True, False]:
                if cv_likelihood and method != 'gmm':
                    assert_raises(ValueError, compute_stability_fold,
                                  dss, idx_train, idx_test,
                                  max_k=20, method=method,
                                  stack=stack, cv_likelihood=cv_likelihood)
                else:
                    result = compute_stability_fold(
                        dss, idx_train, idx_test, max_k=20, method=method,
                        stack=stack, cv_likelihood=cv_likelihood,
                        ground_truth=ground_truth)
                    assert_true(len(result), 6)


def test_generate_random_labeling():
    a = generate_random_labeling(5, 10)
    assert_equal(len(np.unique(a)), 5)
    assert_raises(ValueError, generate_random_labeling, 5, 3)


def test_permute():
    a = np.arange(9)
    b = np.arange(9)[::-1]
    assert_array_equal(a, permute(b, b))


def test_get_optimal_permutation():
    # simple permutation
    a = np.array([0, 1, 2, 3])
    b = np.array([3, 2, 1, 0])
    perm_real = b
    assert_array_equal(perm_real, get_optimal_permutation(a, b))
    # no permutation
    assert_array_equal(a, get_optimal_permutation(a, a))
    # random perms
    for k in xrange(20):
        a = generate_random_labeling(5, 10)
        perm_real = np.arange(len(np.unique(a)))
        np.random.shuffle(perm_real)
        b = a.copy()
        for i in xrange(len(b)):
            b[i] = perm_real[b[i]]
        assert_array_equal(perm_real, get_optimal_permutation(a, b))


def test_stability_score():
    a = np.arange(9)
    b = np.arange(9)
    assert_equal(stability_score(a, b), 0)
    assert_equal(stability_score(a, b[::-1]), 0)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        score = stability_score(a, b)
        assert_less_equal(score, 1.)
        assert_greater_equal(score, 0.)


def test_norm_stability_score():
    a = np.arange(9)
    b = np.arange(9)
    s = 20
    assert_equal(norm_stability_score(a, b, s), 0)
    assert_equal(norm_stability_score(a, b[::-1], s), 0)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        score = norm_stability_score(a, b, s)
        assert_greater_equal(score, 0.)
