from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_is_none, assert_is_instance, assert_less_equal, \
    assert_greater_equal, assert_almost_equal

import numpy as np
from numpy.testing import assert_array_equal

from reprclust.cluster_metrics import (generate_random_labeling, permute,
                                       get_optimal_permutation,
                                       stability_score,
                                       rand_stability_score,
                                       norm_stability_score,
                                       correlation_score,
                                       correlation)


def test_generate_random_labeling():
    a = generate_random_labeling(5, 10)
    assert_equal(len(np.unique(a)), 5)
    assert_raises(ValueError, generate_random_labeling, 5, 3)


def test_permute():
    a = np.arange(9)
    b = np.arange(9)[::-1]
    assert_array_equal(a, permute(b, b))

    a = np.array([0, 0, 1, 2, 1, 3])
    b = np.array([1, 1, 3, 2, 3, 0])
    p = np.array([3, 0, 2, 1])
    assert_array_equal(a, permute(b, p))


def test_get_optimal_permutation():
    # simple permutation
    a = np.array([0, 1, 2, 3])
    b = np.array([3, 2, 1, 0])
    perm_real = b
    assert_array_equal(perm_real, get_optimal_permutation(a, b, 4))
    # no permutation
    assert_array_equal(a, get_optimal_permutation(a, a, 4))
    # random labellings
    k = 5
    for i in xrange(20):
        # a = p(b) -- find p
        b = generate_random_labeling(k, 10)
        perm_real = np.arange(len(np.unique(b)))
        np.random.shuffle(perm_real)
        a = b.copy()
        for j in xrange(len(a)):
            a[j] = perm_real[a[j]]
        assert_array_equal(perm_real, get_optimal_permutation(a, b, k))
    # case when permutation is possible but a and b have different labels
    a = np.array([1, 1, 2, 2, 0])
    b = np.array([3, 3, 4, 4, 0])
    perm = get_optimal_permutation(a, b, k)
    assert_array_equal(perm, np.array([0, 4, 3, 1, 2]))

    a = np.array([0, 0, 2, 2])
    b = np.array([3, 3, 1, 1])
    perm = get_optimal_permutation(a, b, 4)
    assert_array_equal(perm, np.array([3, 2, 1, 0]))

    # case when we have different number of unique labels
    a = np.array([1, 1, 2, 2, 0])
    b = np.array([3, 3, 4, 4, 4])
    # in this case order matters
    # mapping a = p(b)
    perm = get_optimal_permutation(a, b, 5)
    assert_array_equal(perm, np.array([4, 3, 0, 1, 2]))
    # mapping b = p(a)
    perm = get_optimal_permutation(b, a, 5)
    assert_array_equal(perm, np.array([0, 3, 4, 2, 1]))


def test_stability_score():
    a = np.arange(9)
    b = np.arange(9)
    assert_equal(stability_score(a, b, 9), 0)
    assert_equal(stability_score(a, b, 9), stability_score(b, a, 9))
    assert_equal(stability_score(a, b[::-1], 9), 0)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        score = stability_score(a, b, 10)
        assert_less_equal(score, 1.)
        assert_greater_equal(score, 0.)
        # test it's symmetric
        assert_equal(score, stability_score(b, a, 10))


def test_rand_stability_score():
    for i in xrange(20):
        k = np.random.randint(2, 10)
        n = np.random.randint(k, 20)
        s = np.random.randint(1, 20)
        rand_score = rand_stability_score(k, n, s)
        assert_less_equal(rand_score, 1.)
        assert_greater_equal(rand_score, 0.)


def test_norm_stability_score():
    a = np.arange(9)
    b = np.arange(9)
    s = 20
    rand_score = rand_stability_score(len(np.unique(a)), len(a), s)
    assert_equal(norm_stability_score(a, b, rand_score, 9), 0)
    assert_equal(norm_stability_score(a, b[::-1], rand_score, 9), 0)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        rand_score = rand_stability_score(len(np.unique(a)), len(a), s)
        score = norm_stability_score(a, b, rand_score, 10)
        assert_greater_equal(score, 0.)


def test_correlation():
    for i in xrange(10):
        x = np.random.normal(4, 2, size=10)
        y = np.random.normal(-3, 2, size=10)
        assert_almost_equal(correlation(x, y), np.corrcoef(x, y)[0, 1])


def test_correlation_score():
    clust1 = np.random.normal(-5, 1, size=(5, 10))
    clust2 = np.random.normal(5, 1, size=(5, 10))
    true_clust = np.hstack((np.ones(10), np.zeros(10))).astype(int)
    d = np.hstack((clust1, clust2))
    assert_equal(correlation_score(true_clust, true_clust, d), 1.0)
    assert_equal(correlation_score(true_clust, true_clust, d,
                                   corr_score='spearman'), 1.0)
