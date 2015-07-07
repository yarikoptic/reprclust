from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_is_none, assert_is_instance, assert_less_equal, \
    assert_greater_equal, assert_almost_equal

import numpy as np
from numpy.testing import assert_array_equal

from reprclust.cluster_metrics import (_permute,
                                       _get_optimal_permutation,
                                       _instability_score,
                                       rand_instability_score,
                                       AMI, ARI, CorrelationScore,
                                       InstabilityScore)


def generate_random_labeling(k, size):
    """Generate a valid random labeling"""
    if size < k:
        raise ValueError('To have a labeling size cannot be lower than high')
    while True:
        a = np.random.randint(0, k, size)
        if len(np.unique(a)) == k:
            break
    return a


def test_generate_random_labeling():
    a = generate_random_labeling(5, 10)
    assert_equal(len(np.unique(a)), 5)
    assert_raises(ValueError, generate_random_labeling, 5, 3)


def test_permute():
    a = np.arange(9)
    b = np.arange(9)[::-1]
    assert_array_equal(a, _permute(b, b))

    a = np.array([0, 0, 1, 2, 1, 3])
    b = np.array([1, 1, 3, 2, 3, 0])
    p = np.array([3, 0, 2, 1])
    assert_array_equal(a, _permute(b, p))


def test_get_optimal_permutation():
    # simple permutation
    a = np.array([0, 1, 2, 3])
    b = np.array([3, 2, 1, 0])
    perm_real = b
    assert_array_equal(perm_real, _get_optimal_permutation(a, b, 4))
    # no permutation
    assert_array_equal(a, _get_optimal_permutation(a, a, 4))
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
        assert_array_equal(perm_real, _get_optimal_permutation(a, b, k))
    # case when permutation is possible but a and b have different labels
    a = np.array([1, 1, 2, 2, 0])
    b = np.array([3, 3, 4, 4, 0])
    perm = _get_optimal_permutation(a, b, k)
    assert_array_equal(perm, np.array([0, 4, 3, 1, 2]))

    a = np.array([0, 0, 2, 2])
    b = np.array([3, 3, 1, 1])
    perm = _get_optimal_permutation(a, b, 4)
    assert_array_equal(perm, np.array([3, 2, 1, 0]))

    # case when we have different number of unique labels
    a = np.array([1, 1, 2, 2, 0])
    b = np.array([3, 3, 4, 4, 4])
    # in this case order matters
    # mapping a = p(b)
    perm = _get_optimal_permutation(a, b, 5)
    assert_array_equal(perm, np.array([4, 3, 0, 1, 2]))
    # mapping b = p(a)
    perm = _get_optimal_permutation(b, a, 5)
    assert_array_equal(perm, np.array([0, 3, 4, 2, 1]))


def test_instability_score():
    a = np.arange(9)
    b = np.arange(9)
    assert_equal(_instability_score(a, b, 9), 0)
    assert_equal(_instability_score(a, b, 9), _instability_score(b, a, 9))
    assert_equal(_instability_score(a, b[::-1], 9), 0)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        score = _instability_score(a, b, 10)
        assert_less_equal(score, 1.)
        assert_greater_equal(score, 0.)
        # test it's symmetric
        assert_equal(score, _instability_score(b, a, 10))


def test_rand_instability_score():
    s = 20
    ks = np.arange(2, 21)
    n = 30
    rand_scores = rand_instability_score(ks, n, s)
    assert_array_equal(rand_scores.keys(), ks)
    for rand_score in rand_scores.values():
        assert_less_equal(rand_score, 1.)
        assert_greater_equal(rand_score, 0.)


def test_instability_class():
    a = np.arange(9)
    b = np.arange(9)
    s = 20
    ks = np.arange(2, len(a)+1)
    rand_scores = rand_instability_score(ks, len(a), s)
    instability_score = InstabilityScore(rand_scores)
    assert_equal(instability_score(a, b, k=9), 0)
    assert_raises(ValueError, instability_score, a, b)
    for i in xrange(20):
        a = generate_random_labeling(5, 10)
        b = generate_random_labeling(5, 10)
        ks = np.arange(2, 11)
        rand_scores = rand_instability_score(ks, len(a), s)
        instability_score = InstabilityScore(rand_scores)
        for k in ks:
            score = instability_score(a, b, k=k)
            assert_greater_equal(score, 0.)


def test_correlation_method():
    cs = CorrelationScore()
    for i in xrange(10):
        x = np.random.normal(4, 2, size=10)
        y = np.random.normal(-3, 2, size=10)
        assert_almost_equal(cs._correlation(x, y), np.corrcoef(x, y)[0, 1])


def test_correlation_score():
    # (n_features, n_samples) in PyMVPA terminology
    clust1 = np.random.normal(-5, 1, size=(50, 10))
    clust2 = np.random.normal(5, 1, size=(50, 10))
    true_clust = np.hstack((np.ones(50), np.zeros(50))).astype(int)
    d = np.vstack((clust1, clust2))
    for corr_type in ('spearman', 'pearson'):
        correlation_score = CorrelationScore(corr_type=corr_type)
        assert_raises(ValueError, correlation_score, true_clust, true_clust)
        assert_almost_equal(correlation_score(true_clust, true_clust, data=d),
                            1.0)
