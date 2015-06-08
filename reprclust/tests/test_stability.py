from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_is_none, assert_is_instance, assert_less_equal, assert_greater_equal
from numpy.testing import assert_array_equal
import numpy as np
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
from reprclust.stability import (cut_tree_scipy, compute_stability_fold,
                                 compute_stability,
                                 get_optimal_permutation, permute,
                                 generate_random_labeling,
                                 stability_score, norm_stability_score,
                                 rand_stability_score)

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
    ks, ari, ami, stab, likelihood, ari_gt, ami_gt = \
        compute_stability_fold(dss, idx_train, idx_test, max_k=20)
    assert_array_equal(ks, np.arange(2, 21))
    assert_array_equal(ari, np.ones(ari.shape))
    assert_array_equal(ami, np.ones(ami.shape))
    assert_array_equal(stab, np.zeros(stab.shape))
    assert_is_none(likelihood)
    assert_is_none(ari_gt)
    assert_is_none(ami_gt)

    ground_truth = np.hstack((np.zeros(10), np.ones(10)))
    # smoke test with all possible parameters
    for method in ['ward', 'complete', 'kmeans', 'gmm']:
        for stack in [True, False]:
            for cv_likelihood in [True, False]:
                for stability in [True, False]:
                    if cv_likelihood and method != 'gmm':
                        assert_raises(ValueError, compute_stability_fold,
                                      dss, idx_train, idx_test,
                                      max_k=20, method=method,
                                      stack=stack,
                                      stability=stability,
                                      cv_likelihood=cv_likelihood)
                    else:
                        result = compute_stability_fold(
                            dss, idx_train, idx_test, max_k=20, method=method,
                            stack=stack,
                            stability=stability,
                            cv_likelihood=cv_likelihood,
                            ground_truth=ground_truth)
                        assert_true(len(result), 7)


def test_compute_stability():
    # add some noise and transpose
    dss = [(data + np.random.randn(*data.shape)).T for i in xrange(10)]
    # fake test, we should get a value of 1 for k=2
    idx_train = idx_test = range(10)
    splitter = [(idx_train, idx_test)]
    result = compute_stability(splitter, dss, max_k=20, stability=True,
                               rand_stab_rep=20)
    assert_array_equal(result[0], np.arange(2, dss[0].shape[1]+1))
    for i in range(1, 3):
        assert_array_equal(result[i], np.ones(result[i].shape))
    assert_array_equal(result[3], np.zeros(result[3].shape))
    assert_equal((None, None, None), result[4:])


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
