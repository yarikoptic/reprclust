from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_is_none, assert_is_instance, assert_less_equal, \
    assert_greater_equal, assert_almost_equal
from numpy.testing import assert_array_equal
import numpy as np
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from reprclust.stability import (cut_tree_scipy, compute_stability_fold,
                                 compute_stability,
                                 get_optimal_permutation, permute,
                                 generate_random_labeling,
                                 stability_score, norm_stability_score,
                                 rand_stability_score, correlation,
                                 correlation_score)

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
    ks, ari, ami, stab, likelihood, ari_gt, ami_gt, stab_gt, corr, corr_gt = \
        compute_stability_fold(dss, idx_train, idx_test, max_k=20)
    assert_array_equal(ks, np.arange(2, 21))
    assert_array_equal(ari, np.ones(ari.shape))
    assert_array_equal(ami, np.ones(ami.shape))
    assert_array_equal(stab, np.zeros(stab.shape))
    assert_is_none(likelihood)
    assert_is_none(ari_gt)
    assert_is_none(ami_gt)
    assert_is_none(stab_gt)
    assert_is_none(corr)
    assert_is_none(corr_gt)

    ground_truth = np.hstack((np.zeros(10, dtype=int), np.ones(10, dtype=int)))
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
                        assert_true(len(result), 10)


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
    assert_equal((None, None, None, None, None, None), result[4:])


