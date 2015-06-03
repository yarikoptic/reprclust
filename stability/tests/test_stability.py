from nose.tools import assert_equal, assert_true, assert_raises, \
    assert_is_none, assert_is_instance
from numpy.testing import assert_array_equal
import numpy as np
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
from stability.stability import (cut_tree_scipy, compute_stability_fold)

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
    ks, ari, ami, likelihood, ari_gt, ami_gt = \
        compute_stability_fold(dss, idx_train, idx_test, max_k=20)
    assert_array_equal(ks, np.arange(2, 21))
    assert_array_equal(ari, np.ones(ari.shape))
    assert_array_equal(ami, np.ones(ami.shape))
    assert_is_none(likelihood)
    assert_is_none(ari_gt)
    assert_is_none(ami_gt)

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
                    ks, ari, ami, likelihood, ari_gt, ami_gt = \
                        compute_stability_fold(
                            dss, idx_train, idx_test, max_k=20, method=method,
                            stack=stack, cv_likelihood=cv_likelihood,
                            ground_truth=ground_truth)
                    assert_array_equal(ks, np.arange(2, 21))
                    assert_is_instance(ari, np.ndarray)
                    assert_is_instance(ami, np.ndarray)
                    if cv_likelihood:
                        assert_is_instance(likelihood, np.ndarray)
                    else:
                        assert_is_none(likelihood)
                    assert_is_instance(ari_gt, np.ndarray)
                    assert_is_instance(ami_gt, np.ndarray)
