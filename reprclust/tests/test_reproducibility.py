import numpy as np

from numpy.testing import assert_array_equal

from nose.tools import assert_is_none, assert_equal

from reprclust.cluster_methods import WardClusterMethod
from reprclust.reproducibility import Reproducibility

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
data = np.vstack((blob1, blob2))

# add some noise and transpose
dss = [(data + np.random.randn(*data.shape)) for i in xrange(10)]
# fake test, we should get a value of 1 for k=2
idx_train = idx_test = range(10)
fake_splitter = [(idx_train, idx_test)]
ground_truth = np.hstack((np.zeros(10, dtype=int), np.ones(10, dtype=int)))


def test_run_method():
    reprod = Reproducibility(dss, fake_splitter, WardClusterMethod(), ks=20,
                             ground_truth=ground_truth)
    assert_is_none(reprod.run(n_jobs=1, verbose=0))
    for key, value in reprod.scores.items():
        assert_array_equal(value[0], np.arange(2, 21))
        if not key.endswith('_gt'):
            # this works only with ARI and AMI -- default
            assert_array_equal(value[1], np.ones(value[1].shape))
        else:
            # only the first is 1.
            assert_equal(value[1, 0], 1.)
