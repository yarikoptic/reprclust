from mvpa2.datasets.base import Dataset
import numpy as np
from numpy.testing import assert_array_equal

from nose.tools import assert_is_none, assert_equal, assert_raises

from reprclust.cluster_methods import WardClusterMethod
from reprclust.reproducibility import reproducibility, _run_fold

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

dss = np.hstack(dss)
dss = Dataset(dss, sa={'chunks': range(20)},
              fa={'subjects': np.repeat(range(10), 2)})

def test_run_method():
    scores = reproducibility(dss, fake_splitter, WardClusterMethod(),
                             ks=np.arange(2, 21),
                             ground_truth=ground_truth, verbose=0)
    for key, value in scores.items():
        assert_array_equal(value[0], np.arange(2, 21))
        if not key.endswith('_gt'):
            # this works only with ARI and AMI -- default
            assert_array_equal(value[1], np.ones(value[1].shape))
        else:
            # only the first is 1.
            assert_equal(value[1, 0], 1.)

def test_run_fold():
    common_args = (fake_splitter[0][0], fake_splitter[0][1], WardClusterMethod, [2, 3])
    # check raises
    assert_raises(TypeError, _run_fold, blob1, *common_args)
    assert_raises(ValueError, _run_fold, dss, *common_args, fa_space='subjects', sa_space='chunks')
    assert_raises(KeyError, _run_fold, dss, *common_args, fa_space='runs')
    assert_raises(KeyError, _run_fold, dss, *common_args, fa_space=None, sa_space='runs')

