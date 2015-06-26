import numpy as np

from nose.tools import assert_is_none

from reprclust.cluster_methods import WardClusterMethod
from reprclust.cluster_metrics import ari
from reprclust.reproducibility import Reproducibility

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
data = np.vstack((blob1, blob2))

# add some noise and transpose
dss = [(data + np.random.randn(*data.shape)) for i in xrange(10)]
# fake test, we should get a value of 1 for k=2
idx_train = idx_test = range(10)
fake_splitter = [(idx_train, idx_test)] * 4

def test_run_fold_method():
    repr = Reproducibility(dss, fake_splitter, WardClusterMethod, ks=20)
    assert_is_none(repr.run(n_jobs=1, verbose=0))
    import pdb; pdb.set_trace()
