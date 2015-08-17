from nose.tools import (assert_raises, assert_is_none,
                        assert_is_not_none, assert_false,
                        assert_equal, assert_is,
                        assert_true, assert_is_instance)
import numpy as np
from numpy.testing import assert_array_equal
from reprclust.cluster_methods import (ClusterMethod, GMMClusterMethod,
                                       WardClusterMethod, KMeansClusterMethod,
                                       CompleteClusterMethod)

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
data = np.vstack((blob1, blob2))

def test_clustermethod():
    cm = ClusterMethod()

    assert_raises(NotImplementedError, cm.train, 2)
    assert_raises(NotImplementedError, cm.predict, 2)
    assert_false(cm.is_trained)

def _test_clustermethods(cm):
    for k in range(2, 10):
        assert_is_none(cm.train(data, k))
        assert_array_equal(cm.predict(data, k),
                           cm.predict(data[::-1], k)[::-1])
        # XXX: temporary fix. apparently GMM might return less than the
        # number of components requested
        if not isinstance(cm, GMMClusterMethod):
            assert_equal(len(np.unique(cm.predict(data, k))), k)


def test_implemented_clustermethods():
    for cm in [WardClusterMethod,
               KMeansClusterMethod,
               GMMClusterMethod,
               CompleteClusterMethod]:
        yield _test_clustermethods, cm()
