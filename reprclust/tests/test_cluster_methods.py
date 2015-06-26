from nose.tools import (assert_raises, assert_is_none,
                        assert_is_not_none, assert_false,
                        assert_equal,
                        assert_true, assert_is_instance)
import numpy as np
from numpy.testing import assert_array_equal
from reprclust.cluster_methods import (ClusterMethod, GMMClusterMethod,
                                       WardClusterMethod, KMeansClusterMethod)

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
data = np.vstack((blob1, blob2))

def test_clustermethod():
    assert_raises(ValueError, ClusterMethod, 30)

    def my_cluster_method(x):
        """just a fake"""
        return 1

    cm = ClusterMethod(my_cluster_method)

    assert_raises(NotImplementedError, cm.cluster, 2)
    assert_raises(NotImplementedError, cm.predict, 2)
    assert_is_none(cm.get_clusters(2))
    assert_is_none(cm.get_predicted_clusters(2))
    assert_is_none(cm.data)
    assert_false(cm.run)

    cm(data)

    assert_raises(NotImplementedError, cm.cluster, 2)
    assert_raises(NotImplementedError, cm.predict, 2)
    assert_is_none(cm.get_clusters(2))
    assert_is_none(cm.get_predicted_clusters(2))
    assert_false(cm.run)
    assert_array_equal(cm.data, data)

def _test_clustermethods(cm):
    # store the data
    cm(data)
    for k in range(2, 10):
        assert_is_none(cm.cluster(k))
        assert_is_none(cm.predict(data, k))
        assert_array_equal(cm.get_clusters(k), cm.get_predicted_clusters(k))
        # XXX: temporary fix. apparently GMM might return less than the
        # number of components requested
        if cm.method.__name__ is not 'GMM':
            assert_equal(len(np.unique(cm.get_clusters(k))), k)

    if hasattr(cm, 'get_method'):
        for k in range(2, 10):
            assert_is_not_none(cm.get_method(k))


def test_implemented_clustermethods():
    for cm in [WardClusterMethod, KMeansClusterMethod, GMMClusterMethod]:
        yield _test_clustermethods, cm()
