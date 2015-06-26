from nose.tools import (assert_raises, assert_is_none, assert_false,
                        assert_true)
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
