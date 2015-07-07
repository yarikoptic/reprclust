# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module containing cluster methods to uniform calling
"""
from joblib import Memory

import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import complete

from sklearn.cluster import KMeans
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.mixture import GMM
from sklearn.neighbors import KNeighborsClassifier

from tempfile import mkdtemp

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

# cache ward_tree to save executions
cached_ward_tree = memory.cache(ward_tree)
# cache also complete

def _cut_tree_scipy(Y, k):
    """ Given the output Y of a hierarchical clustering solution from scipy
    and a number k, cuts the tree and returns the labels.
    """
    children = Y[:, 0:2].astype(int)
    # convert children to correct values for _hc_cut
    return _hc_cut(k, children, len(children)+1)

def _predict_knn(train_data, data, labels, n_neighbors=1):
    """Common function to be used to predict clustering solution when method
    doesn't have built-in prediction method"""
    # train a classifier on this clustering
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_data, labels)
    # predict
    predicted_labels = knn.predict(data)
    return predicted_labels


class ClusterMethod(object):
    """Wrapper for clustering methods to add predict method and provide
    solution for different ks

    Attributes
    ----------
    clustering_method : callable
        A function that clusters data.

    Methods
    -------
    train(*args, **kwargs)
        Train clustering method on the data. args and kwargs depend on the
        current method.
    predict(*args, **kwargs)
        Predicts clustering on data based on training previous cluster
        solution.
    """
    def __init__(self):
        self._is_trained = False

    @property
    def is_trained(self):
        return self._is_trained

    def train(self, *args, **kwargs):
        """This method will run the cluster method on the data"""
        raise NotImplementedError('Subclass must define it')

    def predict(self, *args, **kwargs):
        """This method will predict the clustering solution on array"""
        raise NotImplementedError('Subclass must define it')


class WardClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(WardClusterMethod, self).__init__()
        # store args and kwargs here
        self._args = args
        self._kwargs = kwargs
        # output for ward_tree
        self._children = None
        self._n_components = None
        self._n_leaves = None
        self._parents = None
        self._train_data = None

    def train(self, data, k, compute_full=True):
        # if we haven't run it the first time, need to run it
        # XXX: this is not really needed here
        if compute_full:
            self._children, self._n_components, self._n_leaves, \
                self._parents = cached_ward_tree(data, *self._args,
                                                 **self._kwargs)
            self._is_trained = True
            self._train_data = data

    def predict(self, data, k):
        train_prediction = _hc_cut(k, self._children, self._n_leaves)
        # XXX: is there a way to avoid this?
        if np.array_equal(data, self._train_data):
            return train_prediction
        else:
            return _predict_knn(self._train_data, data, train_prediction)


class CompleteClusterMethod(ClusterMethod):
    def __init__(self, metric='correlation', *args, **kwargs):
        super(CompleteClusterMethod, self).__init__()
        self._args = args
        self._kwargs = kwargs
        self._method_output = None
        self._metric = metric
        self._train_data = None
        # cache run complete
        self._run_complete = memory.cache(self._run_complete)

    def _run_complete(self, data):
        """Just to allow caching"""
        return complete(pdist(data, metric=self._metric))

    def train(self, data, k, compute_full=True):
        # if we haven't run it already, run the clustering
        if compute_full:
            self._method_output = self._run_complete(data)
            self._is_trained = True
            self._train_data = data

    def predict(self, data, k):
        train_prediction = _cut_tree_scipy(self._method_output, k)
        # XXX: is there a way to avoid this?
        if np.array_equal(data, self._train_data):
            return train_prediction
        else:
            return _predict_knn(self._train_data, data, train_prediction)


class GMMClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(GMMClusterMethod, self).__init__()
        # store args and kwargs here
        self._args = args
        self._kwargs = kwargs
        # store gmm model
        self._gmm_model = None

    def train(self, data, k, compute_full=True):
        # fit the model -- here probably good to have memoization
        self._gmm_model = GMM(n_components=k, **self._kwargs).fit(data)

    def predict(self, data, k):
        return self._gmm_model.predict(data)


class KMeansClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(KMeansClusterMethod, self).__init__()
        # store args and kwargs here
        self._args = args
        self._kwargs = kwargs
        # store gmm model
        self._kmeans_model = None

    def train(self, data, k, compute_full=True):
        # fit the model -- here probably good to have memoization
        self._kmeans_model = KMeans(n_clusters=k, **self._kwargs).fit(data)

    def predict(self, data, k):
        return self._kmeans_model.predict(data)
