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

from joblib import Parallel, delayed

import numpy as np

from scipy.spatial.distance import pdist, hamming
from scipy.cluster.hierarchy import complete
from scipy.stats import rankdata

from sklearn.cluster import KMeans
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.metrics.cluster import (adjusted_rand_score,
                                     adjusted_mutual_info_score,
                                     contingency_matrix)
from sklearn.mixture import GMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.linear_assignment_ import linear_assignment


def cut_tree_scipy(Y, k):
    """ Given the output Y of a hierarchical clustering solution from scipy
    and a number k, cuts the tree and returns the labels.
    """
    children = Y[:, 0:2].astype(int)
    # convert children to correct values for _hc_cut
    return _hc_cut(k, children, len(children)+1)

def _predict_knn(self, newdata, k, n_neighbors=1):
    """Common function to be used to predict clustering solution when method
    doesn't have built-in prediction method"""
    if k not in self._clusters:
        raise ValueError('Run cluster solution of {0} first'.format(k))
    # if it gets called multiple times, data get rewritten
    labels = self.get_clusters(k)
    # train a classifier on this clustering
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(self.data, labels)
    # predict
    predicted_labels = knn.predict(newdata)
    self._predicted[k] = predicted_labels


def _predict(self, newdata, k):
    """Common function to be used to predict clustering solution when method
    has built-in prediction method"""
    self._predicted[k] = self._methods[k].predict(newdata)


class ClusterMethod(object):
    """Wrapper for clustering methods to add predict method and provide
    solution for different ks

    Attributes
    ----------
    clustering_method : callable
        A function that clusters data.

    Methods
    -------
    cluster(*args, **kwargs)
        Clusters the data. args and kwargs depend on the current method.
    predict(*args, **kwargs)
        Predicts clustering on new data based on previous cluster solution.
        args and kwargs depend on the current method.
    get_clusters(k)
        Returns the cluster solution for k clusters.
    get_predicted_clusters(k)
        Returns the predicted cluster solution for k clusters.
    """
    def __init__(self, clustering_method):
        if not hasattr(clustering_method, '__call__'):
            raise ValueError('clustering_method must be a callable')
        self.method = clustering_method
        self._run = False
        # we'll store the actual clusters here
        self._clusters = {}
        # we'll store the predicted clusters here
        self._predicted = {}
        self.data = None

        self._args = None
        self._kwargs = None

    def __call__(self, data):
        """Just store the input data"""
        self.data = data

    def __repr__(self):
        args = []
        if self._args:
            for arg in self._args:
                args.append(repr(arg))
        if self._kwargs:
            for key, value in self._kwargs.items():
                args.append("{0}={1}".format(key, repr(value)))
        if args:
            args = ", ".join(args)
        else:
            args = ""

        return self.__class__.__name__ + '({0})'.format(args)

    @property
    def run(self):
        return self._run

    def cluster(self, *args, **kwargs):
        """This method will run the cluster method on the data"""
        raise NotImplementedError('Subclass must define it')

    def get_clusters(self, k):
        return self._clusters.get(k, None)

    def predict(self, *args, **kwargs):
        """This method will predict the clustering solution on array"""
        raise NotImplementedError('Subclass must define it')

    def get_predicted_clusters(self, k):
        return self._predicted.get(k, None)


class WardClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(WardClusterMethod, self).__init__(ward_tree)
        # store args and kwargs here
        self._args = args
        self._kwargs = kwargs
        self._method_output = None

    def cluster(self, k):
        # if we haven't run it the first time, need to run it
        if not self.run:
            self._method_output = self.method(self.data, *self._args,
                                              **self._kwargs)
            self._run = True
        # did we already computed the solution for this k?
        if k not in self._clusters:
            self._clusters[k] = _hc_cut(k, self._method_output[0],
                                        self._method_output[2])

    def predict(self, newdata, k):
        _predict_knn(self, newdata, k)


class GMMClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(GMMClusterMethod, self).__init__(GMM)
        self._args = args
        self._kwargs = kwargs
        self._method_output = None
        # we'll have to store the different GMMs in here to predict
        self._methods = {}

    def cluster(self, k):
        if k not in self._clusters:
            gmm = self.method(n_components=k, *self._args, **self._kwargs)
            self._methods[k] = gmm.fit(self.data)
            self._clusters[k] = gmm.predict(self.data)

    def get_method(self, k):
        """Return the fitted GMM with k components"""
        if k in self._methods:
            return self._methods[k]
        else:
            return None

    def predict(self, newdata, k):
        _predict(self, newdata, k)


class KMeansClusterMethod(ClusterMethod):
    def __init__(self, *args, **kwargs):
        super(KMeansClusterMethod, self).__init__(KMeans)
        self._args = args
        self._kwargs = kwargs
        self._method_output = None
        self._methods = {}

    def cluster(self, k):
        if k not in self._clusters:
            kmeans = self.method(n_clusters=k, *self._args, **self._kwargs)
            self._methods[k] = kmeans.fit(self.data)
            self._clusters[k] = kmeans.predict(self.data)

    def get_method(self, k):
        """Return the fitted KMeans with k components"""
        if k in self._methods:
            return self._methods[k]
        else:
            return None

    def predict(self, newdata, k):
        _predict(self, newdata, k)


class CompleteClusterMethod(ClusterMethod):
    def __init__(self, metric='correlation', *args, **kwargs):
        super(CompleteClusterMethod, self).__init__(complete)
        self._args = args
        self._kwargs = kwargs
        self._method_output = None
        self._methods = {}
        self.metric = metric

    def cluster(self, k):
        # if we haven't run it already, run the clustering
        if not self.run:
            dist = pdist(self.data, metric=self.metric)
            self._method_output = self.method(dist)
            self._run = True
        if k not in self._clusters:
            self._clusters[k] = cut_tree_scipy(self._method_output, k)

    def predict(self, newdata, k):
        _predict_knn(self, newdata, k)
