import numpy as np

from scipy.spatial.distance import (pdist, squareform, cdist)
from scipy.cluster.hierarchy import complete, ward

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster.hierarchical import _hc_cut, ward_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM

from joblib import Parallel, delayed
AVAILABLE_METHODS = ['ward', 'complete', 'gmm']


def cut_tree_scipy(Y, k):
    children = Y[:, 0:2]
    return _hc_cut(k, children, len(children)+1)


def compute_stability_fold(samples, train, test, method='ward', 
                         max_k=300, stack=False, random_state_gmm=None,
                         **kwargs):
    """
    General function to compute the stability on a fold.
    
    Parameters:
        samples: a list of arrays containing the samples to cluster, each
                 array has shape (n_samples, n_features) in PyMVPA terminology.
                 We are clustering the features, i.e., the nodes.
        train: a list of indices for the training set
        test: a list of indices for the test set
        method: method to use. atm only 'ward', 'complete', and 'gmm' are implemented.
        max_k: maximum k to compute the stability testing.
        stack: if False, datasets are averaged; if True, they are stacked.
        random_state_gmm: a random state for initialization of GMM.
        kwargs: optional keyword arguments being passed to the clustering method 
                (only for 'ward', and 'gmm')
    
    Attributes:
        result: a (max_k-1, 3) array, where result[:, 0] is the ARI, 
                result[:, 1] is the AMI, and result[:, 2] is the k.
        
    """    
    if method not in AVAILABLE_METHODS:
        raise ValueError('Method {0} not implemented'.format(method))
    
    # preallocate matrix for results
    result = np.zeros((max_k-1, 3))

    # get training and test
    train_set = [samples[x] for x in train]
    test_set = [samples[x] for x in test]
    
    if stack:
        train_ds = np.vstack(train_set)
        test_ds = np.vstack(test_set)
    else:
        train_ds = np.mean(np.dstack(train_set), axis=2)
        test_ds = np.mean(np.dstack(test_set), axis=2)

    # compute clustering on training set
    if method is 'complete': 
        train_ds_dist = pdist(train_ds.T, metric='correlation')
        test_ds_dist = pdist(test_ds.T, metric='correlation')
        # I'm computing the full tree and then cutting afterwards to speed computation
        Y_train = complete(train_ds_dist)
        # same on testing set
        Y_test = complete(test_ds_dist)
    elif method is 'ward':
        (children_train, n_comp_train, 
         n_leaves_train, parents_train) = ward_tree(train_ds.T, **kwargs)
        # same on testing set
        (children_test, n_comp_test, 
         n_leaves_test, parents_test) = ward_tree(test_ds.T, **kwargs)
    elif method is 'gmm':
        pass  # we'll have to run it for each k

    for i_k, k in enumerate(range(2, max_k+1)):
        if method is 'complete':
            # cut the tree with right K for both train and test
            train_label = cut_tree_scipy(Y_train, k)
            test_label = cut_tree_scipy(Y_test, k)

            # train a classifier on this clustering
            #knn = KNeighborsClassifier(metric=correlation)
            knn = KNeighborsClassifier(algorithm='brute', metric='correlation')
            knn.fit(train_ds.T, train_label)
            # predict the clusters in the test set
            prediction_label = knn.predict(test_ds.T)
        elif method is 'ward':
            # cut the tree with right K for both train and test
            train_label = _hc_cut(k, children_train, n_leaves_train)
            test_label = _hc_cut(k, children_test, n_leaves_test)

            # train a classifier on this clustering
            knn = KNeighborsClassifier()
            knn.fit(train_ds.T, train_label)
            # predict the clusters in the test set
            prediction_label = knn.predict(test_ds.T)
        elif method is 'gmm':
            gmm = GMM(n_components=k, random_state=random_state_gmm, 
                      **kwargs)
            # fit on train and predict test
            gmm.fit(train_ds.T)
            prediction_label = gmm.predict(test_ds.T)
            
            # fit on test and get labels
            gmm.fit(test_ds.T)
            test_label = gmm.predict(test_ds.T)
            
        # append results
        result[i_k, 0] = adjusted_rand_score(prediction_label, test_label)
        result[i_k, 1] = adjusted_mutual_info_score(prediction_label, test_label)

    result[:, 2] = range(2, max_k+1)

    return result


def compute_stability(splitter, samples, method='ward', stack=False,
                      max_k=300, n_jobs=1, random_state_gmm=None, **kwargs):
    """
    General function to compute the stability of clustering on a dataset.
    
    Parameters:
        splitter: a generator of training and test set, usually from sklearn.cross_validation
        samples: a list of arrays containing the samples to cluster, each
                 array has shape (n_samples, n_features) in PyMVPA terminology.
                 We are clustering the features, i.e., the voxels/nodes.
        method: method to use. atm only 'ward', 'complete', and 'gmm' are implemented.
        stack: if False, datasets are averaged; if True, they are stacked.
        max_k: maximum k to compute the stability testing.
        n_jobs: number of jobs to run the parallelization. default n_jobs=1
        random_state_gmm: a random state for initialization of GMM.
        kwargs: optional keyword arguments being passed to the clustering method 
                (only for 'ward', and 'gmm'). Useful to pass connectivity matrix (ward)
                or covariance structure (GMM)
    
    Attributes:
        result: a (max_k-1, 3) array, where result[:, 0] is the ARI, 
                result[:, 1] is the AMI, and result[:, 2] is the k.
        
    """    
    
    result = Parallel(n_jobs=n_jobs)(delayed(compute_stability_fold)
                                  (samples, train, test, method=method, 
                                   max_k=max_k, stack=stack, random_state_gmm=random_state_gmm,
                                   **kwargs)
                                   for train, test in splitter)


    result = np.vstack(result)
    
    return result
