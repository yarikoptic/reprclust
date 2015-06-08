# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Misc functions
"""
from mvpa2.support.nibabel import surf
import numpy as np
from scipy import sparse


def triangles2edges(triangles):
    """
    Given triangles, returns edges.
    For example,
    [[1, 2, 3], [1, 3, 4]] -> [[1, 2], [1, 3], [2, 3], [1, 4], [3, 4]]
    """
    edges = set()
    idx = [[0, 1], [0, 2], [1, 2]]
    nrow, ncol = np.shape(triangles)
    for irow in xrange(nrow):
        for i in idx:
            edges.add(tuple(triangles[irow, i]))
    out = [list(t) for t in edges]
    out.sort()
    return out


def build_connectivity_matrix(source_surface_fn, nodes=None):
    """
    Given the brain surface (intermediate) saved in `source_surface_fn` and
    a subset of nodes, returns a sparse adjacency matrix to be used
    by the Ward algorithm as implemented in scikit learn.
    """
    surface = surf.from_any(source_surface_fn)
    triangles = surface.faces
    nvertices = surface.nvertices

    e = triangles2edges(triangles)
    conn = np.zeros((nvertices, nvertices), dtype=bool)
    # set adjacent nodes to 1
    for idx in e:
        conn[idx[0], idx[1]] = 1
        conn[idx[1], idx[0]] = 1
    # set diagonal to 1 -- useless but for completeness
    conn[np.diag_indices_from(conn)] = 1
    # get only nodes we have
    if nodes is not None:
        connectivity2 = conn[nodes, :][:, nodes]
    else:
        connectivity2 = conn
    # check it's really symmetric
    for x in xrange(len(connectivity2)):
        assert(np.array_equal(connectivity2[x, :], connectivity2[:, x]))
    # make it a sparse matrix for ward
    connectivity2_sparse = sparse.csr_matrix(connectivity2)
    return connectivity2_sparse
