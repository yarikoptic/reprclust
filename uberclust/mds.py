#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 noet:
"""Classical MDS implementation (to be migrated later into PyMVPA)

 COPYRIGHT: Uberclust gig 2015

 LICENSE: MIT

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist


def mds_classical(d, ndim=2):
    """Multidimensional scaling (classical implementation)

    Given a matrix of distances between points, find a set of (low)
    dimensional points that have similar distances.

    Parameters
    ----------
    d : np.ndarray
        Square matrix (or a vector of upper-triangular values)
        with distances between points
    ndim : int, optional
        Number of ndim in the project to use. If degenerate (<=0)
        then ndim is taken to be the minimal sufficient to describe
        given similarity structure
    """
    d = np.asanyarray(d)
    if d.ndim == 1:
        d = squareform(d)

    (n, m) = d.shape
    assert(n == m)
    # assert(ndim > 0)

    E = -0.5 * d**2

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = E - E.mean(axis=1)[:, np.newaxis] - E.mean(axis=0) + E.mean()

    U, S, _ = np.linalg.svd(F)

    Y = U * np.sqrt(S)

    if ndim <= 0:
        # automatically figure out the space based on the rank of original
        # similarity matrix
        SA = np.abs(S)  # it should be positive semi-def but just to be sure
        ndim = np.sum(SA >= SA[0]/1e6) # disregard tiny values in the tail

    return Y[:, :ndim]


from nose.tools import assert_equal, assert_greater
from numpy.testing import assert_array_almost_equal
from mvpa2.misc.fx import get_random_rotation

def _get_square_points(size):
    """Return square array of points"""
    return np.array([(i / size, i % size) for i in range(size**2)])


def _test_mds_classical(d, ndim, targetndim):
    points = mds_classical(d, ndim)
    # those points should resemble original distances and be of target dimensionality
    d = np.asanyarray(d)
    if d.ndim == 1:
        dsq = squareform(d)
    else:
        dsq = d
        d = squareform(d)
    npoints = len(dsq)
    assert_equal(len(points), npoints)
    if ndim > 0:
        assert_equal(points.shape[1], ndim)
    else:
        # so it was assessed
        ndim = points.shape[1]
    pointsdist = pdist(points)
    if ndim == targetndim:
        assert_array_almost_equal(d, pointsdist)
    else:
        # if we carry mds into higher dimensionality, "disturbance" should not increase
        # kinda obvious, but let's check regardless
        points_1 = mds_classical(d, ndim+1)
        pointsdist_1 = pdist(points_1)
        assert_greater(np.linalg.norm(pointsdist - d), np.linalg.norm(pointsdist_1 - d))


def test_mds_classical():
    # nothing wrong with a 1D case and 2 points
    yield _test_mds_classical, [1], 1, 1
    # and we could project from 2d to 1d with some success
    yield _test_mds_classical, [1, 1, 1], 1, 2
    yield _test_mds_classical, [1, 1, 1], 2, 2
    # it should figure out correctly
    yield _test_mds_classical, [1, 1, 1], 0, 2
    yield _test_mds_classical, [1, 1, 1], -1, 2
    yield _test_mds_classical, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 2, 2


def test_mds_classical_square():
    # square points on a plane
    points = _get_square_points(4)
    pointsdist = pdist(points)
    yield _test_mds_classical, pointsdist, 1, 2
    yield _test_mds_classical, pointsdist, 2, 2


def test_mds_classical_higher_dimensions():
    # get some random cloud on 2d, rotate into 4d, seek solutions
    for nd in [2, 5]:
        points = np.random.normal(size=(10, nd))
        pointsdist = pdist(points)
        yield _test_mds_classical, pointsdist, nd, nd

        rotation = get_random_rotation(nd, nd+2)
        points_2 = np.dot(points, rotation)
        assert_equal(points_2.shape, (len(points), nd+2))
        pointsdist_2 = pdist(points)
        # they should fall back nicely into nd
        yield _test_mds_classical, pointsdist_2, nd, nd
        # and dimensionality should be figured out correctly as well
        yield _test_mds_classical, pointsdist_2, -1, nd
        # and we should get "reasonable" solution by going higher
        yield _test_mds_classical, pointsdist_2, nd+1, nd+1