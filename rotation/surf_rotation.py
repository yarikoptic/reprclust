"""
Small module to rotate surfaces and interpolate data on them
"""
from mvpa2.misc.fx import get_random_rotation
import numpy as np
from sklearn.neighbors import NearestNeighbors


def rotate_surface(xyz):
    """ Function to rotate surface using a random rigid rotation

    Arguments
    ---------
        xyz : np.ndarray (n_points, 3)
            an array containing the euclidean coordinates of the vertices
            of the surface

    Returns
    -------
        xyz_rnd : np.ndarray (n_points, 3)
            the rotated array with the euclidean coordinates of the rotated
            vertices of the surface

        dist : np.ndarray (n_points, 3)
            for each point of the rotated surface, the distance from the three
            nearest vertices in the original surface

        i : np.ndarray (n_points, 3)
            for each point of the rotated surface, the indices of the three
            nearest vertices in the original surface
    """
    assert xyz.shape[1] == 3, 'I work only with surfaces in 3D spaces'
    # rotate xyz randomly
    rnd_rot = get_random_rotation(xyz.shape[1])
    xyz_rnd = np.dot(xyz, rnd_rot)

    # find three closest neighbors making up the triangle
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(xyz)
    dist, i = nbrs.kneighbors(xyz_rnd)

    return xyz_rnd, dist, i


def area_triangle_cross(ab, ac):
    """ Compute the area of the triangle whose
    two sides are ab and ac
    """
    return .5 * np.sqrt(np.sum(np.cross(ab, ac)**2, axis=1))


def barycentric_interpolation(data, xyz, xyz_rnd, i):
    """ Function to interpolate data from original surface upon rotated surface

    Arguments
    ---------
        data : np.ndarray (n_points, n_features)
            the data to be interpolated. features are columns

        xyz : np.ndarray (n_points, 3)
            an array containing the euclidean coordinates of the vertices
            of the source surface

        xyz_rnd : np.ndarray (n_points, 3)
            the rotated array with the euclidean coordinates of the rotated
            vertices of the surface

        i : np.ndarray (n_points, 3)
            for each point of the rotated surface, the indices of the three
            nearest vertices in the original surface

    Returns
    -------
        data_interp : np.ndarray (n_points, n_features)
            the interpolated data

    """
    # compute areas of the triangles that contain each rotated node
    a = xyz[i[:, 0]]
    b = xyz[i[:, 1]]
    c = xyz[i[:, 2]]

    # now compute areas of triangles
    d = xyz_rnd

    # vectors from vertices to point whose projection is inside the triangle
    da = d - a
    db = d - b
    dc = d - c

    # triangle ACD -- ratio for b
    area_acd = area_triangle_cross(da, dc)
    # triangle CBD -- ratio for a
    area_cbd = area_triangle_cross(dc, db)
    # triangle ABD -- ratio for c
    area_abd = area_triangle_cross(da, db)

    # compute weight total
    # NOTE: this formula comes from SUMA, and it considers the areas of the
    # triangles even if the point doesn't lie on the plane of the triangle
    weight_total = area_acd + area_cbd + area_abd

    # weights for each point
    w_a = area_cbd / weight_total
    w_b = area_acd / weight_total
    w_c = area_abd / weight_total

    # TODO: check shape of our arrays and watchout for broadcasting
    data_interp = data[i[:, 0]] * w_a[:, np.newaxis] + \
                  data[i[:, 1]] * w_b[:, np.newaxis] + \
                  data[i[:, 2]] * w_c[:, np.newaxis]

    return data_interp