import numpy as np
from stability.surf_rotation import (area_triangle_cross,
                             barycentric_interpolation,
                             rotate_surface)


def test_area_triangle_cross():
    # three nice triangles
    ab = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 4, 0]])
    ac = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

    assert(np.array_equal(area_triangle_cross(ab, ac),
                          np.array([0.5,  0.5, 2])))


def test_barycentric_interpolation():
    # identity
    data = np.array([[2], [3], [4]])
    xyz = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    xyz_rnd = xyz
    i = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])

    assert(np.array_equal(
        data, barycentric_interpolation(data, xyz, xyz_rnd, i)))

    # random point lying on xy plane inside the triangle
    data = np.array([[1], [2], [0]])
    xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    for rand_intercept in range(40):
        c = np.random.random()  # in [0, 1.)
        for rand_x in range(10):
            x = np.random.random() * c  # in [0, c)
            y = -x + c
            # fake rotation -- put point inside triangle where we know
            xyz_rnd = np.array([[x, y, 0], [x, y, 0], [x, y, 0]])
            i = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
            # check we get the right numbers
            assert(np.allclose(
                np.dot(xyz_rnd, data),
                barycentric_interpolation(data, xyz, xyz_rnd, i)))


def test_rotate_surface():
    # not really a surface but should work anyway
    xyz = np.random.randn(10, 3)
    xyz_rnd, dist, i = rotate_surface(xyz)

    # just check that we get same shapes
    assert(xyz.shape == xyz_rnd.shape)
    assert(dist.shape[0] == xyz.shape[0])
    assert(i.shape[0] == xyz.shape[0])
