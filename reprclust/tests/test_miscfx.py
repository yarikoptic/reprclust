import numpy as np
from reprclust.miscfx import triangles2edges


def test_triangles2edges():
    # a very simple graph (a square)
    triangles1 = np.array([[1, 2, 3],
                           [1, 3, 4]])
    edges1 = [[1, 2], [1, 3], [2, 3], [1, 4], [3, 4]]
    edges1.sort()
    assert np.all(triangles2edges(triangles1) == edges1)

    # a more complex graph (two triangles with only a shared vertex)
    triangles2 = np.array([[1, 2, 3], [3, 4, 5]])
    edges2 = [[1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5]]
    edges2.sort()
    assert np.all(triangles2edges(triangles2) == edges2)

