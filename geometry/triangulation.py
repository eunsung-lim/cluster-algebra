from itertools import combinations
from typing import List


def test_triangulation(vertices: List[int]):
    """
    Returns an iterable list of all edges of a triangulation of a convex n-gon
    Use the indices from 1 to n
    want to use like range
    for t in triangulation([1, 2, 3, 4, 5]):
        print(t)
    then it should print all edges of all triangulations
    when n is 5,
    first t is [(1, 2), (1, 3), (1, 4)]
    second t is [(1, 2), (1, 3), (3, 5)]
    :return:
    """
    n = len(vertices)
    r = []
    if n <= 3:
        return [[]]
    for i in range(1, n-1):
        for a in test_triangulation(vertices[:i+1]):
            for b in test_triangulation(vertices[i:]):
                if i == 1:
                    r.append([(vertices[i], vertices[n-1])] + b)
                elif i == n-2:
                    r.append([(vertices[0], vertices[i])] + a)
                else:
                    r.append([(vertices[0], vertices[i]), (vertices[i], vertices[n-1])] + a + b)
    return sorted(r)


def test_lamination(frozens: List[int]):
    return list(combinations(frozens, 2))