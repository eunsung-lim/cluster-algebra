from collections import defaultdict
from itertools import combinations
from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay
import matplotlib.tri as tri
from numpy import cos, sin, pi




class Vertex:
    instances = []

    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.index = Vertex.index
        self.name = name if name else f"v{Vertex.index}"
        Vertex.instances.append(self)
        Vertex.index += 1

    def __repr__(self):
        return f"{self.name}"

    def to_tuple(self):
        return self.x, self.y

    def __lt__(self, other):
        return self.index < other.index

    @classmethod
    def reset_index(cls, start_value=1):
        cls.index = start_value

    @classmethod
    def reset_instances(cls):
        cls.instances = []


Vertex.index = 1  # Vertex 클래스의 정적 변수로 인덱스 관리


def get_vertex_by_index(vertex_index):
    for vertex in Vertex.instances:
        if vertex.index == vertex_index:
            return vertex
    raise ValueError(f"Vertex with index {vertex_index} not found.")


def get_edge_by_index(edge_index):
    for edge in Edge.instances:
        if edge.index == edge_index:
            return edge
    raise ValueError(f"Edge with index {edge_index} not found.")


def get_edge_by_vertices(vertex1, vertex2):
    if vertex1 > vertex2:
        vertex1, vertex2 = vertex2, vertex1
    for edge in Edge.instances:
        if edge.vertex1 == vertex1 and edge.vertex2 == vertex2:
            return edge
    raise ValueError(f"Edge with vertices {vertex1.index} and {vertex2.index} not found.")


def get_edge_by_name(edge_name):
    for edge in Edge.instances:
        if edge.name == edge_name:
            return edge
    raise ValueError(f"Edge with name {edge_name} not found.")


def get_cluster_by_index(cluster_index):
    for c in ClusterVariable.instances:
        if c.cluster_index == cluster_index:
            return c
    raise ValueError(f"Vertex with index {cluster_index} not found.")


class Edge:
    instances = []

    def __init__(self, vertex1, vertex2, name=None):
        self.vertex1 = min(vertex1, vertex2)
        self.vertex2 = max(vertex1, vertex2)
        self.index = Edge.index
        self.name = name if name else f"e{Edge.index}"
        self.line = LineString([vertex1.to_tuple(), vertex2.to_tuple()])

        Edge.instances.append(self)
        Edge.index += 1

    def intersects(self, other):
        return self.line.intersects(other.line)

    @classmethod
    def reset_index(cls, start_value=1):
        cls.index = start_value

    @classmethod
    def reset_instances(cls):
        cls.instances = []


Edge.index = 1  # Edge 클래스의 정적 변수로 인덱스 관리


class FrozenVariable(Edge):
    instances = []

    def __init__(self, vertex1, vertex2, name=None):
        super().__init__(vertex1, vertex2, name)
        FrozenVariable.instances.append(self)

    def __repr__(self):
        return f"{self.name}: {self.vertex1.name} - {self.vertex2.name}"


class ClusterVariable(Edge):
    instances = []

    def __init__(self, vertex1, vertex2, name=None):
        super().__init__(vertex1, vertex2, name)

        self.shear = 0

        self.cluster_index = ClusterVariable.index
        self.name = name if name else f"c{ClusterVariable.index}"
        ClusterVariable.index += 1

        ClusterVariable.instances.append(self)

    def reset_shear(self):
        self.shear = 0

    def set_vertices(self, vertex1, vertex2):
        self.vertex1 = min(vertex1, vertex2)
        self.vertex2 = max(vertex1, vertex2)
        self.line = LineString([vertex1.to_tuple(), vertex2.to_tuple()])

    def __repr__(self):
        return f"{self.name}: {self.vertex1.name} - {self.vertex2.name}"

    @classmethod
    def reset_index(cls, start_value=1):
        cls.index = start_value

    @classmethod
    def reset_instances(cls):
        cls.instances = []


ClusterVariable.index = 1


class Lamination:
    instances = []

    def __init__(self, edge_index1, edge_index2, name=None):
        Lamination.instances.append(self)
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2

        self.edge1 = get_edge_by_index(self.edge_index1)
        self.edge2 = get_edge_by_index(self.edge_index2)

        self.index = Lamination.index
        self.name = name if name else f"l{Lamination.index}"
        Lamination.index += 1

        self.start_point = None
        self.end_point = None
        self.line = None

        self.shear: List[Shear] = None

    def set_vertex_positions(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.line = LineString([start_point.to_tuple(), end_point.to_tuple()])

    def __lt__(self, other):
        if self.edge_index1 != other.edge_index1:
            return self.edge_index1 < other.edge_index1
        if (self.edge_index2 - self.edge_index1) != (other.edge_index2 - other.edge_index1):
            return (self.edge_index2 - self.edge_index1) > (other.edge_index2 - other.edge_index1)
        return self.index < other.index

    pass


Lamination.index = 1


class LaminationList:
    def __init__(self, laminations):
        self.laminations = sorted(laminations)

    def set_vertex_positions(self):
        num_laminations = defaultdict(int)
        for lamination in self.laminations:
            num_laminations[lamination.edge_index1] += 1
            num_laminations[lamination.edge_index2] += 1

        for i, lamination in enumerate(self.laminations):
            p = len(list(filter(lambda x: x.edge_index1 == lamination.edge_index1, self.laminations[:i]))) \
                + len(list(filter(lambda x: x.edge_index2 == lamination.edge_index1, self.laminations[:i]))) + 1
            q = len(list(filter(lambda x: x.edge_index2 == lamination.edge_index2, self.laminations[i + 1:]))) + 1

            edge = lamination.edge1
            num = num_laminations[lamination.edge_index1]
            segment_length_x = (edge.vertex2.x - edge.vertex1.x) / (num + 1)
            segment_length_y = (edge.vertex2.y - edge.vertex1.y) / (num + 1)

            start_x = edge.vertex1.x + p * segment_length_x
            start_y = edge.vertex1.y + p * segment_length_y

            edge = lamination.edge2
            num = num_laminations[lamination.edge_index2]
            segment_length_x = (edge.vertex2.x - edge.vertex1.x) / (num + 1)
            segment_length_y = (edge.vertex2.y - edge.vertex1.y) / (num + 1)

            end_x = edge.vertex1.x + q * segment_length_x
            end_y = edge.vertex1.y + q * segment_length_y

            lamination.set_vertex_positions(
                Vertex(start_x, start_y),
                Vertex(end_x, end_y)
            )

    def visualize(self):
        self.set_vertex_positions()

        for lamination in self.laminations:
            points = [lamination.start_point, lamination.end_point]
            line = LineString([point.to_tuple() for point in points])
            plt.plot(*line.xy, color='green')

    pass


def exchange_matrix():
    pass


class Shear(Vertex):
    def __init__(self, x, y,
                 cluster: ClusterVariable,
                 lamination: Lamination,
                 value: int,
                 name=None):
        super().__init__(x, y, name)
        self.cluster = cluster
        self.lamination = lamination
        self.value = value

    def plot(self):
        plt.plot(self.x, self.y, 'bo' if self.value > 0 else 'ro')
        pass

    pass


class Arrow:
    instances = []

    def __init__(self, cluster1, cluster2, name=None):
        self.cluster1 = cluster1
        self.cluster2 = cluster2

        self.index = Arrow.index
        self.name = name if name else f"a{Arrow.index}"
        Arrow.index += 1
        Arrow.instances.append(self)

    def __repr__(self):
        return f"{self.name}: {self.cluster1.name} -> {self.cluster2.name}"

    def plot(self):
        # plot an arrow from midpoint of cluster1 to midpoint of cluster2
        x1 = (self.cluster1.vertex1.x + self.cluster1.vertex2.x) / 2
        y1 = (self.cluster1.vertex1.y + self.cluster1.vertex2.y) / 2
        x2 = (self.cluster2.vertex1.x + self.cluster2.vertex2.x) / 2
        y2 = (self.cluster2.vertex1.y + self.cluster2.vertex2.y) / 2
        start = (x1, y1)
        end = (x2, y2)

        ax = plt.gca()
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', color='red', shrinkA=10, shrinkB=10, mutation_scale=20)
        # plt.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1,)
        ax.add_patch(arrow)
    pass

Arrow.index = 1


def is_connected(v1, v2):
    v1 = get_vertex_by_index(v1)
    v2 = get_vertex_by_index(v2)
    if v1 > v2:
        v1, v2 = v2, v1
    return any(e.vertex1 == v1 and e.vertex2 == v2 for e in Edge.instances)


def remove_vertex(vertex_index):
    for vertex in Vertex.instances:
        if vertex.index == vertex_index:
            Vertex.instances.remove(vertex)
            return
    raise ValueError(f"Vertex with index {vertex_index} not found.")


def remove_edge(edge_index):
    for edge in Edge.instances:
        if edge.index == edge_index:
            Edge.instances.remove(edge)
            if isinstance(edge, ClusterVariable):
                ClusterVariable.instances.remove(edge)
            elif isinstance(edge, FrozenVariable):
                FrozenVariable.instances.remove(edge)
            return
    raise ValueError(f"Edge with index {edge_index} not found.")


class Quiver:
    def __init__(self, vertices, frozens, clusters, laminations):
        self.vertices = vertices
        self.frozens = frozens
        self.clusters = clusters
        self.laminations = laminations
        self.laminations.set_vertex_positions()
        self.shears = []
        self.find_shear()
        self.arrows = []
        self.set_arrows()

    def reset_shear(self):
        self.shears = []
        self.find_shear()

    def flip(self, cluster_index):
        c = get_cluster_by_index(cluster_index)
        v1, v2 = c.vertex1, c.vertex2
        v3 = []
        for v in Vertex.instances:
            if is_connected(v1.index, v.index) and is_connected(v2.index, v.index):
                v3.append(v)
        if len(v3) != 2:
            raise ValueError("There are more than two vertices connected to both v1 and v2.")
        v3, v4 = v3
        c.set_vertices(v3, v4)
        self.reset_shear()
        self.find_shear()
        self.reset_arrows()
        pass

    def plot(self,
        show_vertex_labels=True,
        show_frozen_labels=True,
        show_cluster_labels=True,
        show_arrows=True,
        show_shears=True,
    ):

        G = nx.Graph()

        pos = {i: vertex.to_tuple() for i, vertex in enumerate(self.vertices)}
        vertex_labels = {i: vertex.name for i, vertex in enumerate(self.vertices)} if show_vertex_labels else {}
        G.add_nodes_from(pos.keys())

        frozen_list = [(self.vertices.index(f.vertex1), self.vertices.index(f.vertex2)) for f in self.frozens]
        frozen_labels = {frozen_list[i]: f.name for i, f in enumerate(self.frozens)} if show_frozen_labels else {}
        G.add_edges_from(frozen_list)

        cluster_list = [(self.vertices.index(c.vertex1), self.vertices.index(c.vertex2)) for c in self.clusters]
        cluster_labels = {cluster_list[i]: c.name for i, c in enumerate(self.clusters)} if show_cluster_labels else {}
        G.add_edges_from(cluster_list)

        nx.draw(G, pos, labels=vertex_labels, with_labels=show_vertex_labels, node_color='lightblue', node_size=500,
                font_size=10)

        if show_frozen_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=frozen_labels)

        if show_cluster_labels:
            bbox = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='yellow', alpha=1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=cluster_labels, font_color='red', bbox=bbox)

        if self.laminations:
            self.laminations.visualize()

        if show_arrows:
            self.reset_arrows()
            for arrow in self.arrows:
                arrow.plot()

        if show_shears:
            self.reset_shear()
            for shear in self.shears:
                shear.plot()

        plt.show()

    def find_shear(self):
        for c in self.clusters:
            c.reset_shear()

        for lamination in self.laminations.laminations:
            # for a lamination, find all the clusters that intersect with the lamination
            clusters = [lamination.edge2] + [c for c in self.clusters if lamination.line.intersects(c.line)]
            v = [c.vertex1 for c in clusters] + [c.vertex2 for c in clusters]
            v = list(set(v))
            v1 = lamination.edge1.vertex1 if lamination.edge1.vertex1 not in v else lamination.edge1.vertex2
            v2 = lamination.edge1.vertex2 if lamination.edge1.vertex2 in v else lamination.edge1.vertex1
            vec = np.array([v2.x - v1.x, v2.y - v1.y])
            vec = vec / np.linalg.norm(vec)

            path_v = [v1]
            path_e = [lamination.edge1]

            while True:
                next_clusters = [c for c in clusters if c.vertex1 == v2 or c.vertex2 == v2]
                nv1 = v2
                nv2 = None
                # find the cluster whose inner product with vec is the smallest
                max_dot = -1
                next_edge = None
                for c in next_clusters:
                    v3 = c.vertex1 if c.vertex1 != v2 else c.vertex2
                    new_vec = np.array([v3.x - v2.x, v3.y - v2.y])
                    new_vec = new_vec / np.linalg.norm(new_vec)
                    dot = np.dot(vec, new_vec)
                    if dot > max_dot:
                        max_dot = dot
                        nv1 = v2
                        nv2 = v3
                        next_edge = c
                path_v.append(nv1)
                path_e.append(next_edge)
                v2 = nv2
                vec = np.array([v2.x - v1.x, v2.y - v1.y])
                vec = vec / np.linalg.norm(vec)
                if isinstance(next_edge, FrozenVariable):
                    path_v.append(nv2)
                    break

            if len(path_v) == 2:
                continue

            # clockwise or counterclockwise
            sign = 1 if (path_v[0].index + 1 - path_v[1].index) % len(self.vertices) == 0 else -1
            for i in range(1, len(path_e) - 1):
                path_e[i].shear += sign
                intersection = path_e[i].line.intersection(lamination.line)
                self.shears.append(Shear(
                    intersection.x,
                    intersection.y,
                    path_e[i],
                    lamination,
                    sign
                ))

                sign *= -1

            # print(path_v)
            # print(path_e)
            # for c in self.clusters:
            #     print(c, c.shear)

    def find_triangles(self):
        triangles = []
        for c in combinations(self.vertices, 3):
            if all(is_connected(v1.index, v2.index) for v1, v2 in combinations(c, 2)):
                triangles.append(c)
        return triangles

    def dist(self, v1, v2):
        n = len(self.vertices)
        return min(abs(v1 - v2), n - abs(v1 - v2))

    def set_arrows(self):
        n = len(self.vertices)
        arrows = []
        triangles = self.find_triangles()
        for triangle in triangles:
            a, b, c = sorted(triangle, key=lambda x: x.index)
            if b.index - a.index >= 2 and c.index - b.index >= 2:
                arrows.append(
                    Arrow(
                        get_edge_by_vertices(b, c),
                        get_edge_by_vertices(a, b),
                    )
                )
            if c.index - b.index >= 2 and a.index + n - c.index >= 2:
                arrows.append(
                    Arrow(
                        get_edge_by_vertices(c, a),
                        get_edge_by_vertices(b, c),
                    )
                )
            if a.index + n - c.index >= 2 and b.index - a.index >= 2:
                arrows.append(
                    Arrow(
                        get_edge_by_vertices(a, b),
                        get_edge_by_vertices(c, a),
                    )
                )
        self.arrows = arrows
    pass

    def reset_arrows(self):
        self.arrows = []
        self.set_arrows()
        pass

    def get_exchange_matrix(self):
        n = len(self.clusters)
        mtx = [[0 for _ in range(n)] for _ in range(n+1)]
        for arrow in self.arrows:
            mtx[arrow.cluster1.cluster_index-1][arrow.cluster2.cluster_index-1] = 1
            mtx[arrow.cluster2.cluster_index-1][arrow.cluster1.cluster_index-1] = -1
        for cluster in self.clusters:
            mtx[n][cluster.cluster_index-1] = cluster.shear

        row_names = [c.name for c in self.clusters] + ['Shear']
        col_names = [c.name for c in self.clusters]
        df = pd.DataFrame(mtx, index=row_names, columns=col_names)
        return df

def matrix_mutation(df: pd.DataFrame, k):
    k -= 1
    b = df.to_numpy()
    nb = np.zeros_like(b)
    m = len(b)
    n = len(b[0])
    for i in range(m):
        for j in range(n):
            if i == k or j == k:
                nb[i][j] = -b[i][j]
            elif b[i][k] > 0 and b[k][j] > 0:
                nb[i][j] = b[i][j] + b[i][k] * b[k][j]
            elif b[i][k] < 0 and b[k][j] < 0:
                nb[i][j] = b[i][j] - b[i][k] * b[k][j]
            else:
                nb[i][j] = b[i][j]
    return pd.DataFrame(nb, index=df.index, columns=df.columns)
