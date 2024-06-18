from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from queue import Queue
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from overrides import overrides

from abc import ABC, abstractmethod

from shapely import LineString, Point


class Indexable:
    _index_counter = 1

    def __init__(self):
        cls = self.__class__
        if not hasattr(cls, '_index_counter'):
            cls._index_counter = 1
        self.index = cls._index_counter
        cls._index_counter += 1

    @classmethod
    def reset_index(cls):
        cls._index_counter = 1


class InstanceRegistry:
    instances = []

    @classmethod
    def register(cls, instance):
        cls.instances.append(instance)

    @classmethod
    def getinstances(cls):
        return cls.instances

    @classmethod
    def get_instance_by_name(cls, name):
        for instance in cls.instances:
            if instance.name == name:
                return instance
        return None

    @classmethod
    def resetinstances(cls):
        cls.instances = []

    @property
    def instances(self):
        return self.instances


class Plottable(ABC):
    @abstractmethod
    def plot(self) -> None:
        pass

    @abstractmethod
    def add_plot(self, G: nx.Graph) -> nx.Graph:
        pass


def to_latex(expr: str) -> str:
    return r"$" + expr + r"$"


class Vertex(Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, x, y, name=''):
        super().__init__()
        self.x = x
        self.y = y
        self.name = name
        try:
            self.index = int(name)
        except:
            self.index = None
        self.latex = to_latex(name)
        self.symbol = sp.Symbol(name)
        self.point = Point(x, y)
        Vertex.register(self)

    def __eq__(self, other: Vertex) -> bool:
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def plot(self, **kwargs):
        # Create a graph
        G = nx.Graph()

        # Add the vertex
        self.add_plot(G)

        # Plot the graph
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), **kwargs)

    def add_plot(self, G: nx.Graph):
        G.add_node(self.index, pos=(self.x, self.y), label=self.latex)
        return G

    def to_tuple(self):
        return self.x, self.y

    pass


class Edge(Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, v1: VertexLike, v2: VertexLike, name: str = None):
        # Base Attributes
        super().__init__()
        v1 = process_vertex(v1)
        v2 = process_vertex(v2)
        self.v1 = v1
        self.v2 = v2
        self.line = LineString([v1.to_tuple(), v2.to_tuple()])
        if name:
            self.name = name
            self.latex = to_latex(name)
            self.symbol = sp.Symbol(name)
        else:
            self.name = None
            self.latex = None
            self.symbol = None
        Edge.register(self)

    def set_vertices(self, v1: VertexLike, v2: VertexLike):
        self.v1 = process_vertex(v1)
        self.v2 = process_vertex(v2)
        self.line = LineString([self.v1.to_tuple(), self.v2.to_tuple()])

    def __eq__(self, other):
        return (self.v1 == other.v1 and self.v2 == other.v2) \
            or (self.v1 == other.v2 and self.v2 == other.v1)

    def __repr__(self):
        return f"{self.name} : {self.v1} -> {self.v2}"

    def plot(self, **kwargs):
        # Create a graph
        G = nx.Graph()

        # Add the vertices
        self.v1.add_plot(G)
        self.v2.add_plot(G)

        # Add the edge
        self.add_plot(G)

        # Plot the graph
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), **kwargs)

    def add_plot(self, G: nx.Graph):
        G.add_edge(self.v1.index, self.v2.index)
        return G

    def get_midpoint(self):
        return (self.v1.x + self.v2.x) / 2, (self.v1.y + self.v2.y) / 2

    pass


class FrozenVariable(Edge, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        FrozenVariable.register(self)

    pass


class ClusterVariable(Edge, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.varname = f"x_{{{self.name}}}"
        self.varname_org = self.varname
        self.latex = to_latex(self.varname)
        self.symbol = sp.Symbol(self.varname)

        ClusterVariable.register(self)

    @classmethod
    def get_instance_by_varname(cls, varname):
        for instance in cls.instances:
            if instance.varname == varname:
                return instance
        return None

    def __hash__(self):
        return hash(self.varname)

    def set_name(self, name):
        self.name = name
        self.varname = f"x_{{{self.name}}}"
        self.latex = to_latex(f"x_{{{self.name}}}")
        self.symbol = sp.Symbol(f"x_{{{self.name}}}")
        pass

    def __repr__(self):
        return self.varname

    pass


class Polygon(Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, vertices: List[VertexLike], frozens: List[FrozenLike] = None, name=None):
        # Base Attributes
        super().__init__()
        vertices = [process_vertex(v) for v in vertices]

        self.vertices = vertices
        self.name = name
        if frozens:
            frozens = [process_frozen(f) for f in frozens]
            self.frozens = frozens
        else:
            self.frozens = [FrozenVariable(vertices[i], vertices[(i + 1) % len(vertices)]) for i in
                            range(len(vertices))]
        Polygon.register(self)

    def add_plot(self, G):
        for v in self.vertices:
            v.add_plot(G)
        for e in self.frozens:
            e.add_plot(G)
        return G

    def plot(self, **kwargs):
        # Create a graph
        G = nx.Graph()

        # Add the polygon
        self.add_plot(G)

        # Plot the graph
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), **kwargs)

    pass


def process_vertex(v: VertexLike):
    if isinstance(v, str):
        return Vertex.get_instance_by_name(v)
    return v


def process_edge(e: EdgeLike):
    if isinstance(e, str):
        return Edge.get_instance_by_name(e)
    return e


def process_frozen(f: FrozenLike):
    if isinstance(f, str):
        return FrozenVariable.get_instance_by_name(f)
    return f


def process_cluster(c: ClusterLike):
    if isinstance(c, str):
        r1 = ClusterVariable.get_instance_by_name(c)
        if r1:
            return r1
        return ClusterVariable.get_instance_by_varname(c)
    return c


class TriangulatedPolygon(Polygon, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, vertices: List[VertexLike], frozens: List[FrozenLike], clusters: List[ClusterLike] = None):
        super().__init__(vertices, frozens)
        self.polygon = Polygon(vertices, frozens)

        # Triangulation
        if clusters:
            clusters = [process_cluster(c) for c in clusters]
            self.clusters = clusters
        else:
            self.clusters = [ClusterVariable(vertices[0], vertices[i], name=f"x_{i - 1}")
                             for i in range(2, len(vertices) - 1)]

    def add_plot(self, G):
        for v in self.vertices:
            v.add_plot(G)
        for e in self.frozens:
            e.add_plot(G)
        for c in self.clusters:
            c.add_plot(G)
        return G

    def is_connected(self, v1: Vertex, v2: Vertex):
        return any([{v1, v2} == {e.v1, e.v2} for e in self.clusters + self.frozens])

    def get_triangles(self, cluster: ClusterLike):
        cluster = process_cluster(cluster)
        v1, v2 = cluster.v1, cluster.v2
        triangles = []
        for v in self.vertices:
            if self.is_connected(v, v1) and self.is_connected(v, v2):
                triangles.append(v)
        return triangles

    def flip(self, cluster: ClusterLike):
        cluster = process_cluster(cluster)
        v3, v4 = self.get_triangles(cluster)

        # remove v1-v2 edge and add v3-v4 edge
        self.clusters.remove(cluster)
        self.clusters.append(ClusterVariable(v3, v4, name=cluster.name + "'"))
        pass

    def plot(self,
             show_vertex_labels=True,
             show_frozen_labels=True,
             show_cluster_labels=True,
             show_arrows=True,
             show_shears=True,
             **kwargs):
        G = nx.Graph()

        pos = {i: vertex.to_tuple() for i, vertex in enumerate(self.vertices)}
        vertex_labels = {i: vertex.latex for i, vertex in enumerate(self.vertices)} if show_vertex_labels else {}
        G.add_nodes_from(pos.keys())

        frozen_list = [(self.vertices.index(f.v1), self.vertices.index(f.v2)) for f in self.frozens]
        frozen_labels = {frozen_list[i]: f.latex for i, f in enumerate(self.frozens)} if show_frozen_labels else {}
        G.add_edges_from(frozen_list)

        cluster_list = [(self.vertices.index(c.v1), self.vertices.index(c.v2)) for c in self.clusters]
        cluster_labels = {cluster_list[i]: str(c.latex) for i, c in
                          enumerate(self.clusters)} if show_cluster_labels else {}
        G.add_edges_from(cluster_list)

        nx.draw(G, pos, labels=vertex_labels, with_labels=show_vertex_labels, node_color='lightblue', node_size=50,
                font_size=10)

        if show_frozen_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=frozen_labels)

        if show_cluster_labels:
            bbox = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='yellow', alpha=1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=cluster_labels, font_color='red', bbox=bbox)

    pass


class SingleLamination(Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, start: FrozenLike, end: FrozenLike):
        super().__init__()
        start = process_frozen(start)
        end = process_frozen(end)
        self.start = start
        self.end = end
        self.line = LineString(
            [start.get_midpoint(), end.get_midpoint()]
        )

        self.start_point = None
        self.end_point = None

    def set_vertex_position(self, start_point: Vertex, end_point: Vertex):
        self.start_point = start_point
        self.end_point = end_point
        self.line = LineString(
            [start_point.to_tuple(), end_point.to_tuple()]
        )

    def add_plot(self, G):
        pass

    def plot(self, **kwargs):
        # Create a graph
        G = nx.Graph()

        # Add the lamination
        G.add_node('start', pos=self.start.get_midpoint())
        G.add_node('end', pos=self.end.get_midpoint())
        G.add_edge('start', 'end')

        # Plot the graph
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'),
                node_size=0,
                edge_color='red',
                **kwargs
                )

    pass

    def plot_on_snake(self, sd: SnakeDiagram, fig, ax):
        # first, find all shears
        shears = []
        for c in sd.quiver.clusters:
            if c.line.intersects(self.line):
                shears.append(c)

        shears = sorted(shears, key=lambda c:
        Point(self.start.get_midpoint()).distance(
            c.line.intersection(self.line)
        ))
        shears = [self.start, *shears, self.end]

        # Function to calculate quadratic Bézier curve points
        def quadratic_bezier_curve(P0, P1, P2, num_points=100):
            t = np.linspace(0, 1, num_points)
            curve = ((1 - t) ** 2)[:, None] * np.array(P0) + (2 * (1 - t) * t)[:, None] * np.array(P1) + (t ** 2)[:,
                                                                                                         None] * np.array(
                P2)
            return curve

        for box in sd.boxes:

            idx = None

            for i in range(len(shears) - 2):
                if shears[i] in [y.edge for y in box.boundaries] and shears[i + 2] in [y.edge for y in box.boundaries]:
                    idx = i
                    break

            if idx is not None:
                if box.d.edge == shears[idx]:
                    P0 = (box.point.x + 1 / 2, box.point.y)
                else:
                    P0 = (box.point.x, box.point.y + 1 / 2)
                if box.u.edge == shears[idx + 2]:
                    P2 = (box.point.x + 1 / 2, box.point.y + 1)
                else:
                    P2 = (box.point.x + 1, box.point.y + 1 / 2)

                P1 = (box.point.x + 1 / 2, box.point.y + 1 / 2)
                # Calculate quadratic Bézier curve points
                bezier_points = quadratic_bezier_curve(P0, P1, P2)

                # Create Shapely LineString from Bézier points
                bezier_line = LineString(bezier_points)

                # Control points
                # control_x, control_y = zip(*[P0, P1, P2])
                # ax.plot(control_x, control_y, 'ro--', label='Control Points')

                # Bézier curve
                bezier_x, bezier_y = bezier_line.xy
                ax.plot(bezier_x, bezier_y, 'red', label='Quadratic Bézier Curve')

            idx = None

            for i in range(len(shears) - 1):
                if shears[i] in [y.edge for y in box.boundaries] and shears[i + 1] in [y.edge for y in box.boundaries]:
                    idx = i
                    break

            if idx is not None:
                if box.d.edge == shears[idx]:
                    P0 = (box.point.x + 1 / 2, box.point.y)
                    P2 = (box.point.x, box.point.y + 1 / 2)
                else:
                    P0 = (box.point.x + 1, box.point.y + 1 / 2)
                    P2 = (box.point.x + 1 / 2, box.point.y + 1)

                P1 = (box.point.x + 1 / 2, box.point.y + 1 / 2)
                # Calculate quadratic Bézier curve points
                bezier_points = quadratic_bezier_curve(P0, P1, P2)

                # Create Shapely LineString from Bézier points
                bezier_line = LineString(bezier_points)

                # Control points
                # control_x, control_y = zip(*[P0, P1, P2])
                # ax.plot(control_x, control_y, 'ro--', label='Control Points')

                # Bézier curve
                bezier_x, bezier_y = bezier_line.xy
                ax.plot(bezier_x, bezier_y, 'red', label='Quadratic Bézier Curve')

        pass


class Lamination:
    def __init__(self, arcs: List[SingleLamination], name=''):
        self.arcs = arcs
        self.name = name
        if name == '':
            self.varname = 'y'
        else:
            self.varname = f"y_{name}"
        self.latex = to_latex(self.varname)
        self.symbol = sp.Symbol(self.varname)

    def __len__(self):
        return len(self.arcs)

    def __iter__(self):
        return iter(self.arcs)

    def set_vertex_positions(self):
        num_laminations = defaultdict(int)
        for arc in self.arcs:
            num_laminations[arc.start.name] += 1
            num_laminations[arc.end.name] += 1

        for i, arc in enumerate(self.arcs):
            p = len(list(filter(lambda x: x.start.name == arc.start.name, self.arcs[:i]))) \
                + len(list(filter(lambda x: x.end.name == arc.start.name, self.arcs[:i]))) + 1
            q = len(list(filter(lambda x: x.end.name == arc.end.name, self.arcs[i + 1:]))) + 1

            edge = arc.start
            num = num_laminations[arc.start.name]
            segment_length_x = (edge.v2.x - edge.v1.x) / (num + 1)
            segment_length_y = (edge.v2.y - edge.v1.y) / (num + 1)

            start_x = edge.v1.x + p * segment_length_x
            start_y = edge.v1.y + p * segment_length_y

            edge = arc.end
            num = num_laminations[arc.end.name]
            segment_length_x = (edge.v2.x - edge.v1.x) / (num + 1)
            segment_length_y = (edge.v2.y - edge.v1.y) / (num + 1)

            end_x = edge.v1.x + q * segment_length_x
            end_y = edge.v1.y + q * segment_length_y

            arc.set_vertex_position(
                Vertex(start_x, start_y),
                Vertex(end_x, end_y)
            )

    def plot(self):
        self.set_vertex_positions()

        for arc in self.arcs:
            points = [arc.start_point, arc.end_point]
            line = LineString([point.to_tuple() for point in points])
            plt.plot(*line.xy, color='green')


class Shear(Vertex):
    def __init__(self, x, y,
                 cluster: ClusterVariable,
                 lamination: Lamination,
                 value: int,
                 name=''):
        super().__init__(x, y, name)
        self.cluster = cluster
        self.lamination = lamination
        self.value = value

    def plot(self):
        plt.plot(self.x, self.y, 'bo' if self.value > 0 else 'ro')
        pass

    pass


class Arrow:
    index = 1
    instances = []

    def __init__(self,
                 cluster1: ClusterVariable,
                 cluster2: ClusterVariable,
                 name=''):
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
        x1 = (self.cluster1.v1.x + self.cluster1.v2.x) / 2
        y1 = (self.cluster1.v1.y + self.cluster1.v2.y) / 2
        x2 = (self.cluster2.v1.x + self.cluster2.v2.x) / 2
        y2 = (self.cluster2.v1.y + self.cluster2.v2.y) / 2
        start = (x1, y1)
        end = (x2, y2)

        ax = plt.gca()
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', color='red', shrinkA=10, shrinkB=10, mutation_scale=20)
        # plt.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1,)
        ax.add_patch(arrow)

    pass


class Quiver(TriangulatedPolygon):
    def __init__(self,
                 vertices: List[VertexLike],
                 frozens: List[FrozenLike],
                 clusters: List[ClusterLike],
                 lamination: Lamination,
                 name=None):
        super().__init__(vertices, frozens, clusters)
        self.lamination = lamination
        self.name = name
        self.shears = []
        self.shear_dict = defaultdict(int)
        self.set_shear()
        self.arrows = []
        self.set_arrow()
        self.relations = []
        self.expressions = []
        Quiver.register(self)

    def reset_shear(self):
        self.shear_dict = defaultdict(int)
        self.set_shear()

    def plot(self,
             show_vertex_labels=True,
             show_frozen_labels=True,
             show_cluster_labels=True,
             show_arrows=True,
             show_shears=True,
             target: ClusterVariable = None,
             **kwargs):
        super().plot(**kwargs)
        if target:
            target.plot(
                edge_color="purple",
                node_color="gray",
                node_size=100,
                width=5)
        if self.lamination:
            self.lamination.plot()

        if show_arrows:
            self.reset_arrow()
            for arrow in self.arrows:
                arrow.plot()

        if show_shears:
            self.reset_shear()
            for shear in self.shears:
                shear.plot()
        pass

    def get_crossing_clusters(self, target: ClusterLike):
        crossing_clusters = []
        for cluster in self.clusters:
            if cluster.v1 not in [target.v1, target.v2] \
                    and cluster.v2 not in [target.v1, target.v2] \
                    and cluster.line.intersects(target.line):
                crossing_clusters.append(cluster)

        crossing_clusters.sort(
            key=lambda c:  # sort by distance from target.v1
            target.v1.point.distance(
                c.line.intersection(target.line)
            )
        )
        return crossing_clusters

    def get_exchange_matrix(self):
        m = [c.varname for c in self.clusters] + [self.lamination.varname]
        n = [c.varname for c in self.clusters]
        mtx = {i: {j: 0 for j in n} for i in m}
        for arrow in self.arrows:
            mtx[arrow.cluster1.varname][arrow.cluster2.varname] = 1
            mtx[arrow.cluster2.varname][arrow.cluster1.varname] = -1
        for cluster in self.clusters:
            mtx[self.lamination.varname][cluster.varname] = self.shear_dict[cluster]

        df = pd.DataFrame.from_dict(mtx, orient='index')
        return df

    def get_cluster_by_name(self, name: str):
        for cluster in self.clusters:
            if cluster.name == name:
                return cluster
        return None

        return

    def flip(self, cluster: ClusterLike):
        if isinstance(cluster, str):
            cluster = self.get_cluster_by_name(cluster)
        v3, v4 = self.get_triangles(cluster)

        b = self.get_exchange_matrix().to_dict()[cluster.varname]

        # get row names of b whose value is 1
        row_names = [k for k, v in b.items() if v == 1]
        # get row names of b whose value is -1
        row_names_minus = [k for k, v in b.items() if v == -1]

        term1 = 1
        term2 = 1
        for row_name in row_names:
            if row_name == self.lamination.varname:
                term1 *= self.lamination.symbol
            else:
                term1 *= self.get_cluster_by_varname(row_name).symbol
        for row_name in row_names_minus:
            if row_name == self.lamination.varname:
                term2 *= self.lamination.symbol
            else:
                term2 *= self.get_cluster_by_varname(row_name).symbol

        gamma = cluster.symbol
        cluster.set_vertices(v3, v4)
        cluster.set_name(cluster.name + "'")
        gamma_prime = cluster.symbol

        self.relations.append(sp.Eq(gamma_prime * gamma, term1 + term2))
        self.expressions.append(sp.Eq(gamma_prime, (term1 + term2) / gamma))

        self.reset_shear()
        self.reset_arrow()

        pass

    def express(self, target: ClusterVariable):
        q = deepcopy(self)
        crossing_clusters = q.get_crossing_clusters(target)
        for c in crossing_clusters:
            q.flip(c)

        n = len(q.expressions)
        for i in range(n):
            for j in range(i + 1, n):
                q.expressions[j] = q.expressions[j].subs(q.expressions[i].lhs,
                                                         q.expressions[i].rhs).simplify()

        eqns = []

        for expr in q.expressions:
            # Get the common denominator form
            common_denominator_expr = sp.together(expr.rhs)

            # Extract the numerator and denominator
            numerator, denominator = sp.fraction(common_denominator_expr)

            # Expand only the denominator
            expand_numerator = sp.expand(numerator)

            # Combine the expanded denominator with the numerator
            result = expand_numerator / denominator

            eqn = sp.Eq(expr.lhs, result)
            eqns.append(eqn)
        return eqns

    pass

    def set_shear(self):
        for lam in self.lamination:
            # for a lamination, find all the clusters that intersect with the lamination
            clusters = [lam.end] + [c for c in self.clusters if lam.line.intersects(c.line)]
            v = [c.v1 for c in clusters] + [c.v2 for c in clusters]
            v = list(set(v))
            v1 = lam.start.v1 if lam.start.v1 not in v else lam.start.v2
            v2 = lam.start.v2 if lam.start.v2 in v else lam.start.v1
            vec = np.array([v2.x - v1.x, v2.y - v1.y])
            vec = vec / np.linalg.norm(vec)

            path_v = [v1]
            path_e = [lam.start]

            while True:
                next_clusters = [c for c in clusters if c.v1 == v2 or c.v2 == v2]
                nv1 = v2
                nv2 = None
                # find the cluster whose inner product with vec is the smallest
                max_dot = -1
                next_edge = None
                for c in next_clusters:
                    v3 = c.v1 if c.v1 != v2 else c.v2
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
                intersection = path_e[i].line.intersection(lam.line)

                self.shears.append(Shear(
                    intersection.x,
                    intersection.y,
                    path_e[i],
                    lam,
                    sign
                ))
                self.shear_dict[path_e[i]] += sign

                sign *= -1

    def get_all_triangles(self):
        triangles = []
        for c in combinations(self.vertices, 3):
            if all(self.is_connected(v1, v2) for v1, v2 in combinations(c, 2)):
                triangles.append(c)
        return triangles

    def get_edge_by_vertex(self, v1: Vertex, v2: Vertex):
        for e in self.clusters + self.frozens:
            if {v1, v2} == {e.v1, e.v2}:
                return e
        raise ValueError("No edge found", v1, v2)

    def set_arrow(self):
        n = len(self.vertices)
        arrows = []
        triangles = self.get_all_triangles()
        for triangle in triangles:
            a, b, c = sorted(triangle, key=lambda x: x.index)
            if b.index - a.index >= 2 and c.index - b.index >= 2:
                arrows.append(
                    Arrow(
                        self.get_edge_by_vertex(b, c),
                        self.get_edge_by_vertex(a, b),
                    )
                )
            if c.index - b.index >= 2 and a.index + n - c.index >= 2:
                arrows.append(
                    Arrow(
                        self.get_edge_by_vertex(c, a),
                        self.get_edge_by_vertex(b, c),
                    )
                )
            if a.index + n - c.index >= 2 and b.index - a.index >= 2:
                arrows.append(
                    Arrow(
                        self.get_edge_by_vertex(a, b),
                        self.get_edge_by_vertex(c, a),
                    )
                )
        self.arrows = arrows

    def reset_arrow(self):
        self.arrows = []
        self.set_arrow()
        pass

    def get_cluster_by_varname(self, varname):
        for cluster in self.clusters:
            if cluster.varname == varname:
                return cluster
        return None


class Side(Edge, Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self):
        super().__init__()
        pass


class SnakeEdge(Edge, Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self, edge: EdgeLike, name=None, value=0):
        edge = process_edge(edge)
        super().__init__(edge.v1, edge.v2, edge.name)
        self.edge = edge
        self.is_minimal = False
        self.is_boundary = True
        self.is_colored = False
        self.value = value

    def set_value(self, sign):
        self.value = sign

    pass


class SnakeFrozenEdge(SnakeEdge, FrozenVariable):
    instances = []

    def __init__(self, edge: EdgeLike, name=None):
        super().__init__(edge, name)
        pass

    pass


class SnakeClusterEdge(SnakeEdge, ClusterVariable):
    instances = []

    def __init__(self, edge: EdgeLike, name=None):
        super().__init__(edge, name)
        pass

    pass


class Box(Plottable, Indexable, InstanceRegistry):
    instances = []

    def __init__(self,
                 d: SnakeEdge,
                 l: SnakeEdge,
                 u: SnakeEdge,
                 r: SnakeEdge,
                 diagonal: SnakeEdge,  # / diagonal
                 name=None,
                 direction: str = None,
                 sign: int = 1,
                 ):
        super().__init__()
        self.d = d
        self.l = l
        self.u = u
        self.r = r
        self.boundaries = [d, l, u, r]
        self.diagonal = diagonal
        self.sign = sign

        if direction:
            for dir in direction:
                self.__getattribute__(dir).is_colored = True

        self.name = name
        self.x = None
        self.y = None
        self.point = None
        self.prev = None
        self.prev_direction = None
        self.next = None
        self.next_direction = None
        Box.register(self)

    def set_point(self, x, y):
        self.x = x
        self.y = y
        self.point = Point(x, y)

    def flip(self):
        self.d, self.l = self.l, self.d
        self.u, self.r = self.r, self.u

    def rotate(self):
        self.d, self.u = self.u, self.d
        self.l, self.r = self.r, self.l

    def plot(self):
        pass

    def add_plot(self, G: nx.Graph):
        # draw a box with Edges d, l, u, r and diagonal
        # diagonal is \ diagonal
        # d, l, u, r are in clockwise order
        G.add_node(1, pos=(self.x, self.y))
        G.add_node(2, pos=(self.x, self.y + 1))
        G.add_node(3, pos=(self.x + 1, self.y + 1))
        G.add_node(4, pos=(self.x + 1, self.y))

        G.add_edge(1, 2, name=self.l.latex)
        G.add_edge(2, 3, name=self.u.latex)
        G.add_edge(3, 4, name=self.r.latex)
        G.add_edge(4, 1, name=self.d.latex)
        G.add_edge(2, 4, name=self.diagonal.latex)
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=10)
        nx.draw_networkx_edge_labels(
            G,
            pos=nx.get_node_attributes(G, 'pos'),
            edge_labels=nx.get_edge_attributes(G, 'name'),
            font_color='blue')

        # Color the edges
        G = nx.Graph()
        G.add_node(1, pos=(self.x, self.y))
        G.add_node(2, pos=(self.x, self.y + 1))
        G.add_node(3, pos=(self.x + 1, self.y + 1))
        G.add_node(4, pos=(self.x + 1, self.y))

        edge_thickness = 3
        # Yello RGBA
        # edge_color = (0.9, 0.9, 0.1, 0.5)  # RGBA color (last value is opacity)
        # Green RGBA
        edge_color = (0.1, 0.9, 0.1, 0.5)  # RGBA color (last value is opacity)

        if self.d.is_colored:
            G.add_edge(4, 1)
        if self.l.is_colored:
            G.add_edge(1, 2)
        if self.u.is_colored:
            G.add_edge(2, 3)
        if self.r.is_colored:
            G.add_edge(3, 4)

        nx.draw(
            G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_size=0,
            edge_color=[edge_color] * G.number_of_edges(),
            width=edge_thickness,
        )

        pass

    def set_next(self, next: Box):
        self.next = next
        pass

    def set_prev(self, prev: Box):
        self.prev = prev
        pass

    def __repr__(self):
        return f"Box({self.d.name}, {self.l.name}, {self.u.name}, {self.r.name}, {self.diagonal.name})"

    pass


def cross_product(v1, v2, c=None):
    if c:
        v1x = v1.x - c.x
        v1y = v1.y - c.y
        v2x = v2.x - c.x
        v2y = v2.y - c.y
    else:
        v1x, v1y = v1
        v2x, v2y = v2
    return np.cross(
        (v1x, v1y), (v2x, v2y)
    )


class SnakeDiagram:
    def __init__(self, quiver: Quiver, target: ClusterLike):
        target = process_cluster(target)
        self.quiver = quiver
        self.target = target
        self.boxes = []
        self.adjacents = []
        self.crossing_clusters = []
        self.set_boxes()

    def get_min_perfect_matching(self) -> PerfectMatching:
        if len(self.boxes) == 1:
            return ['ud']
        color_directions = ['' for _ in range(len(self.boxes))]
        for i, box in enumerate(self.boxes):
            if i % 2 == 0:
                if box.d.is_boundary:
                    color_directions[i] += 'd'
                if box.u.is_boundary:
                    color_directions[i] += 'u'
            else:
                if box.l.is_boundary:
                    color_directions[i] += 'l'
                if box.r.is_boundary:
                    color_directions[i] += 'r'
        return PerfectMatching(self, color_directions)

    def get_all_perfect_matchings(self) -> List[PerfectMatching]:
        min_pm = self.get_min_perfect_matching()
        all_pm = [min_pm]
        all_color_directions = [min_pm.get_color_directions()]
        q = Queue()
        q.put(min_pm)
        while not q.empty():
            pm = q.get()
            for i, cbox in enumerate(pm.colored_boxes):
                if pm.is_flippable(cbox):
                    new_pm = deepcopy(pm)
                    new_pm.flip(i)
                    new_pm.height += 1
                    if new_pm.get_color_directions() not in all_color_directions:
                        all_pm.append(new_pm)
                        all_color_directions.append(new_pm.get_color_directions())
                        q.put(new_pm)

                        pm.upper_flips.append(i)
                        new_pm.lower_flips.append(i)

        return all_pm

    def plot_quiver(self, **kwargs):
        # Find all cluster variables from target.v1 to target.v2
        self.quiver.plot()
        self.target.plot(**kwargs)
        pass

    def set_boxes(self):
        self.boxes = []

        self.adjacents = []
        self.crossing_clusters = self.quiver.get_crossing_clusters(self.target)

        x, y = 0, 0
        for i, c in enumerate(self.crossing_clusters):

            v1, v2 = self.quiver.get_triangles(c)
            if i == 0:
                start = self.target.v1
                end = v1 if start == v2 else v2
            elif v1 == self.crossing_clusters[i - 1].v1 or v1 == self.crossing_clusters[i - 1].v2:
                start = v1
                end = v2
            else:
                start = v2
                end = v1
            if cross_product(c.v1, c.v2, start) > 0:
                d = self.quiver.get_edge_by_vertex(start, c.v1)
                l = self.quiver.get_edge_by_vertex(start, c.v2)
            else:
                d = self.quiver.get_edge_by_vertex(start, c.v2)
                l = self.quiver.get_edge_by_vertex(start, c.v1)

            if cross_product(c.v1, c.v2, end) > 0:
                u = self.quiver.get_edge_by_vertex(end, c.v1)
                r = self.quiver.get_edge_by_vertex(end, c.v2)
            else:
                u = self.quiver.get_edge_by_vertex(end, c.v2)
                r = self.quiver.get_edge_by_vertex(end, c.v1)

            d = SnakeClusterEdge(d) if isinstance(d, ClusterVariable) else SnakeFrozenEdge(d)
            l = SnakeClusterEdge(l) if isinstance(l, ClusterVariable) else SnakeFrozenEdge(l)
            u = SnakeClusterEdge(u) if isinstance(u, ClusterVariable) else SnakeFrozenEdge(u)
            r = SnakeClusterEdge(r) if isinstance(r, ClusterVariable) else SnakeFrozenEdge(r)
            diag = SnakeClusterEdge(c)

            # flip if i is odd
            if i % 2 == 1:
                d, l, u, r = l, d, r, u

            if i == 0:
                box = Box(d, l, u, r, diag)

            elif self.boxes[-1].u.edge in [d.edge, u.edge]:
                if self.boxes[-1].u.edge == u.edge:
                    d, l, u, r = u, r, d, l
                d = self.boxes[-1].u
                d.is_boundary = False
                box = Box(d, l, u, r, diag)
                y += 1
            else:
                # elif self.boxes[-1].r.edge in [l.edge, r.edge]:
                if self.boxes[-1].r.edge == r.edge:
                    d, l, u, r = u, r, d, l
                l = self.boxes[-1].r
                l.is_boundary = False
                box = Box(d, l, u, r, diag)
                x += 1

            box.set_point(x, y)
            self.boxes.append(box)

        for i, box in enumerate(self.boxes):
            box.value = 1 if i % 2 == 0 else -1

            box.l.value = 1 if i % 2 == 0 else 0
            box.r.value = 1 if i % 2 == 0 else 0

            box.u.value = 0 if i % 2 == 0 else 1
            box.d.value = 0 if i % 2 == 0 else 1

        for i in range(len(self.boxes) - 1):
            self.boxes[i].set_next(self.boxes[i + 1])
            self.boxes[i + 1].set_prev(self.boxes[i])
            if self.boxes[i].u == self.boxes[i + 1].d:
                self.boxes[i].next_direction = 'u'
                self.boxes[i + 1].prev_direction = 'u'
            else:
                self.boxes[i].next_direction = 'r'
                self.boxes[i + 1].prev_direction = 'l'

    def plot(self, **kwargs):
        xmin = min([box.x for box in self.boxes]) - 1 / 2
        xmax = max([box.x for box in self.boxes]) + 1 / 2
        ymin = min([box.y for box in self.boxes]) - 1 / 2
        ymax = max([box.y for box in self.boxes]) + 1 / 2

        fig, ax = plt.subplots(figsize=(xmax - xmin + 1, ymax - ymin + 1))

        G = nx.Graph()
        idx = 0
        for box in self.boxes:
            box.add_plot(G)
            idx += 4
        pass

        for lam in self.quiver.lamination:
            lam.plot_on_snake(self, fig, ax)

    pass


class ColoredBox(Box):
    def __init__(self, box: Box, direction: str):
        super().__init__(box.d, box.l, box.u, box.r, box.diagonal, direction=direction)
        super().set_point(box.x, box.y)
        # direction is at most two characters of 'udlr'
        self.direction = direction
        pass

    def add_plot(self, G: nx.Graph):
        super().add_plot(G)

        # draw colored box
        # Define edge properties

        # # Draw the graph
        # edges = nx.draw_networkx_edges(G, pos, width=edge_thickness, edge_color=[edge_color] * G.number_of_edges())
        # # nodes = nx.draw_networkx_nodes(G, pos)
        # labels = nx.draw_networkx_labels(G, pos)

    def expr(self):
        formula = 1
        if self.l.is_colored and isinstance(self.l, ClusterVariable):
            formula *= self.l.symbol
        if self.r.is_colored and isinstance(self.r, ClusterVariable):
            formula *= self.r.symbol
        if self.u.is_colored and isinstance(self.u, ClusterVariable):
            formula *= self.u.symbol
        if self.d.is_colored and isinstance(self.d, ClusterVariable):
            formula *= self.d.symbol
        return formula


class PerfectMatching(SnakeDiagram):
    def __init__(self, sd: SnakeDiagram, color_directions: List[str]):
        if len(color_directions) != len(sd.boxes):
            raise ValueError("The number of colors must be equal to the number of boxes.")
        super().__init__(sd.quiver, sd.target)
        self.color_directions = color_directions
        self.colored_boxes = [ColoredBox(box, color_directions[i]) for i, box in enumerate(self.boxes)]

        self.height = 0
        # lower_pm_dict[box's index] = Perfect matching that can be obtained by flipping the box
        # upper_pm_dict[box's index] = Perfect matching that can be obtained by flipping the box
        self.lower_flips = []
        self.upper_flips = []
        pass

    def __eq__(self, other: PerfectMatching):
        return self.get_color_directions() == other.get_color_directions()

    def plot(self):
        G = nx.Graph()
        super().plot()
        for cbox in self.colored_boxes:
            cbox.add_plot(G)
        x, y = self.colored_boxes[0].x, self.colored_boxes[0].y
        plt.text(x, y - 0.5, "Term : " + self.expr(latex=True) + "\t" + f"Height : {self.height}", fontsize=12)
        pass

    def save_plot(self, filename):
        self.plot()
        plt.savefig(filename)

    def expr(self, latex=False):
        y = sp.symbols('y')

        sub_expr = sp.Integer(1)
        for cbox in self.colored_boxes:
            sub_expr = sub_expr * cbox.expr()
        poly = self.quiver.express(self.target)[-1].rhs.as_numer_denom()[0]
        formula = None
        if sub_expr == 1:
            for term in poly.as_ordered_terms():
                if term.subs(y, 1).is_number:
                    formula = term
        else:
            for term in poly.as_ordered_terms():
                if term.has(sub_expr) and not any(var in (term / sub_expr).free_symbols for var in sub_expr.free_symbols):
                    formula = term
                    break
        if latex:
            return "$" + sp.latex(formula) + "$"
        return formula

    def is_flippable(self, cbox: ColoredBox):
        if cbox not in self.colored_boxes:
            raise ValueError("This box is not in the perfect matching.")
        return (cbox.u.is_colored and cbox.d.is_colored) or (cbox.l.is_colored and cbox.r.is_colored)

    def flip(self, cbox: CboxLike):
        if isinstance(cbox, int):
            cbox = self.colored_boxes[cbox]
        if not self.is_flippable(cbox):
            raise ValueError("This box cannot be flipped.")
        if cbox.u.is_colored and cbox.d.is_colored:
            cbox.u.is_colored = False
            cbox.d.is_colored = False
            cbox.l.is_colored = True
            cbox.r.is_colored = True
        elif cbox.l.is_colored and cbox.r.is_colored:
            cbox.l.is_colored = False
            cbox.r.is_colored = False
            cbox.u.is_colored = True
            cbox.d.is_colored = True
        else:
            raise ValueError("This box cannot be flipped.")
        self.color_directions = self.get_color_directions()

    def get_color_directions(self):
        color_directions = ['' for _ in self.boxes]
        for i, cbox in enumerate(self.colored_boxes):
            if cbox.u.is_colored:
                color_directions[i] += 'u'
            if cbox.d.is_colored:
                color_directions[i] += 'd'
            if cbox.l.is_colored:
                color_directions[i] += 'l'
            if cbox.r.is_colored:
                color_directions[i] += 'r'
        return color_directions

    def __lt__(self, other: PerfectMatching):
        if self.height != other.height:
            return self.height < other.height
        return min(self.lower_flips) < min(other.lower_flips)


VertexLike = Union[Vertex, str]
EdgeLike = Union[Edge, str]
FrozenLike = Union[FrozenVariable, str]
ClusterLike = Union[ClusterVariable, str]
CboxLike = Union[ColoredBox, int]
