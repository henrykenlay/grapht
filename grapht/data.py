# AUTOGENERATED! DO NOT EDIT! File to edit: 05_data.ipynb (unless otherwise specified).

__all__ = ['cora', 'make_planar_graph', 'BAGraph', 'SensorGraph', 'CoraGraph']

# Cell
from nbdev.showdoc import *
import numpy as np
import networkx as nx
import scipy

# Cell
def cora():
    """
    Returns A, X, y where
        A : is the adjacency matrix
        X : is the feature matrix
        y : are the node labels
    """
    cora = np.load('data/cora_gnnbench.npz', allow_pickle=True)
    A, X, y = cora['A'].tolist(), cora['X'].tolist(), cora['y']
    return A, X, y

# Cell
def make_planar_graph(n):
    """
    Makes a planar graph with n nodes

    Code adapted from https://stackoverflow.com/questions/26681899/how-to-make-networkx-graph-from-delaunay-preserving-attributes-of-the-input-node
    """
    points = np.random.rand(n, 2)
    delTri = scipy.spatial.Delaunay(points)
    edges = set()
    for n in range(delTri.nsimplex):
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,1], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
    graph = nx.Graph(list(edges))
    pos = pos = dict(zip(range(len(points)), points))
    return graph, pos

# Cell
class BAGraph():

    def __init__(self, n, m):
        self.n = n
        self.m = m

    def generate(self):
        return nx.barabasi_albert_graph(self.n, self.m)

    def number_of_edges(self):
        return self.generate().number_of_edges()

class SensorGraph():
    " KNN sensor graph, this used the github pygsp.graphs.Sensor implementation, not the stable release (i.e. as described in the docs) "

    def __init__(self, n):
        self.n = n
        self.regular = regular

    def generate(self):
        G = pygsp.graphs.Sensor(self.n)
        return nx.Graph(G.W)

    def number_of_edges(self, samples=100):
        graphs = [self.generate() for _ in range(samples)]
        return np.mean([G.number_of_edges() for G in graphs])

class CoraGraph():

    def __init__(self, save_location='/tmp/cora'):
        dataset = torch_geometric.datasets.Planetoid(save_location, 'Cora')
        G = torch_geometric.utils.to_networkx(dataset.data).to_undirected()
        self.cora = max(nx.connected_component_subgraphs(G), key=len)

    def generate(self):
        return self.cora

    def number_of_edges(self):
        return self.cora.number_of_edges()