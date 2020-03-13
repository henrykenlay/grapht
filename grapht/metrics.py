# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_metrics.ipynb (unless otherwise specified).

__all__ = ['sparse_norm', 'sparse_2norm', 'sparse_maxnorm', 'laplacian_distance', 'LineDistances',
           'LineDistancesDataset', 'average_gmdegree', 'edge_degree_gm']

# Cell
from nbdev.showdoc import *
from .graphtools import laplacian
from functools import lru_cache
from pathlib import Path
import networkx as nx
import numpy as np
import scipy.sparse as sp

# Cell
def sparse_norm(A, ord=2):
    """Like scipy.sparse.lingalg.norm but with the 2-norm and max norm implemented.

    If `ord=2` or `ord='max'` a grapht implementation is used, otherwise scipy.sparse.lingalg.norm is used.
    """
    if ord == 2:
        return sparse_2norm(A)
    elif ord == 'max':
        return sparse_maxnorm(A)
    else:
        return sp.linalg.norm(A, ord=ord)

def sparse_2norm(A):
    """Returns the matrix 2-norm of a sparse matrix `A`."""
    return sp.linalg.svds(A, k=1, which='LM', return_singular_vectors=False)[0]

def sparse_maxnorm(A):
    """Returns the max |A_ij| for a sparse matrix `A`."""
    return max(-A.min(), A.max())

# Cell
def laplacian_distance(G, Gp, setdiag=False):
    """Calculates $|| \mathcal{L}(G) -  \mathcal{L}(G_p) ||$ using the matrix 2-norm."""
    L = laplacian(G, setdiag)
    Lp = laplacian(Gp, setdiag)
    E = Lp - L
    return sparse_2norm(E)

# Cell
class LineDistances():
    """
    An object which computes the distances of edges in the graphs line graph.
    """

    def __init__(self, G):
        """G is a networkx graph."""
        self.G = G
        self.line_graph = nx.line_graph(G)

    def __call__(self, edge1, edge2):
        """Calculates the linegraph distance between `edge1` and `edge2`."""
        edge1, edge2 = self.sort_edge(edge1), self.sort_edge(edge2)
        return nx.shortest_path_length(self.line_graph, edge1, edge2)

    def sort_edge(self, edge):
        """Makes sure edges are of the form (u, v) where u <= v."""
        if edge[0] <= edge[1]:
            return edge
        else:
            return (edge[1], edge[0])

    def average_distance(self, edges):
        """Calculates the average linegraph distance between all pairs of edges in `edges`."""
        distances = self.pairwise_distances(edges)
        return np.mean(distances)

    def pairwise_distances(self, edges):
        """Calculates the linegraph distance between all pairs of edges in `edges`."""
        distances = []
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                distances.append(self(edges[i], edges[j]))
        return distances


class LineDistancesDataset(LineDistances):
    """
    A LineDistances object for benchmark datasets.

    The linegraph pairwise distances for all edges have been precomputed.

    This is implemented for cora and citeseer.
    """

    def __init__(self, G, dataset):
        """G is a networkx graph and `dataset` is either `cora` or `citeseer`."""
        super(LineDistancesDataset, self).__init__(G)
        self.line_graph_nodes = list(self.line_graph.nodes())
        self.dataset = dataset
        self.load_dataset()

    def load_dataset(self):
        """Loads a precomputed matrix with linegraph distances."""
        fname = Path(__file__).parents[1].joinpath(f'data/{self.dataset}_linegraph_distances.npy')
        self.all_path_lengths = np.load(open(fname, 'rb'))

    def __call__(self, edge1, edge2):
        """Calculates the linegraph distance between `edge1` and `edge2`."""
        i, j = self.edge_index(edge1), self.edge_index(edge2)
        return self.all_path_lengths[i, j]

    @lru_cache(maxsize=None)
    def edge_index(self, edge):
        """Returns the index of the matrix which corresponds to `edge`."""
        return self.line_graph_nodes.index(edge)

# Cell
def average_gmdegree(G, edges):
    """The average edge degree geometric mean over all edges in `edges`."""
    return np.mean([edge_degree_gm(G, edge) for edge in edges])

def edge_degree_gm(G, edge):
    """For an edge (u, v) with degree du, dv this function returns the geometric mean of du and dv."""
    return np.sqrt(G.degree(edge[0]) * G.degree(edge[1]))