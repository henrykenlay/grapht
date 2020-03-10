import networkx as nx
import numpy as np
from grapht.graphtools import *

G = nx.barabasi_albert_graph(100, 2)
G.add_edge(0, 1) # the initial condition of BA(n, 2) means it can have pendant edges, this stops that happening
G_with_pendant = G.copy()
G_with_pendant.add_node(100)
G_with_pendant.add_edge(0, 100)
G_with_isolate = G.copy()
G_with_isolate.add_node(100)

def test_non_pendant_edges():
    assert non_pendant_edges(G) == list(G.edges())
    assert non_pendant_edges(G_with_isolate) == list(G_with_isolate.edges())
    assert non_pendant_edges(G_with_pendant) != list(G_with_pendant.edges())
    assert set(G_with_pendant.edges()) - set(non_pendant_edges(G_with_pendant)) == {(0, 100)}

def test_is_pendant():
    assert is_pendant(G_with_pendant, (0, 100))
    assert is_pendant(G_with_pendant, (100, 0))
    assert np.sum([is_pendant(G, edge) for edge in G.edges()]) == 0
    assert np.sum([is_pendant(G_with_isolate, edge) for edge in G.edges()]) == 0
    assert np.sum([is_pendant(G_with_pendant, edge) for edge in G_with_pendant.edges()]) == 1


def test_has_isolated_nodes():
    assert not has_isolated_nodes(G)
    assert not has_isolated_nodes(G_with_pendant)
    assert has_isolated_nodes(G_with_isolate)

def test_edges_removed():
    G_with_edge = nx.Graph()
    G_with_edge.add_nodes_from([0, 1])
    G_no_edge = G_with_edge.copy()
    G_with_edge.add_edge(0, 1)
    assert edges_removed(G_with_edge, G_no_edge) == [(0, 1)]
    assert edges_removed(G_no_edge, G_with_edge) == []

def test_laplacian():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3)])
    L = laplacian(G)
    L_manual = np.eye(4)
    L_manual[0][1] = L_manual[1][0] = -1/np.sqrt(3) #edge 0 -- 1
    L_manual[1][2] = L_manual[2][1] = -1/np.sqrt(6) #edge 1 -- 2
    L_manual[1][3] = L_manual[3][1] = -1/np.sqrt(6) #edge 1 -- 3
    L_manual[2][3] = L_manual[3][2] = -1/np.sqrt(4) #edge 2 -- 3
    assert np.allclose(L_manual, L.todense())

    # isolated node case
    G.remove_edge(0, 1)
    L = laplacian(G)
    L_manual[0][0] = 0
    L_manual[0][1] = L_manual[1][0] = 0 #edge 0 -- 1
    L_manual[1][2] = L_manual[2][1] = -1/np.sqrt(4) #edge 1 -- 2
    L_manual[1][3] = L_manual[3][1] = -1/np.sqrt(4) #edge 1 -- 3
    assert np.allclose(L_manual, L.todense())

    # isolated node but with diag set to 1
    L = laplacian(G, setdiag=True)
    L_manual[0][0] = 1
    assert np.allclose(L_manual, L.todense())