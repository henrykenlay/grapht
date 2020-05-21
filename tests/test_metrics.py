from grapht.metrics import *
from grapht.data import get_benchmark
from grapht.sampling import sample_edges
import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
from pathlib import Path


def test_sparse_norm():
    A = sp.csr_matrix(np.random.rand(100, 100))
    for ord in [None, 'fro', np.inf, -np.inf, 1, -1]:
        print(ord)
        assert np.allclose(sparse_norm(A, ord=ord), sp.linalg.norm(A, ord=ord))
    assert np.allclose(sparse_norm(A, ord='max'), sparse_maxnorm(A))
    assert np.allclose(sparse_norm(A, ord=2), sparse_2norm(A))
    
def test_sparse_2norm():
    A = np.random.rand(100, 100)
    assert np.allclose(np.linalg.norm(A, ord=2), sparse_2norm(sp.csr_matrix(A)))
    A = np.random.rand(100, 100)
    A = A + A.T
    assert np.allclose(np.linalg.norm(A, ord=2), sparse_2norm(sp.csr_matrix(A)))

def test_sparse_maxnorm():
    A = np.random.rand(100, 100)
    assert np.allclose(np.abs(A).max(), sparse_maxnorm(sp.csr_matrix(A)))
    A = np.random.rand(100, 100)
    A = A + A.T
    assert np.allclose(np.abs(A).max(), sparse_maxnorm(sp.csr_matrix(A)))

def test_laplacian_distance():
    G = nx.barabasi_albert_graph(100, 3)
    Gp = nx.barabasi_albert_graph(100, 3)
    dist = laplacian_distance(G, Gp)
    assert dist >=0 
    assert dist <=2

def test_LineDistances():
    G = nx.barabasi_albert_graph(1000, 3)
    ld = LineDistances(G)
    edge1, edge2 = sample_edges(G, 2)
    line_distance = ld(edge1, edge2)
    assert np.abs(nx.dijkstra_path_length(G, edge1[0], edge2[0]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[0], edge2[1]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[1], edge2[0]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[1], edge2[1]) - line_distance) <= 1
    # line graph case
    n = 50
    G = nx.path_graph(n)
    nx.draw_networkx(G)
    ld = LineDistances(G)
    line_distances = []
    for edge1 in G.edges():
        for edge2 in G.edges():
            line_distances.append(ld(edge1, edge2))
    assert set(line_distances) == set(np.arange(n-1))
    # complete graph case
    n = 10
    G = nx.complete_graph(n)
    ld = LineDistances(G)
    line_distances = []
    for edge1 in G.edges():
        for edge2 in G.edges():
            if edge1 != edge2:
                line_distances.append(ld(edge1, edge2))
    assert set(line_distances) == set([1, 2])
    
    A, X, y = get_benchmark('cora')
    G = nx.from_scipy_sparse_matrix(A)
    ld = LineDistances(G, precompute=True)
    edge1, edge2 = sample_edges(G, 2)
    line_distance = ld(edge1, edge2)
    assert np.abs(nx.dijkstra_path_length(G, edge1[0], edge2[0]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[0], edge2[1]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[1], edge2[0]) - line_distance) <= 1
    assert np.abs(nx.dijkstra_path_length(G, edge1[1], edge2[1]) - line_distance) <= 1
    

def test_average_gmdegree():
    # line graph
    n = 20
    G = nx.path_graph(n)
    returned_values = []
    for edge in G.edges():
        returned_values.append(edge_degree_gm(G, edge))
    returned_values = sorted(returned_values)
    expected = [np.sqrt(2), np.sqrt(2)] + [2 for _ in range(n-3)]
    assert np.allclose(np.mean(returned_values), np.mean(expected))
    # complete graph
    G = nx.complete_graph(n)
    returned_values = []
    for edge in G.edges():
        returned_values.append(edge_degree_gm(G, edge))
    returned_values = sorted(returned_values)
    expected = [n-1 for _ in range(int(n*(n-1)/2))]
    assert np.allclose(np.mean(returned_values), np.mean(expected))

def test_edge_degree_gm():
    # line graph
    n = 20
    G = nx.path_graph(n)
    returned_values = []
    for edge in G.edges():
        returned_values.append(edge_degree_gm(G, edge))
    returned_values = sorted(returned_values)
    expected = [np.sqrt(2), np.sqrt(2)] + [2 for _ in range(n-3)]
    assert np.allclose(returned_values, expected)
    # complete graph
    G = nx.complete_graph(n)
    returned_values = []
    for edge in G.edges():
        returned_values.append(edge_degree_gm(G, edge))
    returned_values = sorted(returned_values)
    expected = [n-1 for _ in range(int(n*(n-1)/2))]
    assert np.allclose(returned_values, expected)