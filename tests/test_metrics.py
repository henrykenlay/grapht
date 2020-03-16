from grapht.metrics import *
import numpy as np
import scipy.sparse as sp
import networkx as nx

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
    pass

def test_LineDistancesDataset():
    pass

def test_average_gmdegree():
    pass

def test_edge_degree_gm():
    pass