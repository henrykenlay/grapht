import pandas as pd
import networkx as nx
import numpy as np
from grapht.data import *

# this is table 3 from https://arxiv.org/pdf/1811.05868.pdf
table_3 = pd.DataFrame(
    [['cora', 7, 1433, 2485, 5069, 0.0563, 0.0004],
     ['citeseer', 6, 3703, 2110, 3668, 0.0569, 0.0004],
     ['pubmed', 3, 500, 19717, 44324, 0.0030, 0.0001],
     #['cora_full', 67, 8710, 18703, 62421, 0.0745, 0.0001], # errors in table 3
     ['ms_academic_cs', 15, 6805, 18333, 81894, 0.0164, 0.0001],
     ['ms_academic_phy', 5, 8415, 34493, 247962, 0.0029, 0.0001],
     ['amazon_electronics_computers', 10, 767, 13381, 245778, 0.0149, 0.0007],
     ['amazon_electronics_photo', 8, 745, 7487, 119043, 0.0214, 0.0011 ]],
    columns = ['dataset', 'classes', 'features', 'nodes', 'edges', 'label_rate', 'edge_density']
).set_index('dataset')


def test_get_benchmark():
    for dataset in table_3.index:
        A, X, y = get_benchmark(dataset)
        
        # correct dimensions
        assert A.ndim == 2
        assert X.ndim == 2
        assert y.ndim == 1
        
        # using elements of table 3 
        assert len(set(y)) == int(table_3.loc[dataset]['classes'])
        assert int(X.shape[1]) == int(table_3.loc[dataset]['features'])
        assert int(X.shape[0]) == int(table_3.loc[dataset]['nodes'])
        assert int(A.shape[0]) == int(table_3.loc[dataset]['nodes'])
        assert int(A.shape[1]) == int(table_3.loc[dataset]['nodes'])
        assert int(y.shape[0]) == int(table_3.loc[dataset]['nodes'])
        assert int(A.sum()/2) == int(table_3.loc[dataset]['edges'])
        
        # label rate calculation is described in appendix B
        assert round(len(set(y))*20/table_3.loc[dataset]['nodes'], 4) == table_3.loc[dataset]['label_rate'] 
        
        # edge density calculation is described in appendix B, the table looks off by a factor of 4 
        #assert int(A.sum()/2) / (int(A.shape[0]) ** 2 /2) 
        
        # check connectivity
        G = nx.is_connected(nx.from_scipy_sparse_matrix(A))
        
def test_get_planar_graph():
    for n in [10, 20, 40, 80]:
        G = get_planar_graph(n)
        assert G.number_of_nodes() == n
        assert nx.is_connected(G)
        assert not nx.is_directed(G)
        assert nx.algorithms.planarity.check_planarity(G)
        G, pos = get_planar_graph(n, return_pos=True)
        assert type(pos) is dict
        
def test_get_sensor_graph():
    for n in [10, 20, 40, 80]:
        G = get_sensor_graph(n)
        assert G.number_of_nodes() == n
        assert nx.is_connected(G)
        assert not nx.is_directed(G)
        
def test_get_BASBM():
    for m in [2, 3]:
        sizes = np.random.randint(m+1, 15, 5)
        G = get_BASBM(sizes, 0, m)
        expected_total_edges = np.sum([(n-m)*m for n in sizes])
        assert len(G.edges()) == expected_total_edges
        G = get_BASBM(sizes, 1, m)
        expected_total_edges = np.sum([(n-m)*m for n in sizes])
        for i in range(len(sizes)):
            for j in range(i+1,len(sizes)):
                expected_total_edges += sizes[i]*sizes[j]
        assert len(G.edges()) == expected_total_edges
    