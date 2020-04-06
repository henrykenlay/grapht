import networkx as nx
import numpy as np
import pytest
from grapht.sampling import *



def test_sample_node():
    G = nx.barabasi_albert_graph(100, 3)
    node = sample_node(G)
    assert node in G.nodes()
    assert type(node) is int

def test_sample_nodes():
    G = nx.barabasi_albert_graph(100, 3)
    for i in range(G.number_of_nodes()+1):
        nodes = sample_nodes(G, i)
        assert len(nodes) == i
        assert len(nodes) == len(set(nodes))

    with pytest.raises(ValueError): 
        sample_nodes(G, G.number_of_nodes()+1)
        
        
def test_khop_neighbourhood():
    for k in [0, 1, 2, 3]:
        G = nx.barabasi_albert_graph(100, 3)
        node = sample_node(G)
        nbs = khop_neighbourhood(G, k, node)
        assert node in nbs
        shortest_paths = np.array([nx.shortest_path_length(G, node, nb) for nb in nbs])
        assert np.alltrue(shortest_paths <= k)
        if k <= 2:
            assert k in shortest_paths
        if k == 0:
            assert nbs == [node]

def test_sample_edges():
    pass

def test_khop_subgraph():
    pass