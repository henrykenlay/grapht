import networkx as nx
import numpy as np
import pytest
from grapht.sampling import *

def test_sample_nodes():
    G = nx.barabasi_albert_graph(1000, 3)
    for i in range(G.number_of_nodes()+1):
        nodes = sample_nodes(G, i)
        assert len(nodes) == i
        assert len(nodes) == len(set(nodes))

    with pytest.raises(ValueError): 
        sample_nodes(G, G.number_of_nodes()+1)
    
def test_khop_neighbourhood():
    for k in range(10):
        G = nx.barabasi_albert_graph(1000, 3)
        node = sample_nodes(G, 1)[0]
        nbs = khop_neighbourhood(G, node, k)
        assert node in nbs
        shortest_paths = np.array([nx.shortest_path_length(G, node, nb) for nb in nbs])
        assert np.alltrue(shortest_paths <= k)
        if k <= 2:
            assert k in shortest_paths
        
def test_dilate():
    G = nx.barabasi_albert_graph(1000, 3)
    # single node dilation
    for _ in range(100):
        node = sample_nodes(G, 1)
        dilation = dilate(G, set(node))
        assert len(dilation) == len(list(G.neighbors(node[0])))+1
        
# test dilate
# TODO

# test sample_edges
# TODO

# test khop_subgraph
# TODO