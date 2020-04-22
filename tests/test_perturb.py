from grapht.graphtools import has_isolated_nodes
from grapht.perturb import *
from grapht.sampling import sample_edges
import networkx as nx

def test_khop_remove():
    G = nx.barabasi_albert_graph(500, 2)
    r = 5
    for k in range(1, 5):
        Gp, edge_info, node = khop_remove(G, k, r)
        # check only edges were deleted and exactly r were deleted
        assert set(Gp.edges()).issubset(set(G.edges())) 
        assert len(G.edges()) - len(Gp.edges()) == r 
        assert edge_info['type'].unique()[0] == 'remove'
        assert len(edge_info['type'].unique()) == 1
        # make sure edges were from a k-hop neighbourhood
        for u in edge_info['u']:
            assert nx.dijkstra_path_length(G, u, node) <= r
        for v in edge_info['v']:
            assert nx.dijkstra_path_length(G, v, node) <= r
        
        
         
        Gp, _, _ = khop_remove(G, k, r, enforce_connected=True, enforce_no_isolates=True)
        assert nx.is_connected(Gp) and not has_isolated_nodes(Gp)
        Gp, _, _ = khop_remove(G, k, r, enforce_connected=True, enforce_no_isolates=False)
        assert nx.is_connected(Gp)
        Gp, _, _ = khop_remove(G, k, r, enforce_connected=False, enforce_no_isolates=True)
        assert not has_isolated_nodes(Gp)
        

def test_khop_rewire():
    G = nx.barabasi_albert_graph(100, 3)
    k, r = 3, 3
    solution, edge_info, node  = khop_rewire(G, k, r)
    assert len(edge_info) == 2*r
    assert len(edge_info['type'].unique()) == 2

def test_rewire():
    G = nx.barabasi_albert_graph(100, 3)
    stubs_before = set()
    stubs_after = set()
    for _, row in rewire(G, sample_edges(G, 3)).iterrows():
        if row['type'] == 'remove':
            stubs_before.add(row['u'])
            stubs_before.add(row['v'])
        elif row['type'] == 'add':
            stubs_after.add(row['u'])
            stubs_after.add(row['v'])
    assert stubs_before == stubs_after