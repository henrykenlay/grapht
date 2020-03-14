from grapht.plotting import *
from grapht.sampling import sample_edges
from matplotlib.axes._subplots import Axes
from sklearn import datasets
import networkx as nx
import pandas as pd

def test_highlight_edges():
    G = nx.barabasi_albert_graph(100, 3)
    m = G.number_of_edges()
    for num_highlight_edges in [0, m, int(m/2)]:
        edges = sample_edges(G, num_highlight_edges)
        ax = highlight_edges(G, edges)
        assert isinstance(ax, Axes)

def test_heatmap():
    iris = pd.DataFrame(datasets.load_iris()['data'], 
                    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    ax = heatmap(iris, x='sepal_length', y='sepal_width', hue='petal_length')
    assert isinstance(ax, Axes)