import numpy as np
import networkx as nx
matrix = np.array([3, 4])
print(matrix)

G = nx.Graph()
G.add_edge(1, 2)
print(G.edges())
