# cluster_analysis.py
from extract_relationships import G
import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt


partition = community_louvain.best_partition(G)

pos = nx.spring_layout(G)
cmap = plt.cm.get_cmap("viridis", max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
