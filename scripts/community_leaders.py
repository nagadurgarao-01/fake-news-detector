# scripts/community_leaders.py

import community as community_louvain
import networkx as nx

def get_communities(G):
    print("Detecting communities in the graph...")
    if G.number_of_nodes() == 0:
        print("Warning: Graph G has no nodes. Cannot perform community detection.")
        return {} 
    G_undirected = G.to_undirected()


    partition = community_louvain.best_partition(G_undirected) 
    print(f"Detected {len(set(partition.values()))} communities.")

  

    return partition
