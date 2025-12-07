import torch
import networkx as nx
from torch_geometric.data import Data
import pandas as pd 


def networkx_to_pyg_data(G_nx, entity_to_id_map=None):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.

    Args:
        G_nx (nx.Graph or nx.DiGraph): The NetworkX graph.
        entity_to_id_map (dict, optional): A dictionary mapping entity names (nodes in G_nx)
                                           to unique integer IDs. If None, it will create one.

    Returns:
        torch_geometric.data.Data: The PyG Data object.
        dict: The created or passed entity_to_id_map.
    """
    print("Converting NetworkX graph to PyTorch Geometric Data object...")

    if not G_nx.number_of_nodes():
        print("Warning: NetworkX graph is empty. Returning empty PyG Data object.")
        return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=0), {}

    if entity_to_id_map is None:
        entity_to_id_map = {node: i for i, node in enumerate(G_nx.nodes())}
    
    edge_index = []
    for u, v in G_nx.edges():
        if u in entity_to_id_map and v in entity_to_id_map: 
            edge_index.append([entity_to_id_map[u], entity_to_id_map[v]])

    if not edge_index:
        print("Warning: No valid edges found after mapping to integer IDs. Returning empty PyG Data object.")
        return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=len(entity_to_id_map)), entity_to_id_map

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    num_nodes = len(entity_to_id_map)
    x = torch.eye(num_nodes) 

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    print(f"PyG Data object created with {data.num_nodes} nodes and {data.num_edges} edges.")
    return data, entity_to_id_map
