import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNModel(nn.Module):
    """Enhanced Graph Neural Network model"""
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

class EnhancedHybridFakeNewsClassifier(nn.Module):
    """Enhanced hybrid model with attention and batch normalization"""
    def __init__(self, bert_embedding_dim, gnn_entity_embedding_dim, 
                 num_node_features, output_dim=1, dropout_rate=0.5):
        super().__init__()
        
        self.gnn_model = GNNModel(num_node_features, gnn_entity_embedding_dim)
        
        # Add batch normalization
        self.bert_bn = nn.BatchNorm1d(bert_embedding_dim)
        self.gnn_bn = nn.BatchNorm1d(gnn_entity_embedding_dim)
        
        combined_features_dim = bert_embedding_dim + gnn_entity_embedding_dim
        
        # Multi-layer architecture
        self.fc1 = nn.Linear(combined_features_dim, combined_features_dim // 2)
        self.fc2 = nn.Linear(combined_features_dim // 2, combined_features_dim // 4)
        self.fc3 = nn.Linear(combined_features_dim // 4, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Attention mechanism for entity aggregation
        self.attention = nn.MultiheadAttention(gnn_entity_embedding_dim, num_heads=4)

    def forward(self, article_bert_embeddings, pyg_data, article_entity_map_tensor):
        # Move data to correct device
        device = article_bert_embeddings.device
        pyg_data.x = pyg_data.x.to(device)
        pyg_data.edge_index = pyg_data.edge_index.to(device)
        
        # Get GNN embeddings
        all_entity_gnn_embeddings = self.gnn_model(pyg_data)
        
        # Enhanced entity aggregation with attention
        batch_gnn_embeddings = self.aggregate_entity_embeddings_with_attention(
            article_bert_embeddings, all_entity_gnn_embeddings, article_entity_map_tensor
        )
        
        # Apply batch normalization
        article_bert_embeddings = self.bert_bn(article_bert_embeddings)
        batch_gnn_embeddings = self.gnn_bn(batch_gnn_embeddings)
        
        # Combine features
        combined_features = torch.cat((article_bert_embeddings, batch_gnn_embeddings), dim=1)
        
        # Forward pass
        x1 = self.relu(self.fc1(combined_features))
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        
        output = self.sigmoid(self.fc3(x2))
        
        return output
    
    def aggregate_entity_embeddings_with_attention(self, article_bert_embeddings, 
                                                 all_entity_gnn_embeddings, 
                                                 article_entity_map_tensor):
        batch_size = article_bert_embeddings.shape[0]
        embedding_dim = all_entity_gnn_embeddings.shape[1]
        device = article_bert_embeddings.device
        
        batch_gnn_embeddings = torch.zeros(batch_size, embedding_dim, device=device)
        
        for i in range(batch_size):
            mentioned_entity_ids = article_entity_map_tensor[i][article_entity_map_tensor[i] != -1]
            
            if len(mentioned_entity_ids) > 0:
                selected_embeddings = all_entity_gnn_embeddings[mentioned_entity_ids]
                
                if len(mentioned_entity_ids) > 1:
                    # Use attention for aggregation
                    selected_embeddings = selected_embeddings.unsqueeze(1)
                    attended, _ = self.attention(selected_embeddings, selected_embeddings, selected_embeddings)
                    batch_gnn_embeddings[i] = attended.mean(dim=0).squeeze()
                else:
                    batch_gnn_embeddings[i] = selected_embeddings[0]
        
        return batch_gnn_embeddings

# Keep the original class for backward compatibility
class HybridFakeNewsClassifier(nn.Module):
    """Original hybrid model - keeping for compatibility"""
    def __init__(self, bert_embedding_dim, gnn_entity_embedding_dim, 
                 num_node_features, output_dim=1, dropout_rate=0.5):
        super(HybridFakeNewsClassifier, self).__init__()
        
        self.gnn_model = GNNModel(num_node_features=num_node_features,
                                  hidden_channels=gnn_entity_embedding_dim)

        combined_features_dim = bert_embedding_dim + gnn_entity_embedding_dim

        self.fc1 = nn.Linear(combined_features_dim, combined_features_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(combined_features_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, article_bert_embeddings, pyg_data, article_entity_map_tensor):
        pyg_data.x = pyg_data.x.to(article_bert_embeddings.device)
        pyg_data.edge_index = pyg_data.edge_index.to(article_bert_embeddings.device)

        all_entity_gnn_embeddings = self.gnn_model(pyg_data)

        batch_gnn_embeddings = torch.zeros(
            article_bert_embeddings.shape[0],
            all_entity_gnn_embeddings.shape[1],
            device=article_bert_embeddings.device
        )

        for i in range(article_bert_embeddings.shape[0]):
            mentioned_entity_ids = article_entity_map_tensor[i][article_entity_map_tensor[i] != -1]

            if len(mentioned_entity_ids) > 0:
                selected_entity_embeddings = all_entity_gnn_embeddings[mentioned_entity_ids]
                batch_gnn_embeddings[i] = torch.mean(selected_entity_embeddings, dim=0)

        combined_features = torch.cat((article_bert_embeddings, batch_gnn_embeddings), dim=1)

        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output