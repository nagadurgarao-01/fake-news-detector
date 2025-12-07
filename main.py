# main.py

import pandas as pd
import torch
import torch.nn as nn
import spacy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix 
import yaml
import logging
import os
import pickle

# Import all custom modules
from scripts.config_loader import get_config
from scripts.logger_setup import setup_logging
from scripts.performance_monitor import monitor
from scripts.utils import safe_load_data, calculate_metrics, PipelineError
from scripts.preprocess_data import preprocess_data
from scripts.extract_relationships import build_knowledge_graph
from scripts.community_leaders import get_communities
from scripts.sentiment_analysis import analyze_sentiment_by_cluster
from scripts.graph_data_prep import networkx_to_pyg_data
from scripts.get_bert_embeddings import get_bert_article_embeddings
from scripts.hybrid_model import HybridFakeNewsClassifier, GNNModel, EnhancedHybridFakeNewsClassifier

class EarlyStopping:
    """Early stopping to prevent overfitting during training"""
    
    def __init__(self, patience=5, verbose=False, delta=0, path='models/best_hybrid_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def safe_load_dataset(file_path, logger):
    """Safely load and validate dataset"""
    try:
        df, _ = preprocess_data(file_path)
        if df is None:
            logger.error(f"Failed to load data from {file_path}")
            return None
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def load_liar_dataset(file_path, logger):
    """Load and preprocess LIAR dataset"""
    try:
        liar_df = pd.read_csv(file_path, sep='\t', header=None)
        liar_df.columns = [
            "ID", "label", "statement", "subject", "speaker", "speaker_job", 
            "state", "party", "barely_true_counts", "false_counts", "half_true_counts", 
            "mostly_true_counts", "pants_on_fire_counts", "context"
        ]

        label_map = {
            "true": 0,
            "mostly-true": 0,
            "half-true": 0,
            "barely-true": 1,
            "false": 1,
            "pants-fire": 1
        }
        liar_df['label'] = liar_df['label'].map(label_map)
        liar_df = liar_df.dropna(subset=['label']) 
        liar_df = liar_df.rename(columns={'statement': 'text'})
        
        logger.info(f"Successfully loaded LIAR dataset with {len(liar_df)} records")
        return liar_df
        
    except Exception as e:
        logger.error(f"Error loading LIAR dataset from {file_path}: {str(e)}")
        return None

def create_article_entity_mapping_tensor(df_split, entity_map, device, nlp_model):
    """Create article-entity mapping tensor"""
    max_entities_per_article = 0
    article_entity_lists_split = {}
    
    for idx, row in df_split.iterrows():
        title = str(row['text'])
        doc = nlp_model(title)
        entities_in_article_ids = []
        
        for ent in doc.ents:
            if ent.text in entity_map:
                entities_in_article_ids.append(entity_map[ent.text])
                
        article_entity_lists_split[idx] = entities_in_article_ids
        if len(entities_in_article_ids) > max_entities_per_article:
            max_entities_per_article = len(entities_in_article_ids)
    
    if max_entities_per_article == 0 and len(df_split) > 0:
        max_entities_per_article = 1

    article_entity_map_tensor_split = torch.full(
        (len(df_split), max_entities_per_article), -1, dtype=torch.long, device=device
    )
    
    for i, original_idx in enumerate(df_split.index):
        if original_idx in article_entity_lists_split:
            for j, entity_id in enumerate(article_entity_lists_split[original_idx]):
                article_entity_map_tensor_split[i, j] = entity_id
                
    return article_entity_map_tensor_split

# Load SpaCy model
try:
    nlp_map_entities = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy English model 'en_core_web_sm' not found for mapping entities.")
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

def main():
    """Main pipeline function"""
    # Setup logging first
    logger = setup_logging()
    logger.info("Starting Fake News Detection Pipeline...")
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Start performance monitoring
        monitor.start_monitoring()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Get parameters from config (with fallbacks to original values)
        learning_rate = config.get('model', {}).get('learning_rate', 0.0005)
        dropout_rate = config.get('model', {}).get('dropout_rate', 0.4)
        num_epochs = config.get('model', {}).get('num_epochs', 10)
        batch_size = config.get('model', {}).get('batch_size', 64)
        gnn_entity_embedding_dim = config.get('model', {}).get('gnn_entity_embedding_dim', 64)
        early_stopping_patience = config.get('model', {}).get('early_stopping_patience', 5)
        early_stopping_delta = config.get('model', {}).get('early_stopping_delta', 0.0001)
        test_size = config.get('data', {}).get('test_size', 0.3)
        random_state = config.get('data', {}).get('random_state', 42)
        
        logger.info(f"Using parameters: LR={learning_rate}, Dropout={dropout_rate}, "
                   f"Epochs={num_epochs}, Batch Size={batch_size}")

        # Get file paths from config with fallbacks
        paths = config.get('paths', {})
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, 'data')
        
        logger.info("\n--- Step 1: Loading & Preprocessing Labeled Data ---")
        monitor.log_performance("Step 1 Start")
        
        # Load datasets using paths from config with fallbacks
        logger.info("Loading Fake.csv and True.csv...") 
        df_fake = safe_load_dataset(paths.get('fake_data', os.path.join(data_dir, 'Fake.csv')), logger)
        if df_fake is None:
            logger.error("Failed to load fake news data. Exiting.")
            return
        df_fake['label'] = 1 

        df_real = safe_load_dataset(paths.get('true_data', os.path.join(data_dir, 'True.csv')), logger)
        if df_real is None:
            logger.error("Failed to load real news data. Exiting.")
            return
        df_real['label'] = 0 

        df_combined1 = pd.concat([df_fake, df_real], ignore_index=True)
        
        logger.info("Loading gossipcop datasets...")
        df_fake = safe_load_dataset(paths.get('gossipcop_fake', os.path.join(data_dir, 'gossipcop_fake.csv')), logger)
        if df_fake is None:
            logger.warning("Failed to load gossipcop fake data. Continuing without it.")
            df_fake = pd.DataFrame(columns=['text', 'label'])
        df_fake['label'] = 1 
        
        df_real = safe_load_dataset(paths.get('gossipcop_real', os.path.join(data_dir, 'gossipcop_real.csv')), logger)
        if df_real is None:
            logger.warning("Failed to load gossipcop real data. Continuing without it.")
            df_real = pd.DataFrame(columns=['text', 'label'])
        df_real['label'] = 0 
        
        df_combined2 = pd.concat([df_fake, df_real, df_combined1], ignore_index=True)
        
        logger.info("Loading politifact datasets...")
        df_fake = safe_load_dataset(paths.get('politifact_fake', os.path.join(data_dir, 'politifact_fake.csv')), logger)
        if df_fake is None:
            logger.warning("Failed to load politifact fake data. Continuing without it.")
            df_fake = pd.DataFrame(columns=['text', 'label'])
        df_fake['label'] = 1 

        df_real = safe_load_dataset(paths.get('politifact_real', os.path.join(data_dir, 'politifact_real.csv')), logger)
        if df_real is None:
            logger.warning("Failed to load politifact real data. Continuing without it.")
            df_real = pd.DataFrame(columns=['text', 'label'])
        df_real['label'] = 0 
        
        df_combined3 = pd.concat([df_fake, df_real, df_combined2], ignore_index=True)

        logger.info("Loading LIAR dataset...")
        liar_df = load_liar_dataset(paths.get('liar_dataset', os.path.join(data_dir, 'train.tsv')), logger)
        
        if liar_df is not None:
            df_combined4 = pd.concat([df_combined3, liar_df[['text', 'label']]], ignore_index=True)
        else:
            logger.warning("LIAR dataset not loaded. Continuing without it.")
            df_combined4 = df_combined3
            
        logger.info("Loading cleaned_news_dataset...")
        df_cleaned = safe_load_dataset(paths.get('cleaned_news_dataset', os.path.join(data_dir, 'cleaned_news_dataset.csv')), logger)
        if df_cleaned is None:
            logger.warning("Failed to load cleaned news dataset. Continuing without it.")
            df_cleaned = pd.DataFrame(columns=['text', 'label'])
            
        df_combined = pd.concat([df_combined4, df_cleaned], ignore_index=True)
        logger.info(f"Combined dataset size: {len(df_combined)} articles.")
        
        # Sampling to reduce memory usage and training time
        SAMPLE_SIZE = 10000
        if len(df_combined) > SAMPLE_SIZE:
            logger.info(f"Dataset too large. Sampling {SAMPLE_SIZE} articles for training...")
            df_combined = df_combined.sample(n=SAMPLE_SIZE, random_state=random_state).reset_index(drop=True)
            logger.info(f"New dataset size: {len(df_combined)} articles.")

        logger.info("Data preprocessing completed successfully.")
        monitor.log_performance("Step 1 Complete - Data Loading")

        logger.info("\n--- Step 2: Splitting Data into Train/Test Sets ---")
        train_df, test_df = train_test_split(df_combined, test_size=test_size, 
                                           random_state=random_state, stratify=df_combined['label'])
        logger.info(f"Training set size: {len(train_df)} articles.")
        logger.info(f"Test set size: {len(test_df)} articles.")
        monitor.log_performance("Step 2 Complete - Data Splitting")
        
        logger.info("\n--- Step 3: Building Knowledge Graph ---")
        G = build_knowledge_graph(train_df) 

        if G is None or G.number_of_nodes() == 0:
            logger.error("Knowledge graph building failed or resulted in an empty graph. Exiting.")
            return
        logger.info(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        monitor.log_performance("Step 3 Complete - Knowledge Graph")

        logger.info("\n--- Step 4: Performing Community Detection ---")
        partition = get_communities(G)

        if not partition:
            logger.error("Community detection failed or resulted in no communities. Exiting.")
            return
        logger.info(f"Detected {len(set(partition.values()))} communities.")
        monitor.log_performance("Step 4 Complete - Community Detection")

        logger.info("\n--- Step 5: Analyzing Sentiment per Cluster ---")
        analyze_sentiment_by_cluster(train_df, partition) 
        logger.info("Sentiment analysis complete.")
        monitor.log_performance("Step 5 Complete - Sentiment Analysis")

        logger.info("\n--- Step 6: Preparing Graph for GNN ---")
        pyg_data, entity_to_id_map = networkx_to_pyg_data(G)

        if pyg_data.num_nodes == 0:
            logger.error("PyG Data object is empty. Cannot proceed with GNN. Exiting.")
            return
        logger.info("PyG Data object created successfully.")
        logger.info(f"PyG Data object details: {pyg_data}")
        pyg_data = pyg_data.to(device)
        
        # Save graph artifacts for inference
        os.makedirs('models', exist_ok=True)
        logger.info("Saving graph artifacts to 'models/'...")
        torch.save(pyg_data, 'models/pyg_data.pt')
        with open('models/entity_to_id_map.pkl', 'wb') as f:
            pickle.dump(entity_to_id_map, f)
        logger.info("Graph artifacts saved.")
        
        monitor.log_performance("Step 6 Complete - Graph Preparation")

        logger.info("\n--- Step 7: Extracting BERT Article Embeddings ---")
        train_article_bert_embeddings, train_original_indices = get_bert_article_embeddings(train_df, text_column='text')
        train_labels = torch.tensor(train_df.loc[train_original_indices, 'label'].values, 
                                  dtype=torch.float32).view(-1, 1).to(device)

        if train_article_bert_embeddings.numel() == 0:
            logger.error("Train BERT article embedding extraction failed. Exiting.")
            return
        logger.info(f"Extracted BERT embeddings for {train_article_bert_embeddings.shape[0]} training articles.")

        test_article_bert_embeddings, test_original_indices = get_bert_article_embeddings(test_df, text_column='text') 
        test_labels = torch.tensor(test_df.loc[test_original_indices, 'label'].values, 
                                 dtype=torch.float32).view(-1, 1).to(device)

        if test_article_bert_embeddings.numel() == 0:
            logger.error("Test BERT article embedding extraction failed. Exiting.")
            return
        logger.info(f"Extracted BERT embeddings for {test_article_bert_embeddings.shape[0]} test articles.")
        
        train_article_bert_embeddings = train_article_bert_embeddings.to(device)
        test_article_bert_embeddings = test_article_bert_embeddings.to(device)
        monitor.log_performance("Step 7 Complete - BERT Embeddings")

        logger.info("\n--- Step 8: Creating Article-Entity Mapping ---")
        train_article_entity_map_tensor = create_article_entity_mapping_tensor(
            train_df, entity_to_id_map, device, nlp_map_entities)
        test_article_entity_map_tensor = create_article_entity_mapping_tensor(
            test_df, entity_to_id_map, device, nlp_map_entities)

        logger.info(f"Train Article-Entity mapping tensor created with shape: {train_article_entity_map_tensor.shape}")
        logger.info(f"Test Article-Entity mapping tensor created with shape: {test_article_entity_map_tensor.shape}")
        monitor.log_performance("Step 8 Complete - Entity Mapping")

        logger.info("\n--- Step 9: Instantiating Hybrid Model ---")
        bert_embedding_dim = train_article_bert_embeddings.shape[1]
        num_total_graph_nodes = pyg_data.num_nodes

        # Use enhanced model if available, otherwise fall back to original
        try:
            hybrid_model = EnhancedHybridFakeNewsClassifier(
                bert_embedding_dim=bert_embedding_dim,
                gnn_entity_embedding_dim=gnn_entity_embedding_dim,
                num_node_features=num_total_graph_nodes,
                dropout_rate=dropout_rate 
            ).to(device)
            logger.info("Enhanced HybridFakeNewsClassifier model instantiated.")
        except Exception as e:
            logger.warning(f"Enhanced model failed, using original: {str(e)}")
            hybrid_model = HybridFakeNewsClassifier(
                bert_embedding_dim=bert_embedding_dim,
                gnn_entity_embedding_dim=gnn_entity_embedding_dim,
                num_node_features=num_total_graph_nodes,
                dropout_rate=dropout_rate 
            ).to(device)
            logger.info("Original HybridFakeNewsClassifier model instantiated.")
        monitor.log_performance("Step 9 Complete - Model Instantiation")

        logger.info("\n--- Step 10: Defining Loss Function and Optimizer ---")
        num_fake_train = train_labels.sum().item()
        num_real_train = len(train_labels) - num_fake_train
        pos_weight = torch.tensor([1.2], dtype=torch.float32).to(device) 
        logger.info(f"Calculated pos_weight for BCELoss: {pos_weight.item():.2f}")

        criterion = nn.BCELoss() 
        optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=learning_rate)
        logger.info("Loss function and Optimizer defined.") 

        train_dataset = TensorDataset(train_article_bert_embeddings, train_article_entity_map_tensor, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_article_bert_embeddings, test_article_entity_map_tensor, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
        monitor.log_performance("Step 10 Complete - Training Setup")

        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, delta=early_stopping_delta)

        # Training loop
        logger.info("\n--- Step 11: Training Model ---")
        for epoch in range(num_epochs):
            hybrid_model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch_bert_embeds, batch_entity_maps, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = hybrid_model(batch_bert_embeds, pyg_data, batch_entity_maps)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                predicted = (outputs > 0.5).float()
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions * 100

            hybrid_model.eval()
            val_total_loss = 0
            val_correct_predictions = 0
            val_total_predictions = 0

            with torch.no_grad():
                for batch_bert_embeds_val, batch_entity_maps_val, batch_labels_val in test_loader:
                    val_outputs = hybrid_model(batch_bert_embeds_val, pyg_data, batch_entity_maps_val)
                    val_loss = criterion(val_outputs, batch_labels_val)
                    val_total_loss += val_loss.item()
                    val_predicted = (val_outputs > 0.5).float()
                    val_total_predictions += batch_labels_val.size(0)
                    val_correct_predictions += (val_predicted == batch_labels_val).sum().item()

            avg_val_loss = val_total_loss / len(test_loader)
            val_accuracy = val_correct_predictions / val_total_predictions * 100

            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            early_stopping(avg_val_loss, hybrid_model)
            
            if early_stopping.early_stop:
                logger.info("Early stopping triggered!")
                break

        logger.info("Training complete.")
        monitor.log_performance("Step 11 Complete - Model Training")

        logger.info("\n--- Step 12: Final Evaluation on Test Set ---")

        # Load best model - Fixed the model checking logic
        try:
            final_model = EnhancedHybridFakeNewsClassifier(
                bert_embedding_dim=bert_embedding_dim,
                gnn_entity_embedding_dim=gnn_entity_embedding_dim,
                num_node_features=num_total_graph_nodes,
                dropout_rate=dropout_rate
            ).to(device)
        except Exception as e:
            logger.warning(f"Enhanced model not available, using original: {str(e)}")
            final_model = HybridFakeNewsClassifier(
                bert_embedding_dim=bert_embedding_dim,
                gnn_entity_embedding_dim=gnn_entity_embedding_dim,
                num_node_features=num_total_graph_nodes,
                dropout_rate=dropout_rate
            ).to(device)

        try:
            final_model.load_state_dict(torch.load(early_stopping.path, map_location=device))
            logger.info(f"Loaded best model from '{early_stopping.path}'")
        except FileNotFoundError:
            logger.error(f"Error: '{early_stopping.path}' not found. Using current model state.")
            final_model = hybrid_model

        final_model.eval()
        
        test_correct_predictions = 0
        test_total_predictions = 0
        test_all_labels = []
        test_all_predictions = []

        with torch.no_grad():
            for batch_bert_embeds_test, batch_entity_maps_test, batch_labels_test in test_loader:
                test_outputs = final_model(batch_bert_embeds_test, pyg_data, batch_entity_maps_test)
                test_predicted = (test_outputs > 0.5).float()
                
                test_total_predictions += batch_labels_test.size(0)
                test_correct_predictions += (test_predicted == batch_labels_test).sum().item()
                
                test_all_labels.extend(batch_labels_test.cpu().numpy())
                test_all_predictions.extend(test_predicted.cpu().numpy())

        test_accuracy = test_correct_predictions / test_total_predictions * 100
        precision = precision_score(test_all_labels, test_all_predictions)
        recall = recall_score(test_all_labels, test_all_predictions)
        f1 = f1_score(test_all_labels, test_all_predictions)
        conf_matrix = confusion_matrix(test_all_labels, test_all_predictions)

        logger.info(f"Test Set Evaluation:")
        logger.info(f"  Accuracy: {test_accuracy:.2f}%")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Confusion Matrix:\n{conf_matrix}")
        monitor.log_performance("Step 12 Complete - Final Evaluation")

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {str(e)}", exc_info=True)
    finally:
        pass

if __name__ == "__main__":
    main()