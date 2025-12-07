import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_config():
    """Get configuration with error handling"""
    try:
        return load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration...")
        return get_default_config()

def get_default_config():
    """Fallback default configuration"""
    return {
        'model': {
            'learning_rate': 0.0005,
            'dropout_rate': 0.4,
            'num_epochs': 10,
            'batch_size': 64,
            'gnn_entity_embedding_dim': 64,
            'early_stopping_patience': 5,
            'early_stopping_delta': 0.0001
        },
        'data': {
            'test_size': 0.3,
            'random_state': 42,
            'min_text_length': 20,
            'max_sequence_length': 512
        }
    }