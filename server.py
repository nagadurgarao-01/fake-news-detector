from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import spacy
import pickle
import os
import logging
from transformers import AutoTokenizer, AutoModel
from scripts.hybrid_model import EnhancedHybridFakeNewsClassifier, HybridFakeNewsClassifier
from scripts.config_loader import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
pyg_data = None
entity_to_id_map = None
nlp = None
tokenizer = None
bert_model = None
device = None

def load_resources():
    global model, pyg_data, entity_to_id_map, nlp, tokenizer, bert_model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load Graph Artifacts
    if not os.path.exists('models/pyg_data.pt') or not os.path.exists('models/entity_to_id_map.pkl'):
        logger.error("Graph artifacts not found in 'models/'. Please run main.py first.")
        raise FileNotFoundError("Run main.py to generate graph artifacts.")
    
    pyg_data = torch.load('models/pyg_data.pt', map_location=device)
    with open('models/entity_to_id_map.pkl', 'rb') as f:
        entity_to_id_map = pickle.load(f)
    logger.info("Graph artifacts loaded.")

    # Load BERT
    logger.info("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)
    bert_model.eval()

    # Load Spacy
    logger.info("Loading Spacy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("Spacy model not found. Run: python -m spacy download en_core_web_sm")
        raise

    # Load Hybrid Model
    config = get_config()
    bert_embedding_dim = 768 # Standard for bert-base
    gnn_entity_embedding_dim = config.get('model', {}).get('gnn_entity_embedding_dim', 64)
    num_node_features = pyg_data.num_nodes
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.4)

    logger.info("Loading Hybrid Model...")
    try:
        model = EnhancedHybridFakeNewsClassifier(
            bert_embedding_dim=bert_embedding_dim,
            gnn_entity_embedding_dim=gnn_entity_embedding_dim,
            num_node_features=num_node_features,
            dropout_rate=dropout_rate
        ).to(device)
    except:
        model = HybridFakeNewsClassifier(
            bert_embedding_dim=bert_embedding_dim,
            gnn_entity_embedding_dim=gnn_entity_embedding_dim,
            num_node_features=num_node_features,
            dropout_rate=dropout_rate
        ).to(device)

    if os.path.exists('models/best_hybrid_model.pth'):
        model.load_state_dict(torch.load('models/best_hybrid_model.pth', map_location=device))
        logger.info("Model weights loaded.")
    else:
        logger.warning("Model weights not found. Using initialized weights (Untrained!).")

    model.eval()
    logger.info("Server ready.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # 1. BERT Embedding
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        bert_embed = outputs.last_hidden_state[:, 0, :] # [1, 768]

        # 2. Entity Mapping
        doc = nlp(text)
        entities_in_article_ids = []
        for ent in doc.ents:
            if ent.text in entity_to_id_map:
                entities_in_article_ids.append(entity_to_id_map[ent.text])
        
        if not entities_in_article_ids:
            # If no entities found, use a placeholder or handle gracefully
            # We'll use a tensor of shape [1, 1] with value -1 (padding)
            article_entity_map = torch.full((1, 1), -1, dtype=torch.long, device=device)
        else:
            article_entity_map = torch.tensor([entities_in_article_ids], dtype=torch.long, device=device)

        # 3. Prediction
        with torch.no_grad():
            output = model(bert_embed, pyg_data, article_entity_map)
            prediction_score = output.item()
        
        label = "Fake" if prediction_score > 0.5 else "Real"
        confidence = prediction_score if label == "Fake" else 1 - prediction_score

        return jsonify({
            'prediction': label,
            'confidence_score': confidence,
            'raw_score': prediction_score
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_resources()
        app.run(host='127.0.0.1', port=5000)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
