# scripts/get_bert_embeddings.py

import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd 
from tqdm import tqdm

def load_bert_model(model_name="bert-base-multilingual-cased"):
    """Load BERT model and tokenizer with proper error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load BERT model '{model_name}': {str(e)}")

def get_bert_article_embeddings(df, text_column='text', batch_size=32, model_name="bert-base-multilingual-cased"):
    """
    Generates BERT [CLS] token embeddings for each article in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text.
        text_column (str): The name of the column containing the text to embed.
        batch_size (int): Number of texts to process in each batch for efficiency.
        model_name (str): Name of the BERT model to use.

    Returns:
        torch.Tensor: A tensor of BERT embeddings for each article (num_articles, embedding_dim).
        list: A list of the original article indices, corresponding to the embeddings.
    """
    print(f"Extracting BERT embeddings from '{text_column}' column...")
    
    # Validate input
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return torch.empty(0), []
        
    if text_column not in df.columns:
        print(f"Error: '{text_column}' column not found in DataFrame for BERT embedding extraction.")
        return torch.empty(0), []

    # Load model and tokenizer
    try:
        tokenizer, model = load_bert_model(model_name)
    except RuntimeError as e:
        print(f"Error: {e}")
        return torch.empty(0), []

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Processing {len(df)} articles in batches of {batch_size}")

    all_embeddings = []
    article_indices = []
    
    try:
        for i in tqdm(range(0, len(df), batch_size), desc="BERT Embedding"):
            batch_df = df.iloc[i : i + batch_size]
            texts = batch_df[text_column].astype(str).tolist() 
            
            if not texts: 
                continue

            # Tokenize with error handling
            try:
                inputs = tokenizer(
                    texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=256  # Reduced to 256 for speed
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}
            except Exception as e:
                print(f"Warning: Tokenization failed for batch {i//batch_size + 1}: {str(e)}")
                continue

            # Generate embeddings
            try:
                with torch.no_grad(): 
                    outputs = model(**inputs)
                
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu()) 
                article_indices.extend(batch_df.index.tolist())
                
            except Exception as e:
                print(f"Warning: Embedding generation failed for batch {i//batch_size + 1}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error during embedding extraction: {str(e)}")
        return torch.empty(0), []
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("BERT embedding extraction complete.")
    
    if not all_embeddings:
        print("Warning: No embeddings were generated.")
        return torch.empty(0), []
        
    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated embeddings shape: {final_embeddings.shape}")
    
    return final_embeddings, article_indices