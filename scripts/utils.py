import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

def safe_load_data(file_path, preprocess_func):
    """Safely load data with comprehensive error handling"""
    logger = logging.getLogger(__name__)
    try:
        return preprocess_func(file_path)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise PipelineError(f"Data loading failed: {str(e)}")

def calculate_metrics(predictions, labels):
    """Calculate comprehensive evaluation metrics"""
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    # Convert to binary predictions if they're probabilities
    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
        binary_preds = (predictions > 0.5).astype(int)
    else:
        binary_preds = predictions
    
    return {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds),
        'recall': recall_score(labels, binary_preds),
        'f1': f1_score(labels, binary_preds)
    }

def validate_data_quality(df, file_path):
    """Validate and report data quality"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n--- Data Quality Report for {file_path} ---")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")
    
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        logger.info(f"Text length stats: min={text_lengths.min()}, max={text_lengths.max()}, mean={text_lengths.mean():.1f}")
        # Filter out very short texts
        original_len = len(df)
        df = df[text_lengths > 20]
        logger.info(f"Filtered {original_len - len(df)} short texts. New shape: {df.shape}")
    
    return df