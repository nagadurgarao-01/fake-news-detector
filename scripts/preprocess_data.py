import pandas as pd
import random
import nltk
import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    required_data = [
        ('corpora/wordnet.zip', 'wordnet'),
        ('tokenizers/punkt.zip', 'punkt'),
        ('corpora/stopwords.zip', 'stopwords')
    ]
    
    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            print(f"Downloading {download_name}...")
            nltk.download(download_name)

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def synonym_replacement(text, n=1):
    """Replace n words with their synonyms"""
    try:
        words = word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            if num_replaced >= n:
                break
            synonyms = []
            try:
                for syn in wordnet.synsets(random_word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
            except Exception:
                continue  # Skip words that cause issues
            
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
        
        return " ".join(new_words)
    except Exception as e:
        print(f"Warning: Synonym replacement failed: {str(e)}")
        return text  # Return original text if replacement fails

def validate_data_quality(df, file_path):
    """Validate and report data quality"""
    print(f"\n--- Data Quality Report for {file_path} ---")
    print(f"Dataset shape: {df.shape}")
    
    if df.empty:
        print("Warning: Dataset is empty!")
        return df
        
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        print(f"Text length stats: min={text_lengths.min()}, max={text_lengths.max()}, mean={text_lengths.mean():.1f}")
        
        # Filter out very short texts
        original_len = len(df)
        df = df[text_lengths > 20]
        filtered_count = original_len - len(df)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} short texts (< 20 chars)")
        print(f"After filtering short texts: {df.shape}")
        
        # Check if we still have enough data
        if len(df) < 10:
            print("Warning: Very few samples remaining after filtering!")
    
    return df

def preprocess_data(file_path, augment=False, aug_prob=0.5, num_words=2):
    """Enhanced preprocessing with proper error handling and validation"""
    
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    try:
        # Try different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
                
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None, None

    if df.empty:
        print(f"Error: Empty dataset loaded from {file_path}")
        return None, None

    # Check for required columns
    if 'title' not in df.columns and 'text' not in df.columns:
        print(f"Error: Neither 'title' nor 'text' column found in {file_path}.")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    # Standardize column names
    if 'title' in df.columns and 'text' not in df.columns:
        df['text'] = df['title']
    elif 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']
    
    # Clean text
    print("Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    initial_count = len(df)
    df = df[df['text'].str.len() > 0]
    removed_empty = initial_count - len(df)
    
    if removed_empty > 0:
        print(f"Removed {removed_empty} rows with empty text")
    
    # Validate data quality
    df = validate_data_quality(df, file_path)
    
    # Data augmentation
    augmented_data = None
    if augment and not df.empty and len(df) > 0:
        print(f"Applying data augmentation with probability {aug_prob}")
        augmented_rows = []
        
        for _, row in df.iterrows():
            if random.random() < aug_prob:
                try:
                    augmented_text = synonym_replacement(row['text'], num_words)
                    new_row = row.copy()
                    new_row['text'] = augmented_text
                    augmented_rows.append(new_row)
                except Exception as e:
                    print(f"Warning: Augmentation failed for one row: {str(e)}")
                    continue
        
        if augmented_rows:
            augmented_data = pd.DataFrame(augmented_rows)
            print(f"Generated {len(augmented_data)} augmented samples")
        else:
            print("No augmented samples were generated")

    return df, augmented_data