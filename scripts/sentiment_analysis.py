# scripts/sentiment_analysis.py

import pandas as pd
from textblob import TextBlob
import spacy

def load_spacy_model():
    """Load SpaCy model with proper error handling"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise ImportError("SpaCy English model 'en_core_web_sm' not found. "
                         "Please run: python -m spacy download en_core_web_sm")

def safe_sentiment_analysis(text):
    """Safely calculate sentiment with error handling"""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception as e:
        print(f"Warning: Sentiment analysis failed for text: {str(e)}")
        return 0.0  # Neutral sentiment as fallback

def analyze_sentiment_by_cluster(df, partition):
    """
    Analyzes the sentiment of articles and calculates the average sentiment
    for each detected community cluster.
    Args:
        df (pd.DataFrame): The original DataFrame containing article 'title' and 'text'.
        partition (dict): A dictionary mapping entity nodes (from the graph) to their community ID.
    """
    print("Analyzing sentiment per cluster...")

    if df is None or df.empty:
        print("Warning: DataFrame is empty or not loaded. Skipping sentiment analysis.")
        return

    if not partition:
        print("Warning: No communities detected. Skipping sentiment analysis per cluster.")
        return

    # Load SpaCy model
    try:
        nlp = load_spacy_model()
    except ImportError as e:
        print(f"Error: {e}")
        return

    # 1. Calculate sentiment for each article's title (or text if you prefer)
    # Check for text column
    text_column = 'title'
    if 'title' not in df.columns:
        if 'text' in df.columns:
            text_column = 'text'
        else:
            print("Error: Neither 'title' nor 'text' column found in DataFrame for sentiment analysis.")
            return

    print(f"Calculating sentiment using '{text_column}' column...")
    df['sentiment_polarity'] = df[text_column].apply(safe_sentiment_analysis)
    print("Article sentiments calculated.")

    # 2. Map articles to communities.
    # This is crucial: 'partition' maps *entities* (graph nodes) to communities.
    # We need to link each *article* (row in df) to a community.
    # A simple approach: an article is assigned to the community of its first found entity
    # that exists in the 'partition'.

    article_to_community = {}
    community_sentiments_raw = {}  # To store all sentiment scores for each community

    for index, row in df.iterrows():
        text = str(row[text_column])
        try:
            doc = nlp(text)  # Use spaCy again to find entities in this article's text
        except Exception as e:
            print(f"Warning: SpaCy processing failed for article {index}: {str(e)}")
            continue
            
        assigned_community = -1  # Default if no entities found in partition

        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            if entity_text in partition:
                assigned_community = partition[entity_text]
                break  # Assign to the first entity's community found
                
        article_to_community[index] = assigned_community

        # Add the article's sentiment to the list for its assigned community
        if assigned_community != -1:
            community_sentiments_raw.setdefault(assigned_community, []).append(row['sentiment_polarity'])

    # 3. Calculate average sentiment per community
    print("\nAverage Sentiment per Cluster (based on article's first entity mapping):")
    if community_sentiments_raw:
        for comm_id in sorted(community_sentiments_raw.keys()):  # Sort for consistent output
            sentiments = community_sentiments_raw[comm_id]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                print(f"Community {comm_id}: Average Sentiment = {avg_sentiment:.3f} (from {len(sentiments)} articles)")
            else:
                print(f"Community {comm_id}: No sentiment data.")

        # Optional: Report communities from 'partition' that had no articles mapped
        mapped_communities = set(community_sentiments_raw.keys())
        all_communities = set(partition.values())
        unmapped_communities = all_communities - mapped_communities
        if unmapped_communities:
            print("\nNote: The following entity clusters exist but had no articles mapped to them for sentiment analysis:")
            for comm_id in sorted(list(unmapped_communities)):
                print(f"- Community {comm_id}")
    else:
        print("No articles could be mapped to any community for sentiment analysis.")