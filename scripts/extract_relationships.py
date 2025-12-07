import networkx as nx
import spacy
from tqdm import tqdm

def load_spacy_model():
    """Load SpaCy model with proper error handling"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise ImportError("SpaCy English model 'en_core_web_sm' not found. "
                         "Please run: python -m spacy download en_core_web_sm")

def build_knowledge_graph(df):
    """Enhanced knowledge graph construction"""
    # Load SpaCy model
    nlp = load_spacy_model()
    
    G = nx.DiGraph()
    print("Building knowledge graph from text...")

    # Check for available text columns
    text_column = None
    if 'text' in df.columns:
        text_column = 'text'
    elif 'title' in df.columns:
        text_column = 'title'
    else:
        print("Warning: No suitable text column found. Skipping graph building.")
        return G

    entity_counts = {}  # Track entity frequency
    cooccurrence_window = 5  # Words within this window are considered co-occurring

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Building knowledge graph"):
        text = str(row[text_column])
        doc = nlp(text)
        
        # Extract entities and track frequency
        entities_in_text = []
        for ent in doc.ents:
            if ent.text.strip() and len(ent.text.strip()) > 1:
                entity_text = ent.text.strip().lower()
                entities_in_text.append((entity_text, ent.label_))
                entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
                
                # Add node with enhanced attributes
                G.add_node(entity_text, 
                          label=ent.label_, 
                          source_article_id=index,
                          entity_type=ent.label_)
        
        # Create co-occurrence relationships
        for i, (ent1, label1) in enumerate(entities_in_text):
            for j, (ent2, label2) in enumerate(entities_in_text):
                if i != j and abs(i - j) <= cooccurrence_window:
                    if G.has_edge(ent1, ent2):
                        G[ent1][ent2]['weight'] += 1
                    else:
                        G.add_edge(ent1, ent2, 
                                 relation='co-occurrence',
                                 weight=1,
                                 source_article_id=index)

    # Filter out low-frequency entities
    min_frequency = 2
    nodes_to_remove = [node for node, count in entity_counts.items() if count < min_frequency]
    G.remove_nodes_from(nodes_to_remove)
    
    print(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Removed {len(nodes_to_remove)} low-frequency entities")
    
    return G