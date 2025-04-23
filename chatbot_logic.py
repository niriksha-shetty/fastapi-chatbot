import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_embeddings():
    """Load embeddings from the embeddings.pkl file."""
    with open("embeddings.pkl", "rb") as f:
        sentences, sentence_embeddings = pickle.load(f)
    return sentences, sentence_embeddings

def get_best_response(query, embeddings=None):
    """Get the best response based on the query and optional preloaded embeddings."""
    # If embeddings are not passed, load them
    if embeddings is None:
        sentences, sentence_embeddings = load_embeddings()
    else:
        sentences, sentence_embeddings = embeddings
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    best_match_idx = np.argmax(similarities)
    
    return sentences[best_match_idx]
