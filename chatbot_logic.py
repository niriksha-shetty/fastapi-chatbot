import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    sentences, sentence_embeddings = pickle.load(f)

def get_best_response(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    best_match_idx = np.argmax(similarities)
    return sentences[best_match_idx]
