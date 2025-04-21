from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load preprocessed sentences
with open("cleaned_Galethird.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines()]

# Load a pre-trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert sentences to embeddings
sentence_embeddings = model.encode(sentences)

# Save embeddings for future use
with open("embeddings.pkl", "wb") as f:
    pickle.dump((sentences, sentence_embeddings), f)

print("Embeddings generated and saved!")
