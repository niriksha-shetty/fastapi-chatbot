from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm  # For progress tracking
import torch  # For checking GPU availability

# Load preprocessed sentences
with open("cleaned_Galethird.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines()]

# Load pre-trained model with GPU support if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Convert sentences to embeddings using batch processing
batch_size = 32  # Adjust based on memory capacity
sentence_embeddings = []

for i in tqdm(range(0, len(sentences), batch_size), desc="Generating Embeddings"):
    batch = sentences[i : i + batch_size]
    batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    sentence_embeddings.extend(batch_embeddings)

# Convert to numpy array
sentence_embeddings = np.array(sentence_embeddings)

# Save embeddings for future use
with open("embeddings.pkl", "wb") as f:
    pickle.dump((sentences, sentence_embeddings), f)

print("Embeddings generated and saved successfully!")
