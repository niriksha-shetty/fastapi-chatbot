# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import gdown
import pickle
from chatbot_logic import get_best_response  # Your chatbot function

app = FastAPI()

# Enable CORS (allow requests from Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the file ID and output path for embeddings.pkl
file_id = '1Vg5lh0SmGLLXyoBVrGtj6dlJncEFlX-m'
output_file = 'embeddings.pkl'

def download_embeddings():
    """Download embeddings.pkl from Google Drive using gdown."""
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        print("Downloading embeddings.pkl from Google Drive...")
        gdown.download(url, output_file, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading embeddings.pkl: {e}")
        raise HTTPException(status_code=500, detail="Failed to download embeddings.pkl")

def load_embeddings():
    """Load embeddings from embeddings.pkl."""
    if not os.path.exists(output_file):
        download_embeddings()  # Download the file if not found locally

    try:
        with open(output_file, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings.pkl: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embeddings.pkl")

# Request model
class ChatRequest(BaseModel):
    message: str

# Response endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Load embeddings when the app starts or when the chatbot is called
        embeddings = load_embeddings()
        
        # Use embeddings in your chatbot logic
        response = get_best_response(request.message, embeddings)
        return {"reply": response}
    except Exception as e:
        return {"error": str(e)}

