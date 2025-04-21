# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# Request model
class ChatRequest(BaseModel):
    message: str

# Response endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    response = get_best_response(request.message)
    return {"reply": response}
