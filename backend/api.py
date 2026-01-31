from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import chat_once
import main  # Ensure this is the main.py above

app = FastAPI()


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

origins = [
    "http://localhost:8000",
    "http://localhost:5173",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # This runs the initialization once when the server starts
    main.initialize_system()

@app.on_event("shutdown")
def shutdown_event():
    if main.client:
        main.client.close()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # This now uses the pre-initialized rag_chain
    user_msg = req.message
    ai_reply = chat_once(user_msg)
    return {"response": ai_reply}