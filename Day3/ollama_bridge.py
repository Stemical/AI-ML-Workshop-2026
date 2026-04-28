"""
Ollama Bridge API for Aura Headphones Reviews
==============================================

Receives chat requests from the frontend and forwards them to Ollama.

Run with:
    uvicorn ollama_bridge:app --port 8000 --reload

Prerequisites:
    pip install fastapi uvicorn requests
    ollama serve  (or have Ollama running)
    ollama pull gemma3:1b  (or whichever model you want)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests

# ─── Config ──────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"  # change to whichever model you have pulled
# Run `ollama list` to see installed models.
# Good options: gemma3:1b, llama3.2:1b, qwen2.5:1.5b, phi3.5

# ─── App setup ───────────────────────────────────────────────────────
app = FastAPI(title="Ollama Bridge")

# CORS — needed because the HTML page will be opened from file:// or a
# different origin than localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/response schemas ────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    model: Optional[str] = None  # let frontend override if it wants


class ChatResponse(BaseModel):
    reply: str
    model: str


# ─── Routes ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Used by the frontend's API status indicator."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        ollama_ok = r.ok
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "ollama_unreachable",
        "ollama": ollama_ok,
        "default_model": DEFAULT_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Forward a chat request to Ollama and return the reply."""
    model = req.model or DEFAULT_MODEL

    payload = {
        "model": model,
        "messages": [m.dict() for m in req.messages],
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Is it running? Try: ollama serve",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond. Try a smaller model.",
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {e.response.text}",
        )

    data = r.json()
    reply = data.get("message", {}).get("content", "").strip()

    if not reply:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned no content. Raw: {data}",
        )

    return ChatResponse(reply=reply, model=model)


@app.get("/")
def root():
    return {
        "service": "Ollama Bridge",
        "endpoints": {
            "GET /health": "Check Ollama connection",
            "POST /chat": "Send messages to Ollama, get a reply",
        },
        "default_model": DEFAULT_MODEL,
    }
#  uvicorn ollama_bridge:app --port 8000 --reload