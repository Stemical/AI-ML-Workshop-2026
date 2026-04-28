"""
OpenAI Bridge API
=================

Simple bridge that forwards chat requests to the OpenAI API.

Run with:
    uvicorn openai_bridge:app --port 8000 --reload

Prerequisites:
    pip install fastapi uvicorn openai python-dotenv
    .env file with OPENAI_API_KEY=sk-...
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ─── Config ──────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to your .env file.")

DEFAULT_MODEL = "gpt-4o-mini"  # cheap + fast; swap for gpt-4o, gpt-5, etc.

client = OpenAI(api_key=API_KEY)

# ─── App setup ───────────────────────────────────────────────────────
app = FastAPI(title="OpenAI Bridge")

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
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str


# ─── Routes ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick check that the API key is loaded and OpenAI is reachable."""
    try:
        client.models.list()
        return {"status": "ok", "default_model": DEFAULT_MODEL}
    except OpenAIError as e:
        return {"status": "openai_unreachable", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Forward a chat request to OpenAI and return the reply."""
    model = req.model or DEFAULT_MODEL

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[m.dict() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    reply = completion.choices[0].message.content.strip()
    if not reply:
        raise HTTPException(status_code=502, detail="OpenAI returned empty content.")

    return ChatResponse(reply=reply, model=model)


@app.post("/ask")
def ask(prompt: str):
    """Quick one-shot endpoint — send a single prompt as a query string."""
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    return {"reply": completion.choices[0].message.content.strip()}


@app.get("/")
def root():
    return {
        "service": "OpenAI Bridge",
        "endpoints": {
            "GET /health": "Check OpenAI connection",
            "POST /chat": "Send a list of messages, get a reply",
            "POST /ask?prompt=...": "Quick one-shot prompt",
        },
        "default_model": DEFAULT_MODEL,
    }