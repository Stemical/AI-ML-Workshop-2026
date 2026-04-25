"""
Simple Chat API — Qwen2.5-0.5B-Instruct + optional web search
Run:  uvicorn main:app --reload --port 8000
"""

import re
from contextlib import asynccontextmanager
from typing import Optional

import torch
from duckduckgo_search import DDGS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW    = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global model state ─────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_ID} on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    state["model"]     = model
    state["tokenizer"] = tokenizer
    print("Model ready.")
    yield
    state.clear()


app = FastAPI(
    title="Mini Chat API",
    description="ChatGPT-style API powered by Qwen2.5-0.5B-Instruct with optional web search",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str          # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    search: bool = False          # set True to inject a live web search result
    max_tokens: Optional[int] = MAX_NEW
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    reply: str
    search_used: bool
    search_snippet: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 3) -> str:
    """Return a short summary string from DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['body'][:200]}")
    return "\n".join(lines)


def extract_user_query(messages: list[Message]) -> str:
    """Pull the last user message to use as a search query."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def generate(messages: list[Message], max_tokens: int, temperature: float) -> str:
    tokenizer = state["tokenizer"]
    model     = state["model"]

    # Build chat prompt using the model's built-in template
    chat = [{"role": m.role, "content": m.content} for m in messages]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens from the output
    new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return response.strip()


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "endpoints": {
            "POST /chat": "Send a conversation and get a reply",
            "GET  /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded yet")

    messages       = list(req.messages)   # make a mutable copy
    search_snippet = None

    # ── Optional web search ───────────────────────────────────────────────
    if req.search:
        query = extract_user_query(messages)
        if query:
            search_snippet = web_search(query)
            # Inject search results as a system context message
            context_msg = Message(
                role="system",
                content=(
                    f"Here are live web search results for the query '{query}':\n"
                    f"{search_snippet}\n\n"
                    "Use this information to answer the user's question. "
                    "Cite the source titles when helpful."
                ),
            )
            # Insert before the last user message
            messages.insert(len(messages) - 1, context_msg)

    reply = generate(messages, req.max_tokens, req.temperature)
    return ChatResponse(
        reply=reply,
        search_used=req.search,
        search_snippet=search_snippet,
    )
