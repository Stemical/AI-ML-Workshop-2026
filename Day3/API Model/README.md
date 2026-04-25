# Mini Chat API

A local ChatGPT-style API powered by **Qwen2.5-0.5B-Instruct** (~1 GB) with optional live web search.

---

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The model downloads automatically from Hugging Face on first run (~1 GB).  
Subsequent starts load from cache instantly.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Status + available endpoints |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send messages, get a reply |

---

## Usage Examples

### Basic chat (curl)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of Norway?"}
    ]
  }'
```

### Multi-turn conversation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about coffee farming."},
      {"role": "assistant", "content": "Coffee farming involves ..."},
      {"role": "user", "content": "Which countries produce the most?"}
    ]
  }'
```

### With live web search
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What happened in the news today?"}
    ],
    "search": true
  }'
```

### Python client
```python
import requests

API = "http://localhost:8000"

def ask(prompt: str, history: list = [], search: bool = False) -> str:
    messages = history + [{"role": "user", "content": prompt}]
    resp = requests.post(f"{API}/chat", json={
        "messages": messages,
        "search": search,
        "temperature": 0.7,
    })
    resp.raise_for_status()
    data = resp.json()
    return data["reply"]

# Single question
print(ask("Explain machine learning in one paragraph."))

# With web search
print(ask("What are today's top AI news?", search=True))
```

---

## Request Schema

```json
{
  "messages": [
    {"role": "system",    "content": "..."},   // optional system prompt
    {"role": "user",      "content": "..."},   // user message
    {"role": "assistant", "content": "..."}    // prior assistant turn
  ],
  "search": false,          // true = inject DuckDuckGo results
  "max_tokens": 512,        // max tokens to generate
  "temperature": 0.7        // 0 = deterministic, 1 = creative
}
```

## Response Schema

```json
{
  "reply": "...",            // model's response
  "search_used": false,     // whether web search was triggered
  "search_snippet": null    // raw search results (if search=true)
}
```

---

## Model

**Qwen2.5-0.5B-Instruct** by Alibaba Cloud  
- 494M parameters  
- ~1 GB on disk (float32 CPU) / ~500 MB (float16 GPU)  
- Runs fine on CPU (slow) or GPU (fast)  
- Supports system prompts, multi-turn chat, and instruction following  
- Apache 2.0 license  

To swap to a larger model, change `MODEL_ID` in `main.py`:
```python
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # better quality, ~3 GB
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"     # even better, ~6 GB
```
