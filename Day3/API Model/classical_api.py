"""
Spam Detector + AI Reply API
Run with: waitress-serve --port=5000 classical_api:app
"""

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

# spam_detector.pkl lives one level up (Day3/), not inside Day3/API Model/
MODEL_PATH = Path(__file__).parent.parent / "spam_detector.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run sms_spam_detector.ipynb first to generate spam_detector.pkl."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    try:
        probs = model.predict_proba([message])[0]
        label = "spam" if model.predict([message])[0] == 1 else "ham"
        return jsonify({
            "message": message,
            "prediction": label,
            "spam_probability": round(float(probs[1]), 4),
            "ham_probability": round(float(probs[0]), 4),
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-new-key-here":
        return jsonify({"error": "OPENAI_API_KEY not set in .env"}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=data.get("messages", []),
            max_tokens=data.get("max_tokens", 120),
            temperature=data.get("temperature", 0.75),
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + MultinomialNB"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Spam Detector + AI Reply API",
        "endpoints": {
            "POST /predict": "Classify a message as spam or ham",
            "POST /chat": "Get a ChatGPT reply (requires OPENAI_API_KEY in .env)",
            "GET /health": "Health check",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
