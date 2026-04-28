"""
Spam Detector API
Run with: waitress-serve --port=5000 classical_api:app
Test:     curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"message\": \"Win a free prize now!\"}"
"""

import pickle
from pathlib import Path
from flask import Flask, request, jsonify

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


@app.route("/predict", methods=["POST"])
def predict():
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + MultinomialNB"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Spam Detector API",
        "endpoints": {
            "POST /predict": "Classify a message as spam or ham",
            "GET /health": "Health check",
        },
        "example": {
            "url": "POST /predict",
            "body": {"message": "You have won a free prize! Call now"},
        },
    })


if __name__ == "__main__":
    print("Starting Spam Detector API...")
    print("API running at http://localhost:5000")
    print()
    print("Endpoints:")
    print("  GET  /        - API info")
    print("  GET  /health  - Health check")
    print("  POST /predict - Classify a message")
    print()
    print("Test with curl:")
    print('  curl -X POST http://localhost:5000/predict \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"message": "You have won a free prize! Call now"}\'')
    print()
    print("Press CTRL+C to stop the server.")
    print("-" * 50)

    app.run(host="0.0.0.0", port=5000, debug=False)

# run using: waitress-serve --port=5000 classical_api:app