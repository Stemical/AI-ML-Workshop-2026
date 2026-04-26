import pickle
from pathlib import Path

_model = None

def _load():
    global _model
    if _model is None:
        with open(Path(__file__).parent / "spam_detector.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model

def predict(text):
    model = _load()
    label = "spam" if model.predict([text])[0] == 1 else "ham"
    probs = model.predict_proba([text])[0]
    return {
        "prediction": label,
        "spam_probability": round(float(probs[1]), 4),
        "ham_probability": round(float(probs[0]), 4),
    }
