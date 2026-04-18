"""
Twitter Language Classification Pipeline
Sheng vs English | Day 2 Workshop
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_excel("cleaned_language_twitter.xlsx")
print(f"Dataset shape: {df.shape}")
print(df["language"].value_counts())

# ── 2. Text Cleaning ──────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # URLs
    text = re.sub(r"@\w+|#\w+", "", text)           # mentions / hashtags
    text = re.sub(r"[^a-z\s]", "", text)            # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

df["cleaned"] = df["source"].apply(clean_text)
print("\nSample cleaned text:")
print(df[["source", "cleaned"]].head(3).to_string())

# ── 3. Visualisations ─────────────────────────────────────────────────────────
CLASSES = ["English", "Sheng"]
COLORS  = ["#4C72B0", "#DD8452"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Twitter Language Dataset – Text Analysis", fontsize=16, fontweight="bold")

# 3a. Class distribution
ax = axes[0, 0]
counts = df["language"].value_counts()
ax.bar(counts.index, counts.values, color=COLORS)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")

# 3b. Word Cloud (all text)
ax = axes[0, 1]
all_text = " ".join(df["cleaned"])
wc = WordCloud(width=800, height=400, background_color="white",
               colormap="viridis", max_words=100).generate(all_text)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Word Cloud – All Tweets")

# 3c. Top-20 most common words
ax = axes[1, 0]
all_tokens = all_text.split()
word_freq = Counter(all_tokens).most_common(20)
words, freqs = zip(*word_freq)
ax.barh(words[::-1], freqs[::-1], color="#4C72B0")
ax.set_title("Top 20 Most Common Words (after cleaning)")
ax.set_xlabel("Frequency")

# 3d. Top-15 bigrams
ax = axes[1, 1]
bigram_list = list(ngrams(all_tokens, 2))
bigram_freq = Counter([" ".join(b) for b in bigram_list]).most_common(15)
bg_words, bg_freqs = zip(*bigram_freq)
ax.barh(bg_words[::-1], bg_freqs[::-1], color="#DD8452")
ax.set_title("Top 15 Bigrams")
ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("text_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → text_analysis.png")

# ── 4. Per-class word clouds ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, lang, color in zip(axes, CLASSES, ["Blues", "Oranges"]):
    text = " ".join(df[df["language"] == lang]["cleaned"])
    wc = WordCloud(width=700, height=350, background_color="white",
                   colormap=color, max_words=80).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud – {lang}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("wordclouds_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → wordclouds_per_class.png")

# ── 5. Bag of Words Feature Engineering ──────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df["language"])          # English=0, Sheng=1
print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

vectorizer = CountVectorizer(max_features=2000, ngram_range=(1, 2))
X_bow = vectorizer.fit_transform(df["cleaned"]).toarray()
print(f"BoW feature matrix: {X_bow.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 6. Gradient Boosting Classifier ──────────────────────────────────────────
print("\n── Gradient Boosting (XGBoost) ──")
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost – Confusion Matrix")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → xgb_confusion_matrix.png")

# Feature importance (top 20)
importances = xgb_clf.feature_importances_
feature_names = vectorizer.get_feature_names_out()
top_idx = np.argsort(importances)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_names[top_idx][::-1], importances[top_idx][::-1], color="#4C72B0")
ax.set_title("XGBoost – Top 20 Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → xgb_feature_importance.png")

# ── 7. Deep Learning (LSTM) ───────────────────────────────────────────────────
print("\n── Deep Learning (LSTM) ──")
VOCAB_SIZE  = 5000
MAX_LEN     = 50
EMBED_DIM   = 64
EPOCHS      = 20
BATCH_SIZE  = 16

tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tok.fit_on_texts(df["cleaned"])
sequences = tok.texts_to_sequences(df["cleaned"])
X_dl = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

X_tr_dl, X_te_dl, y_tr_dl, y_te_dl = train_test_split(
    X_dl, y, test_size=0.2, random_state=42, stratify=y
)

model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.SpatialDropout1D(0.3),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_tr_dl, y_tr_dl,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
    ax.plot(history.history[metric],       label="Train")
    ax.plot(history.history[f"val_{metric}"], label="Validation")
    ax.set_title(f"LSTM – {title}")
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.savefig("lstm_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lstm_training_curves.png")

# Evaluation
y_pred_dl_prob = model.predict(X_te_dl).flatten()
y_pred_dl = (y_pred_dl_prob >= 0.5).astype(int)
print(classification_report(y_te_dl, y_pred_dl, target_names=le.classes_))

fig, ax = plt.subplots(figsize=(6, 5))
cm_dl = confusion_matrix(y_te_dl, y_pred_dl)
disp = ConfusionMatrixDisplay(cm_dl, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title("LSTM – Confusion Matrix")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lstm_confusion_matrix.png")

# ── 8. Model Comparison ───────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score

results = {
    "XGBoost (BoW)": {
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "F1 (macro)": f1_score(y_test, y_pred_xgb, average="macro"),
    },
    "LSTM (DL)": {
        "Accuracy": accuracy_score(y_te_dl, y_pred_dl),
        "F1 (macro)": f1_score(y_te_dl, y_pred_dl, average="macro"),
    },
}
results_df = pd.DataFrame(results).T
print("\n── Model Comparison ──")
print(results_df.round(4))

fig, ax = plt.subplots(figsize=(8, 5))
results_df.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white")
ax.set_title("Model Comparison – Accuracy & F1")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_xticklabels(results_df.index, rotation=0)
ax.legend(loc="lower right")
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → model_comparison.png")

print("\nAll done!")
