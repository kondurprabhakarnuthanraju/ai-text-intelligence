from pathlib import Path
import re
import json
import random

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "text_detector.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_ROWS = None  # Set to an integer for a quick smoke test; None uses all HC3 rows.


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def flatten_answers(rows):
    texts, labels = [], []
    for row in rows:
        human_answers = row.get("human_answers") or []
        ai_answers = row.get("chatgpt_answers") or []
        if isinstance(human_answers, str):
            human_answers = [human_answers]
        if isinstance(ai_answers, str):
            ai_answers = [ai_answers]

        for answer in human_answers:
            text = clean_text(answer)
            if len(text) >= 40:
                texts.append(text)
                labels.append(0)
        for answer in ai_answers:
            text = clean_text(answer)
            if len(text) >= 40:
                texts.append(text)
                labels.append(1)
    return texts, labels


def main():
    print("Downloading/loading HC3 from Hugging Face...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    rows = list(dataset["train"])
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]

    # Split at the original question-row level before flattening answers.
    # This helps prevent answers to the same question leaking across splits.
    train_rows, test_rows = train_test_split(
        rows, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, y_train = flatten_answers(train_rows)
    X_test, y_test = flatten_answers(test_rows)

    if not X_train or not X_test:
        raise RuntimeError("No usable HC3 answers found; check dataset schema.")

    print(f"Train answer samples: {len(X_train):,}")
    print(f"Test answer samples : {len(X_test):,}")
    print(f"Train AI share      : {np.mean(y_train):.3f}")
    print(f"Test AI share       : {np.mean(y_test):.3f}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_ai": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_ai": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_ai": float(f1_score(y_test, predictions, zero_division=0)),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "dataset": "Hello-SimpleAI/HC3",
        "split_unit": "original question row before answer flattening",
    }
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report (0=Human, 1=AI):")
    print(classification_report(y_test, predictions, target_names=["Human", "AI"], zero_division=0))
    print("Confusion matrix [Human, AI]:")
    print(confusion_matrix(y_test, predictions).tolist())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH, compress=3)
    print(f"\nSaved model to: {MODEL_PATH}")
    print("Before deployment, check artifact size and current Vercel bundle limits.")


if __name__ == "__main__":
    main()
