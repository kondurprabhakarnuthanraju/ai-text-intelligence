from pathlib import Path
import re
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from ensemble_model import Ensemble

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "text_detector.joblib"
RANDOM_STATE = 42


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def flatten_answers(rows):
    texts, labels = [], []

    for row in rows:
        for answer in row.get("human_answers", []) or []:
            answer = clean_text(answer)
            if len(answer) >= 40:
                texts.append(answer)
                labels.append(0)

        for answer in row.get("chatgpt_answers", []) or []:
            answer = clean_text(answer)
            if len(answer) >= 40:
                texts.append(answer)
                labels.append(1)

    return texts, np.array(labels)


def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, predictions, zero_division=0
        ))
    }

    print(f"\n{name} metrics: {result}")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    ))
    return result


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    rows = list(dataset["train"])

    train_rows, test_rows = train_test_split(
        rows, test_size=0.20, random_state=RANDOM_STATE
    )

    X_train, y_train = flatten_answers(train_rows)
    X_test, y_test = flatten_answers(test_rows)

    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print("Labels: 0=Human, 1=AI")

    models = {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode"
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode"
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=50,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics[name] = evaluate(name, model, X_test, y_test)

    ensemble = Ensemble(models)
    metrics["ensemble"] = evaluate(
        "Ensemble", ensemble, X_test, y_test
    )

    # Save the ensemble and individual models.
    # The Ensemble class is defined in ensemble_model.py,
    # so joblib can serialize it correctly.
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    joblib.dump({
        "model": ensemble,
        "models": models,
        "metrics": metrics,
        "label_map": {0: "Human", 1: "AI"}
    }, temporary_path, compress=3)

    temporary_path.replace(MODEL_PATH)

    print(f"\nSUCCESS: Model saved to {MODEL_PATH}")
    print("Prediction classes: 0=Human, 1=AI")


if __name__ == "__main__":
    main()
