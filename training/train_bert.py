from pathlib import Path
import sys
import shutil
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from bert_model import BertTextClassifier
from ensemble_model import Ensemble


SEED = 42
BASE_MODEL = "distilbert/distilbert-base-uncased"
MAX_TRAIN_SAMPLES = 8000
MAX_EVAL_SAMPLES = 2000
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "text_detector.joblib"
BERT_DIR = ARTIFACT_DIR / "bert_text_detector"
BACKUP_PATH = ARTIFACT_DIR / "text_detector_before_bert.joblib"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    import re
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

    return texts, np.asarray(labels, dtype=np.int64)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], int(self.labels[index])


def make_collate(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokens["labels"] = torch.tensor(labels, dtype=torch.long)
        return tokens
    return collate


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
    }


def report(name, model, texts, labels):
    predictions = model.predict(texts)
    metrics = calculate_metrics(labels, predictions)

    print(f"\n{name} metrics: {metrics}")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0,
    ))
    return metrics


def main():
    set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Existing model bundle not found: {MODEL_PATH}"
        )

    # Preserve the currently working two-model bundle.
    if not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Existing backup preserved: {BACKUP_PATH}")

    print("Loading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    rows = list(dataset["train"])

    # Keep the same question-level split used in train_model.py.
    train_rows, test_rows = train_test_split(
        rows, test_size=0.20, random_state=SEED
    )

    X_train, y_train = flatten_answers(train_rows)
    X_test, y_test = flatten_answers(test_rows)

    # Use stratified subsets for a first, more manageable BERT experiment.
    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(MAX_TRAIN_SAMPLES, len(X_train)),
        random_state=SEED,
        stratify=y_train,
    )
    X_eval, _, y_eval, _ = train_test_split(
        X_test,
        y_test,
        train_size=min(MAX_EVAL_SAMPLES, len(X_test)),
        random_state=SEED + 1,
        stratify=y_test,
    )

    print(f"BERT training examples: {len(X_train)}")
    print(f"Shared evaluation examples: {len(X_eval)}")
    print(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")

    print(f"\nLoading pretrained model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1},
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    train_loader = DataLoader(
        TextDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    print("\nFine-tuning DistilBERT...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        print(
            f"Average training loss: "
            f"{total_loss / max(len(train_loader), 1):.4f}"
        )

    print(f"\nSaving BERT model to {BERT_DIR}")
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)

    # Release training model memory before inference evaluation.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bert = BertTextClassifier(
        model_dir="artifacts/bert_text_detector",
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    print("\nEvaluating all models on the same held-out examples...")
    bundle = joblib.load(MODEL_PATH)
    models = bundle.get("models")

    if not models:
        raise ValueError(
            "The saved bundle does not contain individual models."
        )

    metrics = {}
    for name, existing_model in models.items():
        metrics[name] = report(
            name, existing_model, X_eval, y_eval
        )

    metrics["distilbert"] = report(
        "DistilBERT", bert, X_eval, y_eval
    )

    # Test the three-model ensemble. Do not assume it wins:
    # compare these metrics with the individual models above.
    models["distilbert"] = bert
    ensemble = Ensemble(models)
    metrics["ensemble"] = report(
        "Three-model ensemble", ensemble, X_eval, y_eval
    )

    updated_bundle = {
        "model": ensemble,
        "models": models,
        "metrics": metrics,
        "label_map": {0: "Human", 1: "AI"},
    }

    temporary_path = MODEL_PATH.with_name(
        "text_detector_with_bert.tmp"
    )
    joblib.dump(updated_bundle, temporary_path, compress=3)
    temporary_path.replace(MODEL_PATH)

    print("\nSUCCESS: Three-model bundle saved.")
    print(f"Bundle: {MODEL_PATH}")
    print(f"BERT weights: {BERT_DIR}")
    print(f"Backup: {BACKUP_PATH}")
    print("Labels: 0=Human, 1=AI")


if __name__ == "__main__":
    main()
