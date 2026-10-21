from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BertTextClassifier:
    def __init__(self, model_path="artifacts/bert_text_detector",
                 max_length=256, batch_size=8):
        self.model_path = str(
            Path(__file__).resolve().parent / model_path
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cpu")

    def _load(self):
        if getattr(self, "_tokenizer", None) is None:
            path = getattr(
                self,
                "model_path",
                str(Path(__file__).resolve().parent /
                    "artifacts" / "bert_text_detector")
            )
            self.model_path = str(Path(path).resolve())
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )

        if getattr(self, "_model", None) is None:
            self._device = torch.device("cpu")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
                low_cpu_mem_usage=False
            )
            self._model.to(self._device)
            self._model.eval()

    def predict_proba(self, texts):
        self._load()
        texts = list(texts)

        if not texts:
            return np.empty((0, 2), dtype=np.float64)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}

            with torch.no_grad():
                logits = self._model(**tokens).logits
                probabilities = torch.softmax(logits, dim=-1)

            batches.append(probabilities.cpu().numpy())

        return np.concatenate(batches, axis=0)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)
