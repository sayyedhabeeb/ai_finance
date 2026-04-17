"""
FinBERT-based financial sentiment analysis.

Wraps the ProsusAI/finbert model for classifying financial text into
sentiment categories (positive, negative, neutral) with a normalized
score suitable for downstream quantitative usage.

Features:
- Single-text and batch sentiment analysis
- Aggregate scoring over a collection of texts
- Aspect-based sentiment extraction (revenue, growth, risk, etc.)
- Custom classification head fine-tuning
- Normalized scoring: 0 = bearish, 0.5 = neutral, 1 = bullish
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Default model identifier
_DEFAULT_MODEL_NAME = "ProsusAI/finbert"

# FinBERT label mapping (ProsusAI/finbert convention)
_LABEL_MAP: Dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}
_LABEL_MAP_INV: Dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

# Normalized score mapping: positive=1.0, neutral=0.5, negative=0.0
_SCORE_MAP: Dict[str, float] = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}

# Financial aspects to detect for aspect-based sentiment
_FINANCIAL_ASPECTS: List[str] = [
    "revenue",
    "earnings",
    "profit",
    "growth",
    "risk",
    "debt",
    "cash flow",
    "dividend",
    "valuation",
    "innovation",
    "regulation",
    "market share",
    "guidance",
    "outlook",
    "demand",
]


# ---------------------------------------------------------------------------
# Aspect Extraction Utilities
# ---------------------------------------------------------------------------

_ASPECT_PATTERNS: Dict[str, re.Pattern] = {
    aspect: re.compile(
        r"\b(" + aspect + r"[a-z]*)\b",
        re.IGNORECASE,
    )
    for aspect in _FINANCIAL_ASPECTS
}


def _extract_sentences_with_aspect(
    text: str, aspect: str
) -> List[str]:
    """Return sentences from *text* that mention *aspect*."""
    pattern = _ASPECT_PATTERNS[aspect]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matched = [s.strip() for s in sentences if pattern.search(s)]
    return matched


# ---------------------------------------------------------------------------
# Fine-tuning Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Simple dataset for fine-tuning FinBERT on custom labelled data."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Main Analyser
# ---------------------------------------------------------------------------

class FinBERTSentimentAnalyzer:
    """FinBERT-based financial sentiment analysis.

    Loads a pre-trained FinBERT model and provides a high-level API for
    sentiment scoring suitable for integration with trading signals.

    Example
    -------
    >>> analyzer = FinBERTSentimentAnalyzer()
    >>> result = analyzer.analyze("The company reported record earnings this quarter.")
    >>> result["normalized_score"]
    1.0
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        mlflow_experiment: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize the analyzer.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier or local path.
        device : str or None
            ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        mlflow_experiment : str or None
            MLflow experiment name for tracking fine-tuning runs.
        max_length : int
            Maximum token length for the tokenizer.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.mlflow_experiment = mlflow_experiment

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self._model.eval()

        self._finetuned = False

    # ------------------------------------------------------------------
    # Core Analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment of a single piece of financial text.

        Parameters
        ----------
        text : str
            Financial text to analyse.

        Returns
        -------
        dict
            ``{
                "label": "positive" | "negative" | "neutral",
                "score": float,
                "normalized_score": float,
                "probabilities": {"positive": float, "negative": float, "neutral": float},
            }``
        """
        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 0.5,
                "normalized_score": 0.5,
                "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Map model outputs to labels
        label_idx = int(np.argmax(probs))
        label = _LABEL_MAP.get(label_idx, "neutral")
        score = float(probs[label_idx])
        normalized = _SCORE_MAP.get(label, 0.5)

        probabilities = {
            _LABEL_MAP.get(i, f"class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        return {
            "label": label,
            "score": score,
            "normalized_score": normalized,
            "probabilities": probabilities,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyse sentiment for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            List of financial texts.

        Returns
        -------
        list[dict]
            List of analysis results (same format as ``analyze``).
        """
        if not texts:
            return []

        # Tokenize all at once for efficiency
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: List[Dict[str, Any]] = []
        for i in range(len(texts)):
            p = probs[i]
            label_idx = int(np.argmax(p))
            label = _LABEL_MAP.get(label_idx, "neutral")
            score = float(p[label_idx])
            normalized = _SCORE_MAP.get(label, 0.5)
            probabilities = {
                _LABEL_MAP.get(j, f"class_{j}"): float(p[j])
                for j in range(len(p))
            }
            results.append({
                "label": label,
                "score": score,
                "normalized_score": normalized,
                "probabilities": probabilities,
            })

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_sentiment(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate sentiment statistics over a collection of texts.

        Parameters
        ----------
        texts : list[str]
            Financial texts.
        weights : list[float] or None
            Optional per-text weight (e.g., recency or source credibility).

        Returns
        -------
        dict
            ``{
                "overall_label": str,
                "overall_score": float,
                "normalized_score": float,
                "distribution": {"positive": float, "negative": float, "neutral": float},
                "mean_normalized": float,
                "std_normalized": float,
                "count": int,
            }``
        """
        if not texts:
            return {
                "overall_label": "neutral",
                "overall_score": 0.0,
                "normalized_score": 0.5,
                "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
                "mean_normalized": 0.5,
                "std_normalized": 0.0,
                "count": 0,
            }

        batch_results = self.analyze_batch(texts)

        n = len(batch_results)
        if weights is not None:
            weights = np.array(weights, dtype=np.float64)
            weights = weights / weights.sum()
        else:
            weights = np.ones(n, dtype=np.float64) / n

        # Weighted probabilities
        prob_accum = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        norm_scores = []
        for result, w in zip(batch_results, weights):
            for label, prob in result["probabilities"].items():
                prob_accum[label] = prob_accum.get(label, 0.0) + w * prob
            norm_scores.append(result["normalized_score"])

        overall_label = max(prob_accum, key=prob_accum.get)
        overall_score = float(prob_accum[overall_label])

        norm_arr = np.array(norm_scores)
        mean_norm = float(np.average(norm_arr, weights=weights))
        std_norm = float(np.sqrt(np.average((norm_arr - mean_norm) ** 2, weights=weights)))

        return {
            "overall_label": overall_label,
            "overall_score": overall_score,
            "normalized_score": _SCORE_MAP.get(overall_label, 0.5),
            "distribution": prob_accum,
            "mean_normalized": mean_norm,
            "std_normalized": std_norm,
            "count": n,
        }

    # ------------------------------------------------------------------
    # Aspect-Based Sentiment
    # ------------------------------------------------------------------

    def extract_aspect_sentiment(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract per-aspect sentiment from a longer financial text.

        Splits the text into sentences, groups them by financial aspect,
        and analyses each group independently.

        Parameters
        ----------
        text : str
            Financial article or report.

        Returns
        -------
        dict[str, dict]
            Mapping aspect name -> analysis result.  Only aspects that
            are mentioned in the text are included.
        """
        if not text or not text.strip():
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        for aspect in _FINANCIAL_ASPECTS:
            sentences = _extract_sentences_with_aspect(text, aspect)
            if not sentences:
                continue

            # Combine matched sentences and run analysis
            combined = " ".join(sentences)
            analysis = self.analyze(combined)
            results[aspect] = {
                **analysis,
                "n_sentences": len(sentences),
                "sentences": sentences,
            }

        return results

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        mlflow_run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Fine-tune the classification head on custom labelled data.

        Parameters
        ----------
        texts : list[str]
            Training texts.
        labels : list[str]
            Corresponding labels (``"positive"``, ``"negative"``, ``"neutral"``).
        val_texts : list[str] or None
            Validation texts.
        val_labels : list[str] or None
            Validation labels.
        output_dir : str or None
            Directory to save the fine-tuned model.
        epochs : int
            Number of training epochs.
        batch_size : int
            Per-device training batch size.
        learning_rate : float
            Learning rate.
        warmup_ratio : float
            Fraction of steps for LR warm-up.
        weight_decay : float
            Weight decay coefficient.
        mlflow_run_name : str or None
            Optional MLflow run name.

        Returns
        -------
        dict
            Evaluation metrics (accuracy, loss, etc.).
        """
        assert len(texts) == len(labels), "texts and labels must have same length"

        # Convert string labels to integers
        label_ids = [_LABEL_MAP_INV.get(l.lower(), 2) for l in labels]
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        # Prepare datasets
        train_dataset = SentimentDataset(texts, label_ids, self._tokenizer, self.max_length)

        eval_dataset = None
        if val_texts is not None and val_labels is not None:
            val_label_ids = [_LABEL_MAP_INV.get(l.lower(), 2) for l in val_labels]
            val_label_ids = torch.tensor(val_label_ids, dtype=torch.long)
            eval_dataset = SentimentDataset(
                val_texts, val_label_ids, self._tokenizer, self.max_length
            )

        # Training arguments
        if output_dir is None:
            output_dir = os.path.join(
                os.getcwd(), "finbert_finetuned", str(uuid.uuid4())[:8]
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch" if eval_dataset is not None else "no",
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to=["none"],  # We'll log to MLflow manually
        )

        # Compute metrics function
        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            from sklearn.metrics import accuracy_score, f1_score

            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro")),
            }

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # MLflow tracking
        mlflow_active = False
        if self.mlflow_experiment:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                mlflow.start_run(run_name=mlflow_run_name)
                mlflow_active = True
                mlflow.log_params({
                    "base_model": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "warmup_ratio": warmup_ratio,
                    "weight_decay": weight_decay,
                    "n_train_samples": len(texts),
                    "n_val_samples": len(val_texts) if val_texts else 0,
                })
            except Exception as exc:
                logger.warning("MLflow initialization failed: %s", exc)

        # Train
        train_result = trainer.train()

        # Evaluate
        metrics: Dict[str, float] = {}
        if eval_dataset is not None:
            eval_results = trainer.evaluate()
            metrics.update(eval_results)

        metrics["train_loss"] = float(train_result.training_loss)

        if mlflow_active:
            mlflow.log_metrics(metrics)
            mlflow.end_run()

        # Save fine-tuned model
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        self._finetuned = True
        logger.info("Fine-tuned model saved to %s", output_dir)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model and tokenizer to a directory.

        Parameters
        ----------
        path : str
            Directory path.
        """
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "finetuned": self._finetuned,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Sentiment model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "FinBERTSentimentAnalyzer":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Directory containing saved model.
        device : str or None
            Target device.

        Returns
        -------
        FinBERTSentimentAnalyzer
        """
        meta_path = os.path.join(path, "meta.json")
        meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        analyzer = cls.__new__(cls)
        analyzer.model_name = meta.get("model_name", path)
        analyzer.max_length = meta.get("max_length", 512)
        analyzer.mlflow_experiment = None

        if device is None:
            analyzer.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            analyzer.device = device

        analyzer._tokenizer = AutoTokenizer.from_pretrained(path)
        analyzer._model = AutoModelForSequenceClassification.from_pretrained(
            path
        ).to(analyzer.device)
        analyzer._model.eval()
        analyzer._finetuned = meta.get("finetuned", True)

        return analyzer
