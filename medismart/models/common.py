"""Shared helpers for every training script in Phase 3.

Keeps metric computation, result storage and plotting consistent across the
four models so the comparison tables in the report are genuinely comparable.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

# Silence TensorFlow's startup chatter before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Make a training run reproducible across python, numpy and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def classification_metrics(y_true, y_pred, y_proba=None, average: str = "macro") -> dict:
    """Accuracy, precision, recall, F1 and (when possible) ROC-AUC.

    Accuracy alone is misleading on the imbalanced datasets in this project,
    so every model reports the full set.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 1 or proba.shape[1] == 2:
                score = proba if proba.ndim == 1 else proba[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, score))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average=average)
                )
        except Exception:
            # Happens when a fold is missing a class; not worth failing over.
            pass

    return metrics


def top_k_accuracy(y_true, y_proba, k: int = 3) -> float:
    """Fraction of samples whose true class is in the model's top k guesses.

    For a 41-class differential diagnosis this matters more than top-1: a
    pharmacist is shown a short ranked list, not a single answer.
    """
    proba = np.asarray(y_proba)
    top_k = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([t in row for t, row in zip(np.asarray(y_true), top_k)]))


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  metrics written to {path.name}")


def print_table(rows: list[dict], columns: list[str], title: str = "") -> None:
    """Print a comparison table that is readable in a terminal and copy-pasteable."""
    if title:
        print(f"\n  {title}")
    widths = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0)) for c in columns}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# Consistent colours across every figure in the report.
TEAL, CORAL, SLATE, AMBER, PURPLE = "#0f766e", "#e05252", "#475569", "#d97706", "#7c3aed"


def style_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
