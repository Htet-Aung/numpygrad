"""Reproducible classification splits and evaluation metrics."""

from __future__ import annotations

import numpy as np


def stratified_indices(labels: np.ndarray, seed: int, validation_fraction: float = 0.2,
                       test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train, validation and test indices for each class."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0 or validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Provide nonempty labels and fractions whose sum is below one")
    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[], [], []]
    for label in np.unique(labels):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        n_test = int(round(len(class_indices) * test_fraction))
        n_val = int(round(len(class_indices) * validation_fraction))
        if len(class_indices) - n_test - n_val < 1:
            raise ValueError(f"Class {label} has too few examples for this split")
        partitions[0].append(class_indices[:len(class_indices) - n_val - n_test])
        partitions[1].append(class_indices[len(class_indices) - n_val - n_test:len(class_indices) - n_test])
        partitions[2].append(class_indices[len(class_indices) - n_test:])
    return tuple(rng.permutation(np.concatenate(parts)) for parts in partitions)  # type: ignore[return-value]


def classification_metrics(logits: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Measure accuracy, macro F1, NLL, Brier score, and expected calibration error."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or len(logits) != len(labels) or not len(labels) or bins < 1:
        raise ValueError("Expected nonempty (N, C) logits, N labels and positive bins")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("Labels must index a logit column")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = prediction == labels
    f1 = []
    for klass in range(logits.shape[1]):
        tp = np.sum((prediction == klass) & (labels == klass))
        fp = np.sum((prediction == klass) & (labels != klass))
        fn = np.sum((prediction != klass) & (labels == klass))
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    one_hot = np.eye(logits.shape[1])[labels]
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
    }
