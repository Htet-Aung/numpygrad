"""
Evaluation metrics for classification tasks.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np
from numpygrad.core.tensor import Tensor


def accuracy(
    y_pred: Union[Tensor, np.ndarray],
    y_true: Union[Tensor, np.ndarray],
) -> float:
    """
    Computes the classification accuracy score.

    Parameters
    ----------
    y_pred : Union[Tensor, np.ndarray]
        Predicted class labels (1D array of shape (N,)) or raw logits/probabilities
        (2D array of shape (N, C)).
    y_true : Union[Tensor, np.ndarray]
        Ground truth integer class labels (shape (N,) or (N, 1)).

    Returns
    -------
    float
        Fraction of correctly predicted samples in [0.0, 1.0].
    """
    pred_data = y_pred.data if isinstance(y_pred, Tensor) else np.asarray(y_pred)
    true_data = y_true.data if isinstance(y_true, Tensor) else np.asarray(y_true)

    if pred_data.ndim == 2:
        pred_labels = np.argmax(pred_data, axis=-1).reshape(-1)
    else:
        pred_labels = pred_data.reshape(-1)

    true_labels = true_data.reshape(-1)

    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Shape mismatch: y_pred has {len(pred_labels)} samples, "
            f"y_true has {len(true_labels)} samples"
        )

    if len(true_labels) == 0:
        return 0.0

    return float(np.mean(pred_labels == true_labels))


def confusion_matrix(
    y_pred: Union[Tensor, np.ndarray],
    y_true: Union[Tensor, np.ndarray],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Computes the multi-class confusion matrix without external dependencies.

    Row indices represent true classes; column indices represent predicted classes:
        cm[i, j] = number of samples with true class i and predicted class j.

    Parameters
    ----------
    y_pred : Union[Tensor, np.ndarray]
        Predicted class labels (1D) or raw logits (2D).
    y_true : Union[Tensor, np.ndarray]
        Ground truth integer class labels (1D).
    num_classes : Optional[int], default=None
        Total number of classes. If None, inferred as max(y_true, y_pred) + 1.

    Returns
    -------
    np.ndarray
        Integer confusion matrix of shape (num_classes, num_classes) and dtype int64.
    """
    pred_data = y_pred.data if isinstance(y_pred, Tensor) else np.asarray(y_pred)
    true_data = y_true.data if isinstance(y_true, Tensor) else np.asarray(y_true)

    if pred_data.ndim == 2:
        pred_labels = np.argmax(pred_data, axis=-1).reshape(-1).astype(np.int64)
    else:
        pred_labels = pred_data.reshape(-1).astype(np.int64)

    true_labels = true_data.reshape(-1).astype(np.int64)

    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Shape mismatch: y_pred has {len(pred_labels)} samples, "
            f"y_true has {len(true_labels)} samples"
        )

    if len(true_labels) == 0:
        n_cls = 0 if num_classes is None else max(0, num_classes)
        return np.zeros((n_cls, n_cls), dtype=np.int64)

    if num_classes is None:
        num_classes = int(max(np.max(true_labels), np.max(pred_labels)) + 1)
    elif num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if np.any(true_labels < 0) or np.any(true_labels >= num_classes):
        raise ValueError(f"Found true label out of bounds [0, {num_classes - 1}]")
    if np.any(pred_labels < 0) or np.any(pred_labels >= num_classes):
        raise ValueError(f"Found predicted label out of bounds [0, {num_classes - 1}]")

    # Fast bincount confusion matrix calculation
    linear_indices = true_labels * num_classes + pred_labels
    counts = np.bincount(linear_indices, minlength=num_classes * num_classes)
    return counts.reshape((num_classes, num_classes)).astype(np.int64)
