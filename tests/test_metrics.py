"""
Unit tests for NumPyGrad evaluation metrics (accuracy, confusion_matrix).
"""

import numpy as np
import pytest

from numpygrad.core.tensor import Tensor
from numpygrad.metrics import accuracy, confusion_matrix


def test_accuracy_2d_logits_and_tensor():
    # 4 samples, 3 classes
    logits = Tensor([
        [2.5, 0.1, -1.0],  # argmax = 0
        [-0.5, 3.2, 1.0],  # argmax = 1
        [0.0, 1.0, 4.0],   # argmax = 2
        [1.0, 5.0, 2.0],   # argmax = 1
    ])
    targets = Tensor([0, 1, 2, 0])  # predictions are 0, 1, 2, 1 -> 3/4 correct (75%)

    score = accuracy(logits, targets)
    assert score == pytest.approx(0.75)


def test_accuracy_1d_predictions_numpy():
    preds = np.array([0, 1, 2, 3, 0])
    targets = np.array([0, 1, 2, 2, 0])  # 4/5 correct (80%)

    score = accuracy(preds, targets)
    assert score == pytest.approx(0.80)


def test_accuracy_perfect_and_zero():
    preds = np.array([1, 2, 0])
    targets = np.array([1, 2, 0])
    assert accuracy(preds, targets) == 1.0

    targets_wrong = np.array([0, 0, 1])
    assert accuracy(preds, targets_wrong) == 0.0


def test_accuracy_shape_mismatch_raises_error():
    preds = np.array([1, 2, 3])
    targets = np.array([1, 2])
    with pytest.raises(ValueError, match="Shape mismatch"):
        accuracy(preds, targets)


def test_accuracy_empty():
    assert accuracy(np.array([]), np.array([])) == 0.0


def test_confusion_matrix_binary():
    # True classes:  [0, 0, 1, 1, 1]
    # Pred classes:  [0, 1, 1, 1, 0]
    # Expected CM:
    # row 0 (true 0): [1, 1] (1 pred 0, 1 pred 1)
    # row 1 (true 1): [1, 2] (1 pred 0, 2 pred 1)
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    cm = confusion_matrix(y_pred, y_true)
    expected = np.array([
        [1, 1],
        [1, 2],
    ], dtype=np.int64)

    np.testing.assert_array_equal(cm, expected)
    assert cm.shape == (2, 2)


def test_confusion_matrix_multiclass_and_logits():
    # 3 classes with 2D logits
    logits = Tensor([
        [3.0, 0.0, 0.0],  # pred 0, true 0 -> (0, 0)
        [0.0, 2.0, 1.0],  # pred 1, true 0 -> (0, 1)
        [0.0, 0.0, 4.0],  # pred 2, true 2 -> (2, 2)
        [0.0, 5.0, 1.0],  # pred 1, true 1 -> (1, 1)
    ])
    y_true = Tensor([0, 0, 2, 1])

    cm = confusion_matrix(logits, y_true, num_classes=3)
    expected = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.int64)

    np.testing.assert_array_equal(cm, expected)
    assert cm.shape == (3, 3)


def test_confusion_matrix_custom_num_classes():
    # Max class index in data is 1, but num_classes=4
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    cm = confusion_matrix(y_pred, y_true, num_classes=4)
    assert cm.shape == (4, 4)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert np.sum(cm) == 2


def test_confusion_matrix_validation_errors():
    with pytest.raises(ValueError, match="Shape mismatch"):
        confusion_matrix(np.array([0, 1]), np.array([0]))

    with pytest.raises(ValueError, match="num_classes must be positive"):
        confusion_matrix(np.array([0, 1]), np.array([0, 1]), num_classes=0)

    with pytest.raises(ValueError, match="out of bounds"):
        confusion_matrix(np.array([0, 2]), np.array([0, 1]), num_classes=2)
