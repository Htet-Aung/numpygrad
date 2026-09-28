"""
Evaluation metrics package for NumPyGrad.
"""

from numpygrad.metrics.classification import accuracy, confusion_matrix
from numpygrad.metrics.evaluation import classification_metrics, stratified_indices

__all__ = [
    "accuracy",
    "confusion_matrix",
    "classification_metrics",
    "stratified_indices",
]
