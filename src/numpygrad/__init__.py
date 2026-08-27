"""
NumPyGrad: A pure NumPy automatic differentiation & deep learning library.
"""

from numpygrad.core.tensor import (
    Tensor,
    _unbroadcast,
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
)
from numpygrad.data.dataset import Dataset, TensorDataset
from numpygrad.data.dataloader import DataLoader
from numpygrad.serialization import save_model, load_model
from numpygrad.metrics.classification import accuracy, confusion_matrix
from numpygrad.utils.gradcheck import gradcheck
import numpygrad.nn as nn
import numpygrad.optim as optim
import numpygrad.data as data
import numpygrad.utils as utils
import numpygrad.metrics as metrics

__all__ = [
    "Tensor",
    "_unbroadcast",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "save_model",
    "load_model",
    "accuracy",
    "confusion_matrix",
    "gradcheck",
    "nn",
    "optim",
    "data",
    "utils",
    "metrics",
]


