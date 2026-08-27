"""
NumPyGrad: A pure NumPy automatic differentiation & deep learning library.
"""

__version__ = "1.0.0"

from numpygrad.core.tensor import (
    Tensor,
    _unbroadcast,
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
    concat,
    cat,
)
from numpygrad.data.dataset import Dataset, TensorDataset
from numpygrad.data.dataloader import DataLoader
from numpygrad.serialization import save_model, load_model
from numpygrad.metrics.classification import accuracy, confusion_matrix
from numpygrad.nn.layers import Flatten
from numpygrad.nn.convolution import Conv2D, MaxPool2D
from numpygrad.nn.summary import summary
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
    "concat",
    "cat",
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "save_model",
    "load_model",
    "accuracy",
    "confusion_matrix",
    "Flatten",
    "Conv2D",
    "MaxPool2D",
    "summary",
    "gradcheck",
    "nn",
    "optim",
    "data",
    "utils",
    "metrics",
]


