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
from numpygrad.utils.gradcheck import gradcheck
import numpygrad.nn as nn
import numpygrad.optim as optim
import numpygrad.data as data
import numpygrad.utils as utils

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
    "gradcheck",
    "nn",
    "optim",
    "data",
    "utils",
]

