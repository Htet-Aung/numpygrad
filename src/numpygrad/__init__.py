"""
NumPyGrad: A pure NumPy automatic differentiation & deep learning library.
"""

from numpygrad.core.tensor import Tensor, _unbroadcast
from numpygrad.utils.gradcheck import gradcheck
import numpygrad.nn as nn
import numpygrad.optim as optim

__all__ = ["Tensor", "_unbroadcast", "gradcheck", "nn", "optim"]
