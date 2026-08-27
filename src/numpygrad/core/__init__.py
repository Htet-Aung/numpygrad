"""
Core autograd and computational DAG engine.
"""

from numpygrad.core.tensor import Tensor, _unbroadcast

__all__ = ["Tensor", "_unbroadcast"]
