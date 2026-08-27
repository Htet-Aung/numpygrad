"""
Core neural network layers: Linear, Sequential, Dropout, BatchNorm1d, and Activations.
"""

from __future__ import annotations
from typing import Sequence, Union, Optional, List, Iterator
import numpy as np
from numpygrad.core.tensor import Tensor
from numpygrad.nn.module import Module, Parameter, kaiming_uniform_


class Linear(Module):
    """
    Applies an affine linear transformation to the incoming data:
        y = x @ W + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Initialize weights with Kaiming uniform
        weight_data = np.empty((in_features, out_features), dtype=np.float32)
        self.weight: Parameter = Parameter(weight_data)
        kaiming_uniform_(self.weight)

        if bias:
            fan_in = in_features
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            bias_data = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
            self.bias: Optional[Parameter] = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Sequential(Module):
    """
    A sequential container cascading modules in the order they were added.
    """

    def __init__(self, *layers: Union[Module, Sequence[Module]]) -> None:
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layer_list = list(layers[0])
        else:
            layer_list = list(layers)  # type: ignore

        self._layer_list: List[Module] = []
        for idx, layer in enumerate(layer_list):
            if not isinstance(layer, Module):
                raise TypeError(f"Sequential argument must be Module, got {type(layer)}")
            self._modules[str(idx)] = layer
            self._layer_list.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layer_list:
            x = layer(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self._layer_list[idx]

    def __len__(self) -> int:
        return len(self._layer_list)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._layer_list)


class Dropout(Module):
    """
    During training, randomly zeroes some elements with probability `p` using
    samples from a Bernoulli distribution. Employs inverted dropout scaling.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p: float = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_p = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep_p).astype(x.dtype) / keep_p
        mask_t = Tensor(mask, requires_grad=False, dtype=x.dtype)
        return x * mask_t

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm1d(Module):
    """
    Applies Batch Normalization over a 2D or 3D tensor:
        y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = float(eps)
        self.momentum: float = float(momentum)
        self.affine: bool = affine

        if affine:
            self.weight: Optional[Parameter] = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias: Optional[Parameter] = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

        self.running_mean: np.ndarray = np.zeros(num_features, dtype=np.float32)
        self.running_var: np.ndarray = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Batch statistics along axis 0
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff ** 2.0).mean(axis=0, keepdims=True)

            # Update running statistics
            batch_mean = mean.data.reshape(-1)
            batch_var = var.data.reshape(-1)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            std_inv = (var + self.eps) ** -0.5
            x_norm = diff * std_inv
        else:
            # Evaluation mode using stored running statistics
            mean_t = Tensor(self.running_mean, requires_grad=False, dtype=x.dtype)
            var_t = Tensor(self.running_var, requires_grad=False, dtype=x.dtype)
            x_norm = (x - mean_t) / ((var_t + self.eps) ** 0.5)

        if self.affine and self.weight is not None and self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm

    def __repr__(self) -> str:
        return (
            f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )


class ReLU(Module):
    """Rectified Linear Unit activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class GELU(Module):
    """Gaussian Error Linear Unit activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = np.sqrt(2.0 / np.pi)
        inner = coeff * (x + 0.044715 * (x ** 3.0))
        return 0.5 * x * (1.0 + inner.tanh())

    def __repr__(self) -> str:
        return "GELU()"


class Flatten(Module):
    """
    Flattens a contiguous range of dimensions into a single dimension.
    By default, flattens from dimension 1 to -1 (preserving batch dimension 0).
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim: int = int(start_dim)
        self.end_dim: int = int(end_dim)

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        x_t = x if isinstance(x, Tensor) else Tensor(x)
        shape = x_t.shape
        ndim = len(shape)

        if ndim == 0:
            return x_t.reshape(1)

        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim

        if start < 0 or start >= ndim:
            raise IndexError(f"start_dim {self.start_dim} out of range for tensor with {ndim} dimensions")
        if end < 0 or end >= ndim:
            raise IndexError(f"end_dim {self.end_dim} out of range for tensor with {ndim} dimensions")
        if start > end:
            raise ValueError(f"start_dim ({self.start_dim}) cannot be greater than end_dim ({self.end_dim})")

        flattened_size = 1
        for d in range(start, end + 1):
            flattened_size *= shape[d]

        new_shape = shape[:start] + (flattened_size,) + shape[end + 1:]
        return x_t.reshape(new_shape)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"

