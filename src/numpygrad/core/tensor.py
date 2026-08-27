"""
NumPyGrad Core Tensor Engine.

Implements reverse-mode automatic differentiation over dynamic computational
directed acyclic graphs (DAGs) using pure NumPy.
"""

from __future__ import annotations
from typing import Optional, Set, Tuple, Union, Sequence, Callable, Any
import functools
import numpy as np


_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    """Returns True if autograd gradient tracking is currently enabled, False otherwise."""
    return _grad_enabled


def set_grad_enabled(mode: bool) -> None:
    """Enables or disables autograd gradient tracking globally."""
    global _grad_enabled
    _grad_enabled = bool(mode)


class no_grad:
    """
    Context-manager and function decorator that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call `Tensor.backward()`. It reduces memory consumption
    and computational overhead by preventing dynamic DAG edge construction and
    gradient graph retention.

    Examples
    --------
    >>> x = Tensor([1.0, 2.0], requires_grad=True)
    >>> with no_grad():
    ...     y = x * 2.0
    >>> y.requires_grad
    False

    >>> @no_grad()
    ... def predict(model, x):
    ...     return model(x)
    """

    def __init__(self) -> None:
        self.prev: bool = True

    def __enter__(self) -> no_grad:
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _grad_enabled
        _grad_enabled = self.prev

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper


class enable_grad:
    """
    Context-manager and function decorator that enables gradient calculation.

    Enables gradient tracking inside contexts where it was previously disabled
    via `no_grad()`.
    """

    def __init__(self) -> None:
        self.prev: bool = True

    def __enter__(self) -> enable_grad:
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _grad_enabled
        _grad_enabled = self.prev

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduces a gradient array `grad` back to `target_shape` by summing out
    dimensions that were broadcasted during the forward pass.

    Parameters
    ----------
    grad : np.ndarray
        The incoming gradient tensor from downstream operations.
    target_shape : Tuple[int, ...]
        The original shape of the operand before broadcasting.

    Returns
    -------
    np.ndarray
        The unbroadcasted gradient with shape equal to `target_shape`.
    """
    if grad.shape == target_shape:
        return grad

    # Handle 0-D scalar target shape
    if target_shape == () or target_shape == (0,):
        return np.sum(grad).astype(grad.dtype, copy=False)

    # 1. Sum across leading dimensions added via broadcasting
    lead_dims = grad.ndim - len(target_shape)
    if lead_dims > 0:
        grad = np.sum(grad, axis=tuple(range(lead_dims)))

    # 2. Sum across axes where target dimension was 1 but broadcasted to > 1
    dims_to_sum = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape))
        if t_dim == 1 and g_dim > 1
    )
    if dims_to_sum:
        grad = np.sum(grad, axis=dims_to_sum, keepdims=True)

    return grad.reshape(target_shape).astype(grad.dtype, copy=False)


class Tensor:
    """
    Dynamic DAG Node encapsulating a NumPy array with automatic differentiation.

    Attributes
    ----------
    data : np.ndarray
        The underlying multi-dimensional tensor data.
    grad : Optional[np.ndarray]
        Accumulated analytical gradient with identical shape to `data`.
    requires_grad : bool
        Whether this tensor requires gradient tracking.
    _prev : Set[Tensor]
        Parent tensors in the computation graph that produced this node.
    _op : str
        The operation label that created this node.
    _backward : Callable[[], None]
        Closure function executing analytical gradient propagation to parents.
    """

    __array_ufunc__ = None  # Defer numpy binary ufuncs to Tensor __r*__ methods

    def __init__(
        self,
        data: Union[int, float, list, tuple, np.ndarray, Sequence],
        requires_grad: bool = True,
        _prev: Tuple[Tensor, ...] = (),
        _op: str = "",
        dtype: Optional[np.dtype] = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            if dtype is not None:
                self.data: np.ndarray = data.astype(dtype, copy=False)
            else:
                self.data: np.ndarray = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        else:
            target_dtype = dtype if dtype is not None else np.float32
            self.data: np.ndarray = np.array(data, dtype=target_dtype)

        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: Set[Tensor] = set(_prev)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> Tensor:
        """Returns the transpose of this tensor."""
        return self.transpose()

    def zero_grad(self) -> None:
        """Resets the accumulated gradient buffer."""
        self.grad = None

    def numpy(self) -> np.ndarray:
        """Returns the raw NumPy ndarray."""
        return self.data

    def __repr__(self) -> str:
        grad_info = f", grad={self.grad.shape}" if self.grad is not None else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_info})"

    def __len__(self) -> int:
        """Returns the size of the first dimension of the tensor."""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return len(self.data)

    def __getitem__(self, item: Any) -> Tensor:
        """Extracts a slice or sub-tensor along dimensions."""
        indexed_data = self.data[item]
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                indexed_data,
                requires_grad=False,
                _prev=(),
                _op="getitem",
                dtype=self.dtype,
            )

        out = Tensor(
            indexed_data,
            requires_grad=True,
            _prev=(self,),
            _op="getitem",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = np.zeros_like(self.data, dtype=self.dtype)
                np.add.at(gx, item, out.grad)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def __hash__(self) -> int:
        return id(self)

    # -------------------------------------------------------------------------
    # Reverse-Mode Automatic Differentiation Engine
    # -------------------------------------------------------------------------

    def backward(self, gradient: Optional[Union[np.ndarray, Tensor, float]] = None) -> None:
        """
        Computes the gradient of this tensor with respect to graph leaves via
        topological reverse-mode automatic differentiation.

        Parameters
        ----------
        gradient : Optional[Union[np.ndarray, Tensor, float]], optional
            Incoming seed gradient dL/d(self). Defaults to 1.0 (ones array).
        """
        if not self.requires_grad:
            return

        if gradient is None:
            self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        elif isinstance(gradient, Tensor):
            self.grad = gradient.data.astype(self.data.dtype, copy=False)
        elif isinstance(gradient, np.ndarray):
            self.grad = gradient.astype(self.data.dtype, copy=False)
        else:
            self.grad = np.full_like(self.data, fill_value=gradient, dtype=self.data.dtype)

        # Build topological sort DAG
        topo: list[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Propagate gradients in reverse topological order
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()

    # -------------------------------------------------------------------------
    # Mathematical Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        """
        Element-wise addition: z = x + y
        Analytical Gradients: dz/dx = 1, dz/dy = 1
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        res_dtype = np.result_type(self.dtype, other.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        if not req_grad:
            return Tensor(
                self.data + other.data,
                requires_grad=False,
                _prev=(),
                _op="+",
                dtype=res_dtype,
            )

        out = Tensor(
            self.data + other.data,
            requires_grad=True,
            _prev=(self, other),
            _op="+",
            dtype=res_dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = _unbroadcast(out.grad, self.shape)
                self.grad = gx if self.grad is None else self.grad + gx
            if other.requires_grad:
                gy = _unbroadcast(out.grad, other.shape)
                other.grad = gy if other.grad is None else other.grad + gy

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return self.__add__(other)

    def __neg__(self) -> Tensor:
        """
        Unary negation: z = -x
        Analytical Gradient: dz/dx = -1
        """
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                -self.data,
                requires_grad=False,
                _prev=(),
                _op="neg",
                dtype=self.dtype,
            )

        out = Tensor(
            -self.data,
            requires_grad=True,
            _prev=(self,),
            _op="neg",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = -out.grad
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def __sub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        """
        Element-wise subtraction: z = x - y
        Analytical Gradients: dz/dx = 1, dz/dy = -1
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        res_dtype = np.result_type(self.dtype, other.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        if not req_grad:
            return Tensor(
                self.data - other.data,
                requires_grad=False,
                _prev=(),
                _op="-",
                dtype=res_dtype,
            )

        out = Tensor(
            self.data - other.data,
            requires_grad=True,
            _prev=(self, other),
            _op="-",
            dtype=res_dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = _unbroadcast(out.grad, self.shape)
                self.grad = gx if self.grad is None else self.grad + gx
            if other.requires_grad:
                gy = _unbroadcast(-out.grad, other.shape)
                other.grad = gy if other.grad is None else other.grad + gy

        out._backward = _backward
        return out

    def __rsub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        return other.__sub__(self)

    def __mul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        """
        Element-wise multiplication (Hadamard product): z = x * y
        Analytical Gradients: dz/dx = y, dz/dy = x
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        res_dtype = np.result_type(self.dtype, other.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        if not req_grad:
            return Tensor(
                self.data * other.data,
                requires_grad=False,
                _prev=(),
                _op="*",
                dtype=res_dtype,
            )

        out = Tensor(
            self.data * other.data,
            requires_grad=True,
            _prev=(self, other),
            _op="*",
            dtype=res_dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = _unbroadcast(out.grad * other.data, self.shape)
                self.grad = gx if self.grad is None else self.grad + gx
            if other.requires_grad:
                gy = _unbroadcast(out.grad * self.data, other.shape)
                other.grad = gy if other.grad is None else other.grad + gy

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        """
        Element-wise division: z = x / y
        Analytical Gradients: dz/dx = 1/y, dz/dy = -x / (y^2)
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        res_dtype = np.result_type(self.dtype, other.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        if not req_grad:
            return Tensor(
                self.data / other.data,
                requires_grad=False,
                _prev=(),
                _op="/",
                dtype=res_dtype,
            )

        out = Tensor(
            self.data / other.data,
            requires_grad=True,
            _prev=(self, other),
            _op="/",
            dtype=res_dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = _unbroadcast(out.grad / other.data, self.shape)
                self.grad = gx if self.grad is None else self.grad + gx
            if other.requires_grad:
                gy = _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                other.grad = gy if other.grad is None else other.grad + gy

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        return other.__truediv__(self)

    def __matmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Matrix multiplication: Z = X @ Y
        Analytical Gradients:
            dZ/dX = dL/dZ @ Y^T
            dZ/dY = X^T @ dL/dZ
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        res_dtype = np.result_type(self.dtype, other.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        if not req_grad:
            return Tensor(
                self.data @ other.data,
                requires_grad=False,
                _prev=(),
                _op="@",
                dtype=res_dtype,
            )

        out = Tensor(
            self.data @ other.data,
            requires_grad=True,
            _prev=(self, other),
            _op="@",
            dtype=res_dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                other_t = np.swapaxes(other.data, -1, -2) if other.ndim >= 2 else other.data.T
                gx = np.matmul(out.grad, other_t)
                gx = _unbroadcast(gx, self.shape)
                self.grad = gx if self.grad is None else self.grad + gx
            if other.requires_grad:
                self_t = np.swapaxes(self.data, -1, -2) if self.ndim >= 2 else self.data.T
                gy = np.matmul(self_t, out.grad)
                gy = _unbroadcast(gy, other.shape)
                other.grad = gy if other.grad is None else other.grad + gy

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=self.dtype)
        return other.__matmul__(self)

    def __pow__(self, exponent: Union[int, float, Tensor]) -> Tensor:
        """
        Element-wise power: z = x^p
        Analytical Gradient (w.r.t x): dz/dx = p * x^(p - 1)
        Analytical Gradient (w.r.t p): dz/dp = (x^p) * ln(x)
        """
        if isinstance(exponent, Tensor):
            res_dtype = np.result_type(self.dtype, exponent.dtype)
            req_grad = is_grad_enabled() and (self.requires_grad or exponent.requires_grad)

            if not req_grad:
                return Tensor(
                    self.data ** exponent.data,
                    requires_grad=False,
                    _prev=(),
                    _op="**",
                    dtype=res_dtype,
                )

            out = Tensor(
                self.data ** exponent.data,
                requires_grad=True,
                _prev=(self, exponent),
                _op="**",
                dtype=res_dtype,
            )

            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    gx = out.grad * (exponent.data * (self.data ** (exponent.data - 1.0)))
                    gx = _unbroadcast(gx, self.shape)
                    self.grad = gx if self.grad is None else self.grad + gx
                if exponent.requires_grad:
                    safe_x = np.where(self.data > 0, self.data, 1.0)
                    gp = out.grad * out.data * np.log(safe_x)
                    gp = _unbroadcast(gp, exponent.shape)
                    exponent.grad = gp if exponent.grad is None else exponent.grad + gp

            out._backward = _backward
            return out
        else:
            p = float(exponent)
            req_grad = is_grad_enabled() and self.requires_grad

            if not req_grad:
                return Tensor(
                    self.data ** p,
                    requires_grad=False,
                    _prev=(),
                    _op=f"**{p}",
                    dtype=self.dtype,
                )

            out = Tensor(
                self.data ** p,
                requires_grad=True,
                _prev=(self,),
                _op=f"**{p}",
                dtype=self.dtype,
            )

            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    gx = out.grad * (p * (self.data ** (p - 1.0)))
                    gx = _unbroadcast(gx, self.shape)
                    self.grad = gx if self.grad is None else self.grad + gx

            out._backward = _backward
            return out

    # -------------------------------------------------------------------------
    # Activation Functions
    # -------------------------------------------------------------------------

    def relu(self) -> Tensor:
        """
        Rectified Linear Unit: z = max(0, x)
        Analytical Derivative: dz/dx = 1 if x > 0 else 0
        """
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                np.maximum(0.0, self.data),
                requires_grad=False,
                _prev=(),
                _op="relu",
                dtype=self.dtype,
            )

        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=True,
            _prev=(self,),
            _op="relu",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = out.grad * (self.data > 0).astype(self.dtype)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        """
        Numerically stable Sigmoid: sigma(x) = 1 / (1 + exp(-x))
        Analytical Derivative: d sigma(x) / dx = sigma(x) * (1 - sigma(x))
        """
        pos_mask = self.data >= 0
        neg_mask = ~pos_mask
        sig = np.empty_like(self.data, dtype=self.dtype)

        sig[pos_mask] = 1.0 / (1.0 + np.exp(-self.data[pos_mask]))
        exp_x = np.exp(self.data[neg_mask])
        sig[neg_mask] = exp_x / (1.0 + exp_x)

        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                sig,
                requires_grad=False,
                _prev=(),
                _op="sigmoid",
                dtype=self.dtype,
            )

        out = Tensor(
            sig,
            requires_grad=True,
            _prev=(self,),
            _op="sigmoid",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = out.grad * out.data * (1.0 - out.data)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """
        Hyperbolic Tangent: z = tanh(x)
        Analytical Derivative: dz/dx = 1 - tanh(x)^2
        """
        t_data = np.tanh(self.data).astype(self.dtype)
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                t_data,
                requires_grad=False,
                _prev=(),
                _op="tanh",
                dtype=self.dtype,
            )

        out = Tensor(
            t_data,
            requires_grad=True,
            _prev=(self,),
            _op="tanh",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = out.grad * (1.0 - out.data ** 2)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reduction & Shape Operations
    # -------------------------------------------------------------------------

    def sum(
        self,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        """
        Sum reduction along specified axis/axes.
        Analytical Gradient: Broadcasts incoming gradient along summed axes.
        """
        s_data = np.sum(self.data, axis=axis, keepdims=keepdims).astype(self.dtype)
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                s_data,
                requires_grad=False,
                _prev=(),
                _op="sum",
                dtype=self.dtype,
            )

        out = Tensor(
            s_data,
            requires_grad=True,
            _prev=(self,),
            _op="sum",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if keepdims or axis is None:
                    gx = np.broadcast_to(out.grad, self.shape)
                else:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                    expanded_shape = list(self.shape)
                    for a in axes:
                        expanded_shape[a] = 1
                    grad_reshaped = out.grad.reshape(expanded_shape)
                    gx = np.broadcast_to(grad_reshaped, self.shape)
                gx = gx.astype(self.dtype, copy=False)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def mean(
        self,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        """
        Mean reduction along specified axis/axes.
        Analytical Gradient: (1 / N) * Broadcast(incoming gradient).
        """
        m_data = np.mean(self.data, axis=axis, keepdims=keepdims).astype(self.dtype)
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                m_data,
                requires_grad=False,
                _prev=(),
                _op="mean",
                dtype=self.dtype,
            )

        out = Tensor(
            m_data,
            requires_grad=True,
            _prev=(self,),
            _op="mean",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if axis is None:
                    numel = float(self.data.size)
                    gx = np.broadcast_to(out.grad, self.shape) / numel
                else:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                    numel = float(np.prod([self.shape[a] for a in axes]))
                    if keepdims:
                        gx = np.broadcast_to(out.grad, self.shape) / numel
                    else:
                        expanded_shape = list(self.shape)
                        for a in axes:
                            expanded_shape[a] = 1
                        grad_reshaped = out.grad.reshape(expanded_shape)
                        gx = np.broadcast_to(grad_reshaped, self.shape) / numel
                gx = gx.astype(self.dtype, copy=False)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def reshape(self, *shape: Union[int, Sequence[int]]) -> Tensor:
        """Reshapes tensor into a new shape while maintaining DAG history."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)  # type: ignore

        r_data = self.data.reshape(target_shape).astype(self.dtype)
        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                r_data,
                requires_grad=False,
                _prev=(),
                _op="reshape",
                dtype=self.dtype,
            )

        out = Tensor(
            r_data,
            requires_grad=True,
            _prev=(self,),
            _op="reshape",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                gx = out.grad.reshape(self.shape).astype(self.dtype, copy=False)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

    def transpose(self, *axes: Union[int, Sequence[int]]) -> Tensor:
        """Permutes the dimensions of the tensor."""
        if not axes:
            t_data = self.data.T.astype(self.dtype)
            order = None
        elif len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            order = tuple(axes[0])
            t_data = self.data.transpose(order).astype(self.dtype)
        else:
            order = tuple(axes)  # type: ignore
            t_data = self.data.transpose(order).astype(self.dtype)

        req_grad = is_grad_enabled() and self.requires_grad

        if not req_grad:
            return Tensor(
                t_data,
                requires_grad=False,
                _prev=(),
                _op="transpose",
                dtype=self.dtype,
            )

        out = Tensor(
            t_data,
            requires_grad=True,
            _prev=(self,),
            _op="transpose",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if order is None:
                    gx = out.grad.T
                else:
                    inv_axes = tuple(np.argsort(order))
                    gx = out.grad.transpose(inv_axes)
                gx = gx.astype(self.dtype, copy=False)
                self.grad = gx if self.grad is None else self.grad + gx

        out._backward = _backward
        return out

