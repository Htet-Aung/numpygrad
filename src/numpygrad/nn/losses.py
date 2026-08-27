"""
Loss functions: MSELoss, CrossEntropyLoss, and BCEWithLogitsLoss.
"""

from __future__ import annotations
from typing import Optional, Union
import numpy as np
from numpygrad.core.tensor import Tensor, _unbroadcast, is_grad_enabled
from numpygrad.nn.module import Module


class MSELoss(Module):
    """
    Mean Squared Error loss criterion:
        L(x, y) = (x - y)^2
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', choose 'mean', 'sum', or 'none'")
        self.reduction: str = reduction

    def forward(self, pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False, dtype=pred.dtype)
        diff = pred - target
        sq_err = diff ** 2.0

        if self.reduction == "mean":
            return sq_err.mean()
        elif self.reduction == "sum":
            return sq_err.sum()
        else:
            return sq_err

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """
    Calculates the cross-entropy loss between input logits and target labels.
    Combines LogSoftmax and NLLLoss in a single numerically stable formulation
    using the Log-Sum-Exp trick:
        L_i = LSE(z_i) - z_{i, y_i}
        LSE(z_i) = max(z_i) + log(sum(exp(z_i - max(z_i))))
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', choose 'mean', 'sum', or 'none'")
        self.reduction: str = reduction

    def forward(self, logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        target_arr = target.data if isinstance(target, Tensor) else np.asarray(target)
        logits_data = logits.data

        # Ensure 2D shape (N, C)
        if logits_data.ndim == 1:
            logits_data_2d = logits_data.reshape(1, -1)
            target_arr_2d = target_arr.reshape(1, -1) if target_arr.ndim > 0 else target_arr.reshape(1)
            is_1d = True
        else:
            logits_data_2d = logits_data
            target_arr_2d = target_arr
            is_1d = False

        N, C = logits_data_2d.shape

        # Numerically stable Log-Sum-Exp computation
        max_logits = np.max(logits_data_2d, axis=-1, keepdims=True)
        shifted_logits = logits_data_2d - max_logits
        exp_shifted = np.exp(shifted_logits)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        log_sum_exp = max_logits + np.log(sum_exp)
        probs = exp_shifted / sum_exp

        # Convert target to one-hot if provided as integer class indices
        if target_arr_2d.ndim == 1 or (target_arr_2d.ndim == 2 and target_arr_2d.shape[1] == 1):
            target_indices = target_arr_2d.astype(int).flatten()
            target_one_hot = np.zeros((N, C), dtype=logits_data.dtype)
            target_one_hot[np.arange(N), target_indices] = 1.0
            # Target logits: z_{i, y_i}
            target_logits = logits_data_2d[np.arange(N), target_indices]
            loss_per_sample = (log_sum_exp.squeeze(-1) - target_logits).astype(logits.dtype)
        else:
            target_one_hot = target_arr_2d.astype(logits_data.dtype)
            target_logits = np.sum(logits_data_2d * target_one_hot, axis=-1)
            loss_per_sample = (log_sum_exp.squeeze(-1) - target_logits).astype(logits.dtype)

        if is_1d and loss_per_sample.ndim > 0:
            loss_per_sample = loss_per_sample.squeeze(0)

        # Apply reduction
        if self.reduction == "mean":
            loss_data = np.mean(loss_per_sample).astype(logits.dtype)
        elif self.reduction == "sum":
            loss_data = np.sum(loss_per_sample).astype(logits.dtype)
        else:
            loss_data = loss_per_sample

        req_grad = is_grad_enabled() and logits.requires_grad

        if not req_grad:
            return Tensor(
                loss_data,
                requires_grad=False,
                _prev=(),
                _op="cross_entropy",
                dtype=logits.dtype,
            )

        out = Tensor(
            loss_data,
            requires_grad=True,
            _prev=(logits,),
            _op="cross_entropy",
            dtype=logits.dtype,
        )

        def _backward() -> None:
            if out.grad is None or not logits.requires_grad:
                return

            grad_logits = (probs - target_one_hot).astype(logits.dtype)

            if self.reduction == "mean":
                scale = out.grad / float(N)
                grad_logits = grad_logits * scale
            elif self.reduction == "sum":
                grad_logits = grad_logits * out.grad
            else:
                grad_expanded = out.grad.reshape(N, 1) if out.grad.ndim == 1 else out.grad
                grad_logits = grad_logits * grad_expanded

            if is_1d:
                grad_logits = grad_logits.reshape(logits.shape)

            logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits loss:
        L_i = max(x_i, 0) - x_i * y_i + log(1 + exp(-|x_i|))
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', choose 'mean', 'sum', or 'none'")
        self.reduction: str = reduction

    def forward(self, logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False, dtype=logits.dtype)
        x = logits.data
        y = target.data

        # Numerically stable softplus: max(x, 0) - x * y + log(1 + exp(-|x|))
        loss_val = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        loss_val = loss_val.astype(logits.dtype)

        if self.reduction == "mean":
            loss_data = np.mean(loss_val).astype(logits.dtype)
        elif self.reduction == "sum":
            loss_data = np.sum(loss_val).astype(logits.dtype)
        else:
            loss_data = loss_val

        req_grad = is_grad_enabled() and (logits.requires_grad or target.requires_grad)

        if not req_grad:
            return Tensor(
                loss_data,
                requires_grad=False,
                _prev=(),
                _op="bce_with_logits",
                dtype=logits.dtype,
            )

        out = Tensor(
            loss_data,
            requires_grad=True,
            _prev=(logits, target),
            _op="bce_with_logits",
            dtype=logits.dtype,
        )

        def _backward() -> None:
            if out.grad is None or not logits.requires_grad:
                return

            # Sigmoid(x) - y
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            sig = np.empty_like(x)
            sig[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
            exp_x = np.exp(x[neg_mask])
            sig[neg_mask] = exp_x / (1.0 + exp_x)

            grad_logits = (sig - y).astype(logits.dtype)

            if self.reduction == "mean":
                grad_logits = grad_logits * (out.grad / float(logits.data.size))
            elif self.reduction == "sum":
                grad_logits = grad_logits * out.grad
            else:
                grad_logits = grad_logits * out.grad

            logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"

