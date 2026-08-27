"""
Stochastic Gradient Descent (SGD) optimizer with Polyak momentum and weight decay.
"""

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
from numpygrad.nn.module import Parameter
from numpygrad.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Implements Stochastic Gradient Descent optimizer with optional momentum,
    dampening, weight decay, and Nesterov accelerated gradient.

    Parameters
    ----------
    params : Iterable[Parameter]
        Trainable parameters to optimize.
    lr : float, default=1e-3
        Learning rate.
    momentum : float, default=0.0
        Momentum factor (Polyak velocity buffer).
    dampening : float, default=0.0
        Dampening for momentum.
    weight_decay : float, default=0.0
        L2 regularization penalty factor.
    nesterov : bool, default=False
        Enables Nesterov momentum formulation.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a positive momentum and zero dampening")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single SGD optimization step."""
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        weight_decay = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        for p in self.params:
            if p.grad is None:
                continue

            d_p = p.grad.copy()

            # L2 Regularization / Weight Decay
            if weight_decay != 0.0:
                d_p += weight_decay * p.data

            # Momentum buffer update
            if momentum != 0.0:
                param_id = id(p)
                if param_id not in self.state:
                    self.state[param_id] = {"momentum_buffer": d_p.copy()}
                    buf = self.state[param_id]["momentum_buffer"]
                else:
                    buf = self.state[param_id]["momentum_buffer"]
                    buf[:] = momentum * buf + (1.0 - dampening) * d_p

                if nesterov:
                    d_p = d_p + momentum * buf
                else:
                    d_p = buf

            # In-place parameter weight update
            p.data -= (lr * d_p).astype(p.dtype, copy=False)
