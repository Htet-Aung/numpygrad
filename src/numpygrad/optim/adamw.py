"""
AdamW optimizer with decoupled weight decay and bias-corrected moment estimates.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from numpygrad.nn.module import Parameter
from numpygrad.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW algorithm with decoupled weight decay as proposed in
    'Decoupled Weight Decay Regularization' (Loshchilov & Hutter, 2019).

    Parameters
    ----------
    params : Iterable[Parameter]
        Trainable parameters to optimize.
    lr : float, default=1e-3
        Learning rate.
    betas : Tuple[float, float], default=(0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
    eps : float, default=1e-8
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default=1e-2
        Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single AdamW parameter optimization step."""
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            param_id = id(p)

            # Initialize state if not present
            if param_id not in self.state:
                self.state[param_id] = {
                    "step": 0,
                    "exp_avg": np.zeros_like(p.data, dtype=np.float32),
                    "exp_avg_sq": np.zeros_like(p.data, dtype=np.float32),
                }

            state = self.state[param_id]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # 1. Decoupled Weight Decay
            if weight_decay != 0.0:
                p.data -= (lr * weight_decay * p.data).astype(p.dtype, copy=False)

            # 2. Update biased first and second moment estimates
            exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * (grad ** 2)

            # 3. Bias corrections
            bias_correction1 = 1.0 - (beta1 ** step)
            bias_correction2 = 1.0 - (beta2 ** step)

            # 4. Compute parameter step
            hat_m = exp_avg / bias_correction1
            hat_v = exp_avg_sq / bias_correction2
            denom = np.sqrt(hat_v) + eps
            step_update = (lr * hat_m / denom).astype(p.dtype, copy=False)

            p.data -= step_update
