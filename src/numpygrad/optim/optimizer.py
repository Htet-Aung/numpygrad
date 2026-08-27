"""
Base Optimizer class for NumPyGrad.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Any
from numpygrad.nn.module import Parameter


class Optimizer:
    """
    Base class for all first-order gradient-based optimizers.
    """

    def __init__(self, params: Iterable[Parameter], defaults: Dict[str, Any]) -> None:
        self.params: List[Parameter] = list(params)
        self.defaults: Dict[str, Any] = defaults
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self) -> None:
        """Resets the gradients of all managed parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Performs a single parameter optimization step."""
        raise NotImplementedError("Subclasses of Optimizer must implement step()")
