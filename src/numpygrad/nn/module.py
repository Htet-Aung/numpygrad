"""
Neural network base abstractions: Module and Parameter.
"""

from __future__ import annotations
from typing import List, Iterator, Dict, Any, Optional, Union, Sequence
import numpy as np
from numpygrad.core.tensor import Tensor


class Parameter(Tensor):
    """
    A trainable neural network parameter wrapping a dynamic Tensor.

    Parameters are automatically tracked by `Module.parameters()`.
    """

    def __init__(
        self,
        data: Union[int, float, list, tuple, np.ndarray, Sequence],
        requires_grad: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        super().__init__(data=data, requires_grad=requires_grad, dtype=dtype)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fills the input Tensor with values according to the Xavier/Glorot uniform distribution:
        U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    """
    if tensor.ndim < 2:
        fan_in = fan_out = tensor.size
    else:
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]

    std = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data = np.random.uniform(-std, std, size=tensor.shape).astype(tensor.dtype)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fills the input Tensor with values according to the Xavier/Glorot normal distribution:
        N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out))
    """
    if tensor.ndim < 2:
        fan_in = fan_out = tensor.size
    else:
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = np.random.normal(0.0, std, size=tensor.shape).astype(tensor.dtype)
    return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0.0) -> Tensor:
    """
    Fills the input Tensor with values according to the He/Kaiming uniform distribution:
        U(-bound, bound) where bound = sqrt(6 / ((1 + a^2) * fan_in))
    """
    fan_in = tensor.shape[0] if tensor.ndim >= 2 else tensor.size
    gain = np.sqrt(2.0 / (1.0 + a ** 2))
    bound = gain * np.sqrt(3.0 / fan_in)
    tensor.data = np.random.uniform(-bound, bound, size=tensor.shape).astype(tensor.dtype)
    return tensor


def kaiming_normal_(tensor: Tensor, a: float = 0.0) -> Tensor:
    """
    Fills the input Tensor with values according to the He/Kaiming normal distribution:
        N(0, std^2) where std = sqrt(2 / ((1 + a^2) * fan_in))
    """
    fan_in = tensor.shape[0] if tensor.ndim >= 2 else tensor.size
    gain = np.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / np.sqrt(fan_in)
    tensor.data = np.random.normal(0.0, std, size=tensor.shape).astype(tensor.dtype)
    return tensor


class Module:
    """
    Base class for all neural network modules.

    Supports automatic parameter registration, hierarchical submodule discovery,
    training/evaluation mode switching, and gradient resetting.
    """

    def __init__(self) -> None:
        self.training: bool = True
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """
        Recursively collects all unique trainable parameters in this module
        and all its child submodules.
        """
        params: List[Parameter] = []
        visited_ids = set()

        for p in self._parameters.values():
            if id(p) not in visited_ids:
                visited_ids.add(id(p))
                params.append(p)

        for m in self._modules.values():
            for p in m.parameters():
                if id(p) not in visited_ids:
                    visited_ids.add(id(p))
                    params.append(p)

        return params

    def modules(self) -> List[Module]:
        """Returns a list of all submodules in this module hierarchy."""
        mods = [self]
        for m in self._modules.values():
            mods.extend(m.modules())
        return mods

    def zero_grad(self) -> None:
        """Resets gradients of all parameters to None."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Sets the module and all submodules into training mode."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        """Sets the module and all submodules into evaluation (inference) mode."""
        return self.train(False)

    def state_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        """Returns a dictionary containing a whole state of the module."""
        from numpygrad.serialization import get_model_state_dict
        return get_model_state_dict(self, prefix=prefix)

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True) -> None:
        """Copies parameters and buffers from `state_dict` into this module and its descendants."""
        from numpygrad.serialization import load_model_state_dict
        load_model_state_dict(self, state=state_dict, strict=strict)

    def save(self, filepath: str) -> None:
        """Saves the module architecture and parameter weights to a `.ng` container file."""
        from numpygrad.serialization import save_model
        save_model(self, filepath)

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None, verbose: bool = True) -> str:
        """Generates a structured ASCII summary table of this module."""
        from numpygrad.nn.summary import summary as model_summary
        return model_summary(self, input_shape=input_shape, verbose=verbose)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Defines the computation performed at every call."""
        raise NotImplementedError("Subclasses of Module must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatches to the forward computation pass."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = "  " + "\n  ".join(mod_str.split("\n"))
            child_lines.append(f"({key}): {mod_str}")
        main_str = self.__class__.__name__ + "("
        if child_lines:
            main_str += "\n" + "\n".join(child_lines) + "\n"
        main_str += ")"
        return main_str
