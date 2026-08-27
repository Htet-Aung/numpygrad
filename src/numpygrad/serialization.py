"""
NumPyGrad Model Serialization and Deserialization Engine.

Provides utilities for saving and loading neural network models as standalone
`.ng` container archives containing `architecture.json` and `weights.npz`.
"""

from __future__ import annotations
import os
import io
import json
import zipfile
from typing import Dict, Any, Union, List, Optional, Type
import numpy as np

from numpygrad.nn.module import Module, Parameter
import numpygrad.nn.layers as layers_module
import numpygrad.nn.convolution as conv_module


# Registry of known serializable layer classes
_LAYER_REGISTRY: Dict[str, Type[Module]] = {
    "Linear": layers_module.Linear,
    "Sequential": layers_module.Sequential,
    "Dropout": layers_module.Dropout,
    "BatchNorm1d": layers_module.BatchNorm1d,
    "ReLU": layers_module.ReLU,
    "Sigmoid": layers_module.Sigmoid,
    "Tanh": layers_module.Tanh,
    "GELU": layers_module.GELU,
    "Flatten": layers_module.Flatten,
    "Conv2D": conv_module.Conv2D,
    "MaxPool2D": conv_module.MaxPool2D,
}


def register_layer(name: str, cls: Type[Module]) -> None:
    """
    Registers a custom Module class with the serialization engine.

    Parameters
    ----------
    name : str
        The class name / identifier.
    cls : Type[Module]
        The Module class.
    """
    _LAYER_REGISTRY[name] = cls


def get_layer_config(layer: Module) -> Dict[str, Any]:
    """
    Extracts the serializable configuration dictionary for a given layer or container.
    """
    layer_type = layer.__class__.__name__

    if isinstance(layer, layers_module.Linear):
        return {
            "type": "Linear",
            "config": {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            },
        }
    elif isinstance(layer, layers_module.Dropout):
        return {
            "type": "Dropout",
            "config": {
                "p": layer.p,
            },
        }
    elif isinstance(layer, layers_module.BatchNorm1d):
        return {
            "type": "BatchNorm1d",
            "config": {
                "num_features": layer.num_features,
                "eps": layer.eps,
                "momentum": layer.momentum,
                "affine": layer.affine,
            },
        }
    elif isinstance(layer, layers_module.Flatten):
        return {
            "type": "Flatten",
            "config": {
                "start_dim": layer.start_dim,
                "end_dim": layer.end_dim,
            },
        }
    elif isinstance(layer, conv_module.Conv2D):
        return {
            "type": "Conv2D",
            "config": {
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride),
                "padding": list(layer.padding),
                "bias": layer.bias is not None,
            },
        }
    elif isinstance(layer, conv_module.MaxPool2D):
        return {
            "type": "MaxPool2D",
            "config": {
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride),
                "padding": list(layer.padding),
            },
        }
    elif isinstance(layer, (layers_module.ReLU, layers_module.Sigmoid, layers_module.Tanh, layers_module.GELU)):
        return {
            "type": layer_type,
            "config": {},
        }
    elif isinstance(layer, layers_module.Sequential):
        sub_configs = [get_layer_config(sub_layer) for sub_layer in layer]
        return {
            "type": "Sequential",
            "layers": sub_configs,
        }
    elif hasattr(layer, "_modules") and layer._modules:
        # Custom or composite module
        sub_configs = {name: get_layer_config(sub_mod) for name, sub_mod in layer._modules.items()}
        return {
            "type": layer_type,
            "submodules": sub_configs,
        }
    else:
        raise ValueError(f"Unsupported layer type for serialization: {layer_type}")


def build_layer_from_config(
    config: Dict[str, Any],
    custom_objects: Optional[Dict[str, Type[Module]]] = None,
) -> Module:
    """
    Instantiates a Module hierarchy from a configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary describing the architecture.
    custom_objects : Optional[Dict[str, Type[Module]]], optional
        Mapping from class names to custom Module classes.
    """
    if not isinstance(config, dict) or "type" not in config:
        raise ValueError(f"Invalid layer configuration schema: {config}")

    registry = dict(_LAYER_REGISTRY)
    if custom_objects:
        registry.update(custom_objects)

    layer_type = config["type"]

    if layer_type == "Sequential":
        sub_layer_configs = config.get("layers", [])
        sub_layers = [build_layer_from_config(cfg, custom_objects=custom_objects) for cfg in sub_layer_configs]
        return layers_module.Sequential(*sub_layers)

    elif layer_type in registry:
        cls = registry[layer_type]
        kwargs = config.get("config", {})
        try:
            instance = cls(**kwargs)
        except TypeError:
            instance = cls()

        if "submodules" in config:
            for name, sub_cfg in config["submodules"].items():
                setattr(instance, name, build_layer_from_config(sub_cfg, custom_objects=custom_objects))
        return instance

    elif "submodules" in config:
        # Fallback container
        container = Module()
        for name, sub_cfg in config["submodules"].items():
            setattr(container, name, build_layer_from_config(sub_cfg, custom_objects=custom_objects))
        return container

    else:
        raise ValueError(f"Unknown or unsupported layer type: '{layer_type}'")


def get_model_state_dict(model: Module, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    Extracts all parameter arrays and buffer arrays into a flat dictionary.
    """
    state: Dict[str, np.ndarray] = {}

    # 1. Collect direct parameters
    if hasattr(model, "_parameters"):
        for name, param in model._parameters.items():
            if param is not None:
                state[f"{prefix}{name}"] = param.data.copy()

    # 2. Collect layer-specific non-parameter buffers (e.g. BatchNorm1d running stats)
    if isinstance(model, layers_module.BatchNorm1d):
        state[f"{prefix}running_mean"] = model.running_mean.copy()
        state[f"{prefix}running_var"] = model.running_var.copy()

    # 3. Recursively collect from child submodules
    if hasattr(model, "_modules"):
        for name, child in model._modules.items():
            child_state = get_model_state_dict(child, prefix=f"{prefix}{name}.")
            state.update(child_state)

    return state


def load_model_state_dict(model: Module, state: Dict[str, np.ndarray], prefix: str = "", strict: bool = True) -> None:
    """
    Loads parameter arrays and buffers into a model instance with shape validation.
    """
    # 1. Load direct parameters
    if hasattr(model, "_parameters"):
        for name, param in model._parameters.items():
            key = f"{prefix}{name}"
            if key in state:
                loaded_arr = state[key]
                if param is not None:
                    if param.shape != loaded_arr.shape:
                        raise ValueError(
                            f"Shape mismatch for parameter '{key}': expected {param.shape}, got {loaded_arr.shape}"
                        )
                    param.data = loaded_arr.astype(param.dtype, copy=True)
            elif strict and param is not None:
                raise KeyError(f"Missing parameter key in state dict: '{key}'")

    # 2. Load layer-specific buffers
    if isinstance(model, layers_module.BatchNorm1d):
        rm_key = f"{prefix}running_mean"
        rv_key = f"{prefix}running_var"
        if rm_key in state:
            model.running_mean = state[rm_key].astype(model.running_mean.dtype, copy=True)
        elif strict:
            raise KeyError(f"Missing buffer key in state dict: '{rm_key}'")

        if rv_key in state:
            model.running_var = state[rv_key].astype(model.running_var.dtype, copy=True)
        elif strict:
            raise KeyError(f"Missing buffer key in state dict: '{rv_key}'")

    # 3. Recursively load into child submodules
    if hasattr(model, "_modules"):
        for name, child in model._modules.items():
            load_model_state_dict(child, state, prefix=f"{prefix}{name}.", strict=strict)


def save_model(model: Module, filepath: str) -> None:
    """
    Serializes a NumPyGrad model architecture and weights to a `.ng` zip container file.

    Parameters
    ----------
    model : Module
        The neural network model to serialize.
    filepath : str
        Target file path (typically ending with `.ng`).
    """
    if not isinstance(model, Module):
        raise TypeError(f"save_model expected a Module instance, got {type(model).__name__}")

    # 1. Serialize Architecture & Training State
    arch_config = {
        "version": "1.0",
        "training": model.training,
        "architecture": get_layer_config(model),
    }
    arch_json_str = json.dumps(arch_config, indent=2)

    # 2. Serialize Weights to NPZ bytes buffer
    state = get_model_state_dict(model)
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **state)
    npz_bytes = npz_buffer.getvalue()

    # 3. Package into ZIP container
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("architecture.json", arch_json_str)
        zip_file.writestr("weights.npz", npz_bytes)


def load_model(filepath: str, custom_objects: Optional[Dict[str, Type[Module]]] = None) -> Module:
    """
    Deserializes and reconstructs a NumPyGrad model from a `.ng` container file.

    Parameters
    ----------
    filepath : str
        Path to the saved `.ng` model file.
    custom_objects : Optional[Dict[str, Type[Module]]], optional
        Dictionary mapping custom class names to Module classes.

    Returns
    -------
    Module
        The reconstructed model with all parameters and buffers restored.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: '{filepath}'")

    if not zipfile.is_zipfile(filepath):
        raise ValueError(f"Invalid or corrupted model archive: '{filepath}' (not a valid zip container)")

    with zipfile.ZipFile(filepath, "r") as zip_file:
        namelist = zip_file.namelist()

        if "architecture.json" not in namelist:
            raise ValueError(f"Corrupted model container '{filepath}': missing 'architecture.json'")
        if "weights.npz" not in namelist:
            raise ValueError(f"Corrupted model container '{filepath}': missing 'weights.npz'")

        # 1. Read & parse architecture JSON
        arch_bytes = zip_file.read("architecture.json")
        try:
            arch_data = json.loads(arch_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse 'architecture.json' in '{filepath}': {e}") from e

        if not isinstance(arch_data, dict) or "architecture" not in arch_data:
            raise ValueError(f"Invalid architecture schema in '{filepath}'")

        # 2. Build model hierarchy
        model = build_layer_from_config(arch_data["architecture"], custom_objects=custom_objects)
        if "training" in arch_data:
            model.train(arch_data["training"])

        # 3. Load & restore weights from NPZ
        weights_bytes = zip_file.read("weights.npz")
        try:
            with np.load(io.BytesIO(weights_bytes)) as loaded_npz:
                weights_dict = {k: loaded_npz[k] for k in loaded_npz.files}
        except Exception as e:
            raise ValueError(f"Failed to unpack 'weights.npz' in '{filepath}': {e}") from e

        load_model_state_dict(model, weights_dict, strict=True)

    return model
