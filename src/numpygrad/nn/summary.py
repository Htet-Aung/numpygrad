"""
Model inspection and architecture summary utilities for NumPyGrad.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np

from numpygrad.core.tensor import Tensor, no_grad
from numpygrad.nn.module import Module, Parameter
import numpygrad.nn.layers as layers_module


def summary(
    model: Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    verbose: bool = True,
) -> str:
    """
    Generates a structured ASCII summary table for a neural network Module.

    Parameters
    ----------
    model : Module
        The neural network module to inspect.
    input_shape : Optional[Tuple[int, ...]], optional
        Input tensor shape (including or excluding batch dimension), e.g. (1, 28, 28) or (1, 1, 28, 28).
        If provided, executes a dry forward pass under `no_grad()` to determine layer output shapes.
    verbose : bool, default=True
        Whether to print the summary to stdout.

    Returns
    -------
    str
        The formatted ASCII summary report.
    """
    lines: List[str] = []
    divider = "=" * 90
    sub_divider = "-" * 90

    lines.append(divider)
    lines.append(f"{'Layer (type:idx)':<35} {'Output Shape':<25} {'Param #':<15} {'Buffer #':<12}")
    lines.append(divider)

    # Collect leaf layers to display
    if isinstance(model, layers_module.Sequential):
        layer_list = list(model._modules.values())
    elif hasattr(model, "_modules") and model._modules:
        layer_list = list(model._modules.values())
    else:
        layer_list = [model]

    # Dry-run forward pass to capture output shapes if input_shape is provided
    output_shapes: List[Optional[Tuple[int, ...]]] = [None] * len(layer_list)
    total_activation_elements = 0
    input_elements = 0

    if input_shape is not None:
        try:
            x_dummy = Tensor(np.zeros(input_shape, dtype=np.float32), requires_grad=False)
            input_elements = x_dummy.data.size
            with no_grad():
                if isinstance(model, layers_module.Sequential):
                    curr_x = x_dummy
                    for idx, layer in enumerate(layer_list):
                        curr_x = layer(curr_x)
                        output_shapes[idx] = curr_x.shape
                        total_activation_elements += curr_x.data.size
                else:
                    out = model(x_dummy)
                    output_shapes[0] = out.shape if hasattr(out, "shape") else None
                    if hasattr(out, "data"):
                        total_activation_elements = out.data.size
        except Exception as e:
            # If dry run fails (e.g. custom module), fallback gracefully to None
            pass

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    total_buffer_elements = 0

    for idx, layer in enumerate(layer_list):
        layer_name = f"{layer.__class__.__name__}-{idx + 1}"
        layer_type_str = f"{layer_name} ({layer.__class__.__name__})"

        # Calculate parameters for this layer
        p_trainable = sum(p.data.size for p in layer.parameters() if p.requires_grad)
        p_non_trainable = sum(p.data.size for p in layer.parameters() if not p.requires_grad)
        p_total = p_trainable + p_non_trainable

        # Calculate buffer sizes (e.g. BatchNorm1d running_mean, running_var)
        b_count = 0
        if isinstance(layer, layers_module.BatchNorm1d):
            b_count += layer.running_mean.size + layer.running_var.size

        total_params += p_total
        trainable_params += p_trainable
        non_trainable_params += p_non_trainable
        total_buffer_elements += b_count

        out_shape_str = str(list(output_shapes[idx])) if output_shapes[idx] is not None else "--"
        lines.append(f"{layer_type_str:<35} {out_shape_str:<25} {p_total:<15,d} {b_count:<12,d}")

    lines.append(divider)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append(f"Non-trainable params: {non_trainable_params:,}")
    lines.append(f"Total buffer elements: {total_buffer_elements:,}")
    lines.append(sub_divider)

    # Memory estimation (assuming float32 4 bytes/element)
    bytes_per_elem = 4.0
    mb_scale = 1024.0 * 1024.0

    params_mb = (total_params + total_buffer_elements) * bytes_per_elem / mb_scale
    input_mb = input_elements * bytes_per_elem / mb_scale
    act_mb = total_activation_elements * 2 * bytes_per_elem / mb_scale  # 2x for forward + backward activations
    total_est_mb = params_mb + input_mb + act_mb

    lines.append(f"Input size (MB): {input_mb:.4f}")
    lines.append(f"Forward/backward pass size (MB): {act_mb:.4f}")
    lines.append(f"Params size (MB): {params_mb:.4f}")
    lines.append(f"Estimated Total Size (MB): {total_est_mb:.4f}")
    lines.append(divider)

    report = "\n".join(lines)
    if verbose:
        print(report)
    return report
