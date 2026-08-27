"""
Convolutional neural network layers: Conv2D and MaxPool2D.

Implements vectorized convolution via im2col/col2im matrix unfolding for
high-performance forward and backward passes in pure NumPy.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple
import numpy as np
from numpygrad.core.tensor import Tensor, is_grad_enabled
from numpygrad.nn.module import Module, Parameter


def _pair(x) -> Tuple[int, int]:
    """Converts a scalar or 2-tuple to a guaranteed (int, int) pair."""
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


# -----------------------------------------------------------------------------
# im2col / col2im Helpers
# -----------------------------------------------------------------------------

def im2col_indices(
    x: np.ndarray,
    kH: int,
    kW: int,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Extracts image patches into columns for vectorized convolution.

    Parameters
    ----------
    x : np.ndarray, shape (B, C, H, W)
    kH, kW : kernel spatial dimensions
    stride : (sH, sW) convolution strides
    padding : (pH, pW) zero-padding

    Returns
    -------
    cols : np.ndarray, shape (B, C * kH * kW, out_H * out_W)
        Unfolded patch columns for matrix multiplication with reshaped kernels.
    """
    B, C, H, W = x.shape
    pH, pW = padding
    sH, sW = stride

    # Pad input if necessary
    if pH > 0 or pW > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode="constant")
    else:
        x_padded = x

    H_pad, W_pad = x_padded.shape[2], x_padded.shape[3]
    out_H = (H_pad - kH) // sH + 1
    out_W = (W_pad - kW) // sW + 1

    # Use stride tricks for efficient patch extraction
    # Shape: (B, C, out_H, out_W, kH, kW)
    shape = (B, C, out_H, out_W, kH, kW)
    strides_arr = x_padded.strides
    strides = (
        strides_arr[0],                    # batch
        strides_arr[1],                    # channel
        strides_arr[2] * sH,              # output row
        strides_arr[3] * sW,              # output col
        strides_arr[2],                    # kernel row
        strides_arr[3],                    # kernel col
    )
    patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    # Transpose to (B, C, kH, kW, out_H, out_W) to group (C, kH, kW) and (out_H, out_W)
    patches = np.ascontiguousarray(patches.transpose(0, 1, 4, 5, 2, 3))

    # Reshape to (B, C * kH * kW, out_H * out_W)
    cols = patches.reshape(B, C * kH * kW, out_H * out_W)
    return cols


def col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, ...],
    kH: int,
    kW: int,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Inverse of im2col: accumulates column gradients back into the spatial image layout.

    Parameters
    ----------
    cols : np.ndarray, shape (B, C * kH * kW, out_H * out_W)
    x_shape : original (B, C, H, W)
    kH, kW : kernel spatial dimensions
    stride, padding : convolution parameters

    Returns
    -------
    x_grad : np.ndarray, shape (B, C, H, W)
    """
    B, C, H, W = x_shape
    pH, pW = padding
    sH, sW = stride

    H_pad = H + 2 * pH
    W_pad = W + 2 * pW
    out_H = (H_pad - kH) // sH + 1
    out_W = (W_pad - kW) // sW + 1

    # Reshape cols back to patches: (B, C, kH, kW, out_H, out_W)
    cols_reshaped = cols.reshape(B, C, kH, kW, out_H, out_W)

    # Accumulate into padded gradient buffer
    x_padded_grad = np.zeros((B, C, H_pad, W_pad), dtype=cols.dtype)

    for i in range(kH):
        i_max = i + sH * out_H
        for j in range(kW):
            j_max = j + sW * out_W
            x_padded_grad[:, :, i:i_max:sH, j:j_max:sW] += cols_reshaped[:, :, i, j, :, :]

    # Remove padding
    if pH > 0 or pW > 0:
        return x_padded_grad[:, :, pH:H_pad - pH, pW:W_pad - pW]
    return x_padded_grad


# -----------------------------------------------------------------------------
# Conv2D Layer
# -----------------------------------------------------------------------------

class Conv2D(Module):
    """
    2D spatial convolution layer using im2col vectorized matrix multiplication.

    Forward:
        Y = W_reshaped @ im2col(X) + b
        where W_reshaped has shape (out_channels, in_channels * kH * kW)
        and im2col(X) has shape (B, in_channels * kH * kW, out_H * out_W)

    Backward:
        dW = dY_reshaped @ col^T
        db = sum(dY)
        dX = col2im(W^T @ dY_reshaped)

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or (int, int)
        Size of the convolving kernel.
    stride : int or (int, int)
        Stride of the convolution. Default: 1
    padding : int or (int, int)
        Zero-padding added to both sides of the input. Default: 0
    bias : bool
        If True, adds a learnable bias to the output. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        kH, kW = self.kernel_size
        fan_in = in_channels * kH * kW

        # He/Kaiming normal initialization for ReLU networks
        std = np.sqrt(2.0 / fan_in)
        weight_data = np.random.normal(0.0, std, size=(out_channels, in_channels, kH, kW)).astype(np.float32)
        self.weight = Parameter(weight_data)

        if bias:
            self.bias: Optional[Parameter] = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Forward pass: Y[b, f, i, j] = sum_c sum_kh sum_kw W[f, c, kh, kw] * X[b, c, i*sH+kh, j*sW+kw] + bias[f]
        """
        x_t = x if isinstance(x, Tensor) else Tensor(x)
        x_data = x_t.data

        B, C, H, W_in = x_data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        out_channels = self.out_channels

        H_pad = H + 2 * pH
        W_pad = W_in + 2 * pW
        out_H = (H_pad - kH) // sH + 1
        out_W = (W_pad - kW) // sW + 1

        # im2col: (B, C*kH*kW, out_H*out_W)
        cols = im2col_indices(x_data, kH, kW, self.stride, self.padding)

        # Weight matrix: (out_channels, C*kH*kW)
        w_reshaped = self.weight.data.reshape(out_channels, -1)

        # Batched matmul: (B, out_channels, out_H*out_W) = (out_channels, C*kH*kW) @ (B, C*kH*kW, out_H*out_W)
        # We use einsum for clarity: out[b, f, hw] = w[f, ckk] * cols[b, ckk, hw]
        out_data = np.einsum("fc,bcp->bfp", w_reshaped, cols)

        if self.bias is not None:
            out_data += self.bias.data.reshape(1, out_channels, 1)

        out_data = out_data.reshape(B, out_channels, out_H, out_W)

        req_grad = is_grad_enabled() and (x_t.requires_grad or self.weight.requires_grad or
                                           (self.bias is not None and self.bias.requires_grad))

        prev = [x_t, self.weight]
        if self.bias is not None:
            prev.append(self.bias)

        if not req_grad:
            return Tensor(out_data, requires_grad=False, _prev=(), _op="conv2d")

        out = Tensor(out_data, requires_grad=True, _prev=tuple(prev), _op="conv2d")

        # Cache for backward
        _cols = cols
        _x_shape = x_data.shape
        _w_reshaped = w_reshaped

        def _backward() -> None:
            if out.grad is None:
                return
            # out.grad shape: (B, out_channels, out_H, out_W)
            dout = out.grad.reshape(B, out_channels, -1)  # (B, out_channels, out_H*out_W)

            # Weight gradients: dW = dout @ cols^T  => (out_channels, C*kH*kW)
            if self.weight.requires_grad:
                # dW[f, ckk] = sum_b sum_hw dout[b, f, hw] * cols[b, ckk, hw]
                dw = np.einsum("bfp,bcp->fc", dout, _cols)
                dw = dw.reshape(self.weight.shape)
                self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

            # Bias gradients: db = sum over batch and spatial dims
            if self.bias is not None and self.bias.requires_grad:
                db = dout.sum(axis=(0, 2))  # (out_channels,)
                self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

            # Input gradients: dx_cols = W^T @ dout => col2im
            if x_t.requires_grad:
                # dx_cols[b, ckk, hw] = sum_f w[f, ckk] * dout[b, f, hw]
                dx_cols = np.einsum("fc,bfp->bcp", _w_reshaped, dout)
                dx = col2im_indices(dx_cols, _x_shape, kH, kW, self.stride, self.padding)
                x_t.grad = dx if x_t.grad is None else x_t.grad + dx

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None})"
        )


# -----------------------------------------------------------------------------
# MaxPool2D Layer
# -----------------------------------------------------------------------------

class MaxPool2D(Module):
    """
    2D max pooling layer.

    Forward:
        For each (kernel_size x kernel_size) spatial window, output the maximum value.

    Backward:
        Routes upstream gradients exclusively to the position of the maximum value
        within each pooling window (argmax mask routing).

    Parameters
    ----------
    kernel_size : int or (int, int)
        Size of the pooling window. Default: 2
    stride : int or (int, int) or None
        Stride of the pooling window. Default: same as kernel_size
    padding : int or (int, int)
        Zero-padding added to both sides. Default: 0
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Forward pass: Y[b, c, oh, ow] = max over (kh, kw) window of X[b, c, oh*sH+kh, ow*sW+kw]
        """
        x_t = x if isinstance(x, Tensor) else Tensor(x)
        x_data = x_t.data

        B, C, H, W_in = x_data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # Pad input
        if pH > 0 or pW > 0:
            x_padded = np.pad(x_data, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                              mode="constant", constant_values=-np.inf)
        else:
            x_padded = x_data

        H_pad, W_pad = x_padded.shape[2], x_padded.shape[3]
        out_H = (H_pad - kH) // sH + 1
        out_W = (W_pad - kW) // sW + 1

        # Use stride tricks to extract windows: (B, C, out_H, out_W, kH, kW)
        strides_arr = x_padded.strides
        shape = (B, C, out_H, out_W, kH, kW)
        strides = (
            strides_arr[0],
            strides_arr[1],
            strides_arr[2] * sH,
            strides_arr[3] * sW,
            strides_arr[2],
            strides_arr[3],
        )
        windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides).copy()

        # Max over (kH, kW) dimensions
        out_data = windows.max(axis=(4, 5))  # (B, C, out_H, out_W)

        req_grad = is_grad_enabled() and x_t.requires_grad

        if not req_grad:
            return Tensor(out_data, requires_grad=False, _prev=(), _op="maxpool2d")

        out = Tensor(out_data, requires_grad=True, _prev=(x_t,), _op="maxpool2d")

        # Cache for backward: argmax mask
        _windows = windows
        _x_shape = x_data.shape

        def _backward() -> None:
            if out.grad is None:
                return
            if not x_t.requires_grad:
                return

            # Build argmax mask: 1 where window value == max, 0 elsewhere
            # out_data expanded: (B, C, out_H, out_W, 1, 1)
            max_vals = out_data[:, :, :, :, np.newaxis, np.newaxis]
            mask = (_windows == max_vals).astype(out.grad.dtype)

            # Handle ties: normalize so gradients sum correctly
            mask_sum = mask.sum(axis=(4, 5), keepdims=True)
            mask_sum = np.maximum(mask_sum, 1.0)
            mask = mask / mask_sum

            # Distribute gradients: (B, C, out_H, out_W) -> (B, C, out_H, out_W, kH, kW)
            dout_expanded = out.grad[:, :, :, :, np.newaxis, np.newaxis]
            dwindows = mask * dout_expanded  # (B, C, out_H, out_W, kH, kW)

            # Scatter back to padded input shape
            dx_padded = np.zeros((B, C, H_pad, W_pad), dtype=out.grad.dtype)
            for i in range(kH):
                for j in range(kW):
                    dx_padded[:, :, i:i + sH * out_H:sH, j:j + sW * out_W:sW] += dwindows[:, :, :, :, i, j]

            # Remove padding
            if pH > 0 or pW > 0:
                dx = dx_padded[:, :, pH:H_pad - pH, pW:W_pad - pW]
            else:
                dx = dx_padded

            x_t.grad = dx if x_t.grad is None else x_t.grad + dx

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
