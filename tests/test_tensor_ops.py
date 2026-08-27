"""
Unit tests and finite-difference gradient checks for expanded Tensor operations
and descriptive dimension error assertions on neural network layers.
"""

import pytest
import numpy as np

import numpygrad as ng
from numpygrad.core.tensor import Tensor, concat, cat
import numpygrad.nn as nn
from numpygrad.utils.gradcheck import gradcheck


# -----------------------------------------------------------------------------
# Reshape, Transpose & T Property Tests
# -----------------------------------------------------------------------------

def test_tensor_reshape_unpacked_and_tuple():
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float64), requires_grad=True)

    # Unpacked args
    out1 = x.reshape(6, 4)
    assert out1.shape == (6, 4)

    # Tuple arg
    out2 = x.reshape((6, 4))
    assert out2.shape == (6, 4)

    # Gradcheck
    assert gradcheck(lambda t: t.reshape(6, 4).sum(), [x])
    assert gradcheck(lambda t: t.reshape((12, 2)).sum(), [x])


def test_tensor_transpose_unpacked_and_tuple():
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float64), requires_grad=True)

    # Unpacked args
    out1 = x.transpose(2, 0, 1)
    assert out1.shape == (4, 2, 3)

    # Tuple arg
    out2 = x.transpose((2, 0, 1))
    assert out2.shape == (4, 2, 3)

    # Gradcheck
    assert gradcheck(lambda t: t.transpose(2, 0, 1).sum(), [x])


def test_tensor_t_property():
    # 2D matrix
    x = Tensor(np.random.randn(3, 5).astype(np.float64), requires_grad=True)
    out = x.T
    assert out.shape == (5, 3)
    assert gradcheck(lambda t: t.T.sum(), [x])

    # N-D tensor
    x_3d = Tensor(np.random.randn(2, 3, 4).astype(np.float64), requires_grad=True)
    assert x_3d.T.shape == (4, 3, 2)
    assert gradcheck(lambda t: (t.T ** 2.0).sum(), [x_3d])


# -----------------------------------------------------------------------------
# Squeeze & Unsqueeze Tests
# -----------------------------------------------------------------------------

def test_tensor_squeeze_all_and_axis():
    x = Tensor(np.random.randn(1, 3, 1, 4).astype(np.float64), requires_grad=True)

    # Squeeze all singletons
    out1 = x.squeeze()
    assert out1.shape == (3, 4)
    assert gradcheck(lambda t: (t.squeeze() ** 2.0).sum(), [x])

    # Squeeze specific axis
    out2 = x.squeeze(axis=0)
    assert out2.shape == (3, 1, 4)
    assert gradcheck(lambda t: (t.squeeze(axis=0) ** 2.0).sum(), [x])

    out3 = x.squeeze(axis=2)
    assert out3.shape == (1, 3, 4)
    assert gradcheck(lambda t: (t.squeeze(axis=2) ** 2.0).sum(), [x])


def test_tensor_unsqueeze():
    x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)

    out0 = x.unsqueeze(0)
    assert out0.shape == (1, 3, 4)
    assert gradcheck(lambda t: (t.unsqueeze(0) ** 2.0).sum(), [x])

    out1 = x.unsqueeze(1)
    assert out1.shape == (3, 1, 4)
    assert gradcheck(lambda t: (t.unsqueeze(1) ** 2.0).sum(), [x])

    out2 = x.unsqueeze(2)
    assert out2.shape == (3, 4, 1)
    assert gradcheck(lambda t: (t.unsqueeze(2) ** 2.0).sum(), [x])


# -----------------------------------------------------------------------------
# Sum & Mean Reduction Tests
# -----------------------------------------------------------------------------

def test_tensor_sum_reductions():
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float64), requires_grad=True)

    # All axes
    assert gradcheck(lambda t: t.sum(), [x])
    assert gradcheck(lambda t: t.sum(keepdims=True), [x])

    # Single axis
    assert gradcheck(lambda t: t.sum(axis=1).sum(), [x])
    assert gradcheck(lambda t: t.sum(axis=1, keepdims=True).sum(), [x])

    # Multiple axes
    assert gradcheck(lambda t: t.sum(axis=(0, 2)).sum(), [x])
    assert gradcheck(lambda t: t.sum(axis=(0, 2), keepdims=True).sum(), [x])


def test_tensor_mean_reductions():
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float64), requires_grad=True)

    # All axes
    assert gradcheck(lambda t: t.mean(), [x])
    assert gradcheck(lambda t: t.mean(keepdims=True), [x])

    # Single axis
    assert gradcheck(lambda t: t.mean(axis=1).sum(), [x])
    assert gradcheck(lambda t: t.mean(axis=1, keepdims=True).sum(), [x])

    # Multiple axes
    assert gradcheck(lambda t: t.mean(axis=(0, 2)).sum(), [x])
    assert gradcheck(lambda t: t.mean(axis=(0, 2), keepdims=True).sum(), [x])


# -----------------------------------------------------------------------------
# Concat & Cat Tests
# -----------------------------------------------------------------------------

def test_concat_forward_and_backward():
    a = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
    b = Tensor(np.random.randn(2, 5).astype(np.float64), requires_grad=True)
    c = Tensor(np.random.randn(2, 2).astype(np.float64), requires_grad=True)

    # Concatenate along axis=1
    out = concat([a, b, c], axis=1)
    assert out.shape == (2, 10)

    def f(a_t, b_t, c_t):
        return (concat([a_t, b_t, c_t], axis=1) ** 2.0).sum()

    assert gradcheck(f, [a, b, c])


def test_concat_axis_0():
    a = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
    b = Tensor(np.random.randn(5, 4).astype(np.float64), requires_grad=True)

    out = cat([a, b], axis=0)
    assert out.shape == (8, 4)

    def f(a_t, b_t):
        return (cat([a_t, b_t], axis=0) ** 2.0).sum()

    assert gradcheck(f, [a, b])


def test_concat_validation_errors():
    with pytest.raises(ValueError, match="non-empty"):
        concat([])


# -----------------------------------------------------------------------------
# Descriptive Dimension Error Handling Tests
# -----------------------------------------------------------------------------

def test_linear_dimension_errors():
    layer = nn.Linear(in_features=128, out_features=10)

    # 4D image input passed directly to Linear without Flatten
    x_4d = Tensor(np.random.randn(4, 1, 28, 28))
    with pytest.raises(ValueError, match="Flatten"):
        layer(x_4d)

    # 3D tensor passed to Linear
    x_3d = Tensor(np.random.randn(4, 14, 14))
    with pytest.raises(ValueError, match="Flatten"):
        layer(x_3d)

    # Feature size mismatch
    x_mismatch = Tensor(np.random.randn(4, 64))
    with pytest.raises(ValueError, match="in_features=128"):
        layer(x_mismatch)


def test_conv2d_dimension_errors():
    conv = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    # 2D flattened input passed to Conv2D
    x_2d = Tensor(np.random.randn(4, 784))
    with pytest.raises(ValueError, match="4D input"):
        conv(x_2d)

    # Channel mismatch (e.g. single-channel image passed to 3-channel Conv2D)
    x_wrong_channel = Tensor(np.random.randn(4, 1, 28, 28))
    with pytest.raises(ValueError, match="in_channels=3"):
        conv(x_wrong_channel)

    # Spatial dimensions too small for kernel size
    conv_large = nn.Conv2D(in_channels=3, out_channels=8, kernel_size=7, padding=0)
    x_too_small = Tensor(np.random.randn(2, 3, 5, 5))
    with pytest.raises(ValueError, match="smaller than kernel size"):
        conv_large(x_too_small)
