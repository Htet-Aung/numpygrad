"""
Unit tests and gradcheck for the Flatten neural network layer.
"""

import os
import tempfile
import numpy as np
import pytest

from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
from numpygrad.utils.gradcheck import gradcheck
from numpygrad.serialization import save_model, load_model


def test_flatten_default_4d():
    layer = nn.Flatten()
    x = Tensor(np.random.randn(4, 3, 8, 8).astype(np.float32))
    out = layer(x)

    assert out.shape == (4, 3 * 8 * 8)
    assert out.data.shape == (4, 192)
    np.testing.assert_array_equal(out.data, x.data.reshape(4, 192))


def test_flatten_3d():
    layer = nn.Flatten()
    x = Tensor(np.random.randn(8, 16, 16).astype(np.float32))
    out = layer(x)

    assert out.shape == (8, 256)
    np.testing.assert_array_equal(out.data, x.data.reshape(8, 256))


def test_flatten_custom_dims():
    # Flatten intermediate dimensions (start_dim=1, end_dim=2)
    layer = nn.Flatten(start_dim=1, end_dim=2)
    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    out = layer(x)

    assert out.shape == (2, 12, 5)
    np.testing.assert_array_equal(out.data, x.data.reshape(2, 12, 5))


def test_flatten_negative_dims():
    layer = nn.Flatten(start_dim=-2, end_dim=-1)
    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    out = layer(x)

    assert out.shape == (2, 3, 20)
    np.testing.assert_array_equal(out.data, x.data.reshape(2, 3, 20))


def test_flatten_all_dims():
    layer = nn.Flatten(start_dim=0, end_dim=-1)
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    out = layer(x)

    assert out.shape == (24,)
    np.testing.assert_array_equal(out.data, x.data.reshape(24))


def test_flatten_bounds_errors():
    layer_bad_start = nn.Flatten(start_dim=4, end_dim=5)
    x = Tensor(np.random.randn(2, 3, 4))
    with pytest.raises(IndexError, match="start_dim"):
        layer_bad_start(x)

    layer_bad_end = nn.Flatten(start_dim=0, end_dim=10)
    with pytest.raises(IndexError, match="end_dim"):
        layer_bad_end(x)

    layer_inverted = nn.Flatten(start_dim=2, end_dim=1)
    with pytest.raises(ValueError, match="cannot be greater"):
        layer_inverted(x)


def test_flatten_gradcheck():
    layer = nn.Flatten()
    x = Tensor(np.random.randn(3, 2, 4, 4).astype(np.float64), requires_grad=True)

    def f(x_t):
        return (layer(x_t) ** 2.0).sum()

    assert gradcheck(f, [x], eps=1e-5, rtol=1e-3, atol=1e-4)


def test_flatten_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "flatten_model.ng")

        model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(24, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        model.save(filepath)
        loaded = load_model(filepath)

        assert isinstance(loaded[0], nn.Flatten)
        assert loaded[0].start_dim == 1
        assert loaded[0].end_dim == -1

        X = Tensor(np.random.randn(4, 2, 3, 4).astype(np.float32))
        np.testing.assert_allclose(model(X).data, loaded(X).data, atol=1e-6)
