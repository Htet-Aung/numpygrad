"""
Unit tests and gradcheck verification for Conv2D and MaxPool2D layers.
"""

import os
import tempfile
import numpy as np
import pytest

from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.gradcheck import gradcheck
from numpygrad.serialization import save_model, load_model


def test_conv2d_forward_shapes():
    # Standard 3x3 conv with stride=1, padding=1 (preserves spatial dims)
    conv = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(4, 3, 28, 28).astype(np.float32))
    out = conv(x)
    assert out.shape == (4, 16, 28, 28)

    # 5x5 conv with stride=2, padding=2
    conv2 = nn.Conv2D(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2)
    x2 = Tensor(np.random.randn(2, 1, 14, 14).astype(np.float32))
    out2 = conv2(x2)
    # (14 + 4 - 5) // 2 + 1 = 13 // 2 + 1 = 7
    assert out2.shape == (2, 8, 7, 7)

    # 3x3 conv with stride=1, padding=0 (valid)
    conv3 = nn.Conv2D(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0)
    x3 = Tensor(np.random.randn(2, 4, 10, 10).astype(np.float32))
    out3 = conv3(x3)
    assert out3.shape == (2, 8, 8, 8)


def test_conv2d_no_bias():
    conv = nn.Conv2D(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)
    assert conv.bias is None
    assert len(conv.parameters()) == 1

    x = Tensor(np.random.randn(2, 2, 8, 8).astype(np.float32))
    out = conv(x)
    assert out.shape == (2, 4, 8, 8)


def test_conv2d_gradcheck():
    np.random.seed(42)
    conv = nn.Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
    conv.weight.data = conv.weight.data.astype(np.float64)
    conv.bias.data = conv.bias.data.astype(np.float64)

    x = Tensor(np.random.randn(2, 2, 4, 4).astype(np.float64), requires_grad=True)

    def f_input(x_t):
        return (conv(x_t) ** 2.0).sum()

    assert gradcheck(f_input, [x], eps=1e-5, rtol=1e-3, atol=1e-4)

    def f_all(x_t, w_t, b_t):
        conv.weight = w_t
        conv.bias = b_t
        return (conv(x_t) ** 2.0).sum()

    assert gradcheck(f_all, [x, conv.weight, conv.bias], eps=1e-5, rtol=1e-3, atol=1e-4)


def test_maxpool2d_forward_shapes():
    pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
    x = Tensor(np.random.randn(4, 8, 28, 28).astype(np.float32))
    out = pool(x)
    assert out.shape == (4, 8, 14, 14)

    pool3x3 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    x2 = Tensor(np.random.randn(2, 4, 15, 15).astype(np.float32))
    out2 = pool3x3(x2)
    # (15 + 2 - 3) // 2 + 1 = 8
    assert out2.shape == (2, 4, 8, 8)


def test_maxpool2d_gradcheck():
    np.random.seed(42)
    pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

    # Use distinct values to avoid ambiguity at equal maxima
    flat = np.arange(2 * 2 * 4 * 4, dtype=np.float64)
    np.random.shuffle(flat)
    x_data = flat.reshape(2, 2, 4, 4)
    x = Tensor(x_data, requires_grad=True)

    def f(x_t):
        return (pool(x_t) ** 2.0).sum()

    assert gradcheck(f, [x], eps=1e-5, rtol=1e-3, atol=1e-4)


def test_cnn_sequential_dag_backprop():
    np.random.seed(42)
    model = nn.Sequential(
        nn.Conv2D(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Linear(4 * 7 * 7, 10),
    )

    x = Tensor(np.random.randn(4, 1, 14, 14).astype(np.float32), requires_grad=True)
    out = model(x)
    assert out.shape == (4, 10)

    loss = (out ** 2.0).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (4, 1, 14, 14)
    for p in model.parameters():
        assert p.grad is not None


def test_cnn_optimization_step():
    np.random.seed(42)
    model = nn.Sequential(
        nn.Conv2D(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Linear(4 * 7 * 7, 2),
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    x = Tensor(np.random.randn(4, 1, 14, 14).astype(np.float32))
    targets = np.array([0, 1, 0, 1], dtype=np.int64)

    initial_loss = float(criterion(model(x), targets).data)

    for _ in range(10):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    final_loss = float(criterion(model(x), targets).data)
    assert final_loss < initial_loss


def test_cnn_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "cnn_model.ng")

        model = nn.Sequential(
            nn.Conv2D(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            nn.Conv2D(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10),
        )

        model.save(filepath)
        loaded = load_model(filepath)

        assert isinstance(loaded[0], nn.Conv2D)
        assert loaded[0].in_channels == 1
        assert loaded[0].out_channels == 8
        assert isinstance(loaded[2], nn.MaxPool2D)
        assert isinstance(loaded[3], nn.Conv2D)
        assert loaded[3].in_channels == 8
        assert loaded[3].out_channels == 16

        X = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
        np.testing.assert_allclose(model(X).data, loaded(X).data, atol=1e-6)
