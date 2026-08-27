"""
Comprehensive unit tests for NumPyGrad neural network modules, layers, losses, and optimizers.
"""

import numpy as np
import pytest
from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.gradcheck import gradcheck


# -----------------------------------------------------------------------------
# Module & Parameter Hierarchy Tests
# -----------------------------------------------------------------------------

def test_module_parameter_collection():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 3)
            self.l2 = nn.Linear(3, 2)

        def forward(self, x):
            return self.l2(self.l1(x).relu())

    net = SimpleNet()
    params = net.parameters()
    assert len(params) == 4  # l1.weight, l1.bias, l2.weight, l2.bias
    assert all(isinstance(p, nn.Parameter) for p in params)


def test_module_train_eval_modes():
    net = nn.Sequential(
        nn.Linear(5, 4),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(4),
    )
    assert net.training is True
    assert net[1].training is True
    assert net[2].training is True

    net.eval()
    assert net.training is False
    assert net[1].training is False
    assert net[2].training is False

    net.train()
    assert net.training is True
    assert net[1].training is True
    assert net[2].training is True


def test_initializations():
    p1 = nn.Parameter(np.empty((100, 50), dtype=np.float32))
    nn.xavier_uniform_(p1)
    assert np.abs(p1.data.mean()) < 0.05

    p2 = nn.Parameter(np.empty((100, 50), dtype=np.float32))
    nn.kaiming_normal_(p2)
    assert np.abs(p2.data.mean()) < 0.05
    assert np.abs(p2.data.std() - np.sqrt(2.0 / 100)) < 0.02


# -----------------------------------------------------------------------------
# Linear Layer & Gradchecks
# -----------------------------------------------------------------------------

def test_linear_forward_and_gradcheck():
    layer = nn.Linear(4, 3, bias=True)
    x = Tensor(np.random.randn(5, 4).astype(np.float32), requires_grad=True)

    out = layer(x)
    assert out.shape == (5, 3)

    def f(x_t, w_t, b_t):
        return x_t @ w_t + b_t

    assert gradcheck(f, [x, layer.weight, layer.bias])


def test_linear_no_bias():
    layer = nn.Linear(4, 2, bias=False)
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    out = layer(x)
    assert out.shape == (3, 2)
    assert layer.bias is None


# -----------------------------------------------------------------------------
# Sequential Container
# -----------------------------------------------------------------------------

def test_sequential_forward():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
        nn.Tanh(),
    )
    x = Tensor(np.random.randn(10, 8).astype(np.float32))
    out = model(x)
    assert out.shape == (10, 4)
    assert len(model.parameters()) == 4
    assert len(model) == 4


# -----------------------------------------------------------------------------
# Dropout Layer
# -----------------------------------------------------------------------------

def test_dropout_eval_mode():
    dropout = nn.Dropout(p=0.5)
    dropout.eval()
    x = Tensor(np.ones((20, 20), dtype=np.float32))
    out = dropout(x)
    np.testing.assert_array_equal(out.data, x.data)


def test_dropout_training_mode():
    np.random.seed(0)
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    x = Tensor(np.ones((100, 100), dtype=np.float32), requires_grad=True)
    out = dropout(x)

    zero_fraction = np.mean(out.data == 0.0)
    assert 0.4 < zero_fraction < 0.6  # Approx 50% dropped
    # Non-zero values scaled by 1 / (1 - 0.5) = 2.0
    non_zeros = out.data[out.data != 0.0]
    np.testing.assert_allclose(non_zeros, 2.0)

    out.backward()
    assert x.grad is not None


# -----------------------------------------------------------------------------
# BatchNorm1d Layer
# -----------------------------------------------------------------------------

def test_batchnorm1d_training():
    bn = nn.BatchNorm1d(4)
    bn.train()
    x = Tensor(np.random.randn(50, 4).astype(np.float32) * 3.0 + 5.0, requires_grad=True)
    out = bn(x)

    # In training mode, batch should have approx mean 0 and var 1
    np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(4), atol=1e-2)
    np.testing.assert_allclose(out.data.var(axis=0), np.ones(4), atol=1e-2)

    # Check running statistics updated
    assert not np.allclose(bn.running_mean, np.zeros(4))
    assert not np.allclose(bn.running_var, np.ones(4))


def test_batchnorm1d_eval():
    bn = nn.BatchNorm1d(3)
    bn.running_mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bn.running_var = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    bn.eval()

    x = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    out = bn(x)
    # Normalized should be (x - mean) / sqrt(var + eps) = (0) / 2 = 0
    np.testing.assert_allclose(out.data, np.zeros((1, 3)), atol=1e-4)


def test_batchnorm1d_gradcheck():
    bn = nn.BatchNorm1d(3, momentum=0.1)
    bn.train()
    x = Tensor(np.random.randn(8, 3).astype(np.float32), requires_grad=True)

    def f(x_t, w_t, b_t):
        mean = x_t.mean(axis=0, keepdims=True)
        diff = x_t - mean
        var = (diff ** 2.0).mean(axis=0, keepdims=True)
        x_norm = diff * ((var + 1e-5) ** -0.5)
        return (x_norm * w_t + b_t).sum()

    assert gradcheck(f, [x, bn.weight, bn.bias])


# -----------------------------------------------------------------------------
# Loss Functions & Gradchecks
# -----------------------------------------------------------------------------

def test_mseloss_gradcheck():
    loss_fn = nn.MSELoss()
    pred = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    target = np.random.randn(5, 3).astype(np.float32)

    def f(p):
        return loss_fn(p, target)

    assert gradcheck(f, [pred])


def test_cross_entropy_loss_gradcheck():
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    logits = Tensor(np.random.randn(6, 4).astype(np.float32), requires_grad=True)
    targets = np.array([0, 2, 1, 3, 0, 2], dtype=int)

    def f(z):
        return loss_fn(z, targets)

    assert gradcheck(f, [logits])


def test_cross_entropy_loss_numerical_stability():
    loss_fn = nn.CrossEntropyLoss()
    # Large logits should not overflow / NaN
    large_logits = Tensor(np.array([[1000.0, 1005.0, 990.0]], dtype=np.float32), requires_grad=True)
    target = np.array([1], dtype=int)

    loss = loss_fn(large_logits, target)
    assert not np.isnan(loss.data)
    assert not np.isinf(loss.data)
    loss.backward()
    assert not np.isnan(large_logits.grad).any()


def test_bce_with_logits_loss_gradcheck():
    loss_fn = nn.BCEWithLogitsLoss()
    logits = Tensor(np.random.randn(8, 2).astype(np.float32), requires_grad=True)
    target = np.random.choice([0.0, 1.0], size=(8, 2)).astype(np.float32)

    def f(z):
        return loss_fn(z, target)

    assert gradcheck(f, [logits])


# -----------------------------------------------------------------------------
# Optimizers & Convergence Tests
# -----------------------------------------------------------------------------

def test_sgd_quadratic_convergence():
    # Minimize f(w) = (w - 5)^2 => df/dw = 2(w - 5)
    w = nn.Parameter(np.array([0.0], dtype=np.float32))
    optimizer = optim.SGD([w], lr=0.05, momentum=0.8)

    for _ in range(100):
        optimizer.zero_grad()
        loss = (w - 5.0) ** 2.0
        loss.backward()
        optimizer.step()

    np.testing.assert_allclose(w.data, np.array([5.0]), atol=1e-2)


def test_adamw_quadratic_convergence():
    # Minimize f(w) = (w - 7)^2
    w = nn.Parameter(np.array([0.0], dtype=np.float32))
    optimizer = optim.AdamW([w], lr=0.1, weight_decay=0.0)

    for _ in range(200):
        optimizer.zero_grad()
        loss = (w - 7.0) ** 2.0
        loss.backward()
        optimizer.step()

    np.testing.assert_allclose(w.data, np.array([7.0]), atol=1e-2)


def test_mlp_training_loop_convergence():
    np.random.seed(42)
    # Synthetic classification task: 2 classes
    X = np.random.randn(40, 4).astype(np.float32)
    # Simple separable linear boundary
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    initial_loss = None
    final_loss = None

    for epoch in range(40):
        optimizer.zero_grad()
        x_tensor = Tensor(X)
        logits = model(x_tensor)
        loss = criterion(logits, y)

        if epoch == 0:
            initial_loss = float(loss.data)
        final_loss = float(loss.data)

        loss.backward()
        optimizer.step()

    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.1
