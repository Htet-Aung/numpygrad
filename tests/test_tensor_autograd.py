"""
Comprehensive unit tests and gradient checks for NumPyGrad core Tensor and Autograd engine.
"""

import numpy as np
import pytest
from numpygrad.core.tensor import Tensor, _unbroadcast
from numpygrad.utils.gradcheck import gradcheck


# -----------------------------------------------------------------------------
# Unbroadcasting Logic Tests
# -----------------------------------------------------------------------------

def test_unbroadcast_identical_shape():
    grad = np.ones((2, 3), dtype=np.float32)
    out = _unbroadcast(grad, (2, 3))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, grad)


def test_unbroadcast_to_scalar():
    grad = np.full((3, 4), 2.0, dtype=np.float32)
    out = _unbroadcast(grad, ())
    assert out.shape == ()
    assert float(out) == 24.0


def test_unbroadcast_leading_dims():
    grad = np.ones((2, 4, 3), dtype=np.float32)
    out = _unbroadcast(grad, (3,))
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.full((3,), 8.0, dtype=np.float32))


def test_unbroadcast_singleton_dims():
    grad = np.ones((2, 3), dtype=np.float32)
    out = _unbroadcast(grad, (2, 1))
    assert out.shape == (2, 1)
    np.testing.assert_array_equal(out, np.full((2, 1), 3.0, dtype=np.float32))


def test_unbroadcast_complex_multidim():
    grad = np.ones((5, 2, 4, 3), dtype=np.float32)
    out = _unbroadcast(grad, (1, 4, 1))
    assert out.shape == (1, 4, 1)
    np.testing.assert_array_equal(out, np.full((1, 4, 1), 30.0, dtype=np.float32))


# -----------------------------------------------------------------------------
# Elementary Arithmetic Operations & Gradchecks
# -----------------------------------------------------------------------------

def test_add_gradcheck():
    def f(x, y):
        return x + y

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_add_broadcast_gradcheck():
    def f(x, y):
        return x + y

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_radd_gradcheck():
    def f(x):
        return 3.5 + x

    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_sub_gradcheck():
    def f(x, y):
        return x - y

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_rsub_gradcheck():
    def f(x):
        return 5.0 - x

    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_mul_gradcheck():
    def f(x, y):
        return x * y

    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_rmul_gradcheck():
    def f(x):
        return 2.5 * x

    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_mul_broadcast_gradcheck():
    def f(x, y):
        return x * y

    x = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(1, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_div_gradcheck():
    def f(x, y):
        return x / y

    x = Tensor(np.random.uniform(1.0, 3.0, size=(3, 2)).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.uniform(1.0, 3.0, size=(3, 2)).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_rdiv_gradcheck():
    def f(x):
        return 4.0 / x

    x = Tensor(np.random.uniform(1.0, 3.0, size=(2, 2)).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_matmul_gradcheck():
    def f(x, y):
        return x @ y

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_batched_matmul_gradcheck():
    def f(x, y):
        return x @ y

    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(2, 4, 5).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, y])


def test_pow_constant_gradcheck():
    def f(x):
        return x ** 3.0

    x = Tensor(np.random.uniform(1.0, 3.0, size=(3, 3)).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_pow_tensor_gradcheck():
    def f(x, p):
        return x ** p

    x = Tensor(np.random.uniform(1.0, 3.0, size=(2, 3)).astype(np.float32), requires_grad=True)
    p = Tensor(np.random.uniform(1.0, 3.0, size=(2, 3)).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x, p])


def test_neg_gradcheck():
    def f(x):
        return -x

    x = Tensor(np.random.randn(3, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


# -----------------------------------------------------------------------------
# Activation Functions & Gradchecks
# -----------------------------------------------------------------------------

def test_relu_gradcheck():
    def f(x):
        return x.relu()

    # Avoid exact 0 for clean non-subgradient finite differences
    data = np.array([[-2.0, -0.5, 0.5], [1.5, 2.5, -1.2]], dtype=np.float32)
    x = Tensor(data, requires_grad=True)
    assert gradcheck(f, [x])


def test_sigmoid_gradcheck():
    def f(x):
        return x.sigmoid()

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_sigmoid_numerical_stability():
    # Extreme values should not produce NaN
    large_vals = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    x = Tensor(large_vals, requires_grad=True)
    out = x.sigmoid()
    assert not np.isnan(out.data).any()
    assert not np.isinf(out.data).any()
    np.testing.assert_allclose(out.data[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(out.data[-1], 1.0, atol=1e-7)
    out.backward()
    assert not np.isnan(x.grad).any()


def test_tanh_gradcheck():
    def f(x):
        return x.tanh()

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


# -----------------------------------------------------------------------------
# Reductions & Reshaping
# -----------------------------------------------------------------------------

def test_sum_all_gradcheck():
    def f(x):
        return x.sum()

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_sum_axis_gradcheck():
    def f(x):
        return x.sum(axis=1)

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_sum_multiple_axes_gradcheck():
    def f(x):
        return x.sum(axis=(0, 2), keepdims=True)

    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_mean_all_gradcheck():
    def f(x):
        return x.mean()

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_mean_axis_gradcheck():
    def f(x):
        return x.mean(axis=0)

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_mean_multiple_axes_gradcheck():
    def f(x):
        return x.mean(axis=(1, 2))

    x = Tensor(np.random.randn(2, 4, 3).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_reshape_gradcheck():
    def f(x):
        return x.reshape(2, 6)

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_transpose_gradcheck():
    def f(x):
        return x.transpose(1, 0)

    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


def test_transpose_3d_gradcheck():
    def f(x):
        return x.transpose(2, 0, 1)

    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    assert gradcheck(f, [x])


# -----------------------------------------------------------------------------
# Complex DAG, Requires-Grad & Multi-Path Gradient Accumulation
# -----------------------------------------------------------------------------

def test_requires_grad_behavior():
    a = Tensor(np.array([2.0, 3.0]), requires_grad=False)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = a * b
    c.backward()

    assert a.grad is None
    assert b.grad is not None
    np.testing.assert_array_equal(b.grad, a.data)


def test_zero_grad():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_multipath_dag_gradient_accumulation():
    # y = x * x + x => dy/dx = 2x + 1
    x_val = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    x = Tensor(x_val, requires_grad=True)
    y = x * x + x
    y.backward()

    expected_grad = 2.0 * x_val + 1.0
    np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_deep_neural_network_dag_gradcheck():
    # Simulate a mini 2-layer neural network forward & backward
    np.random.seed(42)
    x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=False)
    w1 = Tensor(np.random.randn(8, 6).astype(np.float32) * 0.1, requires_grad=True)
    b1 = Tensor(np.zeros(6, dtype=np.float32), requires_grad=True)
    w2 = Tensor(np.random.randn(6, 2).astype(np.float32) * 0.1, requires_grad=True)
    b2 = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    target = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=False)

    def model_loss(w1_t, b1_t, w2_t, b2_t):
        h = (x @ w1_t + b1_t).tanh()
        logits = h @ w2_t + b2_t
        diff = logits - target
        loss = (diff * diff).mean()
        return loss

    assert gradcheck(model_loss, [w1, b1, w2, b2], eps=1e-5, atol=1e-4, rtol=1e-3)
