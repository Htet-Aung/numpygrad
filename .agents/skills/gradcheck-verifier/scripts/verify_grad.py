"""
Automated Centered Finite-Difference Gradcheck Diagnostic Utility.

Verifies analytical gradients of NumPyGrad operations, layers, and losses against
two-sided numerical approximations and outputs formatted diagnostic tables.
"""

from __future__ import annotations
import sys
import argparse
from typing import Callable, Sequence, List, Optional, Dict, Any
import numpy as np

# Ensure src/ is in sys.path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpygrad as ng
from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn


def run_gradcheck_diagnostic(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    names: Optional[Sequence[str]] = None,
    eps: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    title: str = "Gradcheck Verification",
) -> bool:
    """
    Computes analytical vs numerical gradients for each input tensor and displays
    a detailed diagnostic table.

    Parameters
    ----------
    func : Callable[..., Tensor]
        The differentiable function returning a scalar or reduction tensor.
    inputs : Sequence[Tensor]
        List of input tensors (with requires_grad=True).
    names : Optional[Sequence[str]]
        Human-readable labels for each input.
    eps : float
        Finite difference perturbation step size.
    rtol : float
        Relative error tolerance.
    atol : float
        Absolute error tolerance.
    title : str
        Header title for the diagnostic report.

    Returns
    -------
    bool
        True if all inputs passed tolerance thresholds, False otherwise.
    """
    if names is None:
        names = [f"Input[{i}]" for i in range(len(inputs))]

    # 1. Forward and analytical backward in float64 for high-precision validation
    active_inputs = [
        Tensor(t.data.copy(), requires_grad=t.requires_grad, dtype=np.float64)
        for t in inputs
    ]
    out = func(*active_inputs)
    loss = out.sum() if out.shape != () else out
    loss.backward()

    analytical_grads = [
        t.grad.copy() if t.grad is not None else np.zeros_like(t.data)
        for t in active_inputs
    ]

    # Print Table Header
    print(f"\n{'=' * 88}")
    print(f" DIAGNOSTIC REPORT: {title.upper()}")
    print(f"{'=' * 88}")
    print(
        f"{'Target / Parameter':<22} | {'Shape':<12} | {'Max Abs Diff':<14} | "
        f"{'Rel Error':<12} | {'Norm (A/N)':<14} | {'Status':<8}"
    )
    print(f"{'-' * 22}-+-{'-' * 12}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 14}-+-{'-' * 8}")

    all_passed = True

    # 2. Centered finite-difference evaluation per input
    for idx, (name, t, a_grad) in enumerate(zip(names, inputs, analytical_grads)):
        if not t.requires_grad:
            shape_str = str(t.shape)
            print(f"{name:<22} | {shape_str:<12} | {'N/A (no grad)':<14} | {'N/A':<12} | {'N/A':<14} | {'SKIPPED':<8}")
            continue

        num_grad = np.zeros_like(t.data, dtype=np.float64)
        it = np.nditer(t.data, flags=["multi_index"])

        while not it.finished:
            m_idx = it.multi_index

            # f(x + eps)
            pos_inputs = [
                Tensor(other.data.copy(), requires_grad=False, dtype=np.float64)
                for other in inputs
            ]
            pos_inputs[idx].data[m_idx] += eps
            pos_out = func(*pos_inputs)
            pos_loss = float(pos_out.data.sum()) if pos_out.shape != () else float(pos_out.data)

            # f(x - eps)
            neg_inputs = [
                Tensor(other.data.copy(), requires_grad=False, dtype=np.float64)
                for other in inputs
            ]
            neg_inputs[idx].data[m_idx] -= eps
            neg_out = func(*neg_inputs)
            neg_loss = float(neg_out.data.sum()) if neg_out.shape != () else float(neg_out.data)

            # Central finite difference formula
            num_grad[m_idx] = (pos_loss - neg_loss) / (2.0 * eps)
            it.iternext()

        a_grad_64 = a_grad.astype(np.float64)
        abs_diff = np.abs(a_grad_64 - num_grad)
        max_abs_diff = float(np.max(abs_diff))

        norm_a = float(np.linalg.norm(a_grad_64))
        norm_n = float(np.linalg.norm(num_grad))
        rel_error = float(np.linalg.norm(a_grad_64 - num_grad) / (norm_a + norm_n + 1e-8))

        is_close = np.allclose(a_grad_64, num_grad, rtol=rtol, atol=atol)
        status = "PASSED" if is_close else "FAILED"
        if not is_close:
            all_passed = False

        shape_str = str(t.shape)
        norm_str = f"{norm_a:.2f}/{norm_n:.2f}"
        print(
            f"{name:<22} | {shape_str:<12} | {max_abs_diff:<14.6e} | "
            f"{rel_error:<12.6e} | {norm_str:<14} | {status:<8}"
        )

    print(f"{'=' * 88}\n")
    return all_passed


# =============================================================================
# Built-in Primitive Targets for Verification
# =============================================================================

def verify_linear(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    layer = nn.Linear(in_features=6, out_features=4, bias=True)
    x = Tensor(np.random.randn(8, 6).astype(np.float32), requires_grad=True)

    def f(x_t, w_t, b_t):
        return (x_t @ w_t + b_t).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x, layer.weight, layer.bias],
        names=["Input X (8, 6)", "Weight W (6, 4)", "Bias b (4,)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Linear Layer (Affine Transformation)",
    )


def verify_relu(eps: float, rtol: float, atol: float) -> bool:
    # Use non-zero points to avoid the non-differentiable kink at 0
    data = np.array([[-2.5, -0.8, 0.4], [1.2, 3.1, -1.9]], dtype=np.float32)
    x = Tensor(data, requires_grad=True)

    def f(x_t):
        return x_t.relu().sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Input X (2, 3)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="ReLU Activation Layer",
    )


def verify_cross_entropy(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    logits = Tensor(np.random.randn(6, 5).astype(np.float32), requires_grad=True)
    targets = np.array([0, 2, 4, 1, 3, 0], dtype=np.int64)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def f(z):
        return loss_fn(z, targets)

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[logits],
        names=["Logits (6, 5)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="CrossEntropyLoss (Log-Sum-Exp Softmax)",
    )


def verify_batchnorm(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    bn = nn.BatchNorm1d(num_features=4)
    bn.train()
    x = Tensor(np.random.randn(8, 4).astype(np.float32) * 2.0 + 1.0, requires_grad=True)

    def f(x_t, w_t, b_t):
        mean = x_t.mean(axis=0, keepdims=True)
        diff = x_t - mean
        var = (diff ** 2.0).mean(axis=0, keepdims=True)
        x_norm = diff * ((var + bn.eps) ** -0.5)
        return (x_norm * w_t + b_t).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x, bn.weight, bn.bias],
        names=["Input X (8, 4)", "Gamma Weight (4,)", "Beta Bias (4,)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="BatchNorm1d Layer",
    )


def verify_slicing(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    x = Tensor(np.random.randn(5, 4, 3).astype(np.float32), requires_grad=True)

    def f(x_t):
        return (x_t[1:4, 0:2, :] ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Tensor X (5, 4, 3)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Tensor Slicing & Sub-Indexing",
    )


def verify_flatten(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    layer = nn.Flatten(start_dim=1, end_dim=-1)
    x = Tensor(np.random.randn(4, 3, 4, 4).astype(np.float64), requires_grad=True)

    def f(x_t):
        return (layer(x_t) ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Tensor X (4, 3, 4, 4)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Flatten Layer (4D to 2D)",
    )


def verify_conv2d(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    conv = nn.Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
    conv.weight.data = conv.weight.data.astype(np.float64)
    conv.bias.data = conv.bias.data.astype(np.float64)
    x = Tensor(np.random.randn(2, 2, 5, 5).astype(np.float64), requires_grad=True)

    def f(x_t, w_t, b_t):
        conv.weight = w_t
        conv.bias = b_t
        out = conv(x_t)
        return (out ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x, conv.weight, conv.bias],
        names=["Input X (2, 2, 5, 5)", "Conv2D Weight (3, 2, 3, 3)", "Conv2D Bias (3,)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Conv2D Layer (Vectorized im2col/col2im)",
    )


def verify_maxpool2d(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
    # Ensure distinct values to avoid subgradient ambiguity at ties
    flat = np.arange(2 * 2 * 6 * 6, dtype=np.float64)
    np.random.shuffle(flat)
    x_data = flat.reshape(2, 2, 6, 6)
    x = Tensor(x_data, requires_grad=True)

    def f(x_t):
        out = pool(x_t)
        return (out ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Input X (2, 2, 6, 6)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="MaxPool2D Layer (Spatial Max Routing)",
    )


def verify_squeeze(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 1, 3, 1, 4).astype(np.float64), requires_grad=True)

    def f(x_t):
        return (x_t.squeeze() ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Tensor X (2, 1, 3, 1, 4)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Tensor Squeeze (Singleton Removal)",
    )


def verify_unsqueeze(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)

    def f(x_t):
        return (x_t.unsqueeze(1) ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[x],
        names=["Tensor X (3, 4)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Tensor Unsqueeze (Singleton Insertion)",
    )


def verify_concat(eps: float, rtol: float, atol: float) -> bool:
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
    b = Tensor(np.random.randn(3, 2).astype(np.float64), requires_grad=True)
    c = Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)

    def f(a_t, b_t, c_t):
        out = ng.concat([a_t, b_t, c_t], axis=1)
        return (out ** 2.0).sum()

    return run_gradcheck_diagnostic(
        func=f,
        inputs=[a, b, c],
        names=["Tensor A (3, 4)", "Tensor B (3, 2)", "Tensor C (3, 3)"],
        eps=eps,
        rtol=rtol,
        atol=atol,
        title="Tensor Concat (Multi-Tensor Join)",
    )


def main():
    parser = argparse.ArgumentParser(description="NumPyGrad Numerical Gradient Verifier")
    parser.add_argument(
        "--layer",
        type=str,
        default="all",
        choices=["all", "linear", "relu", "cross_entropy", "batchnorm", "slicing", "flatten", "conv2d", "maxpool2d", "squeeze", "unsqueeze", "concat"],
        help="Target operation or layer to verify (default: all)",
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Finite difference perturbation epsilon")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative error tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute error tolerance")

    args = parser.parse_args()

    targets: Dict[str, Callable[[float, float, float], bool]] = {
        "linear": verify_linear,
        "relu": verify_relu,
        "cross_entropy": verify_cross_entropy,
        "batchnorm": verify_batchnorm,
        "slicing": verify_slicing,
        "flatten": verify_flatten,
        "conv2d": verify_conv2d,
        "maxpool2d": verify_maxpool2d,
        "squeeze": verify_squeeze,
        "unsqueeze": verify_unsqueeze,
        "concat": verify_concat,
    }

    if args.layer == "all":
        results = [fn(args.eps, args.rtol, args.atol) for fn in targets.values()]
        overall_success = all(results)
    else:
        overall_success = targets[args.layer](args.eps, args.rtol, args.atol)

    if overall_success:
        print("[SUCCESS] All analytical gradients successfully verified against numerical finite differences!")
        sys.exit(0)
    else:
        print("[ERROR] Gradient verification failed for one or more operations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
