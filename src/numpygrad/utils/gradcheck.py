"""
Numerical Gradient Checking Utility.

Implements two-sided finite-difference approximations to mathematically
verify analytical gradient implementations:
    df/dx_i ≈ (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
"""

from typing import Callable, Sequence, Union, List
import numpy as np
from numpygrad.core.tensor import Tensor


def gradcheck(
    func: Callable[..., Tensor],
    inputs: Union[Tensor, Sequence[Tensor]],
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """
    Validates analytical gradients of `func` with respect to `inputs` using
    centered finite-difference approximations.

    Parameters
    ----------
    func : Callable[..., Tensor]
        A differentiable function taking Tensor arguments and returning a Tensor.
    inputs : Union[Tensor, Sequence[Tensor]]
        Input tensor(s) to compute gradients with respect to.
    eps : float, default=1e-5
        Finite difference step size.
    atol : float, default=1e-4
        Absolute error tolerance threshold.
    rtol : float, default=1e-3
        Relative error tolerance threshold.
    raise_exception : bool, default=True
        Whether to raise an AssertionError on verification failure.

    Returns
    -------
    bool
        True if all analytical gradients match numerical gradients within tolerances.
    """
    if isinstance(inputs, Tensor):
        tensor_inputs: List[Tensor] = [inputs]
    else:
        tensor_inputs = list(inputs)

    # 1. Forward and analytical backward in float64 for exact precision verification
    active_inputs = [
        Tensor(t.data.copy(), requires_grad=t.requires_grad, dtype=np.float64)
        for t in tensor_inputs
    ]
    out = func(*active_inputs)
    loss = out.sum() if out.shape != () else out
    loss.backward()

    # Collect analytical gradients
    analytical_grads = [
        t.grad.copy() if t.grad is not None else np.zeros_like(t.data)
        for t in active_inputs
    ]

    # 2. Compute numerical gradients via centered finite differences
    for idx, (t, a_grad) in enumerate(zip(tensor_inputs, analytical_grads)):
        if not t.requires_grad:
            continue

        num_grad = np.zeros_like(t.data, dtype=np.float64)
        it = np.nditer(t.data, flags=["multi_index"])

        while not it.finished:
            m_idx = it.multi_index

            # f(x + eps)
            pos_inputs = [
                Tensor(other.data.copy(), requires_grad=False, dtype=np.float64)
                for other in tensor_inputs
            ]
            pos_inputs[idx].data[m_idx] += eps
            pos_out = func(*pos_inputs)
            pos_loss = float(pos_out.data.sum()) if pos_out.shape != () else float(pos_out.data)

            # f(x - eps)
            neg_inputs = [
                Tensor(other.data.copy(), requires_grad=False, dtype=np.float64)
                for other in tensor_inputs
            ]
            neg_inputs[idx].data[m_idx] -= eps
            neg_out = func(*neg_inputs)
            neg_loss = float(neg_out.data.sum()) if neg_out.shape != () else float(neg_out.data)

            # Central difference formula
            num_grad[m_idx] = (pos_loss - neg_loss) / (2.0 * eps)
            it.iternext()

        # Compare analytical vs numerical gradients
        a_grad_64 = a_grad.astype(np.float64)
        denom = np.maximum(np.abs(a_grad_64), np.abs(num_grad)) + 1e-15
        rel_error = np.abs(a_grad_64 - num_grad) / denom
        max_rel_error = float(np.max(rel_error))
        abs_error = np.abs(a_grad_64 - num_grad)
        max_abs_error = float(np.max(abs_error))

        is_close = np.allclose(a_grad_64, num_grad, rtol=rtol, atol=atol)
        if not is_close:
            msg = (
                f"Gradcheck failed for input index {idx}!\n"
                f"Max relative error: {max_rel_error:.6e} (tolerance rtol={rtol})\n"
                f"Max absolute error: {max_abs_error:.6e} (tolerance atol={atol})\n"
                f"Analytical Grad:\n{a_grad_64}\n"
                f"Numerical Grad:\n{num_grad}"
            )
            if raise_exception:
                raise AssertionError(msg)
            return False

    return True
