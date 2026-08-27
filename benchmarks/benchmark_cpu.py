"""
NumPyGrad vs. PyTorch CPU Workload Benchmark.

Evaluates forward/backward pass throughput, per-step latency, and loss convergence
between NumPyGrad (Pure NumPy) and PyTorch (CPU) on identical MLP architectures.
"""

import sys
import os
import time
from typing import Dict, Any, List

# Ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim

# On Windows, register system DLL directories for PyTorch C++ runtime
if sys.platform == "win32":
    for path in [
        r"C:\Python314",
        os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"),
    ]:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def run_numpygrad_benchmark(
    batch_size: int = 128,
    in_features: int = 32,
    hidden_dim: int = 128,
    out_features: int = 10,
    num_steps: int = 100,
    warmup_steps: int = 20,
) -> Dict[str, Any]:
    """Benchmarks NumPyGrad on an MLP workload."""
    # Synthetic batch
    np.random.seed(42)
    X_np = np.random.randn(batch_size, in_features).astype(np.float32)
    y_np = np.random.randint(0, out_features, size=(batch_size,))

    # 3-Layer MLP
    model = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_features),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    total_params = sum(p.data.size for p in model.parameters())

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(Tensor(X_np))
        loss = criterion(logits, y_np)
        loss.backward()
        optimizer.step()

    # Benchmark Timings
    forward_times: List[float] = []
    backward_times: List[float] = []
    total_step_times: List[float] = []
    final_loss = 0.0

    for _ in range(num_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        # Forward
        t_fwd_start = time.perf_counter()
        X_tensor = Tensor(X_np)
        logits = model(X_tensor)
        loss = criterion(logits, y_np)
        t_fwd_end = time.perf_counter()
        forward_times.append(t_fwd_end - t_fwd_start)

        # Backward
        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd_end = time.perf_counter()
        backward_times.append(t_bwd_end - t_bwd_start)

        # Optimize Step
        optimizer.step()
        t_step_end = time.perf_counter()
        total_step_times.append(t_step_end - t0)
        final_loss = float(loss.data)

    avg_fwd_ms = np.mean(forward_times) * 1000.0
    avg_bwd_ms = np.mean(backward_times) * 1000.0
    avg_step_ms = np.mean(total_step_times) * 1000.0
    throughput = (batch_size * num_steps) / sum(total_step_times)

    return {
        "framework": "NumPyGrad (Pure NumPy)",
        "params": total_params,
        "fwd_ms": avg_fwd_ms,
        "bwd_ms": avg_bwd_ms,
        "step_ms": avg_step_ms,
        "throughput_samples_sec": throughput,
        "final_loss": final_loss,
    }


def run_pytorch_benchmark(
    batch_size: int = 128,
    in_features: int = 32,
    hidden_dim: int = 128,
    out_features: int = 10,
    num_steps: int = 100,
    warmup_steps: int = 20,
) -> Dict[str, Any]:
    """Benchmarks PyTorch CPU on the identical MLP workload."""
    if not TORCH_AVAILABLE:
        return {}

    torch.manual_seed(42)
    X_torch = torch.randn(batch_size, in_features, dtype=torch.float32)
    y_torch = torch.randint(0, out_features, (batch_size,), dtype=torch.long)

    model = torch_nn.Sequential(
        torch_nn.Linear(in_features, hidden_dim),
        torch_nn.ReLU(),
        torch_nn.Linear(hidden_dim, hidden_dim),
        torch_nn.ReLU(),
        torch_nn.Linear(hidden_dim, out_features),
    )
    criterion = torch_nn.CrossEntropyLoss()
    optimizer = torch_optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(X_torch)
        loss = criterion(logits, y_torch)
        loss.backward()
        optimizer.step()

    # Benchmark Timings
    forward_times: List[float] = []
    backward_times: List[float] = []
    total_step_times: List[float] = []
    final_loss = 0.0

    for _ in range(num_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        # Forward
        t_fwd_start = time.perf_counter()
        logits = model(X_torch)
        loss = criterion(logits, y_torch)
        t_fwd_end = time.perf_counter()
        forward_times.append(t_fwd_end - t_fwd_start)

        # Backward
        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd_end = time.perf_counter()
        backward_times.append(t_bwd_end - t_bwd_start)

        # Optimize Step
        optimizer.step()
        t_step_end = time.perf_counter()
        total_step_times.append(t_step_end - t0)
        final_loss = float(loss.item())

    avg_fwd_ms = np.mean(forward_times) * 1000.0
    avg_bwd_ms = np.mean(backward_times) * 1000.0
    avg_step_ms = np.mean(total_step_times) * 1000.0
    throughput = (batch_size * num_steps) / sum(total_step_times)

    return {
        "framework": "PyTorch (CPU)",
        "params": total_params,
        "fwd_ms": avg_fwd_ms,
        "bwd_ms": avg_bwd_ms,
        "step_ms": avg_step_ms,
        "throughput_samples_sec": throughput,
        "final_loss": final_loss,
    }


def print_comparison_table(res_numpygrad: Dict[str, Any], res_pytorch: Dict[str, Any]):
    """Outputs a formatted Markdown comparison table."""
    print("\n" + "=" * 82)
    print("                NumPyGrad vs. PyTorch CPU Workload Benchmark")
    print("=" * 82)
    print(f"Batch Size: 128 | Input Dim: 32 | Hidden Dim: 128 | Output Dim: 10 | Steps: 100\n")

    headers = ["Metric / Specification", "NumPyGrad (Pure NumPy)", "PyTorch (CPU)"]
    row_fmt = "| {:<30} | {:<22} | {:<22} |"

    print("|" + "-" * 32 + "|" + "-" * 24 + "|" + "-" * 24 + "|")
    print(row_fmt.format(headers[0], headers[1], headers[2]))
    print("|" + "-" * 32 + "|" + "-" * 24 + "|" + "-" * 24 + "|")

    params_ng = f"{res_numpygrad['params']:,}"
    params_pt = f"{res_pytorch.get('params', 'N/A'):,}" if res_pytorch else "N/A"
    print(row_fmt.format("Trainable Parameters", params_ng, params_pt))

    fwd_ng = f"{res_numpygrad['fwd_ms']:.3f} ms"
    fwd_pt = f"{res_pytorch.get('fwd_ms', 0):.3f} ms" if res_pytorch else "N/A"
    print(row_fmt.format("Forward Pass Latency", fwd_ng, fwd_pt))

    bwd_ng = f"{res_numpygrad['bwd_ms']:.3f} ms"
    bwd_pt = f"{res_pytorch.get('bwd_ms', 0):.3f} ms" if res_pytorch else "N/A"
    print(row_fmt.format("Backward Pass Latency", bwd_ng, bwd_pt))

    step_ng = f"{res_numpygrad['step_ms']:.3f} ms"
    step_pt = f"{res_pytorch.get('step_ms', 0):.3f} ms" if res_pytorch else "N/A"
    print(row_fmt.format("Full Step (Fwd+Bwd+Opt)", step_ng, step_pt))

    tp_ng = f"{res_numpygrad['throughput_samples_sec']:,.1f} samples/s"
    tp_pt = f"{res_pytorch.get('throughput_samples_sec', 0):,.1f} samples/s" if res_pytorch else "N/A"
    print(row_fmt.format("Training Throughput", tp_ng, tp_pt))

    loss_ng = f"{res_numpygrad['final_loss']:.4f}"
    loss_pt = f"{res_pytorch.get('final_loss', 0):.4f}" if res_pytorch else "N/A"
    print(row_fmt.format("Final Loss (100 steps)", loss_ng, loss_pt))

    print("|" + "-" * 32 + "|" + "-" * 24 + "|" + "-" * 24 + "|")
    print("=" * 82)


def main():
    print("[BENCHMARK] Executing NumPyGrad CPU benchmark...")
    res_ng = run_numpygrad_benchmark()

    res_pt = {}
    if TORCH_AVAILABLE:
        print("[BENCHMARK] Executing PyTorch CPU benchmark...")
        res_pt = run_pytorch_benchmark()
    else:
        print("[NOTICE] PyTorch not installed. Comparative timing omitted.")

    print_comparison_table(res_ng, res_pt)


if __name__ == "__main__":
    main()
