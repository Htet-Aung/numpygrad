"""
Flagship MNIST CNN Training, Evaluation, and Comparison Pipeline.

Demonstrates:
- Vectorized 2D Convolution (im2col/col2im) and Max Pooling in pure NumPy.
- End-to-end CNN training on MNIST reaching high accuracy with fewer parameters than MLP.
- Side-by-side architectural and performance comparison between MLP and CNN.
- Model persistence to `examples/mnist_cnn.ng` and reload verification.
"""

from __future__ import annotations
import os
import sys
import time
import gzip
import struct
import urllib.request
import numpy as np

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpygrad as ng
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.data import Dataset, DataLoader
from numpygrad.metrics import accuracy
from numpygrad.core.tensor import Tensor, no_grad
from numpygrad.serialization import save_model, load_model


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_and_extract_mnist(cache_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads and loads the MNIST dataset into NumPy arrays with local disk caching.
    Reshapes images into (N, 1, 28, 28) 4D tensor format for convolution.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".numpygrad", "mnist")
    os.makedirs(cache_dir, exist_ok=True)

    files = {}
    for key, url in MNIST_URLS.items():
        filename = os.path.basename(url)
        filepath = os.path.join(cache_dir, filename)

        if not os.path.exists(filepath):
            print(f"[DOWNLOAD] Fetching {filename} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "NumPyGrad/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(filepath, "wb") as f:
                f.write(resp.read())

        files[key] = filepath

    # Parse Training Images (60,000 x 28 x 28) -> (60,000, 1, 28, 28)
    with gzip.open(files["train_images"], "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 1, rows, cols)

    # Parse Training Labels (60,000,)
    with gzip.open(files["train_labels"], "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # Parse Test Images (10,000 x 28 x 28) -> (10,000, 1, 28, 28)
    with gzip.open(files["test_images"], "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 1, rows, cols)

    # Parse Test Labels (10,000,)
    with gzip.open(files["test_labels"], "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # Normalize to [0.0, 1.0] float32
    X_train = (train_images.astype(np.float32) / 255.0)
    y_train = train_labels
    X_test = (test_images.astype(np.float32) / 255.0)
    y_test = test_labels

    return X_train, y_train, X_test, y_test


def evaluate_model(model: nn.Module, data_loader: DataLoader) -> tuple[float, float]:
    """
    Evaluates the model over a DataLoader and returns (avg_loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with no_grad():
        for batch_x, batch_y in data_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y.data)
            batch_size = len(batch_y)
            total_loss += float(loss.data) * batch_size

            preds = np.argmax(logits.data, axis=-1)
            total_correct += int(np.sum(preds == batch_y.data))
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def measure_step_latency(model: nn.Module, sample_batch: Tensor, targets: np.ndarray) -> tuple[float, float]:
    """Measures average forward and backward step latency in milliseconds."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Warm-up
    for _ in range(2):
        optimizer.zero_grad()
        out = model(sample_batch)
        loss = criterion(out, targets)
        loss.backward()

    # Benchmark forward
    n_iters = 10
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(sample_batch)
    t_fwd = (time.perf_counter() - t0) / n_iters * 1000.0

    # Benchmark backward
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad()
        out = model(sample_batch)
        loss = criterion(out, targets)
        loss.backward()
    t_bwd = (time.perf_counter() - t0) / n_iters * 1000.0 - t_fwd

    return t_fwd, max(0.0, t_bwd)


def train_mnist_cnn(epochs: int = 3, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
    np.random.seed(seed)
    print("=" * 80)
    print("             NUMPYGRAD CONVOLUTIONAL NEURAL NETWORK (CNN) PIPELINE")
    print("=" * 80)

    # 1. Ingest Data
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = download_and_extract_mnist()
    t_load = time.perf_counter() - t0
    print(f"[DATA] MNIST loaded in {t_load:.2f}s | Train: {X_train.shape} | Test: {X_test.shape}")

    # Build Datasets and DataLoaders
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. Instantiate CNN Architecture
    # Input: (B, 1, 28, 28)
    # Conv1: (B, 8, 28, 28) -> ReLU -> Pool1: (B, 8, 14, 14)
    # Conv2: (B, 16, 14, 14) -> ReLU -> Pool2: (B, 16, 7, 7)
    # Flatten -> (B, 784) -> Linear(784, 64) -> ReLU -> Linear(64, 10)
    cnn_model = nn.Sequential(
        nn.Conv2D(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Conv2D(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    cnn_params = sum(p.data.size for p in cnn_model.parameters())
    print(f"[MODEL] CNN Architecture: Conv2D(1->8, 3x3) -> MaxPool(2) -> Conv2D(8->16, 3x3) -> MaxPool(2) -> Flatten -> Linear(784, 64) -> Linear(64, 10)")
    print(f"[MODEL] Total CNN Parameters: {cnn_params:,}")

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Measure step latency on sample batch
    sample_x = Tensor(X_train[:batch_size])
    sample_y = y_train[:batch_size]
    cnn_fwd_ms, cnn_bwd_ms = measure_step_latency(cnn_model, sample_x, sample_y)
    print(f"[LATENCY] CNN Step Time (batch={batch_size}): Forward {cnn_fwd_ms:.2f}ms | Backward {cnn_bwd_ms:.2f}ms | Total {cnn_fwd_ms + cnn_bwd_ms:.2f}ms")

    print("-" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Test Loss':^11} | {'Test Acc':^10} | {'Throughput':^14} | {'Time':^7}")
    print("-" * 80)

    # 5. Training Loop
    total_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        cnn_model.train()
        epoch_loss = 0.0
        samples_processed = 0
        t_epoch_start = time.perf_counter()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = cnn_model(batch_x)
            loss = criterion(logits, batch_y.data)
            loss.backward()
            optimizer.step()

            batch_n = len(batch_y)
            epoch_loss += float(loss.data) * batch_n
            samples_processed += batch_n

        t_epoch = time.perf_counter() - t_epoch_start
        throughput = samples_processed / max(1e-6, t_epoch)
        avg_train_loss = epoch_loss / max(1, samples_processed)

        # Evaluate on test set
        test_loss, test_acc = evaluate_model(cnn_model, test_loader)

        print(
            f"{epoch:^7d} | {avg_train_loss:^12.4f} | {test_loss:^11.4f} | "
            f"{test_acc * 100:^9.2f}% | {throughput:^10.1f} smp/s | {t_epoch:^6.2f}s"
        )

    total_time = time.perf_counter() - total_train_start
    print("-" * 80)
    print(f"[DONE] Completed {epochs} epochs in {total_time:.2f}s | Final CNN Test Accuracy: {test_acc * 100:.2f}%")

    # 6. Benchmark Comparison with MLP
    mlp_model = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    mlp_params = sum(p.data.size for p in mlp_model.parameters())
    mlp_sample_x = Tensor(X_train[:batch_size].reshape(batch_size, 28, 28))
    mlp_fwd_ms, mlp_bwd_ms = measure_step_latency(mlp_model, mlp_sample_x, sample_y)

    print("\n" + "=" * 80)
    print("                ARCHITECTURAL COMPARISON: MLP vs. CNN")
    print("=" * 80)
    print(f"{'Metric':<30} | {'MLP (Milestone 4)':<22} | {'CNN (Milestone 6)':<20}")
    print("-" * 80)
    print(f"{'Trainable Parameters':<30} | {mlp_params:<22,} | {cnn_params:<20,}")
    print(f"{'Parameter Reduction':<30} | {'Baseline':<22} | {f'-{(1 - cnn_params / mlp_params)*100:.1f}%':<20}")
    print(f"{'Test Accuracy':<30} | {'97.55%':<22} | {f'{test_acc * 100:.2f}%':<20}")
    print(f"{'Forward Step Latency':<30} | {f'{mlp_fwd_ms:.2f} ms':<22} | {f'{cnn_fwd_ms:.2f} ms':<20}")
    print(f"{'Backward Step Latency':<30} | {f'{mlp_bwd_ms:.2f} ms':<22} | {f'{cnn_bwd_ms:.2f} ms':<20}")
    print(f"{'Total Step Latency':<30} | {f'{mlp_fwd_ms + mlp_bwd_ms:.2f} ms':<22} | {f'{cnn_fwd_ms + cnn_bwd_ms:.2f} ms':<20}")
    print("=" * 80)

    # 7. Model Persistence to .ng container
    artifact_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.ng")
    cnn_model.save(artifact_path)
    print(f"\n[SAVE] CNN model serialized successfully to: {artifact_path}")

    # 8. Reload & Verify Round-trip Inference
    loaded_model = load_model(artifact_path)
    loaded_model.eval()
    _, reloaded_acc = evaluate_model(loaded_model, test_loader)
    print(f"[RELOAD] Model successfully reloaded. Round-trip Test Accuracy: {reloaded_acc * 100:.2f}%\n")
    assert abs(reloaded_acc - test_acc) < 1e-6, "Reloaded model accuracy mismatch!"

    return cnn_model, test_acc


if __name__ == "__main__":
    train_mnist_cnn()
