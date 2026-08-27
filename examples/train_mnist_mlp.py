"""
Flagship MNIST Training and Evaluation Pipeline.

Demonstrates:
- Resilient ingestion and caching of the standard 70,000-sample MNIST dataset.
- Pure NumPy data loading, normalization ([0.0, 1.0]), batching, and shuffling.
- Deep MLP architecture with Flatten, Linear, and ReLU layers.
- Fast, vector-accelerated training using CrossEntropyLoss and AdamW optimizer.
- Validation achieving >=95.0% accuracy on the 10,000-sample test set.
- Standalone model persistence to `examples/mnist_mlp.ng` and reload verification.
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

    # Parse Training Images (60,000 x 28 x 28)
    with gzip.open(files["train_images"], "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    # Parse Training Labels (60,000,)
    with gzip.open(files["train_labels"], "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # Parse Test Images (10,000 x 28 x 28)
    with gzip.open(files["test_images"], "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

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


def train_mnist(epochs: int = 5, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
    np.random.seed(seed)
    print("=" * 72)
    print("             NUMPYGRAD MNIST MLP TRAINING PIPELINE")
    print("=" * 72)

    # 1. Ingest Data
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = download_and_extract_mnist()
    t_load = time.perf_counter() - t0
    print(f"[DATA] MNIST loaded in {t_load:.2f}s | Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. Build Datasets and DataLoaders
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 3. Instantiate Model Architecture
    model = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    num_params = sum(p.data.size for p in model.parameters())
    print(f"[MODEL] MLP Architecture: Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)")
    print(f"[MODEL] Total Trainable Parameters: {num_params:,}")

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("-" * 72)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Test Loss':^11} | {'Test Acc':^10} | {'Throughput':^14} | {'Time':^7}")
    print("-" * 72)

    # 5. Training Loop
    total_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples_processed = 0
        t_epoch_start = time.perf_counter()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
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
        test_loss, test_acc = evaluate_model(model, test_loader)

        print(
            f"{epoch:^7d} | {avg_train_loss:^12.4f} | {test_loss:^11.4f} | "
            f"{test_acc * 100:^9.2f}% | {throughput:^10.1f} smp/s | {t_epoch:^6.2f}s"
        )

    total_time = time.perf_counter() - total_train_start
    print("-" * 72)
    print(f"[DONE] Completed {epochs} epochs in {total_time:.2f}s | Final Test Accuracy: {test_acc * 100:.2f}%")

    # 6. Accuracy Assertion
    assert test_acc >= 0.95, f"Test accuracy {test_acc:.4f} did not meet required threshold of 0.9500"
    print(f"[SUCCESS] Accuracy Requirement Met: {test_acc * 100:.2f}% >= 95.00%")

    # 7. Model Persistence to .ng container
    artifact_path = os.path.join(os.path.dirname(__file__), "mnist_mlp.ng")
    model.save(artifact_path)
    print(f"[SAVE] Model serialized successfully to: {artifact_path}")

    # 8. Reload & Verify Round-trip Inference
    loaded_model = load_model(artifact_path)
    loaded_model.eval()
    _, reloaded_acc = evaluate_model(loaded_model, test_loader)
    print(f"[RELOAD] Model successfully reloaded. Round-trip Test Accuracy: {reloaded_acc * 100:.2f}%\n")
    assert abs(reloaded_acc - test_acc) < 1e-6, "Reloaded model accuracy mismatch!"

    return model, test_acc


if __name__ == "__main__":
    train_mnist()
