"""
End-to-end training, evaluation, and persistence pipeline on the Iris dataset.

Demonstrates:
- Tabular multi-class classification on the canonical 150-sample Iris dataset.
- Pure NumPy data loading, stratified train/test split, and z-score normalization.
- Mini-batch training with Dataset, DataLoader, CrossEntropyLoss, and AdamW.
- Evaluation with accuracy and confusion matrix metrics.
- Model persistence to `.ng` container and reload verification.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpygrad as ng
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.data import Dataset, DataLoader
from numpygrad.metrics import accuracy, confusion_matrix
from numpygrad.core.tensor import no_grad, Tensor
from numpygrad.serialization import save_model, load_model


def load_iris_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Returns the canonical 150-sample Iris dataset (features, labels, feature_names, target_names).
    """
    # 150 samples: 50 Setosa (0), 50 Versicolor (1), 50 Virginica (2)
    # Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
    raw_data = np.array([
        [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
        [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0],
        [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
        [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0],
        [5.7, 4.4, 1.5, 0.4, 0], [5.4, 3.9, 1.3, 0.4, 0], [5.1, 3.5, 1.4, 0.3, 0],
        [5.7, 3.8, 1.7, 0.3, 0], [5.1, 3.8, 1.5, 0.3, 0], [5.4, 3.4, 1.7, 0.2, 0],
        [5.1, 3.7, 1.5, 0.4, 0], [4.6, 3.6, 1.0, 0.2, 0], [5.1, 3.3, 1.7, 0.5, 0],
        [4.8, 3.4, 1.9, 0.2, 0], [5.0, 3.0, 1.6, 0.2, 0], [5.0, 3.4, 1.6, 0.4, 0],
        [5.2, 3.5, 1.5, 0.2, 0], [5.2, 3.4, 1.4, 0.2, 0], [4.7, 3.2, 1.6, 0.2, 0],
        [4.8, 3.1, 1.6, 0.2, 0], [5.4, 3.4, 1.5, 0.4, 0], [5.2, 4.1, 1.5, 0.1, 0],
        [5.5, 4.2, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0],
        [5.5, 3.5, 1.3, 0.2, 0], [4.9, 3.6, 1.4, 0.1, 0], [4.4, 3.0, 1.3, 0.2, 0],
        [5.1, 3.4, 1.5, 0.2, 0], [5.0, 3.5, 1.3, 0.3, 0], [4.5, 2.3, 1.3, 0.3, 0],
        [4.4, 3.2, 1.3, 0.2, 0], [5.0, 3.5, 1.6, 0.6, 0], [5.1, 3.8, 1.9, 0.4, 0],
        [4.8, 3.0, 1.4, 0.3, 0], [5.1, 3.8, 1.6, 0.2, 0], [4.6, 3.2, 1.4, 0.2, 0],
        [5.3, 3.7, 1.5, 0.2, 0], [5.0, 3.3, 1.4, 0.2, 0],
        [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1],
        [5.5, 2.3, 4.0, 1.3, 1], [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1],
        [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1], [6.6, 2.9, 4.6, 1.3, 1],
        [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1],
        [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1],
        [6.7, 3.1, 4.4, 1.4, 1], [5.6, 3.0, 4.5, 1.5, 1], [5.8, 2.7, 4.1, 1.0, 1],
        [6.2, 2.2, 4.5, 1.5, 1], [5.6, 2.5, 3.9, 1.1, 1], [5.9, 3.2, 4.8, 1.8, 1],
        [6.1, 2.8, 4.0, 1.3, 1], [6.3, 2.5, 4.9, 1.5, 1], [6.1, 2.8, 4.7, 1.2, 1],
        [6.4, 2.9, 4.3, 1.3, 1], [6.6, 3.0, 4.4, 1.4, 1], [6.8, 2.8, 4.8, 1.4, 1],
        [6.7, 3.0, 5.0, 1.7, 1], [6.0, 2.9, 4.5, 1.5, 1], [5.7, 2.6, 3.5, 1.0, 1],
        [5.5, 2.4, 3.8, 1.1, 1], [5.5, 2.4, 3.7, 1.0, 1], [5.8, 2.7, 3.9, 1.2, 1],
        [6.0, 2.7, 5.1, 1.6, 1], [5.4, 3.0, 4.5, 1.5, 1], [6.0, 3.4, 4.5, 1.6, 1],
        [6.7, 3.1, 4.7, 1.5, 1], [6.3, 2.3, 4.4, 1.3, 1], [5.6, 3.0, 4.1, 1.3, 1],
        [5.5, 2.5, 4.0, 1.3, 1], [5.5, 2.6, 4.4, 1.2, 1], [6.1, 3.0, 4.6, 1.4, 1],
        [5.8, 2.6, 4.0, 1.2, 1], [5.0, 2.3, 3.3, 1.0, 1], [5.6, 2.7, 4.2, 1.3, 1],
        [5.7, 3.0, 4.2, 1.2, 1], [5.7, 2.9, 4.2, 1.3, 1], [6.2, 2.9, 4.3, 1.3, 1],
        [5.1, 2.5, 3.0, 1.1, 1], [5.7, 2.8, 4.1, 1.3, 1],
        [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2],
        [6.3, 2.9, 5.6, 1.8, 2], [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2],
        [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2], [6.7, 2.5, 5.8, 1.8, 2],
        [7.2, 3.6, 6.1, 2.5, 2], [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2],
        [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2],
        [6.4, 3.2, 5.3, 2.3, 2], [6.5, 3.0, 5.5, 1.8, 2], [7.7, 3.8, 6.7, 2.2, 2],
        [7.7, 2.6, 6.9, 2.3, 2], [6.0, 2.2, 5.0, 1.5, 2], [6.9, 3.2, 5.7, 2.3, 2],
        [5.6, 2.8, 4.9, 2.0, 2], [7.7, 2.8, 6.7, 2.0, 2], [6.3, 2.7, 4.9, 1.8, 2],
        [6.7, 3.3, 5.7, 2.1, 2], [7.2, 3.2, 6.0, 1.8, 2], [6.2, 2.8, 4.8, 1.8, 2],
        [6.1, 3.0, 4.9, 1.8, 2], [6.4, 2.8, 5.6, 2.1, 2], [7.2, 3.0, 5.8, 1.6, 2],
        [7.4, 2.8, 6.1, 1.9, 2], [7.9, 3.8, 6.4, 2.0, 2], [6.4, 2.8, 5.6, 2.2, 2],
        [6.3, 2.8, 5.1, 1.5, 2], [6.1, 2.6, 5.6, 1.4, 2], [7.7, 3.0, 6.1, 2.3, 2],
        [6.3, 3.4, 5.6, 2.4, 2], [6.4, 3.1, 5.5, 1.8, 2], [6.0, 3.0, 4.8, 1.8, 2],
        [6.9, 3.1, 5.4, 2.1, 2], [6.7, 3.1, 5.6, 2.4, 2], [6.9, 3.1, 5.1, 2.3, 2],
        [5.8, 2.7, 5.1, 1.9, 2], [6.8, 3.2, 5.9, 2.3, 2], [6.7, 3.3, 5.7, 2.5, 2],
        [6.7, 3.0, 5.2, 2.3, 2], [6.3, 2.5, 5.0, 1.9, 2], [6.5, 3.0, 5.2, 2.0, 2],
        [6.2, 3.4, 5.4, 2.3, 2], [5.9, 3.0, 5.1, 1.8, 2]
    ], dtype=np.float32)

    X = raw_data[:, :4]
    y = raw_data[:, 4].astype(np.int64)
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    target_names = ["setosa", "versicolor", "virginica"]
    return X, y, feature_names, target_names


def train_iris_pipeline(epochs: int = 100, lr: float = 0.01, batch_size: int = 16, seed: int = 42):
    np.random.seed(seed)
    X, y, feature_names, target_names = load_iris_data()

    # 1. Stratified 80/20 train/test split (40 train / 10 test per class)
    train_indices, test_indices = [], []
    for cls in range(3):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        train_indices.extend(cls_idx[:40])
        test_indices.extend(cls_idx[40:])

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # 2. Z-score feature normalization based on training statistics
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-7
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # 3. Create Dataset & DataLoader
    train_dataset = Dataset(X_train_norm, y_train)
    test_dataset = Dataset(X_test_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)

    # 4. Construct 3-layer MLP Architecture
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n[TRAIN] Training 3-layer MLP on Iris dataset (Train: {len(X_train)}, Test: {len(X_test)})...")
    print("-" * 65)

    # 5. Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y.data)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            batches += 1

        avg_loss = epoch_loss / max(1, batches)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            # Training accuracy snapshot
            with no_grad():
                train_logits = model(Tensor(X_train_norm))
                train_acc = accuracy(train_logits, y_train)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

    print("-" * 65)

    # 6. Evaluation on Test Set
    model.eval()
    with no_grad():
        test_logits = model(Tensor(X_test_norm))
        test_acc = accuracy(test_logits, y_test)
        cm = confusion_matrix(test_logits, y_test, num_classes=3)

    # Print ASCII Confusion Matrix
    print("\n" + "=" * 65)
    print("                     IRIS EVALUATION REPORT")
    print("=" * 65)
    print(f"Train Accuracy:  {train_acc * 100:.2f}%")
    print(f"Test Accuracy:   {test_acc * 100:.2f}% ({int(test_acc * len(y_test))}/{len(y_test)} correct)")
    print("\nConfusion Matrix (Rows: True Class, Columns: Predicted Class):")
    print(f"{'':>15} | {'Setosa':>10} | {'Versicolor':>10} | {'Virginica':>10}")
    print("-" * 15 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10)
    for i, name in enumerate(target_names):
        print(f"{name:>15} | {cm[i, 0]:>10} | {cm[i, 1]:>10} | {cm[i, 2]:>10}")
    print("=" * 65 + "\n")

    # 7. Model Persistence to .ng container
    artifact_path = os.path.join(os.path.dirname(__file__), "iris_model.ng")
    model.save(artifact_path)
    print(f"[SAVE] Model serialized successfully to: {artifact_path}")

    # 8. Reload & Verify Round-trip Inference
    loaded_model = load_model(artifact_path)
    loaded_model.eval()
    with no_grad():
        reloaded_test_logits = loaded_model(Tensor(X_test_norm))
        reloaded_acc = accuracy(reloaded_test_logits, y_test)

    assert np.allclose(test_logits.data, reloaded_test_logits.data, atol=1e-6), "Round-trip prediction mismatch!"
    print(f"[RELOAD] Model successfully reloaded. Round-trip inference verified (Accuracy: {reloaded_acc * 100:.2f}%).\n")

    return model, test_acc


if __name__ == "__main__":
    train_iris_pipeline()
