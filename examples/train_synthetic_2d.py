"""
End-to-End Training & Visual Verification on Non-Linear 2D Datasets.

Demonstrates:
- Pure NumPy dataset generators (Two Moons, Concentric Circles, Spirals).
- Multi-Layer Perceptron (MLP) constructed with NumPyGrad nn modules.
- Dynamic Autograd DAG backpropagation with AdamW optimization.
- Mini-batch DataLoader training loop.
- Decision boundary visualization saved to `examples/decision_boundary.png`.
"""

import os
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.data import TensorDataset, DataLoader


# -----------------------------------------------------------------------------
# Pure NumPy Synthetic Dataset Generators
# -----------------------------------------------------------------------------

def make_moons(n_samples: int = 500, noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two interleaving half circles (Moons dataset)."""
    rng = np.random.default_rng(random_state)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1.0 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1.0 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.column_stack([outer_circ_x, outer_circ_y]),
        np.column_stack([inner_circ_x, inner_circ_y]),
    ])
    y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])

    if noise > 0.0:
        X += rng.normal(scale=noise, size=X.shape)

    return X.astype(np.float32), y


def make_circles(n_samples: int = 500, noise: float = 0.08, factor: float = 0.5, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates concentric circles dataset."""
    rng = np.random.default_rng(random_state)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)

    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    X = np.vstack([
        np.column_stack([outer_circ_x, outer_circ_y]),
        np.column_stack([inner_circ_x, inner_circ_y]),
    ])
    y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])

    if noise > 0.0:
        X += rng.normal(scale=noise, size=X.shape)

    return X.astype(np.float32), y


def make_spirals(n_samples: int = 500, noise: float = 0.1, n_classes: int = 2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates multi-arm interleaved spiral dataset."""
    rng = np.random.default_rng(random_state)
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []

    for c in range(n_classes):
        r = np.linspace(0.1, 1.0, samples_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, samples_per_class) + rng.normal(scale=noise, size=samples_per_class)
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(samples_per_class, c, dtype=int))

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list)
    return X, y


# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------

def build_mlp(in_dim: int = 2, hidden_dim: int = 32, out_dim: int = 2) -> nn.Sequential:
    """Instantiates a multi-layer perceptron architecture."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 0.02,
    weight_decay: float = 1e-4,
) -> Tuple[list, list, list]:
    """Trains the model and returns loss, accuracy, and lr history."""
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    acc_history = []
    lr_history = []

    print(f"\n[TRAIN] Training {model.__class__.__name__} for {epochs} epochs on dataset (N={len(X)})...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.data))

        # Evaluation
        model.eval()
        full_logits = model(Tensor(X, requires_grad=False))
        preds = np.argmax(full_logits.data, axis=1)
        accuracy = np.mean(preds == y) * 100.0
        avg_loss = np.mean(epoch_losses)

        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        lr_history.append(lr)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:6.2f}% | LR: {lr:.4f}")

    print("-" * 65)
    print(f"[DONE] Training completed! Final Accuracy: {acc_history[-1]:.2f}%\n")
    return loss_history, acc_history, lr_history


# -----------------------------------------------------------------------------
# Decision Boundary & Visualization
# -----------------------------------------------------------------------------

def plot_results(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_history: list,
    acc_history: list,
    save_path: str = "examples/decision_boundary.png",
) -> None:
    """Generates and saves the decision boundary contour and training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # 1. Decision Boundary Subplot
    ax_boundary = axes[0]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    grid_tensor = Tensor(grid_points, requires_grad=False)
    grid_logits = model(grid_tensor)
    # Class 1 probability via softmax
    exp_logits = np.exp(grid_logits.data - np.max(grid_logits.data, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
    Z = probs.reshape(xx.shape)

    # Plot filled contour & decision threshold
    contour = ax_boundary.contourf(xx, yy, Z, levels=50, cmap="Spectral_r", alpha=0.85)
    ax_boundary.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")
    plt.colorbar(contour, ax=ax_boundary, label="P(Class = 1)")

    # Overlay ground-truth scatter points
    scatter = ax_boundary.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="Spectral_r",
        edgecolors="black",
        linewidths=0.7,
        s=35,
        alpha=0.9,
    )
    ax_boundary.set_title("NumPyGrad: Learned Non-Linear Decision Boundary", fontsize=12, fontweight="bold")
    ax_boundary.set_xlabel("Feature $x_1$")
    ax_boundary.set_ylabel("Feature $x_2$")
    ax_boundary.grid(True, linestyle=":", alpha=0.5)

    # 2. Loss & Accuracy Training Curves Subplot
    ax_metrics = axes[1]
    epochs_range = range(1, len(loss_history) + 1)
    
    color_loss = "#d9534f"
    ax_metrics.set_xlabel("Epoch", fontweight="bold")
    ax_metrics.set_ylabel("CrossEntropy Loss", color=color_loss, fontweight="bold")
    line1 = ax_metrics.plot(epochs_range, loss_history, color=color_loss, linewidth=2.2, label="Loss")
    ax_metrics.tick_params(axis="y", labelcolor=color_loss)
    ax_metrics.grid(True, linestyle=":", alpha=0.5)

    ax_acc = ax_metrics.twinx()
    color_acc = "#0275d8"
    ax_acc.set_ylabel("Accuracy (%)", color=color_acc, fontweight="bold")
    line2 = ax_acc.plot(epochs_range, acc_history, color=color_acc, linewidth=2.2, linestyle="-.", label="Accuracy")
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_acc.set_ylim([0, 105])

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="center right", frameon=True)
    ax_metrics.set_title("Training Loss & Accuracy Progression", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Visualization saved successfully to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------

def main():
    # 1. Generate Synthetic Two Moons Dataset
    X, y = make_moons(n_samples=600, noise=0.15, random_state=42)

    # 2. Build Multi-Layer Perceptron
    model = build_mlp(in_dim=2, hidden_dim=32, out_dim=2)

    # 3. Train Model
    loss_hist, acc_hist, lr_hist = train_model(
        model,
        X,
        y,
        epochs=70,
        batch_size=32,
        lr=0.03,
        weight_decay=1e-4,
    )

    # 4. Generate & Save Decision Boundary Visualization
    plot_results(model, X, y, loss_hist, acc_hist, save_path="examples/decision_boundary.png")

    # 5. Assert Convergence Threshold
    assert acc_hist[-1] >= 95.0, f"Model failed to reach 95% accuracy! Final: {acc_hist[-1]:.2f}%"
    print(f"[SUCCESS] Verification Succeeded: Model achieved {acc_hist[-1]:.2f}% accuracy (target >= 95%).")


if __name__ == "__main__":
    main()
