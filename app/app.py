"""
NumPyGrad Interactive Deep Learning & Autograd Studio.

A pure NumPy automatic differentiation playground built with Streamlit.
Zero external deep learning framework dependencies.

Tabs:
  1. 2D Decision Boundaries - Live contour training on synthetic datasets.
  2. Handwritten Digit Recognition - Interactive MNIST canvas inference.
"""

import sys
import os
import time
from typing import Tuple, List, Dict, Any

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from numpygrad.core.tensor import Tensor, no_grad
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.data import TensorDataset, DataLoader
from numpygrad.serialization import load_model


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NumPyGrad Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Header styling - transparent background */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
        z-index: 99 !important;
    }

    /* Hide deploy button and internal status decorations */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    div[data-testid="stToolbarActions"] > button:first-child,
    div[data-testid="stHeaderActionElements"] > button:first-child,
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    footer {
        display: none !important;
    }

    /* Position sidebar header absolute in top-right of sidebar so user controls start immediately at the top */
    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: 0.8rem !important;
        right: 0.6rem !important;
        padding: 0 !important;
        margin: 0 !important;
        z-index: 100 !important;
        background: transparent !important;
        width: auto !important;
    }

    [data-testid="stLogoSpacer"] {
        display: none !important;
    }

    /* Reset outer sidebar wrapper padding to avoid multiplying nested padding */
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] .block-container {
        padding: 0 !important;
    }

    /* Clean balanced padding on sidebar user content without extra right gap */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Keep sidebar collapse and expand chevron buttons fully functional */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
    }

    /* Align three-dots options menu with the main studio title */
    #MainMenu,
    [data-testid="stMainMenu"],
    [data-testid="stHeaderActionElements"],
    [data-testid="stToolbar"] {
        visibility: visible !important;
        display: block !important;
    }

    #MainMenu,
    [data-testid="stMainMenu"] {
        margin-top: 2.6rem !important;
        margin-right: 1rem !important;
    }

    /* Compact main block container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Typography & card styles */
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 0rem !important;
        margin-bottom: 0.1rem !important;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 0rem !important;
        margin-bottom: 0.65rem !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid #e9ecef;
    }

    /* Reduce vertical margins between headers and elements */
    h1, h2, h3, h4 {
        margin-top: 0rem !important;
        margin-bottom: 0.25rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }

    /* Prediction callout card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .prediction-digit {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }
    .prediction-confidence {
        font-size: 1.3rem;
        font-weight: 600;
        opacity: 0.95;
        margin-top: 4px;
    }
    .prediction-label {
        font-size: 0.85rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    /* Top-3 ranking cards */
    .rank-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .rank-digit {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .rank-prob {
        font-size: 0.9rem;
        color: #6c757d;
        margin: 0;
    }

    /* Force visibility of canvas undo/redo/clear icons */
    div[data-testid="stDrawableCanvas"] ~ div button,
    div[data-testid="stDrawableCanvas"] button {
        background-color: #31333F !important;
        border: 1px solid #4A4A5A !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        margin: 2px !important;
    }
    div[data-testid="stDrawableCanvas"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    div[data-testid="stDrawableCanvas"] svg path {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Synthetic Dataset Generators (Pure NumPy)
# -----------------------------------------------------------------------------

def generate_dataset(
    dataset_name: str,
    n_samples: int = 500,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2D classification datasets using pure NumPy."""
    rng = np.random.default_rng(random_state)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    if dataset_name == "Two Moons":
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1.0 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1.0 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

        X = np.vstack([
            np.column_stack([outer_circ_x, outer_circ_y]),
            np.column_stack([inner_circ_x, inner_circ_y]),
        ])
        y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])

    elif dataset_name == "Concentric Circles":
        factor = 0.5
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

    elif dataset_name == "Spirals":
        n_classes = 2
        samples_per_class = n_samples // n_classes
        X_list, y_list = [], []

        for c in range(n_classes):
            r = np.linspace(0.1, 1.0, samples_per_class)
            t = np.linspace(c * 4.0, (c + 1) * 4.0, samples_per_class) + rng.normal(scale=noise, size=samples_per_class)
            x1 = r * np.sin(t)
            x2 = r * np.cos(t)
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(samples_per_class, c, dtype=int))

        X = np.vstack(X_list)
        y = np.hstack(y_list)
        return X.astype(np.float32), y

    if noise > 0.0:
        X += rng.normal(scale=noise, size=X.shape)

    return X.astype(np.float32), y


# -----------------------------------------------------------------------------
# Dynamic Model Architecture Construction
# -----------------------------------------------------------------------------

def build_model(
    num_layers: int,
    hidden_dim: int,
    activation_name: str,
    use_batchnorm: bool = False,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    """Builds a sequential neural network dynamically."""
    layers: List[nn.Module] = []
    in_dim = 2

    for layer_idx in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        # Activation selection
        if activation_name == "ReLU":
            layers.append(nn.ReLU())
        elif activation_name == "Tanh":
            layers.append(nn.Tanh())
        elif activation_name == "Sigmoid":
            layers.append(nn.Sigmoid())
        elif activation_name == "GELU":
            layers.append(nn.GELU())

        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))

        in_dim = hidden_dim

    # Final classification head (2 output classes)
    layers.append(nn.Linear(hidden_dim, 2))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Decision Boundary & Training Curves Plotting
# -----------------------------------------------------------------------------

def plot_dashboard_figures(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_hist: List[float],
    acc_hist: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """Generates two separate figures for the decision boundary and metrics."""
    # 1. Decision Boundary Figure
    fig_boundary, ax_b = plt.subplots(figsize=(6, 5), dpi=120)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    grid_tensor = Tensor(grid_points, requires_grad=False)
    grid_logits = model(grid_tensor)

    # Softmax probabilities for class 1
    exp_logits = np.exp(grid_logits.data - np.max(grid_logits.data, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
    Z = probs.reshape(xx.shape)

    contour = ax_b.contourf(xx, yy, Z, levels=40, cmap="Spectral_r", alpha=0.85)
    ax_b.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")
    fig_boundary.colorbar(contour, ax=ax_b, label="P(Class = 1)")

    ax_b.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="Spectral_r",
        edgecolors="black",
        linewidths=0.6,
        s=30,
        alpha=0.9,
    )
    ax_b.set_title("Learned Decision Boundary", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Feature x1", fontsize=10)
    ax_b.set_ylabel("Feature x2", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.4)
    fig_boundary.tight_layout()

    # 2. Loss & Accuracy Progression Figure
    fig_metrics, ax_loss = plt.subplots(figsize=(6, 5), dpi=120)
    epochs_range = range(1, len(loss_hist) + 1)
    
    color_loss = "#d9534f"
    ax_loss.set_xlabel("Epoch", fontweight="bold")
    ax_loss.set_ylabel("Loss", color=color_loss, fontweight="bold")
    line1 = ax_loss.plot(epochs_range, loss_hist, color=color_loss, linewidth=2.0, label="Loss")
    ax_loss.tick_params(axis="y", labelcolor=color_loss)
    ax_loss.grid(True, linestyle=":", alpha=0.4)

    ax_acc = ax_loss.twinx()
    color_acc = "#0275d8"
    ax_acc.set_ylabel("Accuracy (%)", color=color_acc, fontweight="bold")
    line2 = ax_acc.plot(epochs_range, acc_hist, color=color_acc, linewidth=2.0, linestyle="-.", label="Accuracy")
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_acc.set_ylim([0, 105])

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc="center right")
    ax_loss.set_title("Loss & Accuracy Convergence", fontsize=11, fontweight="bold")
    fig_metrics.tight_layout()

    return fig_boundary, fig_metrics


def plot_gradient_norms(model: nn.Module) -> plt.Figure:
    """Plots a layer-by-layer gradient norm diagnostic bar chart."""
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    layer_names = []
    grad_norms = []

    for idx, p in enumerate(model.parameters()):
        name = f"Param {idx} {p.data.shape}"
        norm = float(np.linalg.norm(p.grad)) if p.grad is not None else 0.0
        layer_names.append(name)
        grad_norms.append(norm)

    bars = ax.barh(layer_names, grad_norms, color="#5bc0de", edgecolor="#0275d8", alpha=0.85)
    ax.set_xlabel("Gradient L2 Norm", fontweight="bold")
    ax.set_title("Layer-by-Layer Gradient Magnitude Diagnostic", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# MNIST Canvas Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess_canvas_image(canvas_result) -> np.ndarray | None:
    """
    Extracts, centers, and normalizes a drawn digit from the Streamlit canvas
    into a (1, 28, 28) float32 array matching MNIST preprocessing conventions.

    Returns None if the canvas is blank.
    """
    if canvas_result is None or canvas_result.image_data is None:
        return None

    # Extract the stroke channel (white #FFFFFF stroke on black #000000 background)
    raw_stroke = canvas_result.image_data[:, :, 0].astype(np.float32)

    # Check if canvas contains drawn content (max pixel intensity > 20)
    if np.max(raw_stroke) <= 20.0:
        return None

    # Find bounding box of pixels > 20
    y_indices, x_indices = np.where(raw_stroke > 20.0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())

    # Crop the digit strictly to the bounding box
    cropped = raw_stroke[ymin:ymax + 1, xmin:xmax + 1]
    h, w = cropped.shape
    if h <= 0 or w <= 0:
        return None

    # Resize cropped box to fit inside a 20x20 box preserving aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))

    pil_img = Image.fromarray(cropped.astype(np.uint8), mode="L").resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    resized = np.array(pil_img, dtype=np.float32)

    # Place the 20x20 digit into the center of a 28x28 zero-filled (black) array
    digit_28x28 = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    digit_28x28[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Center using center-of-mass (standard MNIST format)
    total_mass = digit_28x28.sum()
    if total_mass > 0:
        gy, gx = np.mgrid[0:28, 0:28]
        cy = (gy * digit_28x28).sum() / total_mass
        cx = (gx * digit_28x28).sum() / total_mass
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))
        shift_y = int(np.clip(shift_y, -4, 4))
        shift_x = int(np.clip(shift_x, -4, 4))

        shifted = np.zeros_like(digit_28x28)
        for r in range(28):
            for c in range(28):
                nr, nc = r + shift_y, c + shift_x
                if 0 <= nr < 28 and 0 <= nc < 28:
                    shifted[nr, nc] = digit_28x28[r, c]
        digit_28x28 = shifted

    # Normalize strictly to [0.0, 1.0] (0.0 = black background, 1.0 = white stroke)
    digit_28x28 = np.clip(digit_28x28 / 255.0, 0.0, 1.0)

    # Reshape to (1, 28, 28) batch tensor
    return digit_28x28.reshape(1, 28, 28)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax using the Log-Sum-Exp trick."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def plot_probability_distribution(probs: np.ndarray) -> plt.Figure:
    """Creates a horizontal bar chart of class probabilities 0-9."""
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    classes = np.arange(10)
    colors = ["#667eea" if p < np.max(probs) else "#764ba2" for p in probs]

    bars = ax.barh(classes, probs * 100, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(classes)
    ax.set_yticklabels([str(c) for c in classes], fontsize=11, fontweight="bold")
    ax.set_xlabel("Probability (%)", fontweight="bold", fontsize=10)
    ax.set_title("Class Probability Distribution", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")

    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{p * 100:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Tab 1: 2D Decision Boundaries
# -----------------------------------------------------------------------------

def render_decision_boundary_tab():
    """Renders the 2D Decision Boundary training studio."""
    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("Experiment Controls")
        
        # 1. Dataset Settings
        st.subheader("1. Dataset Configuration")
        dataset_name = st.selectbox(
            "Dataset Topology",
            ["Two Moons", "Concentric Circles", "Spirals"],
            index=0,
        )
        n_samples = st.slider("Sample Count", min_value=100, max_value=1000, value=500, step=50)
        noise = st.slider("Noise Level", min_value=0.0, max_value=0.3, value=0.12, step=0.02)
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

        # 2. Architecture Settings
        st.subheader("2. Model Architecture")
        num_layers = st.slider("Hidden Layers", min_value=1, max_value=4, value=2, step=1)
        hidden_dim = st.select_slider("Hidden Dimension", options=[8, 16, 32, 64, 128], value=32)
        activation_name = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid", "GELU"], index=0)
        use_batchnorm = st.checkbox("Enable BatchNorm1d", value=False)
        dropout_p = st.slider("Dropout Probability", min_value=0.0, max_value=0.5, value=0.0, step=0.1)

        # 3. Optimization Settings
        st.subheader("3. Optimization & Training")
        optimizer_name = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0)
        lr = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
            value=0.03,
        )
        momentum = 0.9
        if optimizer_name == "SGD":
            momentum = st.slider("Polyak Momentum", min_value=0.0, max_value=0.99, value=0.9, step=0.05)

        weight_decay = st.select_slider("Weight Decay", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        epochs = st.slider("Epochs", min_value=10, max_value=150, value=60, step=5)
        update_freq = st.slider("UI Refresh Interval (Epochs)", min_value=1, max_value=10, value=2, step=1)

    # ---------------- Data Preparation ----------------
    X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=seed)

    # Top Control Bar
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_training = st.button("Start Training", type="primary", use_container_width=True)
    with col_info:
        total_params = (2 * hidden_dim + hidden_dim) + (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * 2 + 2)
        st.info(f"Dataset: **{dataset_name}** (N={n_samples}) | Architecture: **{num_layers}x {hidden_dim}d** ({total_params} Trainable Parameters) | Optimizer: **{optimizer_name}** (lr={lr})")

    # ---------------- Dashboard Layout Placeholders ----------------
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    col_left, col_right = st.columns(2)
    with col_left:
        plot_left = st.empty()
    with col_right:
        plot_right = st.empty()

    diag_placeholder = st.empty()

    # Initial static plot before training starts
    initial_model = build_model(num_layers, hidden_dim, activation_name, use_batchnorm, dropout_p)
    fig_b, fig_m = plot_dashboard_figures(initial_model, X, y, [0.693], [50.0])
    plot_left.pyplot(fig_b)
    plot_right.pyplot(fig_m)
    plt.close(fig_b)
    plt.close(fig_m)

    # ---------------- Interactive Training Loop ----------------
    if start_training:
        model = build_model(num_layers, hidden_dim, activation_name, use_batchnorm, dropout_p)
        criterion = nn.CrossEntropyLoss()

        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history: List[float] = []
        acc_history: List[float] = []
        start_time = time.time()

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

            # Epoch evaluation
            model.eval()
            full_logits = model(Tensor(X, requires_grad=False))
            preds = np.argmax(full_logits.data, axis=1)
            accuracy = float(np.mean(preds == y) * 100.0)
            avg_loss = float(np.mean(epoch_losses))

            loss_history.append(avg_loss)
            acc_history.append(accuracy)

            # Update UI at regular intervals
            if epoch % update_freq == 0 or epoch == epochs or epoch == 1:
                elapsed = time.time() - start_time
                progress_bar.progress(epoch / float(epochs))

                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Epoch", f"{epoch} / {epochs}")
                    m2.metric("CrossEntropy Loss", f"{avg_loss:.4f}", delta=f"{-(loss_history[0] - avg_loss):.4f}" if epoch > 1 else None, delta_color="inverse")
                    m3.metric("Classification Accuracy", f"{accuracy:.2f}%", delta=f"{accuracy - acc_history[0]:+.2f}%" if epoch > 1 else None)
                    m4.metric("Elapsed Time", f"{elapsed:.2f}s")

                fig_b, fig_m = plot_dashboard_figures(model, X, y, loss_history, acc_history)
                plot_left.pyplot(fig_b)
                plot_right.pyplot(fig_m)
                plt.close(fig_b)
                plt.close(fig_m)

        st.success(f"Training converged in {time.time() - start_time:.2f}s | Final Accuracy: **{acc_history[-1]:.2f}%** | Final Loss: **{loss_history[-1]:.4f}**")

        # ---------------- Post-Training Gradient Norm Diagnostic ----------------
        with diag_placeholder.container():
            st.subheader("Post-Training Diagnostics: Gradient Propagation")
            fig_grad = plot_gradient_norms(model)
            st.pyplot(fig_grad)
            plt.close(fig_grad)


# -----------------------------------------------------------------------------
# Tab 2: Handwritten Digit Recognition
# -----------------------------------------------------------------------------

@st.cache_resource
def load_mnist_model():
    """Loads the pre-trained MNIST MLP model from the .ng artifact."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "examples", "mnist_mlp.ng")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        return None

    model = load_model(model_path)
    model.eval()
    return model


def render_mnist_inference_tab():
    """Renders the Handwritten Digit Recognition inference studio."""
    from streamlit_drawable_canvas import st_canvas

    st.markdown(
        "Draw a single digit (0-9) on the canvas below. "
        "The pre-trained MNIST MLP model will classify your handwriting in real time."
    )

    # Load model
    model = load_mnist_model()
    if model is None:
        st.error(
            "Pre-trained MNIST model not found at `examples/mnist_mlp.ng`. "
            "Run `python examples/train_mnist_mlp.py` first to train and save the model."
        )
        return

    # Layout: canvas on left, results on right
    col_canvas, col_results = st.columns([1, 1.4], gap="large")

    with col_canvas:
        st.subheader("Drawing Canvas")

        # Canvas controls
        brush_width = st.slider(
            "Brush Width", min_value=8, max_value=36, value=20, step=2,
            key="mnist_brush_width",
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=brush_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_digit_canvas",
        )

        st.button("Classify Drawing", type="primary", use_container_width=True)

        # Preprocess and show thumbnail
        processed = preprocess_canvas_image(canvas_result)

        if processed is not None:
            st.subheader("Preprocessed Input (28x28)")
            st.image(processed[0], clamp=True, width=140, caption="28x28 Centered MNIST Input")

    with col_results:
        if processed is not None:
            # Run inference
            input_tensor = Tensor(processed.astype(np.float32), requires_grad=False)

            with no_grad():
                logits = model(input_tensor)

            probs = stable_softmax(logits.data[0])
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            # Top-3 predictions
            top3_indices = np.argsort(probs)[::-1][:3]

            # Prediction callout card
            st.markdown(
                f"""
                <div class="prediction-card">
                    <p class="prediction-label">Predicted Digit</p>
                    <p class="prediction-digit">{predicted_class}</p>
                    <p class="prediction-confidence">{confidence * 100:.1f}% Confidence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top-3 ranking cards
            st.subheader("Top-3 Predictions")
            rank_cols = st.columns(3)
            rank_labels = ["1st", "2nd", "3rd"]
            for i, col in enumerate(rank_cols):
                idx = top3_indices[i]
                p = probs[idx]
                with col:
                    st.markdown(
                        f"""
                        <div class="rank-card">
                            <p style="font-size:0.75rem;color:#6c757d;margin:0;">{rank_labels[i]}</p>
                            <p class="rank-digit">{idx}</p>
                            <p class="rank-prob">{p * 100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Full probability distribution bar chart
            st.subheader("Probability Distribution")
            fig_probs = plot_probability_distribution(probs)
            st.pyplot(fig_probs)
            plt.close(fig_probs)
        else:
            st.info("Draw a digit on the canvas to see predictions.")


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">NumPyGrad Interactive Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A Pure NumPy Dynamic Computational Graph & Deep Learning Playground with Zero External DL Frameworks</div>',
        unsafe_allow_html=True,
    )

    selected_tab = st.segmented_control(
        "Studio Navigation",
        options=["2D Decision Boundaries", "Handwritten Digit Recognition"],
        default="2D Decision Boundaries",
        label_visibility="collapsed",
    )
    if not selected_tab:
        selected_tab = "2D Decision Boundaries"

    if selected_tab == "2D Decision Boundaries":
        render_decision_boundary_tab()
    else:
        with st.sidebar:
            st.markdown("### MNIST Inference Mode")
            st.info("The pre-trained model (`mnist_mlp.ng`) is loaded. Draw on the canvas to evaluate.")
        render_mnist_inference_tab()


if __name__ == "__main__":
    main()
