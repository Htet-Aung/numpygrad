"""
NumPyGrad Interactive Deep Learning & Autograd Studio.

A pure NumPy automatic differentiation playground built with Streamlit.
Zero external deep learning framework dependencies.
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

from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.data import TensorDataset, DataLoader


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NumPyGrad Studio",
    page_icon="⚡",
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

    /* Remove dead padding at top of sidebar content so 'Experiment Controls' sits right next to the collapse button */
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 3rem !important;
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
    plt.colorbar(contour, ax=ax_b, label="P(Class = 1)")

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
    ax_b.set_xlabel("Feature $x_1$")
    ax_b.set_ylabel("Feature $x_2$")
    ax_b.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

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
    plt.tight_layout()

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
    ax.set_xlabel(r"Gradient $\ell_2$ Norm ($\|\nabla_\theta\|_2$)", fontweight="bold")
    ax.set_title("Layer-by-Layer Gradient Magnitude Diagnostic", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">⚡ NumPyGrad Interactive Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A Pure NumPy Dynamic Computational Graph & Deep Learning Playground with Zero External DL Frameworks</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("⚙️ Experiment Controls")
        
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
        start_training = st.button("🚀 Start Training", type="primary", use_container_width=True)
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

        st.success(f"🎉 Training converged in {time.time() - start_time:.2f}s! Final Accuracy: **{acc_history[-1]:.2f}%** | Final Loss: **{loss_history[-1]:.4f}**")

        # ---------------- Post-Training Gradient Norm Diagnostic ----------------
        with diag_placeholder.container():
            st.subheader("🔬 Post-Training Diagnostics: Gradient Propagation")
            fig_grad = plot_gradient_norms(model)
            st.pyplot(fig_grad)
            plt.close(fig_grad)


if __name__ == "__main__":
    main()
