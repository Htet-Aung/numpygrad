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
from typing import Tuple, List, Dict, Any, Optional, Union

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Eager top-level imports to prevent thread-level lazy import race conditions in Plotly validators
try:
    import pandas as pd
except ImportError:
    pd = None

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from numpygrad.core.tensor import Tensor, no_grad
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.utils.data import TensorDataset, DataLoader
from numpygrad.utils.pathfinding import simulate_rover_path, find_safe_waypoints
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
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .rank-digit {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .rank-prob {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38BDF8;
        margin: 0;
    }

    /* Standardize all action buttons for pixel-perfect vertical alignment */
    div.stButton > button {
        border-radius: 8px !important;
        padding: 0.55rem 1.15rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        min-height: 42px !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Primary action buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1.5px solid #818CF8 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
        border-color: #C7D2FE !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.55) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }

    /* Secondary action buttons */
    div.stButton > button[kind="secondary"] {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1.5px solid #334155 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #334155 !important;
        border-color: #64748B !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Center the drawable canvas container relative to column and buttons */
    div[data-testid="stElementContainer"]:has(iframe[title*="drawable_canvas"]),
    div.stElementContainer:has(iframe[title*="drawable_canvas"]) {
        display: flex !important;
        justify-content: center !important;
    }
    iframe[data-testid="stCustomComponentV1"][title*="drawable_canvas"],
    iframe[title*="drawable_canvas"] {
        width: 284px !important;
        height: 284px !important;
        display: block !important;
        margin: 0 auto 0.5rem auto !important;
        border: 2px solid #3B4252 !important;
        border-radius: 8px !important;
        background-color: #000000 !important;
    }

    /* Model Ready Status Banner */
    .model-ready-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        color: #F8FAFC;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.35);
    }
    .model-ready-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
    }
    .model-ready-subtitle {
        font-size: 0.86rem;
        color: #94A3B8;
        margin-bottom: 0.6rem;
    }
    .model-ready-badge {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.28);
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 6px;
    }

    /* Probability distribution bars */
    .prob-bar-container {
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .prob-bar-bg {
        background-color: #E2E8F0;
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
    }
    .prob-bar-fill-0 {
        background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%);
        height: 100%;
        border-radius: 6px;
    }
    .prob-bar-fill-1 {
        background: linear-gradient(90deg, #EF4444 0%, #F87171 100%);
        height: 100%;
        border-radius: 6px;
    }

    /* Prevent stMetric value and label truncation in multi-column layouts */
    div[data-testid="stMetricValue"] > div {
        font-size: 1.25rem !important;
        white-space: nowrap !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.82rem !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Curated Geometric Preset & Dataset Generators (Pure NumPy)
# -----------------------------------------------------------------------------

def load_preset_island_in_moat(
    n_samples: int = 300, noise: float = 0.05, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """Generates an Inner Circular Island (Class 0) surrounded by a Circular Hazard Moat (Class 1) and Outer Shore (Class 0)."""
    rng = np.random.default_rng(random_state)
    pts_X: List[List[float]] = []
    pts_y: List[int] = []

    # Class 0: Inner Core Island
    n_core = max(10, n_samples // 3)
    r_core = rng.uniform(0.0, 0.48, n_core)
    th_core = rng.uniform(0, 2 * np.pi, n_core)
    for r, th in zip(r_core, th_core):
        pts_X.append([
            round(float(r * np.cos(th) + rng.normal(0, noise)), 3),
            round(float(r * np.sin(th) + rng.normal(0, noise)), 3),
        ])
        pts_y.append(0)

    # Class 1: Moat Hazard Ring
    n_moat = max(10, n_samples // 3)
    r_moat = rng.uniform(0.92, 1.38, n_moat)
    th_moat = rng.uniform(0, 2 * np.pi, n_moat)
    for r, th in zip(r_moat, th_moat):
        pts_X.append([
            round(float(r * np.cos(th) + rng.normal(0, noise)), 3),
            round(float(r * np.sin(th) + rng.normal(0, noise)), 3),
        ])
        pts_y.append(1)

    # Class 0: Outer Perimeter
    n_outer = max(10, n_samples - n_core - n_moat)
    r_outer = rng.uniform(1.78, 2.28, n_outer)
    th_outer = rng.uniform(0, 2 * np.pi, n_outer)
    for r, th in zip(r_outer, th_outer):
        pts_X.append([
            round(float(r * np.cos(th) + rng.normal(0, noise)), 3),
            round(float(r * np.sin(th) + rng.normal(0, noise)), 3),
        ])
        pts_y.append(0)

    return pts_X, pts_y


def load_preset_checkerboard_xor(
    n_samples: int = 300, noise: float = 0.05, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """Generates 4 Alternating Quadrant Blocks testing sharp non-linear XOR decision boundaries."""
    rng = np.random.default_rng(random_state)
    pts_X: List[List[float]] = []
    pts_y: List[int] = []
    per_quad = max(8, n_samples // 4)

    quads = [
        (0.20, 2.15, 0.20, 2.15, 1),    # Q1: Red (Class 1)
        (-2.15, -0.20, 0.20, 2.15, 0),  # Q2: Blue (Class 0)
        (-2.15, -0.20, -2.15, -0.20, 1),# Q3: Red (Class 1)
        (0.20, 2.15, -2.15, -0.20, 0),  # Q4: Blue (Class 0)
    ]
    for x_min, x_max, y_min, y_max, label in quads:
        xs = rng.uniform(x_min, x_max, per_quad)
        ys = rng.uniform(y_min, y_max, per_quad)
        for x, y in zip(xs, ys):
            pts_X.append([
                round(float(x + rng.normal(0, noise)), 3),
                round(float(y + rng.normal(0, noise)), 3),
            ])
            pts_y.append(label)

    return pts_X, pts_y


def load_preset_s_bend_corridor(
    n_samples: int = 480, noise: float = 0.02, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """
    Generates a Multi-Turn Interlocking Serpentine Corridor Maze on [-2.0, 2.0] x [-2.0, 2.0]
    with outer perimeter boundaries, 3 horizontal baffle walls, and a continuous 4-channel winding path.
    """
    rng = np.random.default_rng(random_state)
    pts_X: List[List[float]] = []
    pts_y: List[int] = []

    # -------------------------------------------------------------
    # 1. Hazard Walls (Class 1 / Red)
    # -------------------------------------------------------------
    def add_wall_segment(p1, p2, count):
        xs = np.linspace(p1[0], p2[0], count)
        ys = np.linspace(p1[1], p2[1], count)
        for x, y in zip(xs, ys):
            pts_X.append([
                round(float(x + rng.normal(0, noise)), 3),
                round(float(y + rng.normal(0, noise)), 3),
            ])
            pts_y.append(1)

    # Outer perimeter with entry (top-left) & exit (bottom-right) openings
    # Top wall: x in [-1.15, 2.0], y = 2.0
    add_wall_segment((-1.15, 2.0), (2.0, 2.0), 32)
    # Bottom wall: x in [-2.0, 1.15], y = -2.0
    add_wall_segment((-2.0, -2.0), (1.15, -2.0), 32)
    # Left wall: x = -2.0, y in [-2.0, 1.15]
    add_wall_segment((-2.0, -2.0), (-2.0, 1.15), 32)
    # Right wall: x = 2.0, y in [-1.15, 2.0]
    add_wall_segment((2.0, -1.15), (2.0, 2.0), 32)

    # Interior Baffle Walls:
    # Wall 1 at y = 1.0, x in [-2.0, 0.85] (opening on right: x in [0.85, 2.0])
    add_wall_segment((-2.0, 1.0), (0.85, 1.0), 32)
    # Wall 2 at y = 0.0, x in [-0.85, 2.0] (opening on left: x in [-2.0, -0.85])
    add_wall_segment((-0.85, 0.0), (2.0, 0.0), 32)
    # Wall 3 at y = -1.0, x in [-2.0, 0.85] (opening on right: x in [0.85, 2.0])
    add_wall_segment((-2.0, -1.0), (0.85, -1.0), 32)

    # -------------------------------------------------------------
    # 2. Free Space Corridor (Class 0 / Blue)
    # -------------------------------------------------------------
    def add_corridor_lane(p1, p2, count, offset_spread=0.10):
        xs = np.linspace(p1[0], p2[0], count)
        ys = np.linspace(p1[1], p2[1], count)
        for x, y in zip(xs, ys):
            # Center track
            pts_X.append([round(float(x + rng.normal(0, noise)), 3), round(float(y + rng.normal(0, noise)), 3)])
            pts_y.append(0)
            # Parallel tracks for corridor clearance
            for s in [-offset_spread, offset_spread]:
                if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]):
                    # Horizontal channel -> offset in y
                    pts_X.append([round(float(x + rng.normal(0, noise)), 3), round(float(y + s + rng.normal(0, noise)), 3)])
                else:
                    # Vertical chute -> offset in x
                    pts_X.append([round(float(x + s + rng.normal(0, noise)), 3), round(float(y + rng.normal(0, noise)), 3)])
                pts_y.append(0)

    # Channel 1 (Top Entrance): x in [-1.6, 1.45], y = 1.5
    add_corridor_lane((-1.6, 1.5), (1.45, 1.5), 14, offset_spread=0.12)
    # Turn 1 (Right Chute): x = 1.45, y in [0.5, 1.5]
    add_corridor_lane((1.45, 1.5), (1.45, 0.5), 8, offset_spread=0.12)

    # Channel 2 (Upper-Mid): x in [1.45, -1.45], y = 0.5
    add_corridor_lane((1.45, 0.5), (-1.45, 0.5), 14, offset_spread=0.12)
    # Turn 2 (Left Chute): x = -1.45, y in [-0.5, 0.5]
    add_corridor_lane((-1.45, 0.5), (-1.45, -0.5), 8, offset_spread=0.12)

    # Channel 3 (Lower-Mid): x in [-1.45, 1.45], y = -0.5
    add_corridor_lane((-1.45, -0.5), (1.45, -0.5), 14, offset_spread=0.12)
    # Turn 3 (Right Chute): x = 1.45, y in [-1.5, -0.5]
    add_corridor_lane((1.45, -0.5), (1.45, -1.5), 8, offset_spread=0.12)

    # Channel 4 (Bottom Exit): x in [-1.45, 1.6], y = -1.5
    add_corridor_lane((-1.45, -1.5), (1.6, -1.5), 14, offset_spread=0.12)

    # Extra Entrance & Exit buffer clusters
    for _ in range(8):
        pts_X.append([round(float(-1.6 + rng.normal(0, 0.06)), 3), round(float(1.5 + rng.normal(0, 0.06)), 3)])
        pts_y.append(0)
    for _ in range(8):
        pts_X.append([round(float(1.6 + rng.normal(0, 0.06)), 3), round(float(-1.5 + rng.normal(0, 0.06)), 3)])
        pts_y.append(0)

    return pts_X, pts_y


def load_preset_double_cross(
    n_samples: int = 260, noise: float = 0.04, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """Generates a Large Perpendicular Plus/Cross Obstacle (Class 1) with 4 Quadrant Pockets (Class 0)."""
    rng = np.random.default_rng(random_state)
    pts_X: List[List[float]] = []
    pts_y: List[int] = []

    # Class 1 (Red Plus Shape)
    for y_val in np.linspace(-2.0, 2.0, 30):
        pts_X.append([round(float(0.0 + rng.normal(0, noise)), 3), round(float(y_val + rng.normal(0, noise)), 3)])
        pts_y.append(1)
    for x_val in np.linspace(-2.0, 2.0, 30):
        pts_X.append([round(float(x_val + rng.normal(0, noise)), 3), round(float(0.0 + rng.normal(0, noise)), 3)])
        pts_y.append(1)

    # Class 0 (Blue Quadrant Clusters)
    for cx, cy in [(-1.15, 1.15), (1.15, 1.15), (-1.15, -1.15), (1.15, -1.15)]:
        for _ in range(15):
            pts_X.append([
                round(float(cx + rng.normal(0, 0.28)), 3),
                round(float(cy + rng.normal(0, 0.28)), 3),
            ])
            pts_y.append(0)

    return pts_X, pts_y


def load_preset_spirals(
    n_samples: int = 300, noise: float = 0.08, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """Generates Two Interleaved Archimedean Spiral Arms."""
    rng = np.random.default_rng(random_state)
    pts_X: List[List[float]] = []
    pts_y: List[int] = []
    n_per_arm = max(15, n_samples // 2)

    # Arm 0 (Class 0, Blue)
    r0 = np.linspace(0.15, 2.15, n_per_arm)
    t0 = np.linspace(0, 4.0, n_per_arm)
    for r, t in zip(r0, t0):
        pts_X.append([
            round(float(r * np.sin(t) + rng.normal(0, noise)), 3),
            round(float(r * np.cos(t) + rng.normal(0, noise)), 3),
        ])
        pts_y.append(0)

    # Arm 1 (Class 1, Red)
    r1 = np.linspace(0.15, 2.15, n_per_arm)
    t1 = np.linspace(np.pi, np.pi + 4.0, n_per_arm)
    for r, t in zip(r1, t1):
        pts_X.append([
            round(float(r * np.sin(t) + rng.normal(0, noise)), 3),
            round(float(r * np.cos(t) + rng.normal(0, noise)), 3),
        ])
        pts_y.append(1)

    return pts_X, pts_y


# Aliases for backward compatibility
load_preset_corridor_maze = load_preset_s_bend_corridor
load_custom_maze_preset = load_preset_s_bend_corridor
load_custom_ring_preset = load_preset_island_in_moat

PRESET_GENERATOR_MAP = {
    "Island in Moat": load_preset_island_in_moat,
    "Checkerboard XOR": load_preset_checkerboard_xor,
    "Corridor Maze": load_preset_s_bend_corridor,
    "S-Bend Corridor": load_preset_s_bend_corridor,
    "Double Cross": load_preset_double_cross,
    "Spirals": load_preset_spirals,
    "Concentric Circles": load_preset_island_in_moat,
}


def generate_dataset(
    dataset_name: str,
    n_samples: int = 500,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2D classification datasets using pure NumPy or retrieves custom canvas points."""
    if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
        custom_X = st.session_state.get("custom_points_X")
        custom_y = st.session_state.get("custom_points_y")
        if custom_X is None or custom_y is None or len(custom_X) == 0:
            init_X, init_y = load_preset_checkerboard_xor(n_samples=200, noise=0.04, random_state=random_state)
            st.session_state["custom_points_X"] = init_X
            st.session_state["custom_points_y"] = init_y
            custom_X, custom_y = init_X, init_y
        return np.array(custom_X, dtype=np.float32), np.array(custom_y, dtype=np.int64)

    if dataset_name in PRESET_GENERATOR_MAP:
        gen_fn = PRESET_GENERATOR_MAP[dataset_name]
        pts_X, pts_y = gen_fn(n_samples=n_samples, noise=noise, random_state=random_state)
        return np.array(pts_X, dtype=np.float32), np.array(pts_y, dtype=np.int64)

    # Two Moons fallback
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


def get_architecture_summary(model: nn.Sequential) -> str:
    """Returns a concise arrow notation of network layer dimensions (e.g., '2 → 32 → 32 → 2')."""
    dims = [2]
    for layer in model:
        if isinstance(layer, nn.Linear):
            dims.append(layer.out_features)
    return " → ".join(str(d) for d in dims)


def predict_point(model: nn.Module, x: float, y: float) -> Tuple[int, float, np.ndarray]:
    """
    Evaluates a single 2D coordinate through the trained model under no_grad().

    Args:
        model: Trained neural network model.
        x: Feature x1 coordinate.
        y: Feature x2 coordinate.

    Returns:
        predicted_class (int): 0 or 1.
        confidence (float): Probability of the predicted class in [0, 1].
        probs (np.ndarray): 1D array of shape (2,) with class probabilities [P(Class 0), P(Class 1)].
    """
    model.eval()
    input_tensor = Tensor(np.array([[x, y]], dtype=np.float32), requires_grad=False)
    with no_grad():
        logits = model(input_tensor)
    probs = stable_softmax(logits.data[0])
    pred_class = int(np.argmax(probs))
    conf = float(probs[pred_class])
    return pred_class, conf, probs


def trace_forward_pass(model: nn.Sequential, x: float, y: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes a step-by-step forward pass through each layer of the model,
    capturing intermediate activations, shapes, and summary statistics.

    Returns:
        steps: List of step dictionaries with layer names, shapes, and activation stats (PyArrow-safe).
        softmax_details: Breakdown of intermediate values during stable softmax calculation.
    """
    model.eval()
    curr = Tensor(np.array([[x, y]], dtype=np.float32), requires_grad=False)
    
    steps: List[Dict[str, Any]] = [
        {
            "Step": 0,
            "Layer / Operation": "Input Coordinate Tensor",
            "Layer Type": "Tensor",
            "Output Shape": str(list(curr.shape)),
            "Min": float(round(float(curr.data.min()), 4)),
            "Max": float(round(float(curr.data.max()), 4)),
            "Mean": float(round(float(curr.data.mean()), 4)),
            "L2 Norm": float(round(float(np.linalg.norm(curr.data)), 4)),
        }
    ]
    
    with no_grad():
        for idx, layer in enumerate(model):
            curr = layer(curr)
            data = curr.data
            steps.append({
                "Step": idx + 1,
                "Layer / Operation": f"Layer {idx}: {layer.__class__.__name__}",
                "Layer Type": layer.__class__.__name__,
                "Output Shape": str(list(data.shape)),
                "Min": float(round(float(data.min()), 4)),
                "Max": float(round(float(data.max()), 4)),
                "Mean": float(round(float(data.mean()), 4)),
                "L2 Norm": float(round(float(np.linalg.norm(data)), 4)),
            })

    final_logits = curr.data[0]
    max_logit = float(np.max(final_logits))
    shifted = final_logits - max_logit
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))
    probs = exp_shifted / sum_exp

    softmax_details = {
        "raw_logits": [round(float(v), 4) for v in final_logits],
        "max_logit": round(max_logit, 4),
        "shifted_logits": [round(float(v), 4) for v in shifted],
        "exp_shifted": [round(float(v), 4) for v in exp_shifted],
        "sum_exp": round(sum_exp, 4),
        "probabilities": [round(float(v), 4) for v in probs],
    }

    return steps, softmax_details


def get_parameter_diagnostics(model: nn.Sequential) -> List[Dict[str, Any]]:
    """
    Collects parameter shapes, element counts, norms, and distribution statistics
    across all registered parameters in the model.
    Returns PyArrow-safe primitive data types for tabular Streamlit rendering.
    """
    diagnostics: List[Dict[str, Any]] = []
    for layer_idx, layer in enumerate(model):
        if hasattr(layer, "weight") and layer.weight is not None:
            w_data = layer.weight.data
            diagnostics.append({
                "Layer": f"Layer {layer_idx} ({layer.__class__.__name__})",
                "Param": "weight (W)",
                "Shape": str(list(w_data.shape)),
                "Count": int(w_data.size),
                "Mean": float(round(float(np.mean(w_data)), 4)),
                "Std": float(round(float(np.std(w_data)), 4)),
                "L2 Norm": float(round(float(np.linalg.norm(w_data)), 4)),
                "Sparsity": f"{(np.sum(np.abs(w_data) < 1e-4) / w_data.size) * 100:.1f}%",
            })
        if hasattr(layer, "bias") and layer.bias is not None:
            b_data = layer.bias.data
            diagnostics.append({
                "Layer": f"Layer {layer_idx} ({layer.__class__.__name__})",
                "Param": "bias (b)",
                "Shape": str(list(b_data.shape)),
                "Count": int(b_data.size),
                "Mean": float(round(float(np.mean(b_data)), 4)),
                "Std": float(round(float(np.std(b_data)), 4)),
                "L2 Norm": float(round(float(np.linalg.norm(b_data)), 4)),
                "Sparsity": f"{(np.sum(np.abs(b_data) < 1e-4) / b_data.size) * 100:.1f}%",
            })
    return diagnostics


def get_layer_raw_weights(model: nn.Sequential) -> Dict[str, np.ndarray]:
    """Extracts raw parameter ndarrays keyed by human-readable parameter identifier."""
    raw_weights: Dict[str, np.ndarray] = {}
    for layer_idx, layer in enumerate(model):
        if hasattr(layer, "weight") and layer.weight is not None:
            raw_weights[f"Layer {layer_idx} ({layer.__class__.__name__}) - weight"] = layer.weight.data
        if hasattr(layer, "bias") and layer.bias is not None:
            raw_weights[f"Layer {layer_idx} ({layer.__class__.__name__}) - bias"] = layer.bias.data
    return raw_weights


def get_model_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Captures the L2 norm of every parameter gradient across the model layers."""
    grad_norms: Dict[str, float] = {}
    if isinstance(model, nn.Sequential):
        for layer_idx, layer in enumerate(model):
            if hasattr(layer, "weight") and layer.weight is not None and layer.weight.grad is not None:
                grad_norms[f"Layer {layer_idx} ({layer.__class__.__name__}) - weight"] = float(np.linalg.norm(layer.weight.grad))
            if hasattr(layer, "bias") and layer.bias is not None and layer.bias.grad is not None:
                grad_norms[f"Layer {layer_idx} ({layer.__class__.__name__}) - bias"] = float(np.linalg.norm(layer.bias.grad))
    else:
        for idx, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad_norms[f"Param {idx} {list(param.shape)}"] = float(np.linalg.norm(param.grad))
    return grad_norms


# -----------------------------------------------------------------------------
# Decision Boundary & Training Curves Plotting
# -----------------------------------------------------------------------------

def plot_plotly_custom_canvas(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Custom 2D Canvas (Click anywhere to drop data points)",
) -> Optional[Any]:
    """Generates an interactive canvas for custom data point placement with direct click support."""
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()

    # 1. Dark Grid Background Surface
    fig.add_shape(
        type="rect",
        x0=-2.5, y0=-2.5, x1=2.5, y1=2.5,
        fillcolor="#0B1120",
        line=dict(color="#334155", width=1.5),
        layer="below",
    )

    # 2. Dense Invisible Click-Capture Grid (enables direct click placement anywhere)
    grid_click_x = np.linspace(-2.45, 2.45, 50)
    grid_click_y = np.linspace(-2.45, 2.45, 50)
    cg_x, cg_y = np.meshgrid(grid_click_x, grid_click_y)
    fig.add_trace(
        go.Scatter(
            x=cg_x.ravel(),
            y=cg_y.ravel(),
            mode="markers",
            name="Click Capture Grid",
            marker=dict(size=20, color="rgba(255, 255, 255, 0.01)"),
            hoverinfo="none",
            showlegend=False,
            selected=dict(marker=dict(opacity=0.01)),
            unselected=dict(marker=dict(opacity=0.01)),
        )
    )

    # 3. Placed Data Points
    if len(X) > 0 and len(y) > 0:
        mask_0 = (y == 0)
        mask_1 = (y == 1)

        if np.any(mask_0):
            fig.add_trace(
                go.Scatter(
                    x=X[mask_0, 0],
                    y=X[mask_0, 1],
                    mode="markers",
                    name=f"Class 0 (Blue) [N={int(np.sum(mask_0))}]",
                    marker=dict(
                        color="#3B82F6",
                        size=9,
                        line=dict(width=1.5, color="#FFFFFF"),
                        opacity=0.95,
                    ),
                    selected=dict(marker=dict(opacity=0.95)),
                    unselected=dict(marker=dict(opacity=0.95)),
                    hovertext=[f"Class 0 ({X[i, 0]:.2f}, {X[i, 1]:.2f})" for i in np.where(mask_0)[0]],
                    hoverinfo="text",
                )
            )

        if np.any(mask_1):
            fig.add_trace(
                go.Scatter(
                    x=X[mask_1, 0],
                    y=X[mask_1, 1],
                    mode="markers",
                    name=f"Class 1 (Red) [N={int(np.sum(mask_1))}]",
                    marker=dict(
                        color="#EF4444",
                        size=9,
                        line=dict(width=1.5, color="#FFFFFF"),
                        symbol="diamond",
                        opacity=0.95,
                    ),
                    selected=dict(marker=dict(opacity=0.95)),
                    unselected=dict(marker=dict(opacity=0.95)),
                    hovertext=[f"Class 1 ({X[i, 0]:.2f}, {X[i, 1]:.2f})" for i in np.where(mask_1)[0]],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#F8FAFC"),
            y=0.98,
            x=0.02,
            xanchor="left",
            yanchor="top",
            pad=dict(b=10),
        ),
        xaxis=dict(
            title=dict(text="Feature x1", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#1E293B",
            zeroline=True,
            zerolinecolor="#475569",
            fixedrange=True,
            range=[-2.5, 2.5],
        ),
        yaxis=dict(
            title=dict(text="Feature x2", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#1E293B",
            zeroline=True,
            zerolinecolor="#475569",
            fixedrange=True,
            range=[-2.5, 2.5],
        ),
        clickmode="event+select",
        margin=dict(l=35, r=35, t=85, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#CBD5E1"),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=500,
    )
    return fig


def plot_plotly_decision_boundary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    test_point: Optional[Tuple[float, float]] = None,
    title: str = "Learned Decision Boundary (Click to Predict)",
) -> Optional[Any]:
    """Generates an interactive Plotly decision boundary contour plot with point click selection."""
    if not HAS_PLOTLY:
        return None

    # Full-span grid covering the complete visible axis domain
    grid_x = np.linspace(-2.5, 2.5, 150).astype(np.float32)
    grid_y = np.linspace(-2.5, 2.5, 150).astype(np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    grid_tensor = Tensor(grid_points, requires_grad=False)
    with no_grad():
        grid_logits = model(grid_tensor)

    # Softmax probability of class 1
    exp_logits = np.exp(grid_logits.data - np.max(grid_logits.data, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
    Z = probs.reshape(xx.shape)

    fig = go.Figure()

    # 1. Smooth Decision Boundary Surface (No black line banding)
    fig.add_trace(
        go.Contour(
            x=grid_x,
            y=grid_y,
            z=Z,
            colorscale="Spectral",
            reversescale=True,
            opacity=0.85,
            showscale=True,
            contours=dict(
                start=0.0,
                end=1.0,
                size=0.04,
                coloring="heatmap",
                showlines=False,
            ),
            line=dict(width=0),
            colorbar=dict(
                title=dict(text="P(Class 1)", font=dict(size=11, color="white")),
                tickfont=dict(color="white"),
            ),
            hoverinfo="x+y+z",
            name="Decision Surface",
        )
    )

    # 1b. Clean single threshold contour line at P = 0.5
    fig.add_trace(
        go.Contour(
            x=grid_x,
            y=grid_y,
            z=Z,
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.0,
                coloring="none",
                showlines=True,
            ),
            line=dict(color="#FFFFFF", width=2, dash="dash"),
            name="Decision Border (P=0.5)",
            hoverinfo="skip",
        )
    )

    # 2. Dataset Scatter Points (Class 0 and Class 1)
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    fig.add_trace(
        go.Scatter(
            x=X[mask_0, 0],
            y=X[mask_0, 1],
            mode="markers",
            name="Class 0",
            marker=dict(
                color="#3B82F6",
                size=8,
                line=dict(width=1, color="#0F172A"),
                opacity=0.95,
            ),
            hovertext=[f"Class 0 ({X[i, 0]:.2f}, {X[i, 1]:.2f})" for i in np.where(mask_0)[0]],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X[mask_1, 0],
            y=X[mask_1, 1],
            mode="markers",
            name="Class 1",
            marker=dict(
                color="#EF4444",
                size=8,
                line=dict(width=1, color="#0F172A"),
                opacity=0.95,
            ),
            hovertext=[f"Class 1 ({X[i, 0]:.2f}, {X[i, 1]:.2f})" for i in np.where(mask_1)[0]],
            hoverinfo="text",
        )
    )

    # 3. Active Test Point Overlay (Gold Star)
    if test_point is not None:
        tx, ty = test_point
        fig.add_trace(
            go.Scatter(
                x=[tx],
                y=[ty],
                mode="markers",
                name=f"Active ({tx:.2f}, {ty:.2f})",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#FFD700",
                    line=dict(width=2, color="#000000"),
                ),
                hovertext=[f"Active Test Point: ({tx:.2f}, {ty:.2f})"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#F8FAFC"),
            y=0.98,
            x=0.02,
            xanchor="left",
            yanchor="top",
            pad=dict(b=10),
        ),
        xaxis=dict(
            title=dict(text="Feature x1", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155",
            zeroline=False,
            fixedrange=True,
            range=[-2.5, 2.5],
        ),
        yaxis=dict(
            title=dict(text="Feature x2", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155",
            zeroline=False,
            fixedrange=True,
            range=[-2.5, 2.5],
        ),
        dragmode=False,
        margin=dict(l=35, r=35, t=85, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#CBD5E1"),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=480,
    )

    return fig


def plot_plotly_gradient_norms(grad_norms: Dict[str, float], title: str = "Layer-by-Layer Gradient Flow Telemetry") -> Optional[Any]:
    """Generates an interactive horizontal bar chart of gradient L2 norms using Plotly."""
    if not HAS_PLOTLY or not grad_norms:
        return None

    names = list(grad_norms.keys())
    values = list(grad_norms.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Viridis",
                line=dict(color="#38BDF8", width=1.5),
            ),
            text=[f"{v:.5f}" for v in values],
            textposition="auto",
            hoverinfo="x+y",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#F8FAFC")),
        xaxis=dict(
            title=dict(text="Gradient L2 Norm", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155",
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(text="Parameter / Layer", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155",
            autorange="reversed",
            fixedrange=True,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        margin=dict(l=20, r=20, t=35, b=20),
        height=max(220, len(names) * 35 + 80),
        dragmode=False,
    )
    return fig


def plot_dashboard_figures(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_hist: List[float],
    acc_hist: List[float],
    test_point: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Generates two separate figures for the decision boundary and metrics."""
    # 1. Decision Boundary Figure
    fig_boundary, ax_b = plt.subplots(figsize=(6, 5), dpi=100, layout="constrained")
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 120), np.linspace(-2.5, 2.5, 120))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    grid_tensor = Tensor(grid_points, requires_grad=False)
    with no_grad():
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

    if test_point is not None:
        tx, ty = test_point
        ax_b.scatter(
            tx,
            ty,
            c="#FFD700",
            edgecolors="#000000",
            marker="*",
            s=320,
            linewidths=2.0,
            zorder=15,
            label=f"Active Point ({tx:.2f}, {ty:.2f})",
        )
        ax_b.legend(loc="upper right", framealpha=0.88, fontsize=8)

    ax_b.set_title("Learned Decision Boundary", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Feature x1", fontsize=10)
    ax_b.set_ylabel("Feature x2", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    # 2. Loss & Accuracy Progression Figure
    fig_metrics, ax_loss = plt.subplots(figsize=(6, 5), dpi=100, layout="constrained")
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

    return fig_boundary, fig_metrics


def plot_gradient_norms(grad_norms: Dict[str, float]) -> plt.Figure:
    """Plots a layer-by-layer gradient norm diagnostic bar chart."""
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(grad_norms) * 0.45 + 1.0)), dpi=100, layout="constrained")
    names = list(grad_norms.keys())
    values = list(grad_norms.values())

    bars = ax.barh(names, values, color="#38BDF8", edgecolor="#0284C7", alpha=0.85)
    ax.set_xlabel("Gradient L2 Norm", fontweight="bold", fontsize=9)
    ax.set_title("Layer-by-Layer Gradient Magnitude Diagnostic", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.0005, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=8)

    return fig


def plot_comparison_dashboard_figures(
    model_a: nn.Module,
    model_b: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_hist_a: List[float],
    acc_hist_a: List[float],
    loss_hist_b: List[float],
    acc_hist_b: List[float],
    test_point: Optional[Tuple[float, float]] = None,
    title_a: str = "Model A",
    title_b: str = "Model B",
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Generates two separate decision boundary figures and one comparative training curve figure."""
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 120), np.linspace(-2.5, 2.5, 120))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # 1. Model A Decision Boundary
    fig_a, ax_a = plt.subplots(figsize=(5.5, 4.2), dpi=100, layout="constrained")
    model_a.eval()
    with no_grad():
        logits_a = model_a(Tensor(grid_points, requires_grad=False))
    exp_a = np.exp(logits_a.data - np.max(logits_a.data, axis=-1, keepdims=True))
    probs_a = (exp_a / np.sum(exp_a, axis=-1, keepdims=True))[:, 1]
    Z_a = probs_a.reshape(xx.shape)

    c_a = ax_a.contourf(xx, yy, Z_a, levels=40, cmap="Spectral_r", alpha=0.85)
    ax_a.contour(xx, yy, Z_a, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")
    fig_a.colorbar(c_a, ax=ax_a, label="P(Class = 1)")
    ax_a.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral_r", edgecolors="black", linewidths=0.6, s=28, alpha=0.9)
    if test_point is not None:
        tx, ty = test_point
        ax_a.scatter(tx, ty, c="#FFD700", edgecolors="#000000", marker="*", s=300, linewidths=2.0, zorder=15, label=f"Point ({tx:.2f}, {ty:.2f})")
        ax_a.legend(loc="upper right", framealpha=0.88, fontsize=8)
    ax_a.set_title(title_a, fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Feature x1", fontsize=9)
    ax_a.set_ylabel("Feature x2", fontsize=9)
    ax_a.grid(True, linestyle=":", alpha=0.4)

    # 2. Model B Decision Boundary
    fig_b, ax_b = plt.subplots(figsize=(5.5, 4.2), dpi=100, layout="constrained")
    model_b.eval()
    with no_grad():
        logits_b = model_b(Tensor(grid_points, requires_grad=False))
    exp_b = np.exp(logits_b.data - np.max(logits_b.data, axis=-1, keepdims=True))
    probs_b = (exp_b / np.sum(exp_b, axis=-1, keepdims=True))[:, 1]
    Z_b = probs_b.reshape(xx.shape)

    c_b = ax_b.contourf(xx, yy, Z_b, levels=40, cmap="Spectral_r", alpha=0.85)
    ax_b.contour(xx, yy, Z_b, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")
    fig_b.colorbar(c_b, ax=ax_b, label="P(Class = 1)")
    ax_b.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral_r", edgecolors="black", linewidths=0.6, s=28, alpha=0.9)
    if test_point is not None:
        tx, ty = test_point
        ax_b.scatter(tx, ty, c="#FFD700", edgecolors="#000000", marker="*", s=300, linewidths=2.0, zorder=15, label=f"Point ({tx:.2f}, {ty:.2f})")
        ax_b.legend(loc="upper right", framealpha=0.88, fontsize=8)
    ax_b.set_title(title_b, fontsize=10, fontweight="bold")
    ax_b.set_xlabel("Feature x1", fontsize=9)
    ax_b.set_ylabel("Feature x2", fontsize=9)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    # 3. Comparative Curves
    fig_curves, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100, layout="constrained")
    epochs_range_a = range(1, len(loss_hist_a) + 1)
    epochs_range_b = range(1, len(loss_hist_b) + 1)

    # Loss plot
    ax_loss.plot(epochs_range_a, loss_hist_a, color="#EF4444", linewidth=2.0, label="Model A Loss")
    ax_loss.plot(epochs_range_b, loss_hist_b, color="#3B82F6", linewidth=2.0, linestyle="--", label="Model B Loss")
    ax_loss.set_xlabel("Epoch", fontweight="bold", fontsize=9)
    ax_loss.set_ylabel("CrossEntropy Loss", fontweight="bold", fontsize=9)
    ax_loss.set_title("Loss Convergence Comparison", fontsize=10, fontweight="bold")
    ax_loss.grid(True, linestyle=":", alpha=0.4)
    ax_loss.legend(loc="upper right", fontsize=8)

    # Accuracy plot
    ax_acc.plot(epochs_range_a, acc_hist_a, color="#EF4444", linewidth=2.0, label="Model A Accuracy")
    ax_acc.plot(epochs_range_b, acc_hist_b, color="#3B82F6", linewidth=2.0, linestyle="--", label="Model B Accuracy")
    ax_acc.set_xlabel("Epoch", fontweight="bold", fontsize=9)
    ax_acc.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=9)
    ax_acc.set_title("Accuracy Progression Comparison", fontsize=10, fontweight="bold")
    ax_acc.set_ylim([0, 105])
    ax_acc.grid(True, linestyle=":", alpha=0.4)
    ax_acc.legend(loc="lower right", fontsize=8)

    return fig_a, fig_b, fig_curves


def plot_plotly_rover_path(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    ray_history: Optional[List[List[Dict[str, Any]]]] = None,
    start_pos: Tuple[float, float] = (-1.80, 1.20),
    target_pos: Tuple[float, float] = (1.20, 0.70),
    show_rays: bool = True,
    step_index: Optional[int] = None,
    title: str = "Autonomous Neural Rover Simulation",
    base_fig: Optional[go.Figure] = None,
) -> Optional[Any]:
    """Generates an interactive Plotly visualization of the rover path on the neural potential field."""
    if not HAS_PLOTLY:
        return None

    if base_fig is not None:
        fig = go.Figure(base_fig)
    else:
        # Full-span decision surface
        grid_x = np.linspace(-2.5, 2.5, 150).astype(np.float32)
        grid_y = np.linspace(-2.5, 2.5, 150).astype(np.float32)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

        model.eval()
        with no_grad():
            grid_logits = model(Tensor(grid_points, requires_grad=False))
        exp_logits = np.exp(grid_logits.data - np.max(grid_logits.data, axis=-1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
        Z = probs.reshape(xx.shape)

        fig = go.Figure()

        # 1. Background Obstacle Potential Field
        fig.add_trace(
            go.Contour(
                x=grid_x,
                y=grid_y,
                z=Z,
                colorscale="Spectral",
                reversescale=True,
                opacity=0.80,
                showscale=True,
                contours=dict(
                    start=0.0,
                    end=1.0,
                    size=0.04,
                    coloring="heatmap",
                    showlines=False,
                ),
                line=dict(width=0),
                colorbar=dict(
                    title=dict(text="Hazard P(Class 1)", font=dict(size=11, color="white")),
                    tickfont=dict(color="white"),
                ),
                hoverinfo="x+y+z",
                name="Hazard Surface",
            )
        )

        # 1b. Boundary contour at 0.5
        fig.add_trace(
            go.Contour(
                x=grid_x,
                y=grid_y,
                z=Z,
                showscale=False,
                contours=dict(start=0.5, end=0.5, size=0.0, coloring="none", showlines=True),
                line=dict(color="#FFFFFF", width=2, dash="dash"),
                name="Obstacle Perimeter (P=0.5)",
                hoverinfo="skip",
            )
        )

        # 2. Dataset Scatter Points (faded background reference)
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        fig.add_trace(
            go.Scatter(
                x=X[mask_0, 0],
                y=X[mask_0, 1],
                mode="markers",
                name="Free Space Data",
                marker=dict(color="#3B82F6", size=5, opacity=0.35),
                selected=dict(marker=dict(opacity=0.35)),
                unselected=dict(marker=dict(opacity=0.35)),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=X[mask_1, 0],
                y=X[mask_1, 1],
                mode="markers",
                name="Obstacle Data",
                marker=dict(color="#EF4444", size=5, opacity=0.35),
                selected=dict(marker=dict(opacity=0.35)),
                unselected=dict(marker=dict(opacity=0.35)),
                hoverinfo="skip",
            )
        )

        # 3. Dense Invisible Click-Capture Grid (enables direct click waypoint placement anywhere on canvas)
        grid_click_x = np.linspace(-2.45, 2.45, 50)
        grid_click_y = np.linspace(-2.45, 2.45, 50)
        cg_x, cg_y = np.meshgrid(grid_click_x, grid_click_y)
        fig.add_trace(
            go.Scatter(
                x=cg_x.ravel(),
                y=cg_y.ravel(),
                mode="markers",
                name="Click Capture Grid",
                marker=dict(size=20, color="rgba(255, 255, 255, 0.01)"),
                hoverinfo="none",
                showlegend=False,
                selected=dict(marker=dict(opacity=0.01)),
                unselected=dict(marker=dict(opacity=0.01)),
            )
        )

        # 6. Start Marker (Green circle)
        fig.add_trace(
            go.Scatter(
                x=[start_pos[0]],
                y=[start_pos[1]],
                mode="markers+text",
                name="Start (S)",
                text=["START"],
                textposition="top center",
                textfont=dict(color="#10B981", size=11),
                marker=dict(symbol="circle", size=16, color="#10B981", line=dict(color="#FFFFFF", width=2)),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                hovertext=[f"Start Position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})"],
                hoverinfo="text",
            )
        )

        # 7. Target Marker (Red Diamond / Goal)
        fig.add_trace(
            go.Scatter(
                x=[target_pos[0]],
                y=[target_pos[1]],
                mode="markers+text",
                name="Target (T)",
                text=["TARGET"],
                textposition="top center",
                textfont=dict(color="#EF4444", size=11),
                marker=dict(symbol="diamond", size=16, color="#EF4444", line=dict(color="#FFFFFF", width=2)),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                hovertext=[f"Target Goal: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"],
                hoverinfo="text",
            )
        )

        fig.update_layout(
            uirevision="rover_sim_viewport",
            title=dict(
                text=title,
                font=dict(size=13.5, color="#F8FAFC"),
                y=0.985,
                x=0.015,
                xanchor="left",
                yanchor="top",
                pad=dict(b=14),
            ),
            xaxis=dict(
                title=dict(text="Coordinate x1", font=dict(color="#CBD5E1")),
                tickfont=dict(color="#CBD5E1"),
                gridcolor="#334155",
                zeroline=False,
                fixedrange=True,
                range=[-2.5, 2.5],
            ),
            yaxis=dict(
                title=dict(text="Coordinate x2", font=dict(color="#CBD5E1")),
                tickfont=dict(color="#CBD5E1"),
                gridcolor="#334155",
                zeroline=False,
                fixedrange=True,
                range=[-2.5, 2.5],
            ),
            clickmode="event+select",
            margin=dict(l=45, r=45, t=115, b=45),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=9.5, color="#CBD5E1"),
                bgcolor="rgba(15, 23, 42, 0.75)",
                bordercolor="#334155",
                borderwidth=1,
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            height=560,
        )

    has_sim = trajectory is not None and len(trajectory) > 0

    if has_sim:
        # Determine slice of trajectory to render
        if step_index is not None:
            clamped_step = max(0, min(step_index, len(trajectory) - 1))
            active_trajectory = trajectory[:clamped_step + 1]
        else:
            clamped_step = len(trajectory) - 1
            active_trajectory = trajectory

        # 4. Radar Sensor Rays
        def get_ray_color(h: float) -> str:
            if h < 0.25:
                return "#10B981"  # Green: Clear
            elif h <= 0.55:
                return "#FBBF24"  # Yellow: Warning
            return "#EF4444"      # Red: Breach

        if show_rays and ray_history:
            if step_index is not None:
                # Render prominent radar cone for the active inspected step
                if clamped_step < len(ray_history):
                    p_origin = trajectory[clamped_step]
                    for r_idx, r in enumerate(ray_history[clamped_step]):
                        r_end = r["endpoint"]
                        h_val = r["hazard"]
                        r_col = get_ray_color(h_val)
                        fig.add_trace(
                            go.Scatter(
                                x=[p_origin[0], r_end[0]],
                                y=[p_origin[1], r_end[1]],
                                mode="lines+markers",
                                line=dict(color=r_col, width=2.8, dash="solid"),
                                marker=dict(size=6, color=r_col, symbol="diamond"),
                                selected=dict(marker=dict(opacity=1.0)),
                                unselected=dict(marker=dict(opacity=1.0)),
                                hovertext=[f"Origin ({p_origin[0]:.2f}, {p_origin[1]:.2f})", f"Ray #{r_idx+1}: Hazard P={h_val*100:.1f}%"],
                                hoverinfo="text",
                                showlegend=False,
                            )
                        )
            else:
                # Render sampled historical radar cones across entire mission
                step_stride = max(1, len(ray_history) // 8)
                for s_idx in range(0, len(ray_history), step_stride):
                    p_origin = trajectory[s_idx]
                    for r in ray_history[s_idx]:
                        r_end = r["endpoint"]
                        h_val = r["hazard"]
                        r_col = get_ray_color(h_val)
                        fig.add_trace(
                            go.Scatter(
                                x=[p_origin[0], r_end[0]],
                                y=[p_origin[1], r_end[1]],
                                mode="lines",
                                line=dict(color=r_col, width=1.4, dash="dot"),
                                hoverinfo="skip",
                                showlegend=False,
                                opacity=0.65,
                            )
                        )

        # 5. Rover Path Trajectory
        traj_x = [p[0] for p in active_trajectory]
        traj_y = [p[1] for p in active_trajectory]
        fig.add_trace(
            go.Scatter(
                x=traj_x,
                y=traj_y,
                mode="lines+markers",
                name="Rover Path",
                line=dict(color="#FBBF24", width=3.5),
                marker=dict(color="#F59E0B", size=6, symbol="circle"),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                hovertext=[f"Step {idx}: ({p[0]:.2f}, {p[1]:.2f})" for idx, p in enumerate(active_trajectory)],
                hoverinfo="text",
            )
        )

        # 8. Active Rover Position Marker (Glowing gold cross at current step)
        curr_rover_pos = active_trajectory[-1]
        fig.add_trace(
            go.Scatter(
                x=[curr_rover_pos[0]],
                y=[curr_rover_pos[1]],
                mode="markers+text",
                name="Rover Position",
                text=[f"Step {clamped_step}"],
                textposition="bottom center",
                textfont=dict(color="#FBBF24", size=11, family="sans-serif"),
                marker=dict(symbol="cross", size=14, color="#FBBF24", line=dict(color="#FFFFFF", width=2)),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                hovertext=[f"Current Rover: ({curr_rover_pos[0]:.2f}, {curr_rover_pos[1]:.2f}) at Step {clamped_step}"],
                hoverinfo="text",
            )
        )

    fig.update_layout(title_text=title, uirevision="rover_sim_viewport")
    return fig


def plot_rover_path_mpl(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    start_pos: Tuple[float, float] = (-1.80, 1.20),
    target_pos: Tuple[float, float] = (1.20, 0.70),
) -> plt.Figure:
    """Matplotlib fallback rendering of the neural rover path."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100, layout="constrained")
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 120), np.linspace(-2.5, 2.5, 120))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    with no_grad():
        grid_logits = model(Tensor(grid_points, requires_grad=False))
    exp_logits = np.exp(grid_logits.data - np.max(grid_logits.data, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
    Z = probs.reshape(xx.shape)

    c = ax.contourf(xx, yy, Z, levels=40, cmap="Spectral_r", alpha=0.80)
    ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=1.8, linestyles="--")
    fig.colorbar(c, ax=ax, label="Hazard P(Class 1)")

    # Rover path (if simulated)
    if trajectory is not None and len(trajectory) > 0:
        tx = [p[0] for p in trajectory]
        ty = [p[1] for p in trajectory]
        ax.plot(tx, ty, color="#F59E0B", linewidth=2.5, marker="o", markersize=3, label="Rover Trajectory")

    # Start & Target
    ax.scatter(start_pos[0], start_pos[1], color="#10B981", s=140, edgecolors="black", zorder=10, label="Start (S)")
    ax.scatter(target_pos[0], target_pos[1], color="#EF4444", s=140, marker="D", edgecolors="black", zorder=10, label="Target (T)")

    ax.set_title("Autonomous Neural Pathfinding", fontsize=11, fontweight="bold")
    ax.set_xlabel("Coordinate x1", fontsize=9)
    ax.set_ylabel("Coordinate x2", fontsize=9)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    return fig


# -----------------------------------------------------------------------------
# MNIST Canvas Preprocessing Pipeline (Multi-Digit Connected Components)
# -----------------------------------------------------------------------------

def label_components_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected component labeling using 8-connectivity BFS in pure Python/NumPy."""
    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    visited = np.zeros((H, W), dtype=bool)

    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        current_label += 1
        queue = [(y, x)]
        visited[y, x] = True
        labeled[y, x] = current_label

        head = 0
        while head < len(queue):
            cy, cx = queue[head]
            head += 1

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labeled[ny, nx] = current_label
                            queue.append((ny, nx))

    return labeled, current_label


def segment_and_preprocess_digits(canvas_result) -> List[np.ndarray]:
    """
    Extracts, segments, centers, and normalizes one or multiple drawn digits
    from the canvas into a list of (28, 28) float32 arrays ordered left-to-right.
    """
    if canvas_result is None or canvas_result.image_data is None:
        return []

    raw_stroke = canvas_result.image_data[:, :, 0].astype(np.float32)
    if np.max(raw_stroke) <= 20.0:
        return []

    mask = raw_stroke > 20.0
    if not np.any(mask):
        return []

    # Connected component labeling
    if HAS_SCIPY:
        labeled, num_features = ndi.label(mask)
    else:
        labeled, num_features = label_components_numpy(mask)

    if num_features == 0:
        return []

    # Extract initial bounding boxes and filter noise
    boxes = []
    for k in range(1, num_features + 1):
        comp_y, comp_x = np.where(labeled == k)
        if len(comp_y) < 15:  # filter noise speckles
            continue
        ymin, ymax = int(comp_y.min()), int(comp_y.max())
        xmin, xmax = int(comp_x.min()), int(comp_x.max())
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        if h < 5 and w < 5:
            continue
        boxes.append({"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax})

    if not boxes:
        return []

    # Merge overlapping / closely stacked strokes belonging to the same digit
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            b1 = dict(boxes[i])
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                b2 = boxes[j]
                x_overlap = min(b1["xmax"], b2["xmax"]) - max(b1["xmin"], b2["xmin"])
                x_dist = max(0, max(b1["xmin"], b2["xmin"]) - min(b1["xmax"], b2["xmax"]))
                y_overlap = min(b1["ymax"], b2["ymax"]) - max(b1["ymin"], b2["ymin"])
                y_dist = max(0, max(b1["ymin"], b2["ymin"]) - min(b1["ymax"], b2["ymax"]))

                # Merge if horizontal overlap exists or close and vertically aligned
                should_merge = False
                if x_overlap >= -4 and (y_overlap >= -8 or (y_dist <= 18 and (x_overlap > 0 or x_dist <= 6))):
                    should_merge = True

                if should_merge:
                    b1 = {
                        "ymin": min(b1["ymin"], b2["ymin"]),
                        "ymax": max(b1["ymax"], b2["ymax"]),
                        "xmin": min(b1["xmin"], b2["xmin"]),
                        "xmax": max(b1["xmax"], b2["xmax"]),
                    }
                    used.add(j)
                    merged = True
            new_boxes.append(b1)
            used.add(i)
        boxes = new_boxes

    # Sort bounding boxes strictly left-to-right
    boxes.sort(key=lambda b: b["xmin"])

    processed_digits: List[np.ndarray] = []
    for b in boxes:
        cropped = raw_stroke[b["ymin"]:b["ymax"] + 1, b["xmin"]:b["xmax"] + 1]
        h, w = cropped.shape
        if h <= 0 or w <= 0:
            continue

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

        digit_28x28 = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        digit_28x28[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        total_mass = digit_28x28.sum()
        if total_mass > 0:
            gy, gx = np.mgrid[0:28, 0:28]
            cy = (gy * digit_28x28).sum() / total_mass
            cx = (gx * digit_28x28).sum() / total_mass
            shift_y = int(np.clip(np.round(14.0 - cy), -4, 4))
            shift_x = int(np.clip(np.round(14.0 - cx), -4, 4))

            shifted = np.zeros_like(digit_28x28)
            for r in range(28):
                for c in range(28):
                    nr, nc = r + shift_y, c + shift_x
                    if 0 <= nr < 28 and 0 <= nc < 28:
                        shifted[nr, nc] = digit_28x28[r, c]
            digit_28x28 = shifted

        digit_28x28 = np.clip(digit_28x28 / 255.0, 0.0, 1.0)
        processed_digits.append(digit_28x28)

    return processed_digits


def preprocess_canvas_image(canvas_result) -> Optional[np.ndarray]:
    """Backward-compatible single-digit preprocessor returning (1, 28, 28) or None."""
    digits = segment_and_preprocess_digits(canvas_result)
    if not digits:
        return None
    return digits[0].reshape(1, 28, 28)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax using the Log-Sum-Exp trick."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def plot_plotly_digit_probabilities(probs: np.ndarray) -> Optional[Any]:
    """Generates an interactive horizontal bar chart of digit probabilities 0-9 using Plotly."""
    if not HAS_PLOTLY:
        return None
    classes = [f"Digit {c}" for c in range(10)]
    probs_pct = [float(p * 100) for p in probs]
    max_idx = int(np.argmax(probs))
    colors = ["#3B82F6" if i != max_idx else "#8B5CF6" for i in range(10)]

    fig = go.Figure(
        go.Bar(
            x=probs_pct,
            y=classes,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
            text=[f"{v:.1f}%" if v > 0.5 else "" for v in probs_pct],
            textposition="auto",
            hoverinfo="x+y",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155",
            range=[0, 105],
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(color="#CBD5E1", size=11),
            gridcolor="#334155",
            autorange="reversed",
            fixedrange=True,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        margin=dict(l=10, r=20, t=10, b=10),
        height=320,
        dragmode=False,
    )
    return fig


def plot_probability_distribution(probs: np.ndarray) -> plt.Figure:
    """Creates a horizontal bar chart of class probabilities 0-9."""
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100, layout="constrained")
    classes = np.arange(10)
    colors = ["#3B82F6" if p < np.max(probs) else "#8B5CF6" for p in probs]

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

    return fig


# -----------------------------------------------------------------------------
# Tab 1: 2D Decision Boundaries
# -----------------------------------------------------------------------------

def render_single_model_studio():
    """Renders the Single Model 2D Decision Boundary training & interactive inference laboratory."""
    # 0. Session state initialization for Custom Canvas
    if "custom_points_X" not in st.session_state or "custom_points_y" not in st.session_state:
        init_X, init_y = load_preset_checkerboard_xor(n_samples=200, noise=0.04)
        st.session_state["custom_points_X"] = init_X
        st.session_state["custom_points_y"] = init_y

    # Process custom canvas clicks before sidebar widgets instantiate
    for c_key in ["tab1_custom_canvas", "custom_2d_canvas"]:
        canvas_event = st.session_state.get(c_key)
        if isinstance(canvas_event, dict):
            pts = canvas_event.get("selection", {}).get("points", [])
            if pts and "x" in pts[0] and "y" in pts[0]:
                cx = float(np.round(pts[0]["x"], 2))
                cy = float(np.round(pts[0]["y"], 2))
                click_tag = (c_key, cx, cy, len(st.session_state.get("custom_points_X", [])))
                if st.session_state.get("_last_canvas_click_tag") != click_tag:
                    st.session_state["_last_canvas_click_tag"] = click_tag
                    tool_mode = st.session_state.get("canvas_brush_mode", "Drop Class 0 (Blue)")
                    density = st.session_state.get("canvas_brush_density", "Single Point")

                    if "Eraser" in tool_mode:
                        # Eraser mode: remove points within distance < 0.28
                        cur_X = st.session_state.get("custom_points_X", [])
                        cur_y = st.session_state.get("custom_points_y", [])
                        if cur_X:
                            pts_arr = np.array(cur_X)
                            dists = np.sqrt(np.sum((pts_arr - np.array([cx, cy])) ** 2, axis=1))
                            within_radius = np.where(dists < 0.28)[0]
                            if len(within_radius) > 0:
                                keep_mask = np.ones(len(cur_X), dtype=bool)
                                keep_mask[within_radius] = False
                                st.session_state["custom_points_X"] = [cur_X[i] for i in range(len(cur_X)) if keep_mask[i]]
                                st.session_state["custom_points_y"] = [cur_y[i] for i in range(len(cur_y)) if keep_mask[i]]
                    else:
                        c_idx = 1 if "Class 1" in tool_mode else 0
                        if "Cluster" in density:
                            rng_c = np.random.default_rng()
                            for _ in range(5):
                                jx = float(np.clip(round(cx + rng_c.normal(0, 0.08), 3), -2.4, 2.4))
                                jy = float(np.clip(round(cy + rng_c.normal(0, 0.08), 3), -2.4, 2.4))
                                st.session_state["custom_points_X"].append([jx, jy])
                                st.session_state["custom_points_y"].append(c_idx)
                        elif "Dense Line" in density:
                            for t in np.linspace(-0.18, 0.18, 8):
                                lx = float(np.clip(round(cx + t, 3), -2.4, 2.4))
                                ly = float(np.clip(round(cy, 3), -2.4, 2.4))
                                st.session_state["custom_points_X"].append([lx, ly])
                                st.session_state["custom_points_y"].append(c_idx)
                        else:
                            st.session_state["custom_points_X"].append([cx, cy])
                            st.session_state["custom_points_y"].append(c_idx)

                    # Invalidate cached model since custom data changed
                    if st.session_state.get("trained_2d_model", {}).get("dataset_name") in ["Custom Canvas", "Custom / Interactive Canvas"]:
                        st.session_state.pop("trained_2d_model", None)

    # Callbacks for custom canvas controls
    def _cb_load_preset_from_selector():
        choice = st.session_state.get("canvas_preset_selector", "-- Select Preset --")
        if choice in PRESET_GENERATOR_MAP:
            p_x, p_y = PRESET_GENERATOR_MAP[choice](n_samples=260, noise=0.04)
            st.session_state["custom_points_X"] = p_x
            st.session_state["custom_points_y"] = p_y
            if st.session_state.get("trained_2d_model", {}).get("dataset_name") in ["Custom Canvas", "Custom / Interactive Canvas"]:
                st.session_state.pop("trained_2d_model", None)

    def _cb_clear_custom_canvas():
        st.session_state["custom_points_X"] = []
        st.session_state["custom_points_y"] = []
        if st.session_state.get("trained_2d_model", {}).get("dataset_name") in ["Custom Canvas", "Custom / Interactive Canvas"]:
            st.session_state.pop("trained_2d_model", None)

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("Experiment Controls")
        
        # 1. Dataset Settings
        st.subheader("1. Dataset Configuration")
        dataset_name = st.selectbox(
            "Dataset Topology",
            ["Island in Moat", "Checkerboard XOR", "Corridor Maze", "Double Cross", "Spirals", "Custom / Interactive Canvas"],
            index=2,
            key="single_dataset_name",
        )
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            st.caption("Interactive Canvas Drawing Controls:")
            st.radio(
                "Tool:",
                ["Drop Class 0 (Blue)", "Drop Class 1 (Red)", "Eraser / Remove Nearest"],
                horizontal=True,
                key="canvas_brush_mode",
                help="Select whether clicking drops Class 0, Class 1, or erases nearby points.",
            )
            st.radio(
                "Density:",
                ["Single Point", "Cluster (5 pts)", "Dense Line (8 pts)"],
                horizontal=True,
                key="canvas_brush_density",
                help="Single point drops 1 coordinate; Cluster drops 5 jittered points; Dense line drops 8 points.",
            )
            st.selectbox(
                "Load Preset Pattern:",
                ["-- Select Preset --", "Island in Moat", "Checkerboard XOR", "Corridor Maze", "Double Cross", "Spirals"],
                key="canvas_preset_selector",
            )
            c_p1, c_p2 = st.columns(2)
            c_p1.button("Load Pattern", width="stretch", key="btn_load_pattern_side", on_click=_cb_load_preset_from_selector, help="Load selected geometric preset.")
            c_p2.button("Clear All Points", width="stretch", key="btn_clear_points_side", on_click=_cb_clear_custom_canvas, help="Clears all custom points from canvas.")
            n_samples = len(st.session_state.get("custom_points_X", []))
            noise = 0.0
            seed = 42
        else:
            n_samples = st.slider("Sample Count", min_value=100, max_value=1000, value=500, step=50, key="single_n_samples")
            noise = st.slider("Noise Level", min_value=0.0, max_value=0.3, value=0.06, step=0.01, key="single_noise")
            seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1, key="single_seed")

        # 2. Architecture Settings
        st.subheader("2. Model Architecture")
        num_layers = st.slider("Hidden Layers", min_value=1, max_value=4, value=2, step=1, key="single_num_layers")
        hidden_dim = st.select_slider("Hidden Dimension", options=[8, 16, 32, 64, 128], value=32, key="single_hidden_dim")
        activation_name = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid", "GELU"], index=0, key="single_act_name")
        use_batchnorm = st.checkbox("Enable BatchNorm1d", value=False, key="single_use_bn")
        dropout_p = st.slider("Dropout Probability", min_value=0.0, max_value=0.5, value=0.0, step=0.1, key="single_dropout_p")

        # 3. Optimization Settings
        st.subheader("3. Optimization & Training")
        optimizer_name = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0, key="single_opt_name")
        lr = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
            value=0.03,
            key="single_lr",
        )
        momentum = 0.9
        if optimizer_name == "SGD":
            momentum = st.slider("Polyak Momentum", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="single_momentum")

        weight_decay = st.select_slider("Weight Decay", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="single_weight_decay")
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32, key="single_batch_size")
        epochs = st.slider("Epochs", min_value=10, max_value=150, value=60, step=5, key="single_epochs")
        update_freq = st.slider("UI Refresh Interval (Epochs)", min_value=1, max_value=10, value=2, step=1, key="single_update_freq")

        if "trained_2d_model" in st.session_state:
            st.divider()
            def _cb_reset_single():
                if "trained_2d_model" in st.session_state:
                    del st.session_state["trained_2d_model"]
                if "slider_x1" in st.session_state:
                    del st.session_state["slider_x1"]
                if "slider_x2" in st.session_state:
                    del st.session_state["slider_x2"]

            st.button("Reset Trained Model", width="stretch", key="single_reset_btn", on_click=_cb_reset_single)

    # ---------------- Data Preparation ----------------
    X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=seed)

    # Top Control Bar
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_training = st.button("Start Training", type="primary", width="stretch", key="single_start_btn")
    with col_info:
        total_params = (2 * hidden_dim + hidden_dim) + (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * 2 + 2)
        st.info(f"Dataset: **{dataset_name}** (N={n_samples}) | Architecture: **{num_layers}x {hidden_dim}d** ({total_params} Trainable Parameters) | Optimizer: **{optimizer_name}** (lr={lr})")

    # ---------------- Active State Resolution ----------------
    has_trained_model = "trained_2d_model" in st.session_state

    # Dashboard Layout Placeholders
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    col_left, col_right = st.columns(2)
    with col_left:
        plot_left = st.empty()
    with col_right:
        plot_right = st.empty()

    inference_placeholder = st.empty()

    # ---------------- Interactive Training Loop ----------------
    if start_training:
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            c0_count = int(np.sum(y == 0)) if len(y) > 0 else 0
            c1_count = int(np.sum(y == 1)) if len(y) > 0 else 0
            if c0_count < 5 or c1_count < 5:
                status_placeholder.warning(
                    f"Insufficient custom points: You currently have {c0_count} Class 0 (Blue) and {c1_count} Class 1 (Red) points. "
                    "Please place at least 5 points for each class on the canvas (or load a preset pattern) before training."
                )
                start_training = False

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
        progress_bar = progress_placeholder.progress(0.0)
        start_time = time.time()
        grad_norms_single: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                if epoch == epochs:
                    grad_norms_single = get_model_gradient_norms(model)
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

        elapsed_total = time.time() - start_time
        progress_placeholder.empty()

        # Architecture & parameter calculation
        arch_summary = get_architecture_summary(model)
        actual_params = sum(p.data.size for p in model.parameters())

        # Store in session state
        st.session_state["gradient_norms"] = grad_norms_single
        st.session_state["trained_2d_model"] = {
            "model": model,
            "dataset_name": dataset_name,
            "X": X,
            "y": y,
            "loss_history": loss_history,
            "acc_history": acc_history,
            "arch_summary": arch_summary,
            "total_params": actual_params,
            "final_loss": loss_history[-1],
            "final_acc": acc_history[-1],
            "elapsed_time": elapsed_total,
            "grad_norms": grad_norms_single,
            "history": st.session_state.get("trained_2d_model", {}).get("history", []),
        }
        has_trained_model = True

    # ---------------- Render Model State (Trained vs Initial) ----------------
    if has_trained_model and "trained_2d_model" in st.session_state:
        saved = st.session_state["trained_2d_model"]
        model = saved["model"]
        data_X = saved["X"]
        data_y = saved["y"]
        loss_hist = saved["loss_history"]
        acc_hist = saved["acc_history"]
        arch_summary = saved["arch_summary"]
        actual_params = saved["total_params"]
        final_loss = saved["final_loss"]
        final_acc = saved["final_acc"]
        elapsed_total = saved["elapsed_time"]

        # Resolve slider coordinates first (handling preset buttons)
        init_x1 = st.session_state.pop("test_x1_val", None)
        init_x2 = st.session_state.pop("test_x2_val", None)
        if init_x1 is not None:
            st.session_state["slider_x1"] = init_x1
        elif "slider_x1" not in st.session_state:
            st.session_state["slider_x1"] = 0.0

        if init_x2 is not None:
            st.session_state["slider_x2"] = init_x2
        elif "slider_x2" not in st.session_state:
            st.session_state["slider_x2"] = 0.0

        active_x1 = float(st.session_state["slider_x1"])
        active_x2 = float(st.session_state["slider_x2"])

        # 1. Model Ready Banner
        with status_placeholder.container():
            st.markdown(
                f"""
                <div class="model-ready-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <div class="model-ready-title">Model Ready: Interactive Inference Mode</div>
                            <div class="model-ready-subtitle">Topology converged on <strong>{saved["dataset_name"]}</strong> dataset. Click any point on the decision boundary or adjust sliders below for real-time evaluation.</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="model-ready-badge">Topology: {arch_summary}</span>
                            <span class="model-ready-badge">Params: {actual_params:,}</span>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1;">
                        Final Accuracy: <strong>{final_acc:.2f}%</strong> &nbsp;|&nbsp; Final Loss: <strong>{final_loss:.4f}</strong> &nbsp;|&nbsp; Training Duration: <strong>{elapsed_total:.2f}s</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Final Training Metrics Display
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Epochs Completed", f"{len(loss_hist)} / {len(loss_hist)}")
            m2.metric("Final Loss", f"{final_loss:.4f}", delta=f"{-(loss_hist[0] - final_loss):.4f}" if len(loss_hist) > 1 else None, delta_color="inverse")
            m3.metric("Final Accuracy", f"{final_acc:.2f}%", delta=f"{final_acc - acc_hist[0]:+.2f}%" if len(acc_hist) > 1 else None)
            m4.metric("Total Duration", f"{elapsed_total:.2f}s")

        # 3. Always render dashboard plots with test point marker overlaid
        if HAS_PLOTLY:
            def _cb_on_single_plotly_select():
                state = st.session_state.get("plotly_single_boundary")
                if isinstance(state, dict):
                    sel_pts = state.get("selection", {}).get("points", [])
                    if sel_pts:
                        clicked_pt = sel_pts[0]
                        if "x" in clicked_pt and "y" in clicked_pt:
                            st.session_state["slider_x1"] = float(np.round(clicked_pt["x"], 2))
                            st.session_state["slider_x2"] = float(np.round(clicked_pt["y"], 2))

            fig_b_plotly = plot_plotly_decision_boundary(
                model, data_X, data_y, test_point=(active_x1, active_x2)
            )
            plot_left.plotly_chart(
                fig_b_plotly,
                on_select=_cb_on_single_plotly_select,
                selection_mode=["points"],
                key="plotly_single_boundary",
                config={"displayModeBar": False, "scrollZoom": False},
                width="stretch",
            )
            _, fig_m = plot_dashboard_figures(
                model, data_X, data_y, loss_hist, acc_hist, test_point=(active_x1, active_x2)
            )
            plot_right.pyplot(fig_m)
            plt.close(fig_m)
        else:
            fig_b, fig_m = plot_dashboard_figures(
                model, data_X, data_y, loss_hist, acc_hist, test_point=(active_x1, active_x2)
            )
            plot_left.pyplot(fig_b)
            plot_right.pyplot(fig_m)
            plt.close(fig_b)
            plt.close(fig_m)

        # 4. Interactive Inference Controls
        with inference_placeholder.container():
            st.markdown("### Interactive 2D Inference: Test New Coordinates")
            st.markdown("Click on any region/point of the decision boundary above, or adjust coordinates $(x_1, x_2)$ below. The model will evaluate a forward pass in real time under `with no_grad():`.")

            # Presets callbacks
            def _cb_set_single_coords(x1: float, x2: float):
                st.session_state["slider_x1"] = x1
                st.session_state["slider_x2"] = x2

            def _cb_set_single_random():
                st.session_state["slider_x1"] = float(np.round(np.random.uniform(-1.8, 1.8), 2))
                st.session_state["slider_x2"] = float(np.round(np.random.uniform(-1.8, 1.8), 2))

            # Presets row
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
            with p_col1:
                st.button("Origin (0.0, 0.0)", key="preset_origin", width="stretch", on_click=_cb_set_single_coords, args=(0.0, 0.0))
            with p_col2:
                st.button("Class 0 (-1.0, 0.5)", key="preset_c0", width="stretch", on_click=_cb_set_single_coords, args=(-1.0, 0.5))
            with p_col3:
                st.button("Class 1 (1.0, -0.5)", key="preset_c1", width="stretch", on_click=_cb_set_single_coords, args=(1.0, -0.5))
            with p_col4:
                st.button("Decision Border (0.5, 0.25)", key="preset_border", width="stretch", on_click=_cb_set_single_coords, args=(0.5, 0.25))
            with p_col5:
                st.button("Random Point", key="preset_random", width="stretch", on_click=_cb_set_single_random)

            c_coord1, c_coord2 = st.columns(2)
            with c_coord1:
                test_x1 = st.slider("Coordinate X1 (Feature 1)", min_value=-2.5, max_value=2.5, step=0.05, key="slider_x1")
            with c_coord2:
                test_x2 = st.slider("Coordinate X2 (Feature 2)", min_value=-2.5, max_value=2.5, step=0.05, key="slider_x2")

            # Point Prediction Pass
            pred_class, confidence, probs = predict_point(model, test_x1, test_x2)

            # Display prediction card & probabilities
            res_col1, res_col2 = st.columns([1, 2], gap="medium")
            with res_col1:
                class_color = "#3B82F6" if pred_class == 0 else "#EF4444"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1.5px solid {class_color}; border-radius: 12px; padding: 1.25rem; text-align: center; color: white;">
                        <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #94A3B8;">Model Prediction</div>
                        <div style="font-size: 2.3rem; font-weight: 800; color: {class_color}; margin: 0.2rem 0;">Class {pred_class}</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #F1F5F9;">{confidence * 100:.1f}% Confidence</div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">Coordinate: ({test_x1:.2f}, {test_x2:.2f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("##### Probability Distribution")
                p0 = float(probs[0])
                p1 = float(probs[1])

                st.markdown(
                    f"""
                    <div class="prob-bar-container">
                        <div class="prob-bar-label">
                            <span style="color: #3B82F6;">Class 0</span>
                            <span>{p0 * 100:.2f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill-0" style="width: {max(0.0, min(100.0, p0 * 100)):.2f}%;"></div>
                        </div>
                    </div>
                    <div class="prob-bar-container" style="margin-top: 0.75rem;">
                        <div class="prob-bar-label">
                            <span style="color: #EF4444;">Class 1</span>
                            <span>{p1 * 100:.2f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill-1" style="width: {max(0.0, min(100.0, p1 * 100)):.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                def _cb_log_single():
                    saved_history = saved.setdefault("history", [])
                    saved_history.insert(0, {
                        "Time": time.strftime("%H:%M:%S"),
                        "X1": round(test_x1, 2),
                        "X2": round(test_x2, 2),
                        "Predicted Class": f"Class {pred_class}",
                        "Confidence": f"{confidence * 100:.1f}%",
                        "P(Class 0)": f"{p0 * 100:.1f}%",
                        "P(Class 1)": f"{p1 * 100:.1f}%",
                    })
                    st.session_state["trained_2d_model"]["history"] = saved_history[:10]

                # Add to history if unique or log button
                hist_col1, hist_col2 = st.columns([1, 2])
                with hist_col1:
                    st.button("Log Test Point", key="btn_log_point", width="stretch", on_click=_cb_log_single)

                if saved.get("history"):
                    st.caption("Recent Point Evaluations:")
                    st.dataframe(saved["history"][:5], width="stretch", hide_index=True)

            # 4. Engine Internals & Computation Trace Drawer
            with st.expander("Engine Internals & Computation Trace", expanded=False):
                st.markdown("#### 1. Forward Pass Computation Trace")
                st.markdown(f"Layer-by-layer tensor activations evaluated for active coordinate $(x_1 = {test_x1:.2f}, x_2 = {test_x2:.2f})$:")
                
                trace_steps, softmax_details = trace_forward_pass(model, test_x1, test_x2)
                st.dataframe(trace_steps, width="stretch", hide_index=True)

                # Softmax Mathematical Breakdown
                st.markdown("##### Softmax Normalization Breakdown (Log-Sum-Exp Trick)")
                raw_logits = softmax_details["raw_logits"]
                max_logit = softmax_details["max_logit"]
                shifted = softmax_details["shifted_logits"]
                sum_exp = softmax_details["sum_exp"]
                probs_val = softmax_details["probabilities"]

                sm_col1, sm_col2, sm_col3 = st.columns(3)
                with sm_col1:
                    st.caption("Raw Output Logits (z)")
                    st.code(f"z = [{raw_logits[0]:.4f}, {raw_logits[1]:.4f}]")
                with sm_col2:
                    st.caption(f"Log-Sum-Exp Shift (z - {max_logit:.4f})")
                    st.code(f"z_shifted = [{shifted[0]:.4f}, {shifted[1]:.4f}]")
                with sm_col3:
                    st.caption("Normalized Probabilities")
                    st.code(f"P = [{probs_val[0]*100:.2f}%, {probs_val[1]*100:.2f}%]")

                st.divider()

                # Parameter Diagnostics
                st.markdown("#### 2. Parameter Tensors & Weight Distributions")
                param_diagnostics = get_parameter_diagnostics(model)
                st.dataframe(param_diagnostics, width="stretch", hide_index=True)

                # Weight Matrix Inspector
                raw_weights = get_layer_raw_weights(model)
                if raw_weights:
                    selected_param_id = st.selectbox("Inspect Raw Weight Matrix", options=list(raw_weights.keys()), key="raw_weight_selector")
                    if selected_param_id in raw_weights:
                        raw_arr = raw_weights[selected_param_id]
                        st.caption(f"Raw Array ({list(raw_arr.shape)}):")
                        if raw_arr.ndim == 1:
                            st.dataframe(raw_arr.reshape(1, -1), width="stretch")
                        else:
                            st.dataframe(raw_arr, width="stretch")

                st.divider()

                # Gradient Telemetry
                st.markdown("#### 3. Gradient Flow Telemetry")
                st.caption("Layer-by-layer gradient L2 norms captured immediately after loss backpropagation on the final training mini-batch:")
                saved_grad_norms = saved.get("grad_norms") or st.session_state.get("gradient_norms")
                if saved_grad_norms:
                    if HAS_PLOTLY:
                        fig_gn = plot_plotly_gradient_norms(saved_grad_norms, title="Final Backpropagation Gradient Flow (Single Model)")
                        st.plotly_chart(fig_gn, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                    else:
                        fig_gn = plot_gradient_norms(saved_grad_norms)
                        st.pyplot(fig_gn)
                        plt.close(fig_gn)
                else:
                    st.info("Train the model to visualize post-backpropagation gradient flow across layers.")

    elif not start_training:
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            c0_count = int(np.sum(y == 0)) if len(y) > 0 else 0
            c1_count = int(np.sum(y == 1)) if len(y) > 0 else 0
            
            with plot_left.container():
                tb_c1, tb_c2 = st.columns(2)
                with tb_c1:
                    st.radio("Tool:", ["Drop Class 0 (Blue)", "Drop Class 1 (Red)", "Eraser / Remove Nearest"], horizontal=True, key="canvas_brush_mode")
                with tb_c2:
                    st.radio("Density:", ["Single Point", "Cluster (5 pts)", "Dense Line (8 pts)"], horizontal=True, key="canvas_brush_density")

                ps_c1, ps_c2, ps_c3 = st.columns([2, 1, 1])
                with ps_c1:
                    st.selectbox("Load Preset Pattern:", ["-- Select Preset --", "Island in Moat", "Checkerboard XOR", "Corridor Maze", "Double Cross", "Spirals"], key="canvas_preset_selector")
                with ps_c2:
                    st.button("Load Pattern", width="stretch", key="btn_load_pattern_main", on_click=_cb_load_preset_from_selector)
                with ps_c3:
                    st.button("Clear All Points", width="stretch", key="btn_clear_points_main", on_click=_cb_clear_custom_canvas)

                if HAS_PLOTLY:
                    fig_custom = plot_plotly_custom_canvas(
                        X, y,
                        title=f"Custom 2D Canvas (Click to Drop/Erase) - Total: {len(X)} pts (Blue: {c0_count}, Red: {c1_count})"
                    )
                    st.plotly_chart(
                        fig_custom,
                        on_select="rerun",
                        selection_mode=["points"],
                        key="tab1_custom_canvas",
                        config={"displayModeBar": False, "scrollZoom": False},
                        width="stretch",
                    )
                else:
                    fig_b, _ = plot_dashboard_figures(None, X, y, [0.693], [50.0])
                    st.pyplot(fig_b)
                    plt.close(fig_b)

            with plot_right.container():
                st.subheader("Interactive Canvas Controls & Studio")
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 10px; padding: 1.25rem; color: white;">
                        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Active Canvas Data</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8; margin: 0.25rem 0;">
                            {len(X)} Total Points Placed
                        </div>
                        <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 0.5rem;">
                            <span style="color: #3B82F6; font-weight: 700;">Class 0 (Blue Free Space):</span> {c0_count} points<br>
                            <span style="color: #EF4444; font-weight: 700;">Class 1 (Red Obstacle):</span> {c1_count} points
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### Canvas Drawing Guide:")
                st.markdown(
                    "1. **Select Tool & Density** in the toolbar above or the sidebar.\n"
                    "2. **Click on the canvas** to drop points or erase nearby points.\n"
                    "3. Load any preset pattern (**Island in Moat**, **Checkerboard XOR**, **Corridor Maze**, **Double Cross**, **Spirals**) as a baseline and customize.\n"
                    "4. Click **Start Training** to build and train the neural network on your custom layout.\n"
                    "5. Open **Tab 2 (Autonomous Neural Pathfinding)** to navigate the Rover across your custom course!"
                )
        else:
            # Initial static plot before training starts for synthetic datasets
            initial_model = build_model(num_layers, hidden_dim, activation_name, use_batchnorm, dropout_p)
            if HAS_PLOTLY:
                fig_b_plotly = plot_plotly_decision_boundary(initial_model, X, y, title="Untrained Initial Decision Boundary")
                plot_left.plotly_chart(
                    fig_b_plotly,
                    key="plotly_single_init",
                    config={"displayModeBar": False, "scrollZoom": False},
                    width="stretch",
                )
                _, fig_m = plot_dashboard_figures(initial_model, X, y, [0.693], [50.0])
                plot_right.pyplot(fig_m)
                plt.close(fig_m)
            else:
                fig_b, fig_m = plot_dashboard_figures(initial_model, X, y, [0.693], [50.0])
                plot_left.pyplot(fig_b)
                plot_right.pyplot(fig_m)
                plt.close(fig_b)
                plt.close(fig_m)

        with inference_placeholder.container():
            st.info("Configure hyperparameters in the sidebar and click **Start Training** to build the model and unlock the interactive coordinate tester.")


def render_model_comparison_studio():
    """Renders the side-by-side Model Capacity Comparison laboratory."""
    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("Comparison Experiment Controls")

        # 1. Dataset Configuration (Shared)
        st.subheader("1. Shared Dataset")
        dataset_name = st.selectbox(
            "Dataset Topology",
            ["Island in Moat", "Checkerboard XOR", "Corridor Maze", "Double Cross", "Spirals", "Custom / Interactive Canvas"],
            index=2,
            key="comp_dataset_name",
        )
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            n_samples = len(st.session_state.get("custom_points_X", []))
            noise = 0.0
            seed = 42
            st.caption(f"Using Custom Canvas points (Total: {n_samples} pts). Customize layout on Single Model tab.")
        else:
            n_samples = st.slider("Sample Count", min_value=100, max_value=1000, value=500, step=50, key="comp_n_samples")
            noise = st.slider("Noise Level", min_value=0.0, max_value=0.3, value=0.06, step=0.01, key="comp_noise")
            seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1, key="comp_seed")

        # 2. Model A Configuration
        st.subheader("2. Model A (Shallow / Baseline)")
        num_layers_a = st.slider("Model A Hidden Layers", min_value=1, max_value=4, value=1, step=1, key="comp_layers_a")
        hidden_dim_a = st.select_slider("Model A Hidden Dimension", options=[4, 8, 16, 32, 64, 128], value=4, key="comp_dim_a")
        act_a = st.selectbox("Model A Activation", ["Tanh", "ReLU", "Sigmoid", "GELU"], index=0, key="comp_act_a")
        bn_a = st.checkbox("Model A BatchNorm1d", value=False, key="comp_bn_a")
        dropout_a = st.slider("Model A Dropout", min_value=0.0, max_value=0.5, value=0.0, step=0.1, key="comp_drop_a")
        opt_a_name = st.selectbox("Model A Optimizer", ["SGD", "AdamW"], index=0, key="comp_opt_a")
        lr_a = st.select_slider("Model A Learning Rate", options=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2], value=0.05, key="comp_lr_a")
        momentum_a = 0.9
        if opt_a_name == "SGD":
            momentum_a = st.slider("Model A Momentum", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="comp_mom_a")
        wd_a = st.select_slider("Model A Weight Decay", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="comp_wd_a")

        # 3. Model B Configuration
        st.subheader("3. Model B (Deep / High Capacity)")
        num_layers_b = st.slider("Model B Hidden Layers", min_value=1, max_value=4, value=2, step=1, key="comp_layers_b")
        hidden_dim_b = st.select_slider("Model B Hidden Dimension", options=[4, 8, 16, 32, 64, 128], value=32, key="comp_dim_b")
        act_b = st.selectbox("Model B Activation", ["ReLU", "Tanh", "Sigmoid", "GELU"], index=0, key="comp_act_b")
        bn_b = st.checkbox("Model B BatchNorm1d", value=False, key="comp_bn_b")
        dropout_b = st.slider("Model B Dropout", min_value=0.0, max_value=0.5, value=0.0, step=0.1, key="comp_drop_b")
        opt_b_name = st.selectbox("Model B Optimizer", ["AdamW", "SGD"], index=0, key="comp_opt_b")
        lr_b = st.select_slider("Model B Learning Rate", options=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2], value=0.01, key="comp_lr_b")
        momentum_b = 0.9
        if opt_b_name == "SGD":
            momentum_b = st.slider("Model B Momentum", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="comp_mom_b")
        wd_b = st.select_slider("Model B Weight Decay", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="comp_wd_b")

        # 4. Shared Training Parameters
        st.subheader("4. Training Schedule")
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32, key="comp_bs")
        epochs = st.slider("Epochs", min_value=10, max_value=150, value=60, step=5, key="comp_epochs")
        update_freq = st.slider("UI Refresh Interval (Epochs)", min_value=1, max_value=10, value=2, step=1, key="comp_freq")

        if "model_comparison" in st.session_state:
            st.divider()
            def _cb_reset_comp():
                if "model_comparison" in st.session_state:
                    del st.session_state["model_comparison"]
                if "comp_slider_x1" in st.session_state:
                    del st.session_state["comp_slider_x1"]
                if "comp_slider_x2" in st.session_state:
                    del st.session_state["comp_slider_x2"]

            st.button("Reset Comparison Models", width="stretch", key="btn_reset_comp", on_click=_cb_reset_comp)

    # ---------------- Data Preparation ----------------
    X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=seed)

    # Calculate theoretical params
    params_a_calc = (2 * hidden_dim_a + hidden_dim_a) + (num_layers_a - 1) * (hidden_dim_a * hidden_dim_a + hidden_dim_a) + (hidden_dim_a * 2 + 2)
    params_b_calc = (2 * hidden_dim_b + hidden_dim_b) + (num_layers_b - 1) * (hidden_dim_b * hidden_dim_b + hidden_dim_b) + (hidden_dim_b * 2 + 2)

    # Top Action Bar
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_training = st.button("Train Both Models (A & B)", type="primary", width="stretch", key="btn_train_comp")
    with col_info:
        st.info(f"Dataset: **{dataset_name}** (N={n_samples}) | **Model A**: {num_layers_a}x {hidden_dim_a}d ({params_a_calc} params, {act_a}) | **Model B**: {num_layers_b}x {hidden_dim_b}d ({params_b_calc} params, {act_b})")

    # Placeholders
    has_comparison = "model_comparison" in st.session_state
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()

    col_plot_a, col_plot_b = st.columns(2)
    with col_plot_a:
        plot_a_holder = st.empty()
    with col_plot_b:
        plot_b_holder = st.empty()

    curves_placeholder = st.empty()
    inference_placeholder = st.empty()

    # ---------------- Dual Training Loop ----------------
    if start_training:
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            c0_count = int(np.sum(y == 0)) if len(y) > 0 else 0
            c1_count = int(np.sum(y == 1)) if len(y) > 0 else 0
            if c0_count < 5 or c1_count < 5:
                status_placeholder.warning(
                    f"Insufficient custom points: You currently have {c0_count} Class 0 (Blue) and {c1_count} Class 1 (Red) points. "
                    "Please configure at least 5 points for each class in the Single Model tab before comparison training."
                )
                start_training = False

    if start_training:
        model_a = build_model(num_layers_a, hidden_dim_a, act_a, bn_a, dropout_a)
        model_b = build_model(num_layers_b, hidden_dim_b, act_b, bn_b, dropout_b)

        criterion_a = nn.CrossEntropyLoss()
        criterion_b = nn.CrossEntropyLoss()

        if opt_a_name == "AdamW":
            opt_a = optim.AdamW(model_a.parameters(), lr=lr_a, weight_decay=wd_a)
        else:
            opt_a = optim.SGD(model_a.parameters(), lr=lr_a, momentum=momentum_a, weight_decay=wd_a)

        if opt_b_name == "AdamW":
            opt_b = optim.AdamW(model_b.parameters(), lr=lr_b, weight_decay=wd_b)
        else:
            opt_b = optim.SGD(model_b.parameters(), lr=lr_b, momentum=momentum_b, weight_decay=wd_b)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_hist_a, acc_hist_a = [], []
        loss_hist_b, acc_hist_b = [], []

        progress_bar = progress_placeholder.progress(0.0)
        start_time = time.time()
        grad_norms_a: Dict[str, float] = {}
        grad_norms_b: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            model_a.train()
            model_b.train()
            epoch_losses_a = []
            epoch_losses_b = []

            for batch_X, batch_y in loader:
                # Step Model A
                opt_a.zero_grad()
                logits_a = model_a(batch_X)
                loss_a = criterion_a(logits_a, batch_y)
                loss_a.backward()
                if epoch == epochs:
                    grad_norms_a = get_model_gradient_norms(model_a)
                opt_a.step()
                epoch_losses_a.append(float(loss_a.data))

                # Step Model B
                opt_b.zero_grad()
                logits_b = model_b(batch_X)
                loss_b = criterion_b(logits_b, batch_y)
                loss_b.backward()
                if epoch == epochs:
                    grad_norms_b = get_model_gradient_norms(model_b)
                opt_b.step()
                epoch_losses_b.append(float(loss_b.data))

            # Epoch evaluation
            model_a.eval()
            model_b.eval()
            full_t = Tensor(X, requires_grad=False)
            
            full_logits_a = model_a(full_t)
            preds_a = np.argmax(full_logits_a.data, axis=1)
            accuracy_a = float(np.mean(preds_a == y) * 100.0)
            avg_loss_a = float(np.mean(epoch_losses_a))
            loss_hist_a.append(avg_loss_a)
            acc_hist_a.append(accuracy_a)

            full_logits_b = model_b(full_t)
            preds_b = np.argmax(full_logits_b.data, axis=1)
            accuracy_b = float(np.mean(preds_b == y) * 100.0)
            avg_loss_b = float(np.mean(epoch_losses_b))
            loss_hist_b.append(avg_loss_b)
            acc_hist_b.append(accuracy_b)

            if epoch % update_freq == 0 or epoch == epochs or epoch == 1:
                elapsed = time.time() - start_time
                progress_bar.progress(epoch / float(epochs))

                with metrics_placeholder.container():
                    m_ep, m_la, m_lb, m_aa, m_ab, m_tm = st.columns(6)
                    m_ep.metric("Epoch", f"{epoch} / {epochs}")
                    m_la.metric("Loss (A)", f"{avg_loss_a:.3f}")
                    m_lb.metric("Loss (B)", f"{avg_loss_b:.3f}")
                    m_aa.metric("Acc (A)", f"{accuracy_a:.1f}%")
                    m_ab.metric("Acc (B)", f"{accuracy_b:.1f}%")
                    m_tm.metric("Elapsed", f"{elapsed:.2f}s")

                arch_a_str = get_architecture_summary(model_a)
                arch_b_str = get_architecture_summary(model_b)
                fig_a, fig_b, fig_c = plot_comparison_dashboard_figures(
                    model_a, model_b, X, y, loss_hist_a, acc_hist_a, loss_hist_b, acc_hist_b,
                    title_a=f"Model A: {arch_a_str} ({act_a}) - {accuracy_a:.1f}%",
                    title_b=f"Model B: {arch_b_str} ({act_b}) - {accuracy_b:.1f}%",
                )
                plot_a_holder.pyplot(fig_a)
                plot_b_holder.pyplot(fig_b)
                curves_placeholder.pyplot(fig_c)
                plt.close(fig_a)
                plt.close(fig_b)
                plt.close(fig_c)

        elapsed_total = time.time() - start_time
        progress_placeholder.empty()

        arch_a_str = get_architecture_summary(model_a)
        arch_b_str = get_architecture_summary(model_b)
        actual_params_a = sum(p.data.size for p in model_a.parameters())
        actual_params_b = sum(p.data.size for p in model_b.parameters())

        st.session_state["gradient_norms_a"] = grad_norms_a
        st.session_state["gradient_norms_b"] = grad_norms_b
        st.session_state["model_comparison"] = {
            "model_a": model_a,
            "model_b": model_b,
            "dataset_name": dataset_name,
            "X": X,
            "y": y,
            "loss_hist_a": loss_hist_a,
            "acc_hist_a": acc_hist_a,
            "loss_hist_b": loss_hist_b,
            "acc_hist_b": acc_hist_b,
            "arch_a": arch_a_str,
            "arch_b": arch_b_str,
            "act_a": act_a,
            "act_b": act_b,
            "params_a": actual_params_a,
            "params_b": actual_params_b,
            "final_loss_a": loss_hist_a[-1],
            "final_acc_a": acc_hist_a[-1],
            "final_loss_b": loss_hist_b[-1],
            "final_acc_b": acc_hist_b[-1],
            "elapsed_time": elapsed_total,
            "grad_norms_a": grad_norms_a,
            "grad_norms_b": grad_norms_b,
        }
        has_comparison = True

    # ---------------- Render Comparison State ----------------
    if has_comparison and "model_comparison" in st.session_state:
        saved = st.session_state["model_comparison"]
        model_a = saved["model_a"]
        model_b = saved["model_b"]
        data_X = saved["X"]
        data_y = saved["y"]
        l_hist_a = saved["loss_hist_a"]
        a_hist_a = saved["acc_hist_a"]
        l_hist_b = saved["loss_hist_b"]
        a_hist_b = saved["acc_hist_b"]
        arch_a_str = saved["arch_a"]
        arch_b_str = saved["arch_b"]
        act_a_str = saved["act_a"]
        act_b_str = saved["act_b"]
        p_a = saved["params_a"]
        p_b = saved["params_b"]
        f_loss_a = saved["final_loss_a"]
        f_acc_a = saved["final_acc_a"]
        f_loss_b = saved["final_loss_b"]
        f_acc_b = saved["final_acc_b"]
        elapsed_total = saved["elapsed_time"]

        # Resolve slider coordinates first (handling preset buttons)
        init_x1 = st.session_state.pop("comp_test_x1_val", None)
        init_x2 = st.session_state.pop("comp_test_x2_val", None)
        if init_x1 is not None:
            st.session_state["comp_slider_x1"] = init_x1
        elif "comp_slider_x1" not in st.session_state:
            st.session_state["comp_slider_x1"] = 0.0

        if init_x2 is not None:
            st.session_state["comp_slider_x2"] = init_x2
        elif "comp_slider_x2" not in st.session_state:
            st.session_state["comp_slider_x2"] = 0.0

        cur_comp_x1 = float(st.session_state["comp_slider_x1"])
        cur_comp_x2 = float(st.session_state["comp_slider_x2"])

        # 1. Comparative Summary Card
        with status_placeholder.container():
            st.markdown(
                f"""
                <div class="model-ready-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <div class="model-ready-title">Model Capacity Comparison Ready</div>
                            <div class="model-ready-subtitle">Evaluated on <strong>{saved["dataset_name"]}</strong> dataset under identical mini-batches. Click either boundary plot or adjust sliders below for real-time comparison.</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="model-ready-badge">Model A: {arch_a_str} ({p_a} params)</span>
                            <span class="model-ready-badge">Model B: {arch_b_str} ({p_b} params)</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Comparative Metrics Table
        with metrics_placeholder.container():
            acc_diff = f_acc_b - f_acc_a
            loss_diff = f_loss_b - f_loss_a
            param_ratio = (p_b / max(1, p_a))
            comp_table = [
                {"Metric": "Architecture Topology", "Model A": f"{arch_a_str} ({act_a_str})", "Model B": f"{arch_b_str} ({act_b_str})", "Comparison": f"Model B has higher capacity"},
                {"Metric": "Trainable Parameters", "Model A": f"{p_a:,}", "Model B": f"{p_b:,}", "Comparison": f"Model B is {param_ratio:.1f}x larger"},
                {"Metric": "Final Accuracy", "Model A": f"{f_acc_a:.2f}%", "Model B": f"{f_acc_b:.2f}%", "Comparison": f"{acc_diff:+.2f}% Accuracy delta"},
                {"Metric": "Final CrossEntropy Loss", "Model A": f"{f_loss_a:.4f}", "Model B": f"{f_loss_b:.4f}", "Comparison": f"{loss_diff:+.4f} Loss delta"},
                {"Metric": "Total Training Time", "Model A": f"{elapsed_total:.2f}s", "Model B": f"{elapsed_total:.2f}s", "Comparison": "Synchronized training"},
            ]
            st.dataframe(comp_table, width="stretch", hide_index=True)

        # 3. Always render comparison dashboard plots with test point marker overlaid
        if HAS_PLOTLY:
            def _cb_on_comp_plotly_select_a():
                state = st.session_state.get("plotly_comp_a")
                if isinstance(state, dict):
                    sel_pts = state.get("selection", {}).get("points", [])
                    if sel_pts:
                        clicked_pt = sel_pts[0]
                        if "x" in clicked_pt and "y" in clicked_pt:
                            st.session_state["comp_slider_x1"] = float(np.round(clicked_pt["x"], 2))
                            st.session_state["comp_slider_x2"] = float(np.round(clicked_pt["y"], 2))

            def _cb_on_comp_plotly_select_b():
                state = st.session_state.get("plotly_comp_b")
                if isinstance(state, dict):
                    sel_pts = state.get("selection", {}).get("points", [])
                    if sel_pts:
                        clicked_pt = sel_pts[0]
                        if "x" in clicked_pt and "y" in clicked_pt:
                            st.session_state["comp_slider_x1"] = float(np.round(clicked_pt["x"], 2))
                            st.session_state["comp_slider_x2"] = float(np.round(clicked_pt["y"], 2))

            fig_a_plotly = plot_plotly_decision_boundary(
                model_a, data_X, data_y, test_point=(cur_comp_x1, cur_comp_x2),
                title=f"Model A: {arch_a_str} ({act_a_str}) - {f_acc_a:.1f}%",
            )
            fig_b_plotly = plot_plotly_decision_boundary(
                model_b, data_X, data_y, test_point=(cur_comp_x1, cur_comp_x2),
                title=f"Model B: {arch_b_str} ({act_b_str}) - {f_acc_b:.1f}%",
            )
            plot_a_holder.plotly_chart(
                fig_a_plotly,
                on_select=_cb_on_comp_plotly_select_a,
                selection_mode=["points"],
                key="plotly_comp_a",
                config={"displayModeBar": False, "scrollZoom": False},
                width="stretch",
            )
            plot_b_holder.plotly_chart(
                fig_b_plotly,
                on_select=_cb_on_comp_plotly_select_b,
                selection_mode=["points"],
                key="plotly_comp_b",
                config={"displayModeBar": False, "scrollZoom": False},
                width="stretch",
            )
            _, _, fig_c = plot_comparison_dashboard_figures(
                model_a, model_b, data_X, data_y, l_hist_a, a_hist_a, l_hist_b, a_hist_b,
                test_point=(cur_comp_x1, cur_comp_x2),
                title_a=f"Model A",
                title_b=f"Model B",
            )
            curves_placeholder.pyplot(fig_c)
            plt.close(fig_c)
        else:
            fig_a, fig_b, fig_c = plot_comparison_dashboard_figures(
                model_a, model_b, data_X, data_y, l_hist_a, a_hist_a, l_hist_b, a_hist_b,
                test_point=(cur_comp_x1, cur_comp_x2),
                title_a=f"Model A: {arch_a_str} ({act_a_str}) - {f_acc_a:.1f}%",
                title_b=f"Model B: {arch_b_str} ({act_b_str}) - {f_acc_b:.1f}%",
            )
            plot_a_holder.pyplot(fig_a)
            plot_b_holder.pyplot(fig_b)
            curves_placeholder.pyplot(fig_c)
            plt.close(fig_a)
            plt.close(fig_b)
            plt.close(fig_c)

        # 4. Synchronized Interactive Inference Controls
        with inference_placeholder.container():
            st.markdown("### Synchronized 2D Inference: Compare Model Predictions")
            st.markdown("Click on either decision boundary above or adjust coordinates $(x_1, x_2)$ below. Both models will evaluate the active point synchronously under `with no_grad():`.")

            # Presets callbacks
            def _cb_set_comp_coords(x1: float, x2: float):
                st.session_state["comp_slider_x1"] = x1
                st.session_state["comp_slider_x2"] = x2

            def _cb_set_comp_random():
                st.session_state["comp_slider_x1"] = float(np.round(np.random.uniform(-1.8, 1.8), 2))
                st.session_state["comp_slider_x2"] = float(np.round(np.random.uniform(-1.8, 1.8), 2))

            # Presets
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
            with p_col1:
                st.button("Origin (0.0, 0.0)", key="comp_preset_orig", width="stretch", on_click=_cb_set_comp_coords, args=(0.0, 0.0))
            with p_col2:
                st.button("Class 0 (-1.0, 0.5)", key="comp_preset_c0", width="stretch", on_click=_cb_set_comp_coords, args=(-1.0, 0.5))
            with p_col3:
                st.button("Class 1 (1.0, -0.5)", key="comp_preset_c1", width="stretch", on_click=_cb_set_comp_coords, args=(1.0, -0.5))
            with p_col4:
                st.button("Decision Border (0.5, 0.25)", key="comp_preset_bord", width="stretch", on_click=_cb_set_comp_coords, args=(0.5, 0.25))
            with p_col5:
                st.button("Random Point", key="comp_preset_rand", width="stretch", on_click=_cb_set_comp_random)

            c_coord1, c_coord2 = st.columns(2)
            with c_coord1:
                test_x1 = st.slider("Coordinate X1 (Feature 1)", min_value=-2.5, max_value=2.5, step=0.05, key="comp_slider_x1")
            with c_coord2:
                test_x2 = st.slider("Coordinate X2 (Feature 2)", min_value=-2.5, max_value=2.5, step=0.05, key="comp_slider_x2")

            # Point Predictions
            pred_a, conf_a, probs_a = predict_point(model_a, test_x1, test_x2)
            pred_b, conf_b, probs_b = predict_point(model_b, test_x1, test_x2)

            # Dual Prediction Cards
            pred_col_a, pred_col_b = st.columns(2, gap="large")
            with pred_col_a:
                color_a = "#3B82F6" if pred_a == 0 else "#EF4444"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1.5px solid {color_a}; border-radius: 12px; padding: 1.15rem; color: white;">
                        <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #94A3B8;">Model A Prediction ({arch_a_str})</div>
                        <div style="font-size: 2.1rem; font-weight: 800; color: {color_a}; margin: 0.15rem 0;">Class {pred_a}</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #F1F5F9;">{conf_a * 100:.1f}% Confidence</div>
                        <div class="prob-bar-container" style="margin-top: 0.6rem;">
                            <div class="prob-bar-label"><span style="color: #3B82F6;">Class 0</span><span>{probs_a[0]*100:.1f}%</span></div>
                            <div class="prob-bar-bg"><div class="prob-bar-fill-0" style="width: {probs_a[0]*100:.1f}%;"></div></div>
                        </div>
                        <div class="prob-bar-container" style="margin-top: 0.4rem;">
                            <div class="prob-bar-label"><span style="color: #EF4444;">Class 1</span><span>{probs_a[1]*100:.1f}%</span></div>
                            <div class="prob-bar-bg"><div class="prob-bar-fill-1" style="width: {probs_a[1]*100:.1f}%;"></div></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with pred_col_b:
                color_b = "#3B82F6" if pred_b == 0 else "#EF4444"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1.5px solid {color_b}; border-radius: 12px; padding: 1.15rem; color: white;">
                        <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #94A3B8;">Model B Prediction ({arch_b_str})</div>
                        <div style="font-size: 2.1rem; font-weight: 800; color: {color_b}; margin: 0.15rem 0;">Class {pred_b}</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #F1F5F9;">{conf_b * 100:.1f}% Confidence</div>
                        <div class="prob-bar-container" style="margin-top: 0.6rem;">
                            <div class="prob-bar-label"><span style="color: #3B82F6;">Class 0</span><span>{probs_b[0]*100:.1f}%</span></div>
                            <div class="prob-bar-bg"><div class="prob-bar-fill-0" style="width: {probs_b[0]*100:.1f}%;"></div></div>
                        </div>
                        <div class="prob-bar-container" style="margin-top: 0.4rem;">
                            <div class="prob-bar-label"><span style="color: #EF4444;">Class 1</span><span>{probs_b[1]*100:.1f}%</span></div>
                            <div class="prob-bar-bg"><div class="prob-bar-fill-1" style="width: {probs_b[1]*100:.1f}%;"></div></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Internals Expander for Comparison
            with st.expander("Comparative Engine Internals & Layer Diagnostics", expanded=False):
                tab_diag_a, tab_diag_b = st.tabs(["Model A Diagnostics", "Model B Diagnostics"])
                with tab_diag_a:
                    st.markdown(f"#### Model A Forward Trace ({test_x1:.2f}, {test_x2:.2f})")
                    steps_a, sm_a = trace_forward_pass(model_a, test_x1, test_x2)
                    st.dataframe(steps_a, width="stretch", hide_index=True)
                    st.caption("Model A Parameters:")
                    st.dataframe(get_parameter_diagnostics(model_a), width="stretch", hide_index=True)
                    raw_a = get_layer_raw_weights(model_a)
                    if raw_a:
                        sel_a = st.selectbox("Inspect Model A Raw Parameter", options=list(raw_a.keys()), key="comp_raw_sel_a")
                        arr_a = raw_a[sel_a]
                        st.caption(f"Array Shape: {list(arr_a.shape)}")
                        st.dataframe(arr_a.reshape(1, -1) if arr_a.ndim == 1 else arr_a, width="stretch")
                    st.caption("Model A Gradient Flow Telemetry:")
                    gn_a = saved.get("grad_norms_a") or st.session_state.get("gradient_norms_a")
                    if gn_a:
                        if HAS_PLOTLY:
                            fig_gna = plot_plotly_gradient_norms(gn_a, title=f"Model A Gradient Flow ({arch_a_str})")
                            st.plotly_chart(fig_gna, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                        else:
                            fig_gna = plot_gradient_norms(gn_a)
                            st.pyplot(fig_gna)
                            plt.close(fig_gna)
                    else:
                        st.info("Train the model to visualize post-backpropagation gradient flow across layers.")

                with tab_diag_b:
                    st.markdown(f"#### Model B Forward Trace ({test_x1:.2f}, {test_x2:.2f})")
                    steps_b, sm_b = trace_forward_pass(model_b, test_x1, test_x2)
                    st.dataframe(steps_b, width="stretch", hide_index=True)
                    st.caption("Model B Parameters:")
                    st.dataframe(get_parameter_diagnostics(model_b), width="stretch", hide_index=True)
                    raw_b = get_layer_raw_weights(model_b)
                    if raw_b:
                        sel_b = st.selectbox("Inspect Model B Raw Parameter", options=list(raw_b.keys()), key="comp_raw_sel_b")
                        arr_b = raw_b[sel_b]
                        st.caption(f"Array Shape: {list(arr_b.shape)}")
                        st.dataframe(arr_b.reshape(1, -1) if arr_b.ndim == 1 else arr_b, width="stretch")
                    st.caption("Model B Gradient Flow Telemetry:")
                    gn_b = saved.get("grad_norms_b") or st.session_state.get("gradient_norms_b")
                    if gn_b:
                        if HAS_PLOTLY:
                            fig_gnb = plot_plotly_gradient_norms(gn_b, title=f"Model B Gradient Flow ({arch_b_str})")
                            st.plotly_chart(fig_gnb, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                        else:
                            fig_gnb = plot_gradient_norms(gn_b)
                            st.pyplot(fig_gnb)
                            plt.close(fig_gnb)
                    else:
                        st.info("Train the model to visualize post-backpropagation gradient flow across layers.")

    elif not start_training:
        if dataset_name in ["Custom Canvas", "Custom / Interactive Canvas"]:
            c0_count = int(np.sum(y == 0)) if len(y) > 0 else 0
            c1_count = int(np.sum(y == 1)) if len(y) > 0 else 0
            if HAS_PLOTLY:
                fig_a_plotly = plot_plotly_custom_canvas(
                    X, y, title=f"Model A (Custom Canvas Untrained) - Total: {len(X)} pts"
                )
                fig_b_plotly = plot_plotly_custom_canvas(
                    X, y, title=f"Model B (Custom Canvas Untrained) - Total: {len(X)} pts"
                )
                plot_a_holder.plotly_chart(
                    fig_a_plotly,
                    key="plotly_comp_init_a",
                    config={"displayModeBar": False, "scrollZoom": False},
                    width="stretch",
                )
                plot_b_holder.plotly_chart(
                    fig_b_plotly,
                    key="plotly_comp_init_b",
                    config={"displayModeBar": False, "scrollZoom": False},
                    width="stretch",
                )
            else:
                fig_a, _ = plot_dashboard_figures(None, X, y, [0.693], [50.0])
                fig_b, _ = plot_dashboard_figures(None, X, y, [0.693], [50.0])
                plot_a_holder.pyplot(fig_a)
                plot_b_holder.pyplot(fig_b)
                plt.close(fig_a)
                plt.close(fig_b)
        else:
            init_a = build_model(num_layers_a, hidden_dim_a, act_a, bn_a, dropout_a)
            init_b = build_model(num_layers_b, hidden_dim_b, act_b, bn_b, dropout_b)
            if HAS_PLOTLY:
                fig_a_plotly = plot_plotly_decision_boundary(init_a, X, y, title="Model A (Untrained Baseline)")
                fig_b_plotly = plot_plotly_decision_boundary(init_b, X, y, title="Model B (Untrained Baseline)")
                plot_a_holder.plotly_chart(
                    fig_a_plotly,
                    key="plotly_comp_init_a",
                    config={"displayModeBar": False, "scrollZoom": False},
                    width="stretch",
                )
                plot_b_holder.plotly_chart(
                    fig_b_plotly,
                    key="plotly_comp_init_b",
                    config={"displayModeBar": False, "scrollZoom": False},
                    width="stretch",
                )
                _, _, fig_c = plot_comparison_dashboard_figures(
                    init_a, init_b, X, y, [0.693], [50.0], [0.693], [50.0],
                    title_a="Model A", title_b="Model B",
                )
                curves_placeholder.pyplot(fig_c)
                plt.close(fig_c)
            else:
                fig_a, fig_b, fig_c = plot_comparison_dashboard_figures(
                    init_a, init_b, X, y, [0.693], [50.0], [0.693], [50.0],
                    title_a=f"Model A (Untrained Baseline)",
                    title_b=f"Model B (Untrained Baseline)",
                )
                plot_a_holder.pyplot(fig_a)
                plot_b_holder.pyplot(fig_b)
                curves_placeholder.pyplot(fig_c)
                plt.close(fig_a)
                plt.close(fig_b)
                plt.close(fig_c)

        with inference_placeholder.container():
            st.info("Configure Model A and Model B in the sidebar and click **Train Both Models (A & B)** to start the comparative experiment.")


def render_decision_boundary_tab():
    """Renders the 2D Decision Boundary laboratory supporting Single Model and Model Comparison modes."""
    studio_submode = st.segmented_control(
        "2D Studio Mode",
        options=["Single Model Studio", "Model Comparison (A vs B)"],
        default="Single Model Studio",
        key="2d_studio_mode_selector",
    )
    if not studio_submode:
        studio_submode = "Single Model Studio"

    if studio_submode == "Single Model Studio":
        render_single_model_studio()
    else:
        render_model_comparison_studio()


# -----------------------------------------------------------------------------
# Tab 2: Autonomous Neural Pathfinding
# -----------------------------------------------------------------------------

def render_neural_pathfinding_tab():
    """Renders the Tab 2 Autonomous Neural Pathfinding laboratory supporting Single Model and Dual-Model Navigation Race."""
    st.header("Autonomous Neural Pathfinding & Obstacle Avoidance")
    st.markdown(
        "Simulate an autonomous rover navigating an environment where the classification decision boundary "
        "of your trained neural network acts as a **dynamic artificial potential field**. The rover emits forward "
        "sensory rays that query the neural network in real time to repel away from high-hazard obstacle regions "
        "while being attracted toward the destination target."
    )

    has_single = "trained_2d_model" in st.session_state and st.session_state["trained_2d_model"] is not None
    has_comp = "model_comparison" in st.session_state and st.session_state["model_comparison"] is not None

    if not has_single and not has_comp:
        st.info(
            "**Model Required:** No 2D classification model is currently trained.\n\n"
            "1. Switch to **Tab 1: 2D Decision Boundaries**.\n"
            "2. Select a topology (e.g. *Spirals*, *Two Moons*, or *Circles*) and click **Start Training** (or train in Comparison Mode).\n"
            "3. Return to this tab to launch the Autonomous Neural Rover across the learned obstacle field!"
        )
        return

    # Navigation Mission Submode Selector
    if has_comp:
        nav_mode = st.segmented_control(
            "Navigation Mission Mode",
            options=["Single Model Mission", "Dual-Model Navigation Race (A vs B)"],
            default="Single Model Mission" if has_single else "Dual-Model Navigation Race (A vs B)",
            key="rover_nav_mode_selector",
        )
        if not nav_mode:
            nav_mode = "Single Model Mission" if has_single else "Dual-Model Navigation Race (A vs B)"
    else:
        nav_mode = "Single Model Mission"

    # Determine single model reference
    if nav_mode == "Single Model Mission":
        if has_single:
            saved = st.session_state["trained_2d_model"]
            model = saved["model"]
            X = saved["X"]
            y = saved["y"]
            dataset_name = saved.get("dataset_name", "2D Dataset")
            arch_str = saved.get("arch_summary", "MLP")
        else:
            saved_comp = st.session_state["model_comparison"]
            model = saved_comp["model_b"]
            X = saved_comp["X"]
            y = saved_comp["y"]
            dataset_name = saved_comp.get("dataset_name", "2D Dataset")
            arch_str = saved_comp.get("arch_b", "Model B")

    active_eval_model = model if nav_mode == "Single Model Mission" else saved_comp["model_a"]

    # Initialize coordinates dynamically if not present
    if "rover_start_x1" not in st.session_state and active_eval_model is not None:
        try:
            auto_s, auto_t = find_safe_waypoints(active_eval_model)
            st.session_state["rover_start_x1"] = auto_s[0]
            st.session_state["rover_start_x2"] = auto_s[1]
            st.session_state["rover_target_x1"] = auto_t[0]
            st.session_state["rover_target_x2"] = auto_t[1]
        except Exception:
            st.session_state["rover_start_x1"] = -1.80
            st.session_state["rover_start_x2"] = 1.20
            st.session_state["rover_target_x1"] = 1.20
            st.session_state["rover_target_x2"] = 0.70

    # Process incoming map click events BEFORE sidebar widgets instantiate
    for map_k in ["rover_plotly_map", "rover_plotly_map_standby", "plotly_rover_a", "plotly_rover_b", "plotly_rover_a_standby", "plotly_rover_b_standby"]:
        event_dict = st.session_state.get(map_k)
        if isinstance(event_dict, dict):
            pts = event_dict.get("selection", {}).get("points", [])
            if pts and "x" in pts[0] and "y" in pts[0]:
                cx = float(np.round(pts[0]["x"], 2))
                cy = float(np.round(pts[0]["y"], 2))
                click_tag = (map_k, cx, cy)
                if st.session_state.get("_last_map_click") != click_tag:
                    st.session_state["_last_map_click"] = click_tag
                    mode = st.session_state.get("rover_waypoint_click_mode") or st.session_state.get("rover_waypoint_click_mode_sb") or "Start Point (S)"
                    if "Start" in mode:
                        st.session_state["rover_start_x1"] = cx
                        st.session_state["rover_start_x2"] = cy
                    else:
                        st.session_state["rover_target_x1"] = cx
                        st.session_state["rover_target_x2"] = cy
                    st.session_state.pop("rover_sim", None)
                    st.session_state.pop("rover_sim_a", None)
                    st.session_state.pop("rover_sim_b", None)

    # Helper callback to update coordinates cleanly BEFORE widgets instantiate
    def _update_waypoint_coords(
        sx1=None, sx2=None, tx1=None, tx2=None,
        dsx1=0.0, dsx2=0.0, dtx1=0.0, dtx2=0.0,
        swap=False, auto=False
    ):
        if auto and active_eval_model is not None:
            try:
                auto_s, auto_t = find_safe_waypoints(active_eval_model)
                st.session_state["rover_start_x1"] = float(auto_s[0])
                st.session_state["rover_start_x2"] = float(auto_s[1])
                st.session_state["rover_target_x1"] = float(auto_t[0])
                st.session_state["rover_target_x2"] = float(auto_t[1])
            except Exception:
                pass
        elif swap:
            cur_s1 = st.session_state.get("rover_start_x1", -1.80)
            cur_s2 = st.session_state.get("rover_start_x2", 1.20)
            cur_t1 = st.session_state.get("rover_target_x1", 1.20)
            cur_t2 = st.session_state.get("rover_target_x2", 0.70)
            st.session_state["rover_start_x1"] = cur_t1
            st.session_state["rover_start_x2"] = cur_t2
            st.session_state["rover_target_x1"] = cur_s1
            st.session_state["rover_target_x2"] = cur_s2
        else:
            if sx1 is not None:
                st.session_state["rover_start_x1"] = float(sx1)
            elif dsx1 != 0.0:
                st.session_state["rover_start_x1"] = max(-2.4, min(2.4, round(st.session_state.get("rover_start_x1", -1.80) + dsx1, 2)))

            if sx2 is not None:
                st.session_state["rover_start_x2"] = float(sx2)
            elif dsx2 != 0.0:
                st.session_state["rover_start_x2"] = max(-2.4, min(2.4, round(st.session_state.get("rover_start_x2", 1.20) + dsx2, 2)))

            if tx1 is not None:
                st.session_state["rover_target_x1"] = float(tx1)
            elif dtx1 != 0.0:
                st.session_state["rover_target_x1"] = max(-2.4, min(2.4, round(st.session_state.get("rover_target_x1", 1.20) + dtx1, 2)))

            if tx2 is not None:
                st.session_state["rover_target_x2"] = float(tx2)
            elif dtx2 != 0.0:
                st.session_state["rover_target_x2"] = max(-2.4, min(2.4, round(st.session_state.get("rover_target_x2", 0.70) + dtx2, 2)))

        st.session_state.pop("rover_sim", None)
        st.session_state.pop("rover_sim_a", None)
        st.session_state.pop("rover_sim_b", None)

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("Rover Navigation Controls")
        st.subheader("1. Route Waypoints")

        # Action buttons
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.button(
                "Auto-Detect Route",
                width="stretch",
                key="btn_auto_safe_route",
                on_click=_update_waypoint_coords,
                kwargs={"auto": True},
                help="Queries the neural network to discover optimal Start and Target waypoints in free space (P < 0.20).",
            )
        with col_act2:
            st.button(
                "Invert Route",
                width="stretch",
                key="btn_swap_route",
                on_click=_update_waypoint_coords,
                kwargs={"swap": True},
                help="Swaps Start (S) and Target (T).",
            )

        # Preset routes
        st.caption("Quick Preset Waypoints:")
        preset_cols = st.columns(2)
        with preset_cols[0]:
            st.button(
                "Moon Ridge",
                width="stretch",
                key="preset_moon_ridge",
                on_click=_update_waypoint_coords,
                kwargs={"sx1": -1.80, "sx2": 1.20, "tx1": 1.20, "tx2": 0.70},
            )
        with preset_cols[1]:
            st.button(
                "Moon Channel",
                width="stretch",
                key="preset_moon_channel",
                on_click=_update_waypoint_coords,
                kwargs={"sx1": -1.50, "sx2": 0.80, "tx1": 0.80, "tx2": -0.20},
            )

        preset_cols_2 = st.columns(2)
        with preset_cols_2[0]:
            st.button(
                "Ring Bypass",
                width="stretch",
                key="preset_ring_bypass",
                on_click=_update_waypoint_coords,
                kwargs={"sx1": -1.80, "sx2": 0.00, "tx1": 1.80, "tx2": 0.00},
            )
        with preset_cols_2[1]:
            st.button(
                "Spiral Run",
                width="stretch",
                key="preset_spiral_run",
                on_click=_update_waypoint_coords,
                kwargs={"sx1": -1.80, "sx2": -1.80, "tx1": 1.80, "tx2": 1.80},
            )

        start_x1 = st.slider("Start Position x1", -2.4, 2.4, st.session_state.get("rover_start_x1", -1.80), 0.05, key="rover_start_x1")
        start_x2 = st.slider("Start Position x2", -2.4, 2.4, st.session_state.get("rover_start_x2", 1.20), 0.05, key="rover_start_x2")
        target_x1 = st.slider("Target Goal x1", -2.4, 2.4, st.session_state.get("rover_target_x1", 1.20), 0.05, key="rover_target_x1")
        target_x2 = st.slider("Target Goal x2", -2.4, 2.4, st.session_state.get("rover_target_x2", 0.70), 0.05, key="rover_target_x2")

        with st.expander("Waypoint Micro-Nudge Controls", expanded=False):
            st.caption("Nudge Start (S) Position (0.10 step):")
            col_ns1, col_ns2 = st.columns(2)
            col_ns1.button(
                "Start X -0.1",
                width="stretch",
                key="nudge_s_xm",
                on_click=_update_waypoint_coords,
                kwargs={"dsx1": -0.10},
                help="Move Start point left along X1",
            )
            col_ns2.button(
                "Start X +0.1",
                width="stretch",
                key="nudge_s_xp",
                on_click=_update_waypoint_coords,
                kwargs={"dsx1": 0.10},
                help="Move Start point right along X1",
            )
            col_ns1.button(
                "Start Y -0.1",
                width="stretch",
                key="nudge_s_ym",
                on_click=_update_waypoint_coords,
                kwargs={"dsx2": -0.10},
                help="Move Start point down along X2",
            )
            col_ns2.button(
                "Start Y +0.1",
                width="stretch",
                key="nudge_s_yp",
                on_click=_update_waypoint_coords,
                kwargs={"dsx2": 0.10},
                help="Move Start point up along X2",
            )

            st.caption("Nudge Target (T) Position (0.10 step):")
            col_nt1, col_nt2 = st.columns(2)
            col_nt1.button(
                "Target X -0.1",
                width="stretch",
                key="nudge_t_xm",
                on_click=_update_waypoint_coords,
                kwargs={"dtx1": -0.10},
                help="Move Target point left along X1",
            )
            col_nt2.button(
                "Target X +0.1",
                width="stretch",
                key="nudge_t_xp",
                on_click=_update_waypoint_coords,
                kwargs={"dtx1": 0.10},
                help="Move Target point right along X1",
            )
            col_nt1.button(
                "Target Y -0.1",
                width="stretch",
                key="nudge_t_ym",
                on_click=_update_waypoint_coords,
                kwargs={"dtx2": -0.10},
                help="Move Target point down along X2",
            )
            col_nt2.button(
                "Target Y +0.1",
                width="stretch",
                key="nudge_t_yp",
                on_click=_update_waypoint_coords,
                kwargs={"dtx2": 0.10},
                help="Move Target point up along X2",
            )

        # Live coordinate guidance and obstacle warning
        if active_eval_model is not None:
            with no_grad():
                pts_t = Tensor(np.array([[start_x1, start_x2], [target_x1, target_x2]], dtype=np.float32), requires_grad=False)
                pts_log = active_eval_model(pts_t).data
                exp_p = np.exp(pts_log - np.max(pts_log, axis=-1, keepdims=True))
                probs_p = exp_p / np.sum(exp_p, axis=-1, keepdims=True)
                start_haz = float(probs_p[0, 1])
                target_haz = float(probs_p[1, 1])

            if target_haz > 0.50:
                st.warning(f"Target is inside an obstacle zone ({target_haz*100:.1f}% hazard). Use 'Auto-Detect Route' to snap to open space.")
            if start_haz > 0.50:
                st.warning(f"Start is inside an obstacle zone ({start_haz*100:.1f}% hazard). Use 'Auto-Detect Route' to snap to open space.")

        with st.expander("Physical Simulation Tuning", expanded=True):
            step_size = st.slider("Step Size (Speed)", 0.04, 0.25, 0.09, 0.01, key="rover_step_size")
            ray_len = st.slider("Sensor Lookahead", 0.15, 0.70, 0.35, 0.05, key="rover_ray_len")
            avoidance_weight = st.slider("Repulsion Weight", 0.5, 6.0, 1.8, 0.25, key="rover_avoid_wt")
            tangent_weight = st.slider("Tangent Wall-Following Weight", 0.0, 4.0, 1.2, 0.25, key="rover_tangent_wt")
            num_rays = st.slider("Sensor Ray Count", 3, 9, 5, 2, key="rover_num_rays")
            max_steps = st.slider("Max Steps", 20, 180, 120, 5, key="rover_max_steps")
            show_sensor_rays = st.checkbox("Show Sensor Radar Rays", value=True, key="rover_show_rays")

        # Playback State Management
        if "anim_playing" not in st.session_state:
            st.session_state["anim_playing"] = False
        if "current_anim_step" not in st.session_state:
            st.session_state["current_anim_step"] = 0

        btn_label = "Launch Rover Simulation" if nav_mode == "Single Model Mission" else "Start Dual-Model Race"
        launch_rover = st.button(btn_label, type="primary", width="stretch", key="launch_rover_btn")

    start_pos = (start_x1, start_x2)
    target_pos = (target_x1, target_x2)

    # Invalidate cached simulations if coordinates changed
    if st.session_state.get("sim_active_start") != start_pos or st.session_state.get("sim_active_target") != target_pos:
        st.session_state.pop("rover_sim", None)
        st.session_state.pop("rover_sim_a", None)
        st.session_state.pop("rover_sim_b", None)
        st.session_state.pop("rover_base_fig", None)
        st.session_state.pop("rover_base_fig_a", None)
        st.session_state.pop("rover_base_fig_b", None)
        st.session_state["anim_playing"] = False
        st.session_state["current_anim_step"] = 0

    def _cb_jump_start():
        st.session_state["anim_playing"] = False
        st.session_state["current_anim_step"] = 0
        st.session_state["rover_step_slider"] = 0

    def _cb_step_back():
        st.session_state["anim_playing"] = False
        cur = st.session_state.get("current_anim_step", 0)
        new_s = max(0, cur - 5)
        st.session_state["current_anim_step"] = new_s
        st.session_state["rover_step_slider"] = new_s

    def _cb_toggle_play():
        st.session_state["anim_playing"] = not st.session_state.get("anim_playing", False)

    def _cb_step_forward(max_s: int):
        st.session_state["anim_playing"] = False
        cur = st.session_state.get("current_anim_step", 0)
        new_s = min(max_s, cur + 5)
        st.session_state["current_anim_step"] = new_s
        st.session_state["rover_step_slider"] = new_s

    def _cb_jump_end(max_s: int):
        st.session_state["anim_playing"] = False
        st.session_state["current_anim_step"] = max_s
        st.session_state["rover_step_slider"] = max_s

    def _cb_slider_step_change():
        st.session_state["anim_playing"] = False
        st.session_state["current_anim_step"] = st.session_state.get("rover_step_slider", 0)

    # ---------------- Mission Action Bar ----------------
    col_act1, col_act2 = st.columns([1, 1], vertical_alignment="bottom")
    with col_act1:
        launch_main = st.button(
            "Launch Full Mission" if nav_mode == "Single Model Mission" else "Start Dual-Model Race",
            type="primary",
            width="stretch",
            key="rover_launch_main",
        )
    with col_act2:
        st.button(
            "Auto-Detect Route",
            width="stretch",
            key="rover_auto_route_main_btn",
            on_click=_update_waypoint_coords,
            kwargs={"auto": True},
        )

    # ---------------- SINGLE MODEL MISSION ----------------
    if nav_mode == "Single Model Mission":
        do_calc = launch_rover or launch_main
        if do_calc:
            sim_result = simulate_rover_path(
                model=model,
                start_pos=start_pos,
                target_pos=target_pos,
                max_steps=max_steps,
                step_size=step_size,
                num_rays=num_rays,
                ray_len=ray_len,
                avoidance_weight=avoidance_weight,
                tangent_weight=tangent_weight,
            )
            st.session_state["rover_sim"] = sim_result
            st.session_state["sim_active_start"] = start_pos
            st.session_state["sim_active_target"] = target_pos
            st.session_state.pop("rover_base_fig", None)
            st.session_state["anim_playing"] = False
            st.session_state["current_anim_step"] = 0
            st.session_state["rover_step_slider"] = 0
        else:
            sim_result = st.session_state.get("rover_sim", None)

        if sim_result is not None:
            trajectory = sim_result["trajectory"]
            ray_history = sim_result["ray_history"]
            hazard_history = sim_result["hazard_history"]
            success = sim_result["success"]
            collisions = sim_result["collisions"]
            steps_taken = sim_result["steps_taken"]
            final_dist = sim_result["final_distance"]

            path_len = 0.0
            for i in range(1, len(trajectory)):
                path_len += float(np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])))

            max_hazard = max(hazard_history) if hazard_history else 0.0
            total_inferences = steps_taken * num_rays + steps_taken

            # Status calculation
            if success and collisions == 0:
                status_text = "Goal Reached"
                status_delta = "Clean Route"
                status_delta_color = "normal"
            elif success and collisions > 0:
                status_text = "Goal Reached (Breaches)"
                status_delta = f"{collisions} hazard breaches"
                status_delta_color = "inverse"
            elif not success and collisions > 0:
                status_text = "Obstacle Collision"
                status_delta = "Path Blocked"
                status_delta_color = "inverse"
            else:
                status_text = "Trapped / Max Steps"
                status_delta = "Timed Out"
                status_delta_color = "inverse"

            # Pre-generate base figure once and cache in session state
            if "rover_base_fig" not in st.session_state or st.session_state["rover_base_fig"] is None:
                if HAS_PLOTLY:
                    st.session_state["rover_base_fig"] = plot_plotly_rover_path(
                        model=model,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Rover Mission on {dataset_name} ({arch_str})",
                    )
            base_fig = st.session_state.get("rover_base_fig", None)

            # ---------------- Full Media Transport Bar ----------------
            max_step_limit = len(trajectory) - 1
            is_playing = st.session_state.get("anim_playing", False)
            c_first, c_prev, c_play_pause, c_next, c_last, c_spd = st.columns([1, 1, 1.2, 1, 1, 2], vertical_alignment="bottom")
            c_first.button("|<< Start", on_click=_cb_jump_start, width="stretch", key="ctrl_start", help="Jump to Step 0")
            c_prev.button("<< -5", on_click=_cb_step_back, width="stretch", key="ctrl_prev", help="Step back 5 frames")
            if is_playing:
                c_play_pause.button("|| Pause", on_click=_cb_toggle_play, type="secondary", width="stretch", key="ctrl_play_pause", help="Pause animation playback")
            else:
                c_play_pause.button("> Play", on_click=_cb_toggle_play, type="primary", width="stretch", key="ctrl_play_pause", help="Start continuous animation playback")
            c_next.button("+5 >>", on_click=_cb_step_forward, args=(max_step_limit,), width="stretch", key="ctrl_next", help="Step forward 5 frames")
            c_last.button("End >>|", on_click=_cb_jump_end, args=(max_step_limit,), width="stretch", key="ctrl_end", help="Jump to final step")
            speed_choice = c_spd.select_slider(
                "Playback Speed",
                options=["Slow-Mo (0.18s)", "Normal (0.08s)", "Fast (0.03s)"],
                value="Normal (0.08s)",
                key="rover_playback_speed",
            )
            speed_delay_map = {
                "Fast (0.03s)": 0.03,
                "Normal (0.08s)": 0.08,
                "Slow-Mo (0.18s)": 0.18,
            }
            playback_delay = speed_delay_map.get(speed_choice, 0.08)

            # Interactive Step Playback Scrubber
            cur_step_val = max(0, min(st.session_state.get("current_anim_step", max_step_limit), max_step_limit))
            step_idx = st.slider(
                "Inspect Simulation Step (Scrubber)",
                min_value=0,
                max_value=max_step_limit,
                value=cur_step_val,
                key="rover_step_slider",
                on_change=_cb_slider_step_change,
                help="Scrub through the rover's trajectory to inspect position, heading, and sensor radar rays at each individual step.",
            )

            # Placement Mode Selector & Waypoint Click Helper
            col_click_mode, col_click_info = st.columns([1, 1])
            with col_click_mode:
                st.radio(
                    "Click on Map Sets:",
                    ["Start Point (S) [Green]", "Target Point (T) [Red]"],
                    horizontal=True,
                    key="rover_waypoint_click_mode",
                    help="Click anywhere on the potential field map to position Start (S) or Target (T).",
                )
            with col_click_info:
                if target_haz > 0.50 or start_haz > 0.50:
                    st.warning("Warning: Waypoint is inside an obstacle zone! Click open blue space or use 'Auto-Detect Route'.")
                else:
                    st.caption("Tip: Click anywhere on the contour map to instantly relocate the selected waypoint.")

            # Dynamic animation placeholders
            telemetry_placeholder = st.empty()
            chart_placeholder = st.empty()

            if st.session_state.get("anim_playing", False):
                start_s = st.session_state.get("current_anim_step", 0)
                if start_s >= max_step_limit:
                    start_s = 0
                    st.session_state["current_anim_step"] = 0
                    st.session_state["rover_step_slider"] = 0
                for s_idx in range(start_s, max_step_limit + 1):
                    if not st.session_state.get("anim_playing", False):
                        break
                    st.session_state["current_anim_step"] = s_idx
                    st.session_state["rover_step_slider"] = s_idx
                    cur_p = trajectory[s_idx]
                    cur_d = float(np.linalg.norm(np.array(cur_p) - np.array(target_pos)))
                    cur_h = hazard_history[s_idx] if s_idx < len(hazard_history) else 0.0
                    cur_inf = s_idx * num_rays + s_idx

                    if s_idx == max_step_limit:
                        s_text = status_text
                        s_delta = status_delta
                        s_color = status_delta_color
                    else:
                        s_text = "Navigating..."
                        s_delta = f"Step {s_idx}/{max_step_limit}"
                        s_color = "normal"

                    with telemetry_placeholder.container():
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Mission Status", s_text, delta=s_delta, delta_color=s_color)
                        m2.metric("Steps Taken", f"{s_idx} / {max_steps}")
                        m3.metric("Distance to Target", f"{cur_d:.2f}")
                        m4.metric("Instant Hazard", f"{cur_h * 100:.1f}%", delta="Safe" if cur_h <= 0.50 else "Hazard Warning", delta_color="normal" if cur_h <= 0.50 else "inverse")
                        m5.metric("Forward Passes", f"{cur_inf}")

                    with chart_placeholder.container():
                        col_c, col_d = st.columns([3, 2])
                        with col_c:
                            if HAS_PLOTLY:
                                fig_live = plot_plotly_rover_path(
                                    model=model,
                                    X=X,
                                    y=y,
                                    trajectory=trajectory,
                                    ray_history=ray_history,
                                    start_pos=start_pos,
                                    target_pos=target_pos,
                                    show_rays=show_sensor_rays,
                                    step_index=s_idx,
                                    title=f"Live Mission: {dataset_name} ({arch_str}) - Step {s_idx}/{max_step_limit}",
                                    base_fig=base_fig,
                                )
                                st.plotly_chart(
                                    fig_live,
                                    config={"displayModeBar": False, "scrollZoom": False},
                                    width="stretch",
                                )
                            else:
                                fig_mpl = plot_rover_path_mpl(model, X, y, trajectory[:s_idx+1], start_pos, target_pos)
                                st.pyplot(fig_mpl)
                                plt.close(fig_mpl)
                        with col_d:
                            st.subheader("Live Mission Telemetry")
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 10px; padding: 1rem; color: white;">
                                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Active Step: {s_idx} / {max_step_limit}</div>
                                    <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; margin: 0.2rem 0;">Current Pos: ({cur_p[0]:.2f}, {cur_p[1]:.2f})</div>
                                    <div style="font-size: 0.85rem; color: #CBD5E1;">
                                        Distance to Goal: <code>{cur_d:.2f}</code> | Instant Hazard: <code>{cur_h*100:.1f}%</code>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                            st.markdown("#### Real-time Sensory Radar Sweep")
                            if s_idx < len(ray_history):
                                ray_vals = [round(r["hazard"], 2) for r in ray_history[s_idx]]
                                st.caption(f"Radar Beams Hazard: {ray_vals}")
                                st.progress(min(1.0, max(0.0, cur_h)))

                    time.sleep(playback_delay)

                st.session_state["anim_playing"] = False
                st.session_state["current_anim_step"] = max_step_limit
                st.session_state["rover_step_slider"] = max_step_limit
                st.rerun()

            # Static / Scrubber View
            cur_step = max(0, min(st.session_state.get("current_anim_step", max_step_limit), max_step_limit))
            with telemetry_placeholder.container():
                m_stat, m_steps, m_dist, m_haz, m_inf = st.columns(5)
                m_stat.metric("Mission Status", status_text, delta=status_delta, delta_color=status_delta_color)
                m_steps.metric("Steps Taken", f"{steps_taken} / {max_steps}")
                m_dist.metric("Path Distance", f"{path_len:.2f}")
                m_haz.metric("Max Hazard", f"{max_hazard * 100:.1f}%", delta="Safe" if max_hazard <= 0.55 else "Breached", delta_color="normal" if max_hazard <= 0.55 else "inverse")
                m_inf.metric("Forward Inferences", f"{total_inferences}")

            with chart_placeholder.container():
                col_chart, col_details = st.columns([3, 2])
                with col_chart:
                    if HAS_PLOTLY:
                        fig_rover = plot_plotly_rover_path(
                            model=model,
                            X=X,
                            y=y,
                            trajectory=trajectory,
                            ray_history=ray_history,
                            start_pos=start_pos,
                            target_pos=target_pos,
                            show_rays=show_sensor_rays,
                            step_index=cur_step,
                            title=f"Rover Mission on {dataset_name} ({arch_str}) - Step {cur_step}/{max_step_limit}",
                            base_fig=base_fig,
                        )
                        st.plotly_chart(
                            fig_rover,
                            on_select="rerun",
                            selection_mode=["points"],
                            key="rover_plotly_map",
                            config={"displayModeBar": False, "scrollZoom": False},
                            width="stretch",
                        )
                    else:
                        fig_rover_mpl = plot_rover_path_mpl(model, X, y, trajectory[:cur_step+1], start_pos, target_pos)
                        st.pyplot(fig_rover_mpl)
                        plt.close(fig_rover_mpl)

                with col_details:
                    st.subheader("Rover Mission Diagnostics")
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 10px; padding: 1rem; color: white;">
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Active Potential Field Model</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; margin: 0.2rem 0;">{arch_str} on {dataset_name}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1;">
                                Start: <code>({start_pos[0]:.2f}, {start_pos[1]:.2f})</code> &rarr; Target: <code>({target_pos[0]:.2f}, {target_pos[1]:.2f})</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("#### Sensor Radar Ray Legend")
                    st.markdown(
                        """
                        - <span style="color:#10B981; font-weight:bold;">Green Rays</span>: $P(\\text{Class } 1) < 0.25$ (Clear path)
                        - <span style="color:#FBBF24; font-weight:bold;">Yellow Rays</span>: $0.25 \\le P(\\text{Class } 1) \\le 0.55$ (Obstacle warning)
                        - <span style="color:#EF4444; font-weight:bold;">Red Rays</span>: $P(\\text{Class } 1) > 0.55$ (Obstacle breach)
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption("How Neural Potential Field Navigation Works:")
                    st.markdown(
                        "1. **Attractive Force ($v_{\\text{goal}}$):** Pulls the rover along the intelligent Neural Geodesic Flow Field.\n"
                        "2. **Sensory Ray-Casting:** Casts forward sensor radar sweeps querying the neural network under `no_grad()`.\n"
                        "3. **Repulsive Force ($v_{\\text{avoid}}$):** Repels the rover away from high-hazard obstacle terrain.\n"
                        "4. **Momentum Smoothing:** Smooths velocity vectors to prevent jitter around corridor corners."
                    )

            # Step-by-Step Telemetry Table
            with st.expander("Step-by-Step Rover Trajectory & Sensor Telemetry", expanded=False):
                table_rows = []
                for s_i, (px, py) in enumerate(trajectory):
                    dist_i = float(np.linalg.norm(np.array([px, py]) - np.array(target_pos)))
                    haz_i = hazard_history[s_i] if s_i < len(hazard_history) else 0.0
                    table_rows.append({
                        "Step": s_i,
                        "Position (x1, x2)": f"({px:.3f}, {py:.3f})",
                        "Distance to Goal": round(dist_i, 3),
                        "Hazard P(Class 1)": f"{haz_i * 100:.1f}%",
                        "State": "Collision Breach" if haz_i > 0.55 else ("At Goal" if dist_i <= step_size else "In Transit"),
                    })
                st.dataframe(table_rows, width="stretch", hide_index=True)

        else:
            # Standby state before "Launch Full Mission" is clicked
            st.info("Mission Waypoints Positioned: Start (S) and Target (T) are set on the field. Click '**Launch Full Mission**' to calculate the autonomous navigation route.")

            # Placement Mode Selector & Waypoint Click Helper (Standby)
            col_click_mode_sb, col_click_info_sb = st.columns([1, 1])
            with col_click_mode_sb:
                st.radio(
                    "Click on Map Sets:",
                    ["Start Point (S) [Green]", "Target Point (T) [Red]"],
                    horizontal=True,
                    key="rover_waypoint_click_mode_sb",
                    help="Click anywhere on the potential field map to position Start (S) or Target (T).",
                )
            with col_click_info_sb:
                if target_haz > 0.50 or start_haz > 0.50:
                    st.warning("Warning: Waypoint is inside an obstacle zone! Click open blue space or use 'Auto-Detect Route'.")
                else:
                    st.caption("Tip: Click anywhere on the contour map to instantly relocate the selected waypoint.")

            col_chart, col_details = st.columns([3, 2])
            with col_chart:
                if HAS_PLOTLY:
                    fig_rover = plot_plotly_rover_path(
                        model=model,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Mission Route Planning on {dataset_name} ({arch_str})",
                    )
                    st.plotly_chart(
                        fig_rover,
                        on_select="rerun",
                        selection_mode=["points"],
                        key="rover_plotly_map_standby",
                        config={"displayModeBar": False, "scrollZoom": False},
                        width="stretch",
                    )
                else:
                    fig_rover_mpl = plot_rover_path_mpl(model, X, y, None, start_pos, target_pos)
                    st.pyplot(fig_rover_mpl)
                    plt.close(fig_rover_mpl)

            with col_details:
                st.subheader("Rover Mission Setup")
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 10px; padding: 1rem; color: white;">
                        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Active Potential Field Model</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; margin: 0.2rem 0;">{arch_str} on {dataset_name}</div>
                        <div style="font-size: 0.85rem; color: #CBD5E1;">
                            Start: <code>({start_pos[0]:.2f}, {start_pos[1]:.2f})</code> &rarr; Target: <code>({target_pos[0]:.2f}, {target_pos[1]:.2f})</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    #### Ready for Launch
                    - Adjust Start / Target positions via the sidebar sliders or presets.
                    - Use **Auto-Detect Route** to automatically position waypoints in verified open free space ($P < 0.20$).
                    - Click **"Launch Full Mission"** to solve the geodesic neural potential field.
                    """
                )

    # ---------------- DUAL-MODEL NAVIGATION RACE (A vs B) ----------------
    else:
        saved_comp = st.session_state["model_comparison"]
        model_a = saved_comp["model_a"]
        model_b = saved_comp["model_b"]
        X = saved_comp["X"]
        y = saved_comp["y"]
        dataset_name = saved_comp.get("dataset_name", "2D Dataset")
        arch_a = saved_comp.get("arch_a", "Model A")
        arch_b = saved_comp.get("arch_b", "Model B")
        params_a = saved_comp.get("params_a", 0)
        params_b = saved_comp.get("params_b", 0)

        # Execute dual simulation
        do_dual = launch_rover or launch_main
        if do_dual:
            sim_a = simulate_rover_path(
                model=model_a,
                start_pos=start_pos,
                target_pos=target_pos,
                max_steps=max_steps,
                step_size=step_size,
                num_rays=num_rays,
                ray_len=ray_len,
                avoidance_weight=avoidance_weight,
                tangent_weight=tangent_weight,
            )
            sim_b = simulate_rover_path(
                model=model_b,
                start_pos=start_pos,
                target_pos=target_pos,
                max_steps=max_steps,
                step_size=step_size,
                num_rays=num_rays,
                ray_len=ray_len,
                avoidance_weight=avoidance_weight,
                tangent_weight=tangent_weight,
            )
            st.session_state["rover_sim_a"] = sim_a
            st.session_state["rover_sim_b"] = sim_b
            st.session_state["sim_active_start"] = start_pos
            st.session_state["sim_active_target"] = target_pos
            st.session_state.pop("rover_base_fig_a", None)
            st.session_state.pop("rover_base_fig_b", None)
            st.session_state["anim_playing"] = False
            st.session_state["current_anim_step"] = 0
            st.session_state["rover_step_slider"] = 0
        else:
            sim_a = st.session_state.get("rover_sim_a", None)
            sim_b = st.session_state.get("rover_sim_b", None)

        if sim_a is not None and sim_b is not None:
            # Calculate metrics for Model A
            traj_a = sim_a["trajectory"]
            path_len_a = sum(float(np.linalg.norm(np.array(traj_a[i]) - np.array(traj_a[i-1]))) for i in range(1, len(traj_a)))
            max_haz_a = max(sim_a["hazard_history"]) if sim_a["hazard_history"] else 0.0
            inferences_a = sim_a["steps_taken"] * num_rays + sim_a["steps_taken"]

            # Calculate metrics for Model B
            traj_b = sim_b["trajectory"]
            path_len_b = sum(float(np.linalg.norm(np.array(traj_b[i]) - np.array(traj_b[i-1]))) for i in range(1, len(traj_b)))
            max_haz_b = max(sim_b["hazard_history"]) if sim_b["hazard_history"] else 0.0
            inferences_b = sim_b["steps_taken"] * num_rays + sim_b["steps_taken"]

            # Pre-generate base figures for Model A & B once
            if "rover_base_fig_a" not in st.session_state or st.session_state["rover_base_fig_a"] is None:
                if HAS_PLOTLY:
                    st.session_state["rover_base_fig_a"] = plot_plotly_rover_path(
                        model=model_a,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Model A ({arch_a}) Potential Field",
                    )
            if "rover_base_fig_b" not in st.session_state or st.session_state["rover_base_fig_b"] is None:
                if HAS_PLOTLY:
                    st.session_state["rover_base_fig_b"] = plot_plotly_rover_path(
                        model=model_b,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Model B ({arch_b}) Potential Field",
                    )
            base_fig_a = st.session_state.get("rover_base_fig_a", None)
            base_fig_b = st.session_state.get("rover_base_fig_b", None)

            # ---------------- Full Media Transport Bar (Race) ----------------
            max_race_steps = max(len(traj_a), len(traj_b)) - 1
            is_race_playing = st.session_state.get("anim_playing", False)
            c_first, c_prev, c_play_pause, c_next, c_last, c_spd = st.columns([1, 1, 1.2, 1, 1, 2], vertical_alignment="bottom")
            c_first.button("|<< Start", on_click=_cb_jump_start, width="stretch", key="ctrl_race_start", help="Jump to Step 0")
            c_prev.button("<< -5", on_click=_cb_step_back, width="stretch", key="ctrl_race_prev", help="Step back 5 frames")
            if is_race_playing:
                c_play_pause.button("|| Pause", on_click=_cb_toggle_play, type="secondary", width="stretch", key="ctrl_race_play_pause", help="Pause race animation playback")
            else:
                c_play_pause.button("> Play", on_click=_cb_toggle_play, type="primary", width="stretch", key="ctrl_race_play_pause", help="Start continuous race animation playback")
            c_next.button("+5 >>", on_click=_cb_step_forward, args=(max_race_steps,), width="stretch", key="ctrl_race_next", help="Step forward 5 frames")
            c_last.button("End >>|", on_click=_cb_jump_end, args=(max_race_steps,), width="stretch", key="ctrl_race_end", help="Jump to final step")
            speed_choice = c_spd.select_slider(
                "Playback Speed",
                options=["Slow-Mo (0.18s)", "Normal (0.08s)", "Fast (0.03s)"],
                value="Normal (0.08s)",
                key="rover_race_playback_speed",
            )
            speed_delay_map = {
                "Fast (0.03s)": 0.03,
                "Normal (0.08s)": 0.08,
                "Slow-Mo (0.18s)": 0.18,
            }
            playback_delay = speed_delay_map.get(speed_choice, 0.08)

            cur_race_val = max(0, min(st.session_state.get("current_anim_step", max_race_steps), max_race_steps))
            race_step = st.slider(
                "Synchronized Race Step Playback (Scrubber)",
                min_value=0,
                max_value=max_race_steps,
                value=cur_race_val,
                key="rover_step_slider",
                on_change=_cb_slider_step_change,
                help="Scrub through both rovers simultaneously to observe decision boundary navigation side by side.",
            )

            dual_chart_placeholder = st.empty()

            if st.session_state.get("anim_playing", False):
                start_r = st.session_state.get("current_anim_step", 0)
                if start_r >= max_race_steps:
                    start_r = 0
                    st.session_state["current_anim_step"] = 0
                    st.session_state["rover_step_slider"] = 0
                for r_idx in range(start_r, max_race_steps + 1):
                    if not st.session_state.get("anim_playing", False):
                        break
                    st.session_state["current_anim_step"] = r_idx
                    st.session_state["rover_step_slider"] = r_idx
                    with dual_chart_placeholder.container():
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.subheader(f"Model A: {arch_a}")
                            if HAS_PLOTLY:
                                fig_a_live = plot_plotly_rover_path(
                                    model=model_a,
                                    X=X,
                                    y=y,
                                    trajectory=traj_a,
                                    ray_history=sim_a["ray_history"],
                                    start_pos=start_pos,
                                    target_pos=target_pos,
                                    show_rays=show_sensor_rays,
                                    step_index=min(r_idx, len(traj_a) - 1),
                                    title=f"Model A ({arch_a}) - Step {min(r_idx, len(traj_a)-1)}/{len(traj_a)-1}",
                                    base_fig=base_fig_a,
                                )
                                st.plotly_chart(fig_a_live, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                        with col_b:
                            st.subheader(f"Model B: {arch_b}")
                            if HAS_PLOTLY:
                                fig_b_live = plot_plotly_rover_path(
                                    model=model_b,
                                    X=X,
                                    y=y,
                                    trajectory=traj_b,
                                    ray_history=sim_b["ray_history"],
                                    start_pos=start_pos,
                                    target_pos=target_pos,
                                    show_rays=show_sensor_rays,
                                    step_index=min(r_idx, len(traj_b) - 1),
                                    title=f"Model B ({arch_b}) - Step {min(r_idx, len(traj_b)-1)}/{len(traj_b)-1}",
                                    base_fig=base_fig_b,
                                )
                                st.plotly_chart(fig_b_live, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                    time.sleep(playback_delay)

                st.session_state["anim_playing"] = False
                st.session_state["current_anim_step"] = max_race_steps
                st.session_state["rover_step_slider"] = max_race_steps
                st.rerun()

            # Static / Scrubber dual render
            cur_race_s = max(0, min(st.session_state.get("current_anim_step", max_race_steps), max_race_steps))
            with dual_chart_placeholder.container():
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader(f"Model A: {arch_a}")
                    st.caption(f"Parameters: {params_a:,} | Outcome: {'Goal Reached' if sim_a['success'] else 'Failed'} | Collisions: {sim_a['collisions']}")
                    if HAS_PLOTLY:
                        fig_a = plot_plotly_rover_path(
                            model=model_a,
                            X=X,
                            y=y,
                            trajectory=traj_a,
                            ray_history=sim_a["ray_history"],
                            start_pos=start_pos,
                            target_pos=target_pos,
                            show_rays=show_sensor_rays,
                            step_index=min(cur_race_s, len(traj_a) - 1),
                            title=f"Model A ({arch_a}) Potential Field",
                            base_fig=base_fig_a,
                        )
                        st.plotly_chart(
                            fig_a,
                            key="plotly_rover_a",
                            config={"displayModeBar": False, "scrollZoom": False},
                            width="stretch",
                        )
                    else:
                        fig_a_mpl = plot_rover_path_mpl(model_a, X, y, traj_a[:min(cur_race_s, len(traj_a)-1)+1], start_pos, target_pos)
                        st.pyplot(fig_a_mpl)
                        plt.close(fig_a_mpl)

                with col_b:
                    st.subheader(f"Model B: {arch_b}")
                    st.caption(f"Parameters: {params_b:,} | Outcome: {'Goal Reached' if sim_b['success'] else 'Failed'} | Collisions: {sim_b['collisions']}")
                    if HAS_PLOTLY:
                        fig_b = plot_plotly_rover_path(
                            model=model_b,
                            X=X,
                            y=y,
                            trajectory=traj_b,
                            ray_history=sim_b["ray_history"],
                            start_pos=start_pos,
                            target_pos=target_pos,
                            show_rays=show_sensor_rays,
                            step_index=min(cur_race_s, len(traj_b) - 1),
                            title=f"Model B ({arch_b}) Potential Field",
                            base_fig=base_fig_b,
                        )
                        st.plotly_chart(
                            fig_b,
                            key="plotly_rover_b",
                            config={"displayModeBar": False, "scrollZoom": False},
                            width="stretch",
                        )
                    else:
                        fig_b_mpl = plot_rover_path_mpl(model_b, X, y, traj_b[:min(cur_race_s, len(traj_b)-1)+1], start_pos, target_pos)
                        st.pyplot(fig_b_mpl)
                        plt.close(fig_b_mpl)

            # Comparative Race Telemetry Summary
            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
            st.subheader("Dual-Model Navigation Battle Telemetry")

            race_table = [
                {"Metric": "Model Architecture", "Model A": arch_a, "Model B": arch_b},
                {"Metric": "Parameter Count", "Model A": f"{params_a:,}", "Model B": f"{params_b:,}"},
                {"Metric": "Mission Status", "Model A": "Goal Reached" if sim_a["success"] else "Failed", "Model B": "Goal Reached" if sim_b["success"] else "Failed"},
                {"Metric": "Steps Taken", "Model A": f"{sim_a['steps_taken']} / {max_steps}", "Model B": f"{sim_b['steps_taken']} / {max_steps}"},
                {"Metric": "Path Distance", "Model A": f"{path_len_a:.2f}", "Model B": f"{path_len_b:.2f}"},
                {"Metric": "Max Hazard Encountered", "Model A": f"{max_haz_a*100:.1f}%", "Model B": f"{max_haz_b*100:.1f}%"},
                {"Metric": "Obstacle Breaches", "Model A": f"{sim_a['collisions']}", "Model B": f"{sim_b['collisions']}"},
                {"Metric": "Forward Inferences", "Model A": f"{inferences_a}", "Model B": f"{inferences_b}"},
            ]
            st.dataframe(race_table, width="stretch", hide_index=True)
        else:
            st.info("Race Waypoints Configured: Start (S) and Target (T) are set on the field. Click '**Start Dual-Model Race**' to benchmark both models.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Model A: {arch_a}")
                if HAS_PLOTLY:
                    fig_a = plot_plotly_rover_path(
                        model=model_a,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Model A ({arch_a}) Potential Field",
                    )
                    st.plotly_chart(
                        fig_a,
                        key="plotly_rover_a_standby",
                        config={"displayModeBar": False, "scrollZoom": False},
                        width="stretch",
                    )
                else:
                    fig_a_mpl = plot_rover_path_mpl(model_a, X, y, None, start_pos, target_pos)
                    st.pyplot(fig_a_mpl)
                    plt.close(fig_a_mpl)

            with col_b:
                st.subheader(f"Model B: {arch_b}")
                if HAS_PLOTLY:
                    fig_b = plot_plotly_rover_path(
                        model=model_b,
                        X=X,
                        y=y,
                        trajectory=None,
                        ray_history=None,
                        start_pos=start_pos,
                        target_pos=target_pos,
                        show_rays=False,
                        title=f"Model B ({arch_b}) Potential Field",
                    )
                    st.plotly_chart(
                        fig_b,
                        key="plotly_rover_b_standby",
                        config={"displayModeBar": False, "scrollZoom": False},
                        width="stretch",
                    )
                else:
                    fig_b_mpl = plot_rover_path_mpl(model_b, X, y, None, start_pos, target_pos)
                    st.pyplot(fig_b_mpl)
                    plt.close(fig_b_mpl)

        st.markdown(
            """
            > [!TIP]
            > **Capacity & Navigation Dynamics:** An underfitted model (or one with insufficient hidden capacity/layers) 
            > produces overly coarse decision boundaries that cut through complex obstacle corridors or fail to separate 
            > interlocking topologies (like spirals). In contrast, a deeper, well-fitted neural network generates smooth, 
            > well-calibrated potential field contours, allowing the autonomous rover's forward sensor rays to steer 
            > precisely through tight passages without collision breaches.
            """
        )


# -----------------------------------------------------------------------------
# Tab 3: Handwritten Digit Recognition
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
    col_canvas, col_results = st.columns(2, gap="large")

    with col_canvas:
        st.subheader("Drawing Canvas")

        # State Management for Undo/Redo/Clear
        if "canvas_key" not in st.session_state:
            st.session_state["canvas_key"] = 0
        if "stroke_history" not in st.session_state:
            st.session_state["stroke_history"] = []
        if "redo_stack" not in st.session_state:
            st.session_state["redo_stack"] = []

        # Action functions for Undo, Redo, and Clear
        def on_undo():
            if st.session_state["stroke_history"]:
                stroke = st.session_state["stroke_history"].pop()
                st.session_state["redo_stack"].append(stroke)
                st.session_state["canvas_key"] += 1

        def on_redo():
            if st.session_state["redo_stack"]:
                stroke = st.session_state["redo_stack"].pop()
                st.session_state["stroke_history"].append(stroke)
                st.session_state["canvas_key"] += 1

        def on_clear():
            st.session_state["stroke_history"] = []
            st.session_state["redo_stack"] = []
            st.session_state["canvas_key"] += 1

        # Canvas controls
        brush_width = st.slider(
            "Brush Width", min_value=8, max_value=36, value=20, step=2,
            key="mnist_brush_width",
        )

        init_drawing = (
            {"objects": st.session_state["stroke_history"]}
            if st.session_state["stroke_history"]
            else None
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=brush_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            display_toolbar=False,
            initial_drawing=init_drawing,
            key=f"mnist_canvas_{st.session_state['canvas_key']}",
        )

        # Synchronize stroke history only when NEW strokes are drawn by the user
        if canvas_result is not None and canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            current_objects = canvas_result.json_data["objects"]
            if len(current_objects) > len(st.session_state["stroke_history"]):
                st.session_state["stroke_history"] = current_objects
                st.session_state["redo_stack"] = []

        # Native Action Buttons Row
        c1, c2, c3 = st.columns(3)
        c1.button(
            "Undo",
            width="stretch",
            key="mnist_undo",
            on_click=on_undo,
            disabled=len(st.session_state["stroke_history"]) == 0,
        )
        c2.button(
            "Redo",
            width="stretch",
            key="mnist_redo",
            on_click=on_redo,
            disabled=len(st.session_state["redo_stack"]) == 0,
        )
        c3.button(
            "Clear",
            width="stretch",
            key="mnist_clear",
            on_click=on_clear,
        )

        # Preprocess and show thumbnails
        digits = segment_and_preprocess_digits(canvas_result)

        if digits:
            if len(digits) == 1:
                st.subheader("Preprocessed Input (28x28)")
                st.image(digits[0], clamp=True, width=140, caption="28x28 Centered MNIST Input")
            else:
                st.subheader(f"Segmented Digits ({len(digits)} Detected)")
                thumb_cols = st.columns(min(len(digits), 5))
                for idx, (col_thumb, d_img) in enumerate(zip(thumb_cols, digits)):
                    with col_thumb:
                        st.image(d_img, clamp=True, width=70, caption=f"Digit #{idx + 1}")

    with col_results:
        if digits:
            # Run sequential inference for each detected digit
            predictions = []
            for d_img in digits:
                input_tensor = Tensor(d_img.reshape(1, 28, 28).astype(np.float32), requires_grad=False)
                with no_grad():
                    logits = model(input_tensor)
                probs = stable_softmax(logits.data[0])
                pred_cls = int(np.argmax(probs))
                conf = float(probs[pred_cls])
                predictions.append((pred_cls, conf, probs))

            if len(predictions) == 1:
                # Single-digit UI
                pred_cls, conf, probs = predictions[0]
                top3_indices = np.argsort(probs)[::-1][:3]

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <p class="prediction-label">Predicted Digit</p>
                        <p class="prediction-digit">{pred_cls}</p>
                        <p class="prediction-confidence">{conf * 100:.1f}% Confidence</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top-3 ranking cards
                st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
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
                                <p style="font-size:0.75rem;color:#94A3B8;margin:0;text-transform:uppercase;letter-spacing:0.5px;">{rank_labels[i]}</p>
                                <p class="rank-digit">{idx}</p>
                                <p class="rank-prob">{p * 100:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Full probability distribution bar chart with vertical separation
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.subheader("Probability Distribution")
                if HAS_PLOTLY:
                    fig_probs = plot_plotly_digit_probabilities(probs)
                    st.plotly_chart(fig_probs, key="plotly_mnist_probs", config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                else:
                    fig_probs = plot_probability_distribution(probs)
                    st.pyplot(fig_probs)
                    plt.close(fig_probs)

            else:
                # Multi-digit sequence UI
                combined_number = "".join(str(p[0]) for p in predictions)
                avg_confidence = float(np.mean([p[1] for p in predictions]))

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <p class="prediction-label">Recognized Number Sequence ({len(predictions)} Digits)</p>
                        <p class="prediction-digit" style="letter-spacing: 4px;">{combined_number}</p>
                        <p class="prediction-confidence">{avg_confidence * 100:.1f}% Avg Confidence</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
                st.subheader("Left-to-Right Digit Breakdown")
                breakdown_cols = st.columns(min(len(predictions), 5))
                for i, (col, (pred_cls, conf, _)) in enumerate(zip(breakdown_cols, predictions)):
                    with col:
                        st.markdown(
                            f"""
                            <div class="rank-card">
                                <p style="font-size:0.75rem;color:#94A3B8;margin:0;text-transform:uppercase;letter-spacing:0.5px;">Position #{i+1}</p>
                                <p class="rank-digit">{pred_cls}</p>
                                <p class="rank-prob">{conf * 100:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Multi-digit probability distribution inspector
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.subheader("Per-Digit Probability Breakdown")
                tabs_digits = st.tabs([f"Digit #{i+1} ('{pred_cls}')" for i, (pred_cls, _, _) in enumerate(predictions)])
                for i, (tab, (pred_cls, conf, probs)) in enumerate(zip(tabs_digits, predictions)):
                    with tab:
                        st.caption(f"Predicted class: **{pred_cls}** with **{conf * 100:.2f}%** confidence")
                        if HAS_PLOTLY:
                            fig_p = plot_plotly_digit_probabilities(probs)
                            st.plotly_chart(fig_p, key=f"plotly_mnist_probs_digit_{i}", config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
                        else:
                            fig_p = plot_probability_distribution(probs)
                            st.pyplot(fig_p)
                            plt.close(fig_p)
        else:
            st.info("Draw one or more digits on the canvas to see predictions.")


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
        options=["2D Decision Boundaries", "Autonomous Neural Pathfinding", "Handwritten Digit Recognition"],
        default="2D Decision Boundaries",
        label_visibility="collapsed",
    )
    if not selected_tab:
        selected_tab = "2D Decision Boundaries"

    if selected_tab == "2D Decision Boundaries":
        render_decision_boundary_tab()
    elif selected_tab == "Autonomous Neural Pathfinding":
        render_neural_pathfinding_tab()
    else:
        with st.sidebar:
            st.markdown("### MNIST Inference Mode")
            st.info("The pre-trained model (`mnist_mlp.ng`) is loaded. Draw on the canvas to evaluate.")
        render_mnist_inference_tab()


if __name__ == "__main__":
    main()
