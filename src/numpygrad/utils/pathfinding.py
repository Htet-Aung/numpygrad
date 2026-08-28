"""
Autonomous Neural Rover Pathfinding and Obstacle Avoidance Simulation Engine.

Simulates a physical agent navigating an environment where a trained neural network's
classification probability surface acts as a dynamic artificial potential / hazard field.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from numpygrad.core.tensor import Tensor, no_grad
import numpygrad.nn as nn


def compute_geodesic_flow(
    model: nn.Module,
    target_pos: Tuple[float, float],
    grid_bounds: Tuple[float, float] = (-2.4, 2.4),
    resolution: int = 45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes an intelligent corridor-aware neural geodesic guidance flow field
    toward `target_pos` over the model's safe probability terrain.

    Parameters
    ----------
    model : nn.Module
        Trained 2D neural network predicting obstacle classification probabilities.
    target_pos : Tuple[float, float]
        Target goal (x, y) coordinates.
    grid_bounds : Tuple[float, float]
        Coordinate span along each spatial dimension.
    resolution : int
        Number of discrete grid points per dimension.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (grid_x, grid_y, flow_x, flow_y) containing coordinate axes and normalized 2D flow vectors.
    """
    import heapq

    model.eval()
    grid_x = np.linspace(grid_bounds[0], grid_bounds[1], resolution, dtype=np.float32)
    grid_y = np.linspace(grid_bounds[0], grid_bounds[1], resolution, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    with no_grad():
        logits = model(Tensor(grid_points, requires_grad=False)).data
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_l / np.sum(exp_l, axis=-1, keepdims=True))[:, 1]
    hazard_map = probs.reshape((resolution, resolution))

    # Cell traversal cost: massive penalty for obstacle regions (Class 1) to prevent wall tunneling
    cost_map = np.where(hazard_map > 0.40, 800.0, 1.0 + 30.0 * (hazard_map ** 2))

    # 8-connected Dijkstra distance transform starting from the grid cell containing target_pos
    tc = int(np.clip(np.argmin(np.abs(grid_x - target_pos[0])), 0, resolution - 1))
    tr = int(np.clip(np.argmin(np.abs(grid_y - target_pos[1])), 0, resolution - 1))

    dist = np.full((resolution, resolution), np.inf, dtype=np.float64)
    dist[tr, tc] = 0.0
    pq = [(0.0, tr, tc)]

    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < resolution and 0 <= nc < resolution:
                step_metric = np.sqrt((dr * dy) ** 2 + (dc * dx) ** 2)
                c_edge = step_metric * 0.5 * (cost_map[r, c] + cost_map[nr, nc])
                if d + c_edge < dist[nr, nc]:
                    dist[nr, nc] = d + c_edge
                    heapq.heappush(pq, (d + c_edge, nr, nc))

    # Negative gradient of distance field points towards goal along lowest-cost corridors
    gy, gx = np.gradient(dist, dy, dx)
    flow_x = -gx.astype(np.float32)
    flow_y = -gy.astype(np.float32)
    fnorm = np.sqrt(flow_x ** 2 + flow_y ** 2)
    valid = fnorm > 1e-6
    flow_x[valid] /= fnorm[valid]
    flow_y[valid] /= fnorm[valid]
    flow_x[~valid] = 0.0
    flow_y[~valid] = 0.0

    return grid_x, grid_y, flow_x, flow_y


def sample_geodesic_vector(
    pos: np.ndarray,
    target: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
) -> np.ndarray:
    """Samples the 2D unit guidance vector at continuous coordinate `pos` via bilinear interpolation."""
    res = len(grid_x)
    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])

    c_flt = (pos[0] - grid_x[0]) / dx
    r_flt = (pos[1] - grid_y[0]) / dy

    if 0 <= c_flt <= res - 1 and 0 <= r_flt <= res - 1:
        c0 = int(np.clip(np.floor(c_flt), 0, res - 2))
        r0 = int(np.clip(np.floor(r_flt), 0, res - 2))
        c1, r1 = c0 + 1, r0 + 1

        wc1 = c_flt - c0
        wc0 = 1.0 - wc1
        wr1 = r_flt - r0
        wr0 = 1.0 - wr1

        vx = wr0 * (wc0 * flow_x[r0, c0] + wc1 * flow_x[r0, c1]) + wr1 * (wc0 * flow_x[r1, c0] + wc1 * flow_x[r1, c1])
        vy = wr0 * (wc0 * flow_y[r0, c0] + wc1 * flow_y[r0, c1]) + wr1 * (wc0 * flow_y[r1, c0] + wc1 * flow_y[r1, c1])
        v = np.array([vx, vy], dtype=np.float32)
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-4:
            return v / v_norm

    to_t = target - pos
    t_norm = float(np.linalg.norm(to_t))
    if t_norm > 1e-6:
        return to_t / t_norm
    return np.array([1.0, 0.0], dtype=np.float32)


def simulate_rover_path(
    model: nn.Module,
    start_pos: Tuple[float, float] = (-1.80, 1.20),
    target_pos: Tuple[float, float] = (1.20, 0.70),
    max_steps: int = 120,
    step_size: float = 0.09,
    num_rays: int = 5,
    ray_len: float = 0.35,
    avoidance_weight: float = 1.8,
    tangent_weight: float = 1.2,
    hazard_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Simulates an autonomous rover navigating from `start_pos` to `target_pos`
    using Neural Geodesic Flow Field guidance, forward ray-casting, and momentum smoothing.

    Parameters
    ----------
    model : nn.Module
        Trained 2D classification neural network predicting class probabilities.
        Class 1 represents obstacle/hazard zones, Class 0 represents free space.
    start_pos : Tuple[float, float]
        Initial (x1, x2) coordinate of the rover.
    target_pos : Tuple[float, float]
        Target goal (x1, x2) coordinate.
    max_steps : int
        Maximum simulation steps before termination.
    step_size : float
        Distance traveled per simulation step.
    num_rays : int
        Number of forward sensory rays cast across [-50 deg, +50 deg].
    ray_len : float
        Physical lookahead reach of each sensor ray.
    avoidance_weight : float
        Strength of repulsive obstacle force relative to attractive goal force.
    tangent_weight : float
        Strength of tangential wall-following force to escape local potential minima.
    hazard_threshold : float
        Class 1 probability threshold indicating an active obstacle collision.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing full simulation telemetry:
        - trajectory: List of (x, y) coordinates visited
        - ray_history: List of sensory ray measurements at each step
        - hazard_history: List of hazard probabilities at each visited point
        - success: Boolean indicating if target was reached
        - collisions: Number of steps where rover entered hazard zone
        - steps_taken: Total steps executed
        - final_distance: Euclidean distance to target at termination
    """
    model.eval()

    current_pos = np.array(start_pos, dtype=np.float32)
    target = np.array(target_pos, dtype=np.float32)

    # 1. Precompute Neural Geodesic Flow Field towards target_pos
    grid_x, grid_y, flow_x, flow_y = compute_geodesic_flow(
        model=model,
        target_pos=target_pos,
        grid_bounds=(-2.4, 2.4),
        resolution=45,
    )

    trajectory: List[Tuple[float, float]] = [tuple(current_pos.tolist())]
    ray_history: List[List[Dict[str, Any]]] = []
    hazard_history: List[float] = []
    collision_count: int = 0
    success: bool = False
    prev_v: Optional[np.ndarray] = None

    # Half-angle spread in radians (50 degrees)
    spread_rad = 50.0 * (np.pi / 180.0)
    ray_angles = np.linspace(-spread_rad, spread_rad, num_rays)

    for _ in range(max_steps):
        # 1. Check goal condition
        to_target = target - current_pos
        dist_to_target = float(np.linalg.norm(to_target))

        # Check hazard at current position
        with no_grad():
            cur_tensor = Tensor(current_pos.reshape(1, 2), requires_grad=False)
            cur_logits = model(cur_tensor).data[0]
            exp_c = np.exp(cur_logits - np.max(cur_logits))
            cur_probs = exp_c / np.sum(exp_c)
            cur_hazard = float(cur_probs[1])

        hazard_history.append(cur_hazard)
        if cur_hazard > hazard_threshold:
            collision_count += 1

        if dist_to_target <= max(step_size, 0.15):
            trajectory.append(tuple(target.tolist()))
            success = True
            break

        # 2. Intelligent Geodesic Corridor Guidance Vector
        v_goal = sample_geodesic_vector(current_pos, target, grid_x, grid_y, flow_x, flow_y)
        base_heading = np.arctan2(v_goal[1], v_goal[0])

        # 3. Cast forward sensory rays
        ray_endpoints = []
        ray_unit_vectors = []
        for angle_offset in ray_angles:
            angle = base_heading + angle_offset
            u_ray = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            r_end = current_pos + ray_len * u_ray
            ray_endpoints.append(r_end)
            ray_unit_vectors.append(u_ray)

        ray_batch = np.array(ray_endpoints, dtype=np.float32)

        # 4. Batch model inference for ray obstacle detection
        with no_grad():
            ray_tensor = Tensor(ray_batch, requires_grad=False)
            ray_logits = model(ray_tensor).data
            exp_rays = np.exp(ray_logits - np.max(ray_logits, axis=-1, keepdims=True))
            ray_probs = exp_rays / np.sum(exp_rays, axis=-1, keepdims=True)
            ray_hazards = ray_probs[:, 1]

        # Record ray history for visualization
        step_rays = []
        v_avoid = np.zeros(2, dtype=np.float32)
        for i in range(num_rays):
            h_val = float(ray_hazards[i])
            u_i = ray_unit_vectors[i]
            r_end = ray_endpoints[i]

            step_rays.append({
                "endpoint": tuple(r_end.tolist()),
                "unit_vector": tuple(u_i.tolist()),
                "hazard": h_val,
            })
            # Repulsive force points away from sensed obstacle breaches
            if h_val > 0.50:
                v_avoid -= (h_val - 0.50) * u_i

        ray_history.append(step_rays)

        # 5. Combine corridor guidance and local obstacle avoidance
        v_avoid_norm = float(np.linalg.norm(v_avoid))
        if v_avoid_norm > 0.05:
            v_avoid = v_avoid / v_avoid_norm
            v_total = v_goal + (avoidance_weight * 0.40) * v_avoid
        else:
            v_total = v_goal

        v_total_norm = float(np.linalg.norm(v_total))
        if v_total_norm > 1e-6:
            v_raw = v_total / v_total_norm
        else:
            v_raw = v_goal

        # 6. Momentum / Velocity Smoothing to prevent jitter near corridor walls
        if prev_v is None:
            v_smooth = v_raw
        else:
            v_smooth = 0.80 * v_raw + 0.20 * prev_v
            v_s_norm = float(np.linalg.norm(v_smooth))
            if v_s_norm > 1e-6:
                v_smooth = v_smooth / v_s_norm
            else:
                v_smooth = v_raw
        prev_v = v_smooth

        # 7. Step rover forward
        next_pos = current_pos + step_size * v_smooth
        next_pos = np.clip(next_pos, -2.45, 2.45)

        current_pos = next_pos
        trajectory.append(tuple(current_pos.tolist()))

    final_distance = float(np.linalg.norm(target - np.array(trajectory[-1])))
    if final_distance <= max(step_size, 0.15):
        success = True

    return {
        "trajectory": trajectory,
        "ray_history": ray_history,
        "hazard_history": hazard_history,
        "success": success,
        "collisions": collision_count,
        "steps_taken": len(trajectory) - 1,
        "final_distance": final_distance,
    }


def find_safe_waypoints(
    model: nn.Module,
    grid_bounds: Tuple[float, float] = (-1.85, 1.85),
    resolution: int = 35,
    max_hazard: float = 0.20,
    min_distance: float = 2.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Dynamically discovers a pair of valid Start and Target waypoints located strictly
    in safe free-space territory (Class 0, P(Class 1) < max_hazard) across the decision field.

    Parameters
    ----------
    model : nn.Module
        Trained 2D neural network classification model.
    grid_bounds : Tuple[float, float]
        Coordinate span [min_val, max_val] along each axis.
    resolution : int
        Number of mesh samples per axis.
    max_hazard : float
        Upper bound on P(Class 1) to qualify as a safe waypoint.
    min_distance : float
        Minimum Euclidean distance desired between Start and Target.

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        (start_pos, target_pos) as rounded (x, y) coordinate pairs.
    """
    model.eval()

    xs = np.linspace(grid_bounds[0], grid_bounds[1], resolution, dtype=np.float32)
    ys = np.linspace(grid_bounds[0], grid_bounds[1], resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    with no_grad():
        t_in = Tensor(grid_points, requires_grad=False)
        logits = model(t_in).data
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        hazards = probs[:, 1]

    # Find safe candidates
    safe_mask = hazards < max_hazard
    if not np.any(safe_mask):
        k = max(2, len(hazards) // 10)
        sorted_indices = np.argsort(hazards)
        safe_indices = sorted_indices[:k]
    else:
        safe_indices = np.where(safe_mask)[0]

    safe_points = grid_points[safe_indices]

    best_pair = None

    # 1. Prefer top-left entrance (x < -0.4, y > 0.3) and bottom-right exit (x > 0.4, y < -0.3)
    tl_mask = (safe_points[:, 0] < -0.4) & (safe_points[:, 1] > 0.3)
    br_mask = (safe_points[:, 0] > 0.4) & (safe_points[:, 1] < -0.3)

    if np.any(tl_mask) and np.any(br_mask):
        tl_pts = safe_points[tl_mask]
        br_pts = safe_points[br_mask]
        if len(tl_pts) > 60:
            tl_pts = tl_pts[:: max(1, len(tl_pts) // 60)]
        if len(br_pts) > 60:
            br_pts = br_pts[:: max(1, len(br_pts) // 60)]

        dists = np.linalg.norm(tl_pts[:, None, :] - br_pts[None, :, :], axis=-1)
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        if dists[max_idx] >= min_distance:
            best_pair = (tl_pts[max_idx[0]], br_pts[max_idx[1]])

    # 2. General opposite sides (left vs right)
    if best_pair is None:
        left_mask = (safe_points[:, 0] < -0.5)
        right_mask = (safe_points[:, 0] > 0.5)

        if np.any(left_mask) and np.any(right_mask):
            left_pts = safe_points[left_mask]
            right_pts = safe_points[right_mask]
            if len(left_pts) > 60:
                left_pts = left_pts[:: max(1, len(left_pts) // 60)]
            if len(right_pts) > 60:
                right_pts = right_pts[:: max(1, len(right_pts) // 60)]

            dists = np.linalg.norm(left_pts[:, None, :] - right_pts[None, :, :], axis=-1)
            max_idx = np.unravel_index(np.argmax(dists), dists.shape)
            if dists[max_idx] >= min_distance:
                best_pair = (left_pts[max_idx[0]], right_pts[max_idx[1]])

    # 3. Maximum distance pair fallback
    if best_pair is None:
        sample_pts = safe_points
        if len(sample_pts) > 100:
            sample_pts = sample_pts[:: max(1, len(sample_pts) // 100)]
        dists = np.linalg.norm(sample_pts[:, None, :] - sample_pts[None, :, :], axis=-1)
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        best_pair = (sample_pts[max_idx[0]], sample_pts[max_idx[1]])

    start_pt = (round(float(best_pair[0][0]), 2), round(float(best_pair[0][1]), 2))
    target_pt = (round(float(best_pair[1][0]), 2), round(float(best_pair[1][1]), 2))

    return start_pt, target_pt
