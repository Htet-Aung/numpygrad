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


def simulate_rover_path(
    model: nn.Module,
    start_pos: Tuple[float, float] = (-1.8, 1.2),
    target_pos: Tuple[float, float] = (1.8, -1.2),
    max_steps: int = 80,
    step_size: float = 0.12,
    num_rays: int = 5,
    ray_len: float = 0.35,
    avoidance_weight: float = 2.0,
    tangent_weight: float = 1.5,
    hazard_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Simulates an autonomous rover navigating from `start_pos` to `target_pos`
    using forward ray-casting and tangential wall-following against a trained neural obstacle field.

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

    trajectory: List[Tuple[float, float]] = [tuple(current_pos.tolist())]
    ray_history: List[List[Dict[str, Any]]] = []
    hazard_history: List[float] = []
    collision_count: int = 0
    success: bool = False

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

        if dist_to_target <= step_size:
            trajectory.append(tuple(target.tolist()))
            success = True
            break

        # 2. Attractive goal unit vector
        v_goal = to_target / (dist_to_target + 1e-9)
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
            # Repulsive force points away from sensed hazard
            v_avoid -= h_val * u_i

        ray_history.append(step_rays)

        # 5. Combine attractive, repulsive, and tangential wall-following potential vectors
        v_avoid_norm = float(np.linalg.norm(v_avoid))
        if v_avoid_norm > 0.1:
            # Perpendicular tangent vector [-v_avoid_y, v_avoid_x]
            v_tangent = np.array([-v_avoid[1], v_avoid[0]], dtype=np.float32)
            v_tan_norm = float(np.linalg.norm(v_tangent))
            if v_tan_norm > 1e-9:
                u_tangent = v_tangent / v_tan_norm
            else:
                u_tangent = np.zeros(2, dtype=np.float32)

            # Choose tangent direction that aligns forward with v_goal
            if float(np.dot(u_tangent, v_goal)) < 0.0:
                u_tangent = -u_tangent

            v_total = v_goal + avoidance_weight * v_avoid + tangent_weight * u_tangent
        else:
            v_total = v_goal

        v_norm = float(np.linalg.norm(v_total))
        if v_norm > 1e-6:
            u_step = v_total / v_norm
        else:
            u_step = v_goal

        # 6. Step rover forward
        next_pos = current_pos + step_size * u_step
        next_pos = np.clip(next_pos, -3.0, 3.0)

        current_pos = next_pos
        trajectory.append(tuple(current_pos.tolist()))

    final_distance = float(np.linalg.norm(target - np.array(trajectory[-1])))
    if final_distance <= step_size:
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
