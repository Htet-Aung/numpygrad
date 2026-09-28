"""Reproduce the classifier-to-navigation study: python examples/study_navigation.py."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpygrad import Tensor, no_grad
import numpygrad.nn as nn
import numpygrad.optim as optim
from numpygrad.data import DataLoader, TensorDataset
from numpygrad.metrics import classification_metrics, stratified_indices
from numpygrad.utils.pathfinding import simulate_rover_path


def obstacle_labels(points: np.ndarray) -> np.ndarray:
    """Label a vertical wall with one traversable opening as class 1."""
    x, y = np.asarray(points).T
    wall = (np.abs(x) <= 0.34) & (np.abs(y) <= 1.7)
    opening = (y >= 0.35) & (y <= 0.95)
    return (wall & ~opening).astype(np.int64)


def sample_map(seed: int, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Sample a balanced map with extra examples near the wall and opening."""
    rng = np.random.default_rng(seed)
    batches = [rng.uniform(-2.2, 2.2, size=(n * 4, 2))]
    batches.append(np.column_stack((rng.uniform(-0.6, 0.6, n * 2), rng.uniform(-1.9, 1.9, n * 2))))
    points = np.concatenate(batches).astype(np.float32)
    labels = obstacle_labels(points)
    positive = rng.choice(np.flatnonzero(labels == 1), size=n // 2, replace=False)
    negative = rng.choice(np.flatnonzero(labels == 0), size=n - n // 2, replace=False)
    indices = rng.permutation(np.concatenate((positive, negative)))
    return points[indices], labels[indices]


def build_model(width: int) -> nn.Sequential:
    """Build a two-hidden-layer classifier for the map experiment."""
    return nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))


class OracleMap:
    """Provide exact obstacle labels through the rover's existing model interface."""

    def eval(self) -> OracleMap:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        labels = obstacle_labels(x.data)
        logits = np.column_stack((1 - labels, labels)).astype(np.float32) * 20.0
        return Tensor(logits, requires_grad=False)


ROUTES = [((-1.8, -1.0), (1.8, -1.0)), ((-1.8, 1.2), (1.8, 1.2))]


def evaluate_routes(model: nn.Module) -> dict[str, float]:
    """Measure route success, true wall breaches, and mean traversed distance."""
    successes, clean_successes, breaches, distances = [], [], [], []
    for start, target in ROUTES:
        result = simulate_rover_path(model, start_pos=start, target_pos=target, max_steps=160)
        trajectory = np.asarray(result["trajectory"], dtype=np.float32)
        # Sample each segment as well as its endpoints to catch a wall crossing.
        samples = np.concatenate([trajectory, (trajectory[:-1] + trajectory[1:]) / 2])
        breach_count = int(np.sum(obstacle_labels(samples)))
        breaches.append(breach_count)
        successes.append(int(result["success"]))
        clean_successes.append(int(result["success"] and breach_count == 0))
        distances.append(float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()))
    return {"route_success": float(np.mean(successes)), "collision_free_success": float(np.mean(clean_successes)),
            "true_wall_breaches": float(np.mean(breaches)),
            "route_length": float(np.mean(distances))}


def run_case(seed: int, width: int, label_noise: float, epochs: int) -> dict[str, float | int]:
    """Train one seeded model and evaluate on clean held-out map examples."""
    np.random.seed(seed)
    x, y = sample_map(seed)
    train, validation, test = stratified_indices(y, seed=seed)
    noisy_train = y[train].copy()
    rng = np.random.default_rng(seed + 10_000)
    flips = rng.random(len(train)) < label_noise
    noisy_train[flips] = 1 - noisy_train[flips]
    model = build_model(width)
    loader = DataLoader(TensorDataset(x[train], noisy_train), batch_size=64, shuffle=True, seed=seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y.data)
            loss.backward()
            optimizer.step()
        model.eval()
        with no_grad():
            val_logits = model(Tensor(x[validation], requires_grad=False)).data
        val_loss = classification_metrics(val_logits, y[validation])["nll"]
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with no_grad():
        test_logits = model(Tensor(x[test], requires_grad=False)).data
    metrics = classification_metrics(test_logits, y[test])
    return {"seed": seed, "width": width, "label_noise": label_noise, "epochs": epochs,
            "train_size": len(train), "validation_size": len(validation), "test_size": len(test),
            **metrics, **evaluate_routes(model)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("docs/results/navigation_study.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.epochs < 1:
        parser.error("--seeds and --epochs must be positive")
    rows = []
    for seed in range(args.seeds):
        for width in (8, 32):
            for noise in (0.0, 0.1, 0.2):
                row = run_case(seed, width, noise, args.epochs)
                rows.append(row)
                print(f"seed={seed} width={width} noise={noise:.1f}: test={row['accuracy']:.3f}, clean_route={row['collision_free_success']:.2f}, breaches={row['true_wall_breaches']:.1f}", flush=True)
    oracle = evaluate_routes(OracleMap())
    payload = {"protocol": "balanced synthetic wall with a known opening; 60/20/20 stratified split; label noise on training data only; best validation NLL checkpoint; two fixed routes", 
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
               "oracle": oracle, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
