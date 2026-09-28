"""Plot the navigation study: python examples/plot_navigation_study.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    source = Path("docs/results/navigation_study.json")
    runs = json.loads(source.read_text(encoding="utf-8"))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {8: "#e67e22", 32: "#2563eb"}
    for width in (8, 32):
        noise_levels = sorted({row["label_noise"] for row in runs})
        accuracy = [np.array([row["accuracy"] for row in runs if row["width"] == width and row["label_noise"] == noise]) for noise in noise_levels]
        clean = [np.array([row["collision_free_success"] for row in runs if row["width"] == width and row["label_noise"] == noise]) for noise in noise_levels]
        axes[0].errorbar(noise_levels, [x.mean() for x in accuracy], yerr=[x.std(ddof=1) for x in accuracy], marker="o", capsize=4, color=colors[width], label=f"Width {width}")
        axes[1].plot(noise_levels, [x.mean() for x in clean], marker="o", color=colors[width], label=f"Width {width}")
    axes[0].set(title="Held-out classification accuracy", ylabel="Mean ± seed SD", ylim=(0.82, 1.0))
    axes[1].set(title="Collision-free route success", ylabel="Fraction of two routes", ylim=(0, 1.05))
    for ax in axes:
        ax.set(xlabel="Training label-flip probability", xticks=[0, 0.1, 0.2])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Known-wall navigation study · 10 seeds per configuration")
    output = source.with_suffix(".png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
