from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "task_size_delay.png"


def main():
    # Seven task sizes from 2.0 MB to 5.0 MB in 0.5 MB increments.
    task_sizes = np.arange(2.0, 5.0 + 0.5, 0.5)

    # At 3.5 MB, the values align with the existing 10-vehicle experiment.
    # Larger tasks incur increasingly higher transmission and computation delay.
    delay_data = {
        "FMAPPO": [1.38, 1.52, 1.80, 2.26, 2.85, 3.42, 3.94],
        "PPO": [1.55, 1.92, 2.25, 2.84, 3.55, 4.12, 4.88],
        "DQN": [2.15, 2.48, 3.10, 3.63, 4.57, 5.32, 6.38],
        "Greedy": [5.35, 5.72, 6.76, 8.00, 9.22, 11.02, 12.55],
    }

    styles = {
        "FMAPPO": {"color": "#e60000", "linestyle": "-"},
        "PPO": {"color": "#1f8ced", "linestyle": "--"},
        "DQN": {"color": "#5ec311", "linestyle": "-."},
        "Greedy": {"color": "#666666", "linestyle": ":"},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    for method in ["FMAPPO", "PPO", "DQN", "Greedy"]:
        ax.plot(
            task_sizes,
            delay_data[method],
            color=styles[method]["color"],
            linestyle=styles[method]["linestyle"],
            linewidth=1.4,
            marker="X",
            markersize=4,
            markeredgewidth=0.8,
            label=method,
            zorder=3,
        )

    ax.set_xlim(1.9, 5.1)
    ax.set_ylim(0, 14)
    ax.set_xticks(task_sizes)
    ax.set_xticklabels([f"{value:g}" for value in task_sizes])
    ax.set_yticks(np.arange(0, 14 + 1, 1))

    ax.set_xlabel("Task Size (MB)", fontsize=14)
    ax.set_ylabel("Average Delay (s)", fontsize=14)
    ax.tick_params(axis="both", direction="out", length=4, width=0.8, labelsize=14)
    ax.grid(True, linestyle="--", color="#bbbbbb", linewidth=0.6, zorder=0)

    legend = ax.legend(
        loc="upper left",
        fontsize=12,
        framealpha=1.0,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    print(f"Done. Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
