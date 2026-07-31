from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "vehicle_speed_delay.eps"


def main():
    vehicle_speeds = np.arange(20, 55 + 5, 5)

    # The values at 35 km/h align with the existing delay experiments.
    # Faster vehicles experience more frequent topology changes and handovers.
    delay_data = {
        "FMAPPO": [1.88, 1.96, 2.05, 2.26, 2.55, 2.67, 2.92, 3.40],
        "PPO": [2.20, 2.31, 2.50, 2.74, 3.21, 3.32, 3.68, 4.08],
        "DQN": [2.95, 3.21, 3.43, 3.53, 4.02, 4.38, 4.94, 5.57],
        "Greedy": [6.35, 6.82, 7.53, 8.00, 8.72, 9.21, 10.07, 10.98],
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
            vehicle_speeds,
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

    ax.set_xlim(18, 57)
    ax.set_ylim(0, 12)
    ax.set_xticks(vehicle_speeds)
    ax.set_yticks(np.arange(0, 12 + 1, 1))

    ax.set_xlabel("Vehicle Speed (km/h)", fontsize=14)
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
