from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "dp_epsilon_system_performance.eps"


def main():
    epsilon_labels = ["0.5", "2", "4", "8", "12", "No DP"]
    x = np.arange(len(epsilon_labels))
    average_delay = [3.42, 2.98, 2.74, 2.52, 2.30, 2.22]
    completion_rate = [86.5, 90.2, 92.8, 95.1, 97.3, 98.0]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_delay = plt.subplots(figsize=(6.5, 4.5))
    ax_completion = ax_delay.twinx()

    for spine in ax_delay.spines.values():
        spine.set_linewidth(0.8)
    for spine in ax_completion.spines.values():
        spine.set_linewidth(0.8)

    delay_line, = ax_delay.plot(
        x,
        average_delay,
        marker="o",
        markersize=6,
        linewidth=2.0,
        linestyle="-",
        label="Average delay",
        color="#3C5488",
        zorder=3,
    )
    completion_line, = ax_completion.plot(
        x,
        completion_rate,
        marker="s",
        markersize=6,
        linewidth=2.0,
        linestyle="--",
        label="Completion rate",
        color="#E64B35",
        zorder=3,
    )

    ax_delay.set_xticks(x)
    ax_delay.set_xticklabels(epsilon_labels)
    ax_delay.set_xlabel(r"Privacy budget $\epsilon$", fontsize=14)
    ax_delay.set_ylabel("Average Delay (s)", fontsize=14)
    ax_completion.set_ylabel("Task Completion Rate (%)", fontsize=14)

    ax_delay.tick_params(axis="both", direction="out", length=4, width=0.8, labelsize=14)
    ax_completion.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=14)
    ax_delay.grid(True, axis="y", linestyle="--", color="#bbbbbb", linewidth=0.6, zorder=0)

    leg = ax_delay.legend(
        [delay_line, completion_line],
        ["Average Delay", "Completion Rate"],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.65),
        fontsize=12,
        framealpha=1.0,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.8,
    )
    leg.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    print(f"Done. Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
