from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "dp_epsilon_delay.png"


def main():
    epsilon_labels = ["0.5", "2.0", "4.0", "8.0", "12.0", "Without DP"]
    x = np.arange(len(epsilon_labels))
    average_delay = [3.42, 2.98, 2.74, 2.52, 2.30, 2.22]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.plot(
        x,
        average_delay,
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="FMAPPO-DP",
        color="#4d4d9f",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_labels)
    ax.set_xlabel(r"Privacy budget $\epsilon$", fontsize=14)
    ax.set_ylabel("Average Delay (s)", fontsize=14)

    ax.tick_params(axis="both", direction="out", length=4, width=0.8, labelsize=14)
    ax.grid(True, linestyle="--", color="#bbbbbb", linewidth=0.6, zorder=0)
    leg = ax.legend(
        loc="upper right",
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
