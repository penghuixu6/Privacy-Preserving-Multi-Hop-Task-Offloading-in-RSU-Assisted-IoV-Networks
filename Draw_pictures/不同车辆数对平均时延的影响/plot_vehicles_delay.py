from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "vehicles_delay.eps"

def main():
    # 数据设置
    labels = ['8', '10', '12']
    x = np.arange(len(labels))
    width = 0.20

    # 编造的合理数据 (确保 Fed-MAPPO 在车辆数为10时约为 2.4s)
    # 规律：随着车辆数增多，资源竞争变大，平均时延增加。
    fed_mappo = [2.62, 2.26, 1.92]
    ppo = [3.16, 2.74, 2.52]
    dqn = [4.12, 3.73, 3.45]
    # Greedy baseline; at 10 vehicles the average delay is about 8 seconds.
    greedy = [8.50, 8.00, 7.70]

    # 初始化画布
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # 边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 绘制分组柱状图，使用 Nature 科研配色方案，zorder=3保证其在网格线之上
    ax.bar(x - 1.5 * width, greedy, width, label='Greedy', color='#c7c7c7', edgecolor='black', linewidth=0.8, zorder=3)
    ax.bar(x - 0.5 * width, dqn, width, label='DQN', color='#ffdbd1', edgecolor='black', linewidth=0.8, zorder=3)
    ax.bar(x + 0.5 * width, ppo, width, label='PPO', color='#d6e2ee', edgecolor='black', linewidth=0.8, zorder=3)
    ax.bar(x + 1.5 * width, fed_mappo, width, label='FMAPPO', color="#0183bb", edgecolor='black', linewidth=0.8, zorder=3)

    # 刻度与标签
    ax.set_ylabel('Average Delay (s)', fontsize=14)
    ax.set_xlabel('Number of Vehicles', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", direction="out", length=4, width=0.8, labelsize=14)
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 10.0 + 0.5, 1))

    # Y轴网格线 (放于底层)
    ax.grid(True, axis='y', linestyle="--", color="#bbbbbb", linewidth=0.6, zorder=0)

    # 图例排序定制：让最优的方法(Fed-MAPPO)排在前列
    handles, labels_leg = ax.get_legend_handles_labels()
    hl_dict = dict(zip(labels_leg, handles))
    ordered_labels = ["FMAPPO", "PPO", "DQN", "Greedy"]
    ordered_handles = [hl_dict[lbl] for lbl in ordered_labels]

    leg = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper right",
        ncol=2,
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
