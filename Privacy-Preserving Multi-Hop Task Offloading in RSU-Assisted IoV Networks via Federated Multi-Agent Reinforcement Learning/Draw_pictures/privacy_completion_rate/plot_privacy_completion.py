from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "privacy_pressure_test.png"

def main():
    # 数据准备
    rho_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    # 把原本百分制数据转换为小数比例以匹配 0.5-1.0 的范围
    fed_mappo = [0.982, 0.972, 0.951, 0.945, 0.907]
    ppo = [0.935, 0.892, 0.861, 0.815, 0.752]
    dqn = [0.891, 0.825, 0.756, 0.642, 0.528]

    # 初始化画布
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # 边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 绘制折线，Nature 科研配色
    ax.plot(rho_values, dqn, marker='^', markersize=8, linewidth=1.5, label='DQN', color='#80dc66', zorder=3)
    ax.plot(rho_values, ppo, marker='s', markersize=8, linewidth=1.5, label='PPO', color='#fc9871', zorder=3)
    ax.plot(rho_values, fed_mappo, marker='o', markersize=8, linewidth=2.0, label='FMAPPO', color='#4d4d9f', zorder=3)

    # 坐标轴刻度设置
    ax.set_xticks(rho_values)
    ax.set_xticklabels([f"{x:.2f}" for x in rho_values])
    ax.set_yticks(np.arange(0.5, 1.1, 0.1))
    ax.set_ylim(0.5, 1.0)

    # Inward ticks for both axes
    ax.tick_params(axis="both", direction="out", length=4, width=0.8, labelsize=14)

    # 网格线 (放于底层)
    ax.grid(True, linestyle="--", color="#bbbbbb", linewidth=0.6, zorder=0)

    # 坐标轴与标题
    ax.set_xlabel('Proportion of Privacy-sensitive Tasks ($\\rho$)', fontsize=14)
    ax.set_ylabel('Task Completion Rate', fontsize=14)
    
    # 图例排序定制：让最优的方法(Fed-MAPPO)排在前列
    handles, labels_leg = ax.get_legend_handles_labels()
    hl_dict = dict(zip(labels_leg, handles))
    ordered_labels = ["FMAPPO", "PPO", "DQN"]
    ordered_handles = [hl_dict[lbl] for lbl in ordered_labels if lbl in hl_dict]

    # 图例
    leg = ax.legend(
        ordered_handles, 
        ordered_labels, 
        loc="lower left", 
        fontsize=12, 
        framealpha=1.0, 
        edgecolor="#cccccc",
        fancybox=True
    )
    leg.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    print(f"Done. Saved figure to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()