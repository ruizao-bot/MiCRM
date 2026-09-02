"""
plot_heatmaps.py
从 scan_balances_results.csv 读取数据，画三张热图：
  - CUE         (合并后群落)
  - Stability   (leading eigenvalue, 越负越稳定)
  - Feasibility (coexistence probability)

运行：python plot_heatmaps.py
输出：heatmaps.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CSV_FILE = "scan_balances_results.csv"
OUT_FILE = "heatmap_cue.png"
THR_EV   = 0.01   # ev < thr 才算稳定，与 main.py 一致

# =============================================================================
# 读取数据
# =============================================================================
df = pd.read_csv(CSV_FILE)
df3 = df[df["community"] == "community3"].copy()

# 每个 (b1, b2) 对取均值
pivot_cue  = df3.groupby(["competition_cooperation_balance1",
                           "competition_cooperation_balance2"])["cue"].mean().reset_index()
def to_matrix(pivot, value_col):
    """长表转方阵，行=b1，列=b2"""
    return pivot.pivot(
        index="competition_cooperation_balance1",
        columns="competition_cooperation_balance2",
        values=value_col
    )

mat_cue  = to_matrix(pivot_cue,  "cue")

b_vals = mat_cue.index.values   # 共用同一套 b 轴

# =============================================================================
# 画图
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

Z = mat_cue.values

im = ax.imshow(
    Z,
    origin="lower",
    aspect="auto",
    cmap="YlGn",
    extent=[b_vals.min(), b_vals.max(),
            b_vals.min(), b_vals.max()],
)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_title("Community CUE", fontsize=13)
ax.set_xlabel("Balance of community 1 ($b_1$)", fontsize=11)
ax.set_ylabel("Balance of community 2 ($b_2$)", fontsize=11)

# 对角线标注（结构相似区域）
ax.plot([b_vals.min(), b_vals.max()],
        [b_vals.min(), b_vals.max()],
        'w--', linewidth=1, alpha=0.6, label='b1=b2')

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"图像已保存到 {OUT_FILE}")
plt.show()
