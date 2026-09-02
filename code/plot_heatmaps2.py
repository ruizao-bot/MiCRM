"""
plot_heatmaps2.py
从 scan_balances_results.csv 画一张热图：
    1. CUE observed       合并后群落完整CUE

运行：python plot_heatmaps2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "scan_balances_results.csv"
OUT_FILE = "heatmaps2.png"

# =============================================================================
# 读取并整理数据
# =============================================================================
df  = pd.read_csv(CSV_FILE)
df3 = df[df["community"] == "community3"].copy()

def make_pivot(col):
    return (df3.groupby(["balance1", "balance2"])[col]
               .mean()
               .reset_index()
               .pivot(index="balance1",
                      columns="balance2",
                      values=col))

mat_cue_obs  = make_pivot("cue_obs")
# mat_cue_base = make_pivot("cue_base")
# mat_delta    = make_pivot("delta_cue")
# mat_ev       = make_pivot("ev")
# mat_feas     = make_pivot("feasibility")

b_vals = mat_cue_obs.index.values
ext    = [b_vals.min(), b_vals.max(), b_vals.min(), b_vals.max()]

# =============================================================================
# 画图
# =============================================================================
panels = [
    (mat_cue_obs,  "CUE observed",          "YlGn"),
    # (mat_delta,    "ΔCUE\n(cross-feeding contribution)", "RdBu"),
]

fig, axes = plt.subplots(1, 1, figsize=(4.8, 4.5))
axes = [axes]

for ax, (mat, title, cmap) in zip(axes, panels):
    Z = mat.values.astype(float)

    # 仅保留 CUE observed 热图，不需要 delta CUE 的颜色中心设置
    vmin, vmax = None, None

    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=ext, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Balance of community 1 ($b_1$)", fontsize=9)
    ax.set_ylabel("Balance of community 2 ($b_2$)", fontsize=9)
    ax.plot([b_vals.min(), b_vals.max()],
            [b_vals.min(), b_vals.max()],
            'w--', lw=1, alpha=0.6)

plt.suptitle(" ",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"图像已保存到 {OUT_FILE}")
plt.show()
