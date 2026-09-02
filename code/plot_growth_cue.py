import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


CSV_PATH = os.path.join(os.path.dirname(__file__), "../results/rmax_cue.csv")
OUT_FIG = os.path.join(os.path.dirname(__file__), "../figure/rmax_cue.pdf")


def plot_from_csv(csv_path, out_path):
    df = pd.read_csv(csv_path)
    df_plot = df.dropna(subset=["growth_CUE", "intrinsic_CUE"]).copy()

    if df_plot.empty:
        raise ValueError("No valid rows to plot after dropping NaN values.")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:regular",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6, 3))

    # Swapped axes: growth_CUE on x, intrinsic_CUE on y.
    ax.scatter(
        df_plot["growth_CUE"],
        df_plot["intrinsic_CUE"],
        s=30,
        alpha=0.5,
        facecolors="#9FB7CC",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)
    ax.tick_params(axis="both", width=0.4, colors="black", pad=4)
    ax.grid(False)

    x_lo = np.min(df_plot["growth_CUE"]) - 0.03
    x_hi = np.max(df_plot["growth_CUE"]) + 0.03
    y_lo = np.min(df_plot["intrinsic_CUE"]) - 0.03
    y_hi = np.max(df_plot["intrinsic_CUE"]) + 0.03
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Set x-axis major ticks at 0.01 intervals
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.01))

    ax.set_xlabel("Measurable CUE", labelpad=6)
    ax.set_ylabel("Theoretical CUE", labelpad=8)
    ax.set_title("", pad=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    plot_from_csv(CSV_PATH, OUT_FIG)
