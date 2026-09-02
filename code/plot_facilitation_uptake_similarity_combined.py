"""Combine community uptake similarity and facilitation plots vertically.

The script reads ``data/coal_hpc.csv`` and writes both PDF and PNG versions to
the project's ``figure`` directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "coal_hpc.csv"
FIGURE_DIR = PROJECT_DIR / "figure"
SURVIVAL_THRESHOLD = 1e-5

COMMUNITIES = ["1", "2", "3"]
COLORS = {
    "1": "#D8A39A",
    "2": "#A8C3A6",
    "3": "#9FB7CC",
}
TITLES = {
    "1": "Community 1",
    "2": "Community 2",
    "3": "Coalesced Community",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def first_unique(series: pd.Series) -> float:
    """Return the first non-missing unique value in a group."""
    values = series.dropna().unique()
    return values[0] if len(values) else np.nan


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis="both", width=0.4, colors="black", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)


def load_community_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_FILE)
    df["Community"] = df["Community"].astype(str)
    df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")
    df = df[df["Abundance"] > SURVIVAL_THRESHOLD].copy()

    similarity = (
        df.groupby(
            [
                "Seed",
                "Community",
                "Competition",
                "Community_CUE_surv",
                "Facilitation",
            ],
            as_index=False,
        )
        .size()
        .drop(columns="size")
    )

    facilitation = (
        df.groupby(["Seed", "Community"], as_index=False)
        .agg(
            Community_CUE_surv=("Community_CUE_surv", first_unique),
            Facilitation=("Facilitation", "mean"),
        )
    )
    return similarity, facilitation


def scatter_row(
    axes: np.ndarray,
    data: pd.DataFrame,
    x_column: str,
    x_label: str,
    scientific_x: bool = False,
) -> None:
    for index, (ax, community) in enumerate(zip(axes, COMMUNITIES)):
        subset = data[data["Community"] == community]
        ax.scatter(
            subset[x_column],
            subset["Community_CUE_surv"],
            s=44,
            alpha=0.6,
            facecolors=COLORS[community],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        ax.set_title(TITLES[community], pad=8)
        ax.set_xlabel(x_label)
        ax.set_ylim(0.530, 0.565)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        if scientific_x:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.get_offset_text().set_fontsize(12)

        if index == 0:
            ax.set_ylabel("Community-level CUE")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        style_axis(ax)


def main() -> None:
    similarity, facilitation = load_community_data()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8.2),
        sharey=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.08},
    )

    scatter_row(
        axes[0],
        facilitation,
        x_column="Facilitation",
        x_label="Facilitation",
        scientific_x=True,
    )
    scatter_row(
        axes[1],
        similarity,
        x_column="Competition",
        x_label="Community uptake similarity",
    )

    axes[0, 0].text(
        -0.22, 1.10, "(a)", transform=axes[0, 0].transAxes, fontweight="bold"
    )
    axes[1, 0].text(
        -0.22, 1.10, "(b)", transform=axes[1, 0].transAxes, fontweight="bold"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = FIGURE_DIR / "facilitation_uptake_similarity_combined"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
