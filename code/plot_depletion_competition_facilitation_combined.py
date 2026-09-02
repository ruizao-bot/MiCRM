"""Combine facilitation-CUE and depletion-competition-CUE in one figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
SPECIES_DATA = ROOT / "data" / "coal.csv"
COMPETITION_DATA = ROOT / "results" / "depletion_community_competition_cue.csv"
OUTPUT_PNG = ROOT / "results" / "depletion_competition_facilitation_combined.png"
OUTPUT_PDF = ROOT / "results" / "depletion_competition_facilitation_combined.pdf"
SURVIVAL_THRESHOLD = 1e-5

COMMUNITIES = (1, 2, 3)
COLORS = {
    1: "#D8A39A",
    2: "#A8C3A6",
    3: "#9FB7CC",
}
TITLES = {
    1: "Community 1",
    2: "Community 2",
    3: "Coalesced Community",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def first_finite(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def load_data():
    species = pd.read_csv(SPECIES_DATA)
    species["Community"] = species["Community"].astype(int)
    survivors = species[species["Abundance"] > SURVIVAL_THRESHOLD].copy()
    facilitation = (
        survivors.groupby(["Seed", "Community"], as_index=False)
        .agg(
            Community_CUE=("Community_CUE_surv", first_finite),
            Facilitation=("Facilitation", "mean"),
        )
    )

    competition = pd.read_csv(COMPETITION_DATA)[
        ["Seed", "Community", "Heterospecific_Competition_Pressure", "Community_CUE"]
    ].copy()
    competition["Community"] = competition["Community"].astype(int)
    return facilitation, competition


def style_axis(ax, column_index):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis="both", width=0.5, colors="black", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)
    if column_index > 0:
        ax.tick_params(axis="y", left=False, labelleft=False)


def scatter_row(axes, data, x_column, x_label, scientific_x=False):
    for column_index, (ax, community) in enumerate(zip(axes, COMMUNITIES)):
        subset = data[data["Community"] == community]
        ax.scatter(
            subset[x_column],
            subset["Community_CUE"],
            s=42,
            alpha=0.68,
            facecolors=COLORS[community],
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_title(TITLES[community], pad=7)
        ax.set_xlabel(x_label)
        ax.set_ylim(0.527, 0.564)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        if scientific_x:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.get_offset_text().set_fontsize(12)

        if column_index == 0:
            ax.set_ylabel("Community-level CUE")
        style_axis(ax, column_index)


def main():
    facilitation, competition = load_data()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8.0),
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
        competition,
        x_column="Heterospecific_Competition_Pressure",
        x_label="Heterospecific competition pressure",
    )

    axes[0, 0].text(-0.22, 1.10, "(a)", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[1, 0].text(-0.22, 1.10, "(b)", transform=axes[1, 0].transAxes, fontweight="bold")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
