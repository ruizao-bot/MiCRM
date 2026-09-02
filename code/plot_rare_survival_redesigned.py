"""Plot rare-invader survival probability against species-level CUE.

Only species from the diluted invader community (Comm2) in the coalesced
community are included. The figure uses fixed-width CUE bins and Wilson 95%
confidence intervals for binomial survival proportions.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "rare.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "pdf" / "rare_survival_redesigned.pdf"

SURVIVAL_THRESHOLD = 1e-5
CUE_BIN_WIDTH = 0.01
CUE_CUTOFF = 0.42
CUE_EDGES = np.arange(CUE_CUTOFF, 0.481, CUE_BIN_WIDTH)
MIN_BIN_COUNT = 50
Z_95 = 1.959963984540054

STYLES = {
    0.01: {
        "label": "Rarity level = 0.01",
        "color": "#0072B2",
        "marker": "o",
        "markersize": 7.2,
        "linestyle": "-",
        "offset": 0.0,
    },
    0.10: {
        "label": "Rarity level = 0.10",
        "color": "#D55E00",
        "marker": "s",
        "markersize": 5.4,
        "linestyle": "--",
        "offset": 0.0,
    },
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
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def wilson_interval(successes: pd.Series, totals: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return lower and upper Wilson score limits for binomial proportions."""
    probability = successes / totals
    denominator = 1.0 + Z_95**2 / totals
    center = (probability + Z_95**2 / (2.0 * totals)) / denominator
    half_width = (
        Z_95
        * np.sqrt(
            probability * (1.0 - probability) / totals
            + Z_95**2 / (4.0 * totals**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def summarize_survival() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)

    # Community 2 is diluted before coalescence, so these are the rare invaders.
    data = data[(data["Community"] == 3) & (data["Origin"] == "Comm2")].copy()
    data["survived"] = data["Abundance"] > SURVIVAL_THRESHOLD
    interval_labels = [
        f"{right:.2f}" for right in CUE_EDGES[1:]
    ]
    category_order = [f"<{CUE_CUTOFF:.2f}", *interval_labels]

    data["cue_group"] = f"<{CUE_CUTOFF:.2f}"
    above_cutoff = data["Species_CUE"] >= CUE_CUTOFF
    cut_bins = pd.cut(
        data.loc[above_cutoff, "Species_CUE"],
        bins=CUE_EDGES,
        right=False,
        include_lowest=True,
        labels=interval_labels,
    )
    data.loc[above_cutoff, "cue_group"] = cut_bins.astype("object")
    data["cue_group"] = pd.Categorical(
        data["cue_group"], categories=category_order, ordered=True
    )

    summary = (
        data.groupby(["DilutionRate", "cue_group"], observed=True)["survived"]
        .agg(survivors="sum", total="size")
        .reset_index()
    )
    summary["survival_probability"] = summary["survivors"] / summary["total"]
    summary["ci_low"], summary["ci_high"] = wilson_interval(
        summary["survivors"], summary["total"]
    )
    summary = summary[summary["total"] >= MIN_BIN_COUNT].copy()
    shown_categories = [
        category
        for category in category_order
        if category in set(summary["cue_group"].astype(str))
    ]
    position_map = {category: index for index, category in enumerate(shown_categories)}
    summary["x_position"] = summary["cue_group"].astype(str).map(position_map)
    summary.attrs["shown_categories"] = shown_categories
    return summary


def main() -> None:
    summary = summarize_survival()
    shown_categories = summary.attrs["shown_categories"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for dilution_rate, style in STYLES.items():
        subset = summary[summary["DilutionRate"] == dilution_rate].sort_values("x_position")
        x = subset["x_position"].to_numpy(dtype=float) + style["offset"]
        y = subset["survival_probability"].to_numpy()
        yerr = np.maximum(
            np.vstack(
                [
                    y - subset["ci_low"].to_numpy(),
                    subset["ci_high"].to_numpy() - y,
                ]
            ),
            0.0,
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
            capsize=3.2,
            capthick=1.0,
            elinewidth=1.0,
            label=style["label"],
            zorder=3,
        )

    ax.set_title("Rare-species survival after community coalescence", pad=13)
    ax.set_xlabel("Species-level CUE")
    ax.set_ylabel("Survival proportion")
    ax.set_xlim(-0.5, len(shown_categories) - 0.5)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xticks(np.arange(len(shown_categories)))
    ax.set_xticklabels(shown_categories, rotation=25, ha="right")
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=5, width=0.7)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#BDBDBD",
        framealpha=0.95,
        handlelength=2.7,
    )
    legend.get_frame().set_linewidth(0.7)

    fig.text(
        0.99,
        0.018,
        "CUE < 0.42 is pooled; other labels show upper bounds of 0.01-wide bins. Error bars are 95% Wilson CIs; n >= 50.",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.22)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
