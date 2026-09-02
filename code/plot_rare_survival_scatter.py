"""Scatter-plot variant of rare-species survival by species-level CUE.

The data filtering, CUE bins, survival threshold, and Wilson confidence
intervals are identical to ``plot_rare_survival_redesigned.py``.  Only the
visual encoding changes: group means are shown as unconnected, horizontally
offset scatter points.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, PercentFormatter

from plot_rare_survival_redesigned import STYLES, summarize_survival


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "pdf" / "rare_survival_scatter.pdf"

X_OFFSETS = {0.01: -0.07, 0.10: 0.07}


def main() -> None:
    summary = summarize_survival()
    shown_categories = summary.attrs["shown_categories"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for dilution_rate, style in STYLES.items():
        subset = summary[summary["DilutionRate"] == dilution_rate].sort_values(
            "x_position"
        )
        x = (
            subset["x_position"].to_numpy(dtype=float)
            + X_OFFSETS[dilution_rate]
        )
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

        # Draw confidence intervals separately so the observations themselves
        # remain a true scatter layer with no implied connection between bins.
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.1,
            capsize=3.2,
            capthick=1.1,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=58 if dilution_rate == 0.01 else 45,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.6,
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
    )
    legend.get_frame().set_linewidth(0.7)

    fig.text(
        0.99,
        0.018,
        "CUE < 0.42 is pooled; other labels show upper bounds of 0.01-wide bins. Points are offset for clarity; error bars are 95% Wilson CIs; n >= 50.",
        ha="right",
        va="bottom",
        fontsize=9.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.22)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
