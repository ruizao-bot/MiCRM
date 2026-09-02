#!/usr/bin/env python3
"""Scatter plot with a compact inset summarizing community CUE distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from plot_residual_vs_cue_publication import (
    DEFAULT_INPUT,
    GROUPS,
    load_data,
    set_publication_style,
    style_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "Residual_vs_CUE_inset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_figure(plot_df):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.09, 3.65))  # approximately 180 x 93 mm

    all_cue = plot_df["Community_CUE"].to_numpy(dtype=float)
    cue_min = float(np.min(all_cue))
    cue_max = float(np.max(all_cue))
    cue_pad = max(0.0015, 0.05 * (cue_max - cue_min))
    cue_limits = (cue_min - cue_pad, cue_max + cue_pad)

    legend_handles = []
    for label, _, _, marker, color in GROUPS:
        group = plot_df[plot_df["Community"] == label]
        ax.scatter(
            group["Community_CUE"],
            group["Residual_resource"],
            s=25,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.54,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=marker,
                markersize=5.8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=label,
            )
        )

    ax.set_xlim(cue_limits)
    ax.set_xlabel("Community CUE")
    ax.set_ylabel("Total residual resources")
    style_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    # The upper-right region contains little data because CUE and residual
    # resources are negatively associated, making it suitable for the inset.
    inset = ax.inset_axes([0.675, 0.56, 0.305, 0.365], zorder=5)
    inset.set_facecolor("white")
    group_values = [
        plot_df.loc[plot_df["Community"] == label, "Community_CUE"].to_numpy()
        for label, *_ in GROUPS
    ]
    boxes = inset.boxplot(
        group_values,
        positions=[1, 2, 3],
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.05},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
        boxprops={"edgecolor": "#333333", "linewidth": 0.7},
        flierprops={
            "marker": "o",
            "markersize": 2.1,
            "markerfacecolor": "white",
            "markeredgecolor": "#555555",
            "markeredgewidth": 0.5,
        },
    )
    for patch, (_, _, _, _, color) in zip(boxes["boxes"], GROUPS):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)

    inset.set_ylim(cue_limits)
    inset.set_xticks([1, 2, 3], ["Parent\n1", "Parent\n2", "Coalesced"])
    inset.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.6, pad=2)
    inset.tick_params(axis="y", labelsize=6.8, length=2.2, width=0.6, pad=2)
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    inset.set_title("CUE distribution", fontsize=8, pad=2.5)
    inset.grid(axis="y", color="#D9D9D9", linewidth=0.35, alpha=0.6)
    inset.set_axisbelow(True)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.6)
    inset.spines["bottom"].set_linewidth(0.6)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.17)
    return fig


def main() -> None:
    args = parse_args()
    plot_df = load_data(args.input)
    fig = make_figure(plot_df)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_stem.with_suffix(".pdf")
    png_path = args.output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    counts = plot_df.groupby("Community")["Seed"].nunique()
    print(f"Loaded {len(plot_df)} observations from {args.input}")
    print("Replicates per group:")
    print(counts.reindex([label for label, *_ in GROUPS]).to_string())
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
