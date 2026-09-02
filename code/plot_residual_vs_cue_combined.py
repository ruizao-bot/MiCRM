#!/usr/bin/env python3
"""Combine the CUE boxplots with the scatter as aligned marginal summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_residual_vs_cue_publication import (
    DEFAULT_INPUT,
    GROUPS,
    load_data,
    set_publication_style,
    style_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "Residual_vs_CUE_combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_figure(plot_df):
    set_publication_style()
    fig = plt.figure(figsize=(7.09, 4.05))  # 180 x 103 mm
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 3.35], hspace=0.035)
    ax_marginal = fig.add_subplot(grid[0])
    ax_scatter = fig.add_subplot(grid[1], sharex=ax_marginal)

    all_cue = plot_df["Community_CUE"].to_numpy(dtype=float)
    cue_min = float(np.min(all_cue))
    cue_max = float(np.max(all_cue))
    cue_pad = max(0.0015, 0.05 * (cue_max - cue_min))
    cue_limits = (cue_min - cue_pad, cue_max + cue_pad)

    # Horizontal boxplots summarize the x distribution and share the scatter x-axis.
    group_values = [
        plot_df.loc[plot_df["Community"] == label, "Community_CUE"].to_numpy()
        for label, *_ in GROUPS
    ]
    positions = [3, 2, 1]
    box = ax_marginal.boxplot(
        group_values,
        positions=positions,
        widths=0.52,
        orientation="horizontal",
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.15},
        whiskerprops={"color": "#333333", "linewidth": 0.85},
        capprops={"color": "#333333", "linewidth": 0.85},
        boxprops={"edgecolor": "#333333", "linewidth": 0.85},
        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "markerfacecolor": "white",
            "markeredgecolor": "#555555",
            "markeredgewidth": 0.6,
        },
    )
    for patch, (_, _, _, _, color) in zip(box["boxes"], GROUPS):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)

    labels = [label for label, *_ in GROUPS]
    ax_marginal.set_yticks(positions, labels)
    for tick_label, (_, _, _, _, color) in zip(ax_marginal.get_yticklabels(), GROUPS):
        tick_label.set_color(color)
    ax_marginal.set_ylim(0.45, 3.55)
    ax_marginal.set_xlim(cue_limits)
    ax_marginal.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_marginal.tick_params(axis="y", length=0, pad=7)
    ax_marginal.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.60)
    ax_marginal.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax_marginal.spines[side].set_visible(False)
    ax_marginal.spines["bottom"].set_color("#777777")
    ax_marginal.spines["bottom"].set_linewidth(0.65)

    # Main scatter: observations only, with no fitted or connecting lines.
    for label, _, _, marker, color in GROUPS:
        group = plot_df[plot_df["Community"] == label]
        ax_scatter.scatter(
            group["Community_CUE"],
            group["Residual_resource"],
            s=34,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.62,
            zorder=3,
        )

    ax_scatter.set_xlim(cue_limits)
    ax_scatter.set_xlabel("Community-level CUE")
    ax_scatter.set_ylabel("Total residual resource concentration")
    style_axis(ax_scatter)

    fig.subplots_adjust(left=0.15, right=0.985, top=0.975, bottom=0.145)
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
