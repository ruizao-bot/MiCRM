#!/usr/bin/env python3
"""Publication-style CUE and residual-resource figure without trend lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "coal_resource_100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "Residual_vs_CUE_publication"

GROUPS = [
    ("Parent 1", "CUE1", "Depletion1", "o", "#D95F59"),
    ("Parent 2", "CUE2", "Depletion2", "^", "#3BA272"),
    ("Coalesced", "CUE3", "Depletion3", "s", "#4C9FD3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension; both PDF and PNG are written.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Seed"}
    for _, cue_column, depletion_column, _, _ in GROUPS:
        required.update([cue_column, depletion_column])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    frames = []
    for order, (label, cue_column, depletion_column, marker, color) in enumerate(GROUPS):
        group = df[["Seed", cue_column, depletion_column]].copy()
        group.columns = ["Seed", "Community_CUE", "Residual_resource"]
        group["Community"] = label
        group["Order"] = order
        group["Marker"] = marker
        group["Color"] = color
        frames.append(group)

    plot_df = pd.concat(frames, ignore_index=True)
    for column in ["Community_CUE", "Residual_resource"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["Community_CUE", "Residual_resource"])

    counts = plot_df.groupby("Community")["Seed"].nunique()
    missing_groups = [label for label, *_ in GROUPS if label not in counts]
    if missing_groups:
        raise ValueError(f"No valid observations for groups: {missing_groups}")
    return plot_df


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(direction="out", pad=3)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def make_figure(plot_df: pd.DataFrame) -> plt.Figure:
    set_publication_style()
    fig, (ax_scatter, ax_box) = plt.subplots(
        1,
        2,
        figsize=(7.09, 3.65),  # 180 x 93 mm, suitable for an A4 full-width figure
        gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.30},
    )

    legend_handles = []
    cue_values = []
    for label, _, _, marker, color in GROUPS:
        group = plot_df[plot_df["Community"] == label]
        cue_values.extend(group["Community_CUE"].tolist())
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
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=marker,
                markersize=6.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=label,
            )
        )

    cue_min = float(np.min(cue_values))
    cue_max = float(np.max(cue_values))
    cue_pad = max(0.0015, 0.05 * (cue_max - cue_min))
    cue_limits = (cue_min - cue_pad, cue_max + cue_pad)

    ax_scatter.set_xlim(cue_limits)
    ax_scatter.set_xlabel("Community-level CUE")
    ax_scatter.set_ylabel("Total residual resource concentration")
    ax_scatter.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        borderaxespad=0.25,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    style_axis(ax_scatter)

    group_values = [
        plot_df.loc[plot_df["Community"] == label, "Community_CUE"].to_numpy()
        for label, *_ in GROUPS
    ]
    box = ax_box.boxplot(
        group_values,
        positions=np.arange(1, len(GROUPS) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.25},
        whiskerprops={"color": "#333333", "linewidth": 0.9},
        capprops={"color": "#333333", "linewidth": 0.9},
        boxprops={"edgecolor": "#333333", "linewidth": 0.9},
    )
    for patch, (_, _, _, _, color) in zip(box["boxes"], GROUPS):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)

    # Overlay every observation; deterministic jitter keeps the figure reproducible.
    rng = np.random.default_rng(20260811)
    for position, (values, (_, _, _, marker, color)) in enumerate(
        zip(group_values, GROUPS), start=1
    ):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax_box.scatter(
            position + jitter,
            values,
            s=13,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.48,
            zorder=3,
        )

    ax_box.set_ylim(cue_limits)
    ax_box.set_xticks([1, 2, 3], ["Parent\n1", "Parent\n2", "Coalesced"])
    ax_box.set_ylabel("Community-level CUE")
    ax_box.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    style_axis(ax_box)

    ax_scatter.text(
        -0.13,
        1.02,
        "a",
        transform=ax_scatter.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax_box.text(
        -0.20,
        1.02,
        "b",
        transform=ax_box.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.18)
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
