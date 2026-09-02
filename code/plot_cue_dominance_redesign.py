#!/usr/bin/env python3
"""Plot CUE difference against signed coalescence dominance.

The input is the replicate-level output produced by ``resource_overlap.py``.
Each point is one simulation.  For each resource-overlap treatment, the curve
is an origin-constrained saturating fit

    signed dominance = amplitude * tanh(rate * CUE difference).

The fit is performed with NumPy only so the plotting script does not require
SciPy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "coal_resource.csv",
    PROJECT_ROOT / "data" / "coal_resource_100.csv",
]
DEFAULT_OVERLAPS = [0.50, 0.75, 1.00]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cue_dominance_overlap_50_75_100"

REQUIRED_COLUMNS = {
    "Overlap",
    "CUE1",
    "CUE2",
    "Similarity_3vs1",
    "Similarity_3vs2",
}

SERIES_COLORS = ["#5B917F", "#D9893B", "#8256C2"]


def fit_origin_tanh(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = amplitude * tanh(rate*x) by a stable one-dimensional search.

    For a given rate, the least-squares amplitude has a closed-form solution.
    Searching rate on a dense logarithmic grid avoids an additional optimizer
    dependency and is deterministic.  Amplitude is constrained to [0, 1]
    because signed Bray-Curtis similarity differences lie in [-1, 1].
    """
    rates = np.geomspace(0.1, 5000.0, 6000)
    best_sse = np.inf
    best_amplitude = np.nan
    best_rate = np.nan

    for rate in rates:
        basis = np.tanh(rate * x)
        denom = float(basis @ basis)
        if denom == 0:
            continue
        amplitude = float(np.clip((basis @ y) / denom, 0.0, 1.0))
        residual = y - amplitude * basis
        sse = float(residual @ residual)
        if sse < best_sse:
            best_sse = sse
            best_amplitude = amplitude
            best_rate = float(rate)

    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - best_sse / total_ss if total_ss > 0 else np.nan
    return best_amplitude, best_rate, r_squared


def load_plot_data(csv_paths: list[Path], overlaps: list[float]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {csv_path}. For the 100% data, run "
                "code/generate_overlap_100.py first."
            )
        frame = pd.read_csv(csv_path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    numeric_columns = sorted(REQUIRED_COLUMNS)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    overlap_mask = np.zeros(len(df), dtype=bool)
    for overlap in overlaps:
        overlap_mask |= np.isclose(df["Overlap"].to_numpy(), overlap)
    df = df.loc[overlap_mask].copy()
    df = df.drop_duplicates(subset=["Seed", "Overlap"], keep="last")
    df["CUE_difference"] = df["CUE1"] - df["CUE2"]
    df["signed_dominance"] = (
        df["Similarity_3vs1"] - df["Similarity_3vs2"]
    )
    df = df[np.isfinite(df["CUE_difference"])]
    df = df[np.isfinite(df["signed_dominance"])]

    if df.empty:
        raise ValueError("No finite observations remain after data cleaning.")

    present = df["Overlap"].unique()
    missing_overlaps = [
        overlap for overlap in overlaps
        if not np.any(np.isclose(present, overlap))
    ]
    if missing_overlaps:
        raise ValueError(f"No observations found for overlap values: {missing_overlaps}")
    return df


def _rounded_symmetric_limit(values: pd.Series, step: float, minimum: float) -> float:
    observed = float(np.nanmax(np.abs(values.to_numpy(dtype=float))))
    return max(minimum, math.ceil(observed * 1.08 / step) * step)


def make_figure(df: pd.DataFrame, title: str) -> tuple[plt.Figure, pd.DataFrame]:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x_limit = _rounded_symmetric_limit(df["CUE_difference"], 0.005, 0.02)
    y_limit = _rounded_symmetric_limit(df["signed_dominance"], 0.1, 0.4)
    x_curve = np.linspace(-x_limit, x_limit, 800)

    # 180 x 127 mm: suitable for a full-width figure on an A4 manuscript page.
    fig, ax = plt.subplots(figsize=(7.09, 5.0))
    fit_rows: list[dict[str, float | int]] = []

    overlaps = sorted(df["Overlap"].unique())
    if len(overlaps) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} overlap series can be plotted.")
    color_by_overlap = dict(zip(overlaps, SERIES_COLORS))

    for overlap in overlaps:
        group = df[df["Overlap"] == overlap]
        x = group["CUE_difference"].to_numpy(dtype=float)
        y = group["signed_dominance"].to_numpy(dtype=float)
        amplitude, rate, r_squared = fit_origin_tanh(x, y)
        color = color_by_overlap[overlap]

        ax.plot(
            x_curve,
            amplitude * np.tanh(rate * x_curve),
            color=color,
            linewidth=2.2,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=38,
            color=color,
            alpha=0.52,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        fit_rows.append(
            {
                "Overlap": float(overlap),
                "n": int(len(group)),
                "amplitude": amplitude,
                "rate": rate,
                "slope_at_origin": amplitude * rate,
                "R_squared": r_squared,
            }
        )

    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.15, zorder=1)
    ax.axvline(0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)

    ax.set_title(title, pad=10)
    ax.set_xlabel(r"Parental community CUE difference, $\Delta E = E_1 - E_2$")
    ax.set_ylabel("Signed coalescence dominance")

    ax.text(
        0.985,
        0.94,
        "Community 1 dominant",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#3F3F3F",
        fontsize=14,
    )
    ax.text(
        0.02,
        0.06,
        "Community 2 dominant",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#3F3F3F",
        fontsize=14,
    )

    overlap_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_overlap[value],
            marker="o",
            markersize=7,
            linewidth=2.2,
            markerfacecolor=color_by_overlap[value],
            alpha=0.9,
            label=f"{int(round(value * 100))}% overlap",
        )
        for value in overlaps
    ]
    overlap_legend = ax.legend(
        handles=overlap_handles,
        title="Shared-resource overlap",
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.96,
    )
    ax.add_artist(overlap_legend)

    element_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.2, label="Saturating fit"),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color="#777777",
            markersize=6,
            alpha=0.8,
            label="Simulation outcome",
        ),
    ]
    ax.legend(
        handles=element_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.96,
    )

    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.17)
    return fig, pd.DataFrame(fit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more input CSV files.",
    )
    parser.add_argument(
        "--overlaps",
        type=float,
        nargs="+",
        default=DEFAULT_OVERLAPS,
        help="Overlap fractions to plot (default: 0.5 0.75 1.0).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension; PNG, PDF, and fit CSV are written.",
    )
    parser.add_argument(
        "--title",
        default="CUE difference predicts coalescence dominance",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_plot_data(args.input, args.overlaps)
    fig, fit_table = make_figure(df, args.title)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    fit_path = args.output_stem.parent / f"{args.output_stem.name}_fit_parameters.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fit_table.to_csv(fit_path, index=False)
    plt.close(fig)

    print(f"Loaded {len(df)} simulation replicates from {len(args.input)} file(s)")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {fit_path}")
    print(fit_table.to_string(index=False, float_format=lambda value: f'{value:.4g}'))


if __name__ == "__main__":
    main()
