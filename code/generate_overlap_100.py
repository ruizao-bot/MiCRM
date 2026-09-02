#!/usr/bin/env python3
"""Generate stratified simulation data for 100% resource overlap only.

This imports the established simulation functions from ``resource_overlap.py``
and writes a separate CSV, so the existing 25%/50%/75% results are preserved.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from resource_overlap import (  # noqa: E402
    BASE_SEED,
    _proxy_cue_diff,
    simulate_overlap,
)


OVERLAP = 1.0
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "coal_resource_100.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--per-bin", type=int, default=10)
    parser.add_argument(
        "--processes",
        type=int,
        default=min(cpu_count(), 2),
        help="Worker processes (default: 2; each simulation is memory intensive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidates < args.bins * args.per_bin:
        raise ValueError("--candidates must be at least --bins * --per-bin")

    seed_generator = np.random.default_rng(BASE_SEED)
    candidate_seeds = seed_generator.integers(
        0, 2**32 - 1, size=args.candidates, dtype=np.uint32
    ).tolist()
    proxy_args = [(seed, OVERLAP) for seed in candidate_seeds]

    print(f"Screening {len(proxy_args)} candidates for 100% overlap ...")
    with Pool(args.processes) as pool:
        proxy_diffs = pool.map(_proxy_cue_diff, proxy_args)

    proxy_df = pd.DataFrame(
        {
            "Seed": candidate_seeds,
            "Overlap": OVERLAP,
            "proxy_diff": proxy_diffs,
        }
    )
    proxy_df["bin_idx"] = pd.qcut(
        proxy_df["proxy_diff"],
        q=args.bins,
        labels=False,
        duplicates="drop",
    ).astype(int)

    rng = np.random.default_rng(BASE_SEED + 1)
    selected = []
    for bin_idx in sorted(proxy_df["bin_idx"].unique()):
        cell = proxy_df[proxy_df["bin_idx"] == bin_idx]
        if len(cell) < args.per_bin:
            raise ValueError(
                f"Only {len(cell)} candidates in bin {bin_idx}; "
                f"need {args.per_bin}. Increase --candidates."
            )
        selected.append(
            cell.sample(
                n=args.per_bin,
                replace=False,
                random_state=int(rng.integers(2**31)),
            )
        )
    selected_df = pd.concat(selected, ignore_index=True)
    simulation_args = list(zip(selected_df["Seed"], selected_df["Overlap"]))

    print(
        f"Running {len(simulation_args)} full simulations for 100% overlap "
        f"on {args.processes} processes ..."
    )
    with Pool(args.processes) as pool:
        results = pool.map(simulate_overlap, simulation_args)

    result_df = pd.DataFrame(results)
    result_df["CUE_diff"] = result_df["CUE1"] - result_df["CUE2"]
    result_df = result_df.merge(
        selected_df[["Seed", "Overlap", "bin_idx"]],
        on=["Seed", "Overlap"],
        how="left",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Saved {len(result_df)} simulations to {args.output}")
    print(result_df.groupby("bin_idx").size().to_string())


if __name__ == "__main__":
    main()
