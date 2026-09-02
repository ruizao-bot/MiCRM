"""Relate equilibrium heterospecific depletion pressure to community CUE.

The script reconstructs each simulated uptake matrix from its saved random seed,
combines it with equilibrium abundances in data/coal.csv, and writes a one-row-
per-community analysis table plus a publication-ready competition-CUE figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

import param


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "coal.csv"
OUTPUT_CSV = ROOT / "results" / "depletion_community_competition_cue.csv"
OUTPUT_PNG = ROOT / "results" / "depletion_community_competition_vs_cue.png"
OUTPUT_PDF = ROOT / "results" / "depletion_community_competition_vs_cue.pdf"

N_POOL = 1000
M_POOL = 100
N1 = N2 = 100
M1 = M2 = 50
N_MODULES = 1
S_RATIO = 1.0
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
LEAKAGE_RATE = 0.2
SURVIVAL_THRESHOLD = 1e-5


def reconstruct_community_parameters(seed):
    """Recreate uptake and selected leakage tensors for all three communities."""
    rng = np.random.default_rng(int(seed))
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)

    # generate_l_tensor draws N_POOL independent M_POOL x M_POOL uniforms.
    # Advancing PCG64 reproduces the subsequent index draws without allocating
    # the approximately 80 MB leakage tensor, which is not needed here.
    rng.bit_generator.advance(N_POOL * M_POOL * M_POOL)

    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)
    resource_indices2 = param.choose_resources_for_second_community(
        M_POOL, M1, M2, resource_indices1, rng
    )
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)

    selected_species = set(np.concatenate([species_indices1, species_indices2]).tolist())
    selected_leakage = {}
    leakage_rng = np.random.default_rng(int(seed))
    param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, leakage_rng)
    for species_idx in range(N_POOL):
        if species_idx in selected_species:
            selected_leakage[species_idx] = param.modular_leakage(
                M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, leakage_rng
            )
        else:
            leakage_rng.bit_generator.advance(M_POOL * M_POOL)

    l1_pool = np.stack([selected_leakage[i] for i in species_indices1])
    l2_pool = np.stack([selected_leakage[i] for i in species_indices2])
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    u3 = u_pool[np.ix_(np.concatenate([species_indices1, species_indices2]), resource_indices1)]
    l1 = l1_pool[:, resource_indices1][:, :, resource_indices1]
    l2 = l2_pool[:, resource_indices2][:, :, resource_indices2]
    l3_pool = np.concatenate([l1_pool, l2_pool], axis=0)
    l3 = l3_pool[:, resource_indices1][:, :, resource_indices1]
    return {1: (u1, l1), 2: (u2, l2), 3: (u3, l3)}


def equilibrium_resources(abundance, uptake, leakage, rho):
    """Solve the linear steady-state resource balance at saved C*."""
    consumer_uptake = abundance[:, None] * uptake
    leakage_flux = np.einsum("ia,iab->ab", consumer_uptake, leakage, optimize=True)
    loss = OMEGA_VALUE + np.sum(consumer_uptake, axis=0)
    system = np.diag(loss) - leakage_flux.T
    return np.linalg.solve(system, np.full(uptake.shape[1], rho, dtype=float))


def first_finite(values):
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.iloc[0]) if len(finite) else np.nan


def compute_table(df):
    records = []

    for seed, seed_df in df.groupby("Seed", sort=True):
        parameters = reconstruct_community_parameters(seed)
        for community in (1, 2, 3):
            group = seed_df[seed_df["Community"].astype(int) == community].sort_values("Species_ID")
            u, leakage = parameters[community]
            abundance = np.maximum(group["Abundance"].to_numpy(dtype=float), 0.0)
            if abundance.shape[0] != u.shape[0]:
                raise ValueError(
                    f"seed {seed}, community {community}: {abundance.shape[0]} rows for {u.shape[0]} species"
                )

            survivors = np.flatnonzero(abundance > SURVIVAL_THRESHOLD)
            rho = RHO_VALUE if community in (1, 2) else 2.0 * RHO_VALUE
            resources = equilibrium_resources(abundance, u, leakage, rho)
            depletion_matrix = param.micrm_depletion_competition_matrix(
                abundance,
                resources,
                u,
                leakage,
                omega=np.full(u.shape[1], OMEGA_VALUE),
                lambda_vec=np.full(u.shape[0], LEAKAGE_RATE),
            )
            competition = param.community_heterospecific_competition_pressure(
                depletion_matrix,
                abundance,
                survivor_idx=survivors,
            )
            records.append(
                {
                    "Seed": int(seed),
                    "Community": community,
                    "Heterospecific_Competition_Pressure": competition,
                    "Community_CUE": first_finite(group["Community_CUE_surv"]),
                    "N_Survivors": int(survivors.size),
                    "Total_Abundance": float(np.sum(abundance[survivors])),
                }
            )
    return pd.DataFrame.from_records(records)


def add_statistics(table):
    table = table.copy()
    stats_rows = []
    for community, group in table.groupby("Community", sort=True):
        valid = group[["Heterospecific_Competition_Pressure", "Community_CUE"]].dropna()
        x = valid["Heterospecific_Competition_Pressure"].to_numpy()
        y = valid["Community_CUE"].to_numpy()
        fit = linregress(x, y)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        stats_rows.append(
            {
                "Community": int(community),
                "n": len(valid),
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.rvalue**2,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )
    return pd.DataFrame(stats_rows)


def plot_relationship(table, stats):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    labels = {1: "Community 1", 2: "Community 2", 3: "Coalesced community"}
    point_color = "#3B82B4"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), sharey=True)

    for ax, community in zip(axes, (1, 2, 3)):
        group = table[table["Community"] == community].dropna(
            subset=["Heterospecific_Competition_Pressure", "Community_CUE"]
        )
        x = group["Heterospecific_Competition_Pressure"].to_numpy()
        y = group["Community_CUE"].to_numpy()
        stat = stats.loc[stats["Community"] == community].iloc[0]

        ax.scatter(x, y, s=34, alpha=0.72, color=point_color, edgecolor="white", linewidth=0.45)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, stat["intercept"] + stat["slope"] * x_line, color="#2F2F2F", lw=1.5)
        ax.set_title(labels[community])
        ax.set_xlabel(r"Heterospecific competition pressure, $\mathcal{C}_{intra}$")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Community-level CUE")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(INPUT_CSV)
    required = {"Seed", "Community", "Species_ID", "Abundance", "Community_CUE_surv"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing)}")

    table = compute_table(df)
    stats = add_statistics(table)
    table = table.merge(stats, on="Community", how="left")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    plot_relationship(table, stats)
    print(stats.to_string(index=False))
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
