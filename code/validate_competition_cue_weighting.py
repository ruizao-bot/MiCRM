"""Validate the competition -> CUE-weight covariance -> community CUE pathway."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

import param


ROOT = Path(__file__).resolve().parents[1]
COMMUNITY_DATA = ROOT / "data" / "coal.csv"
COMPETITION_DATA = ROOT / "results" / "depletion_community_competition_cue.csv"
OUTPUT_DATA = ROOT / "results" / "competition_cue_weighting_validation.csv"
OUTPUT_STATS = ROOT / "results" / "competition_cue_weighting_validation_stats.csv"
OUTPUT_PNG = ROOT / "results" / "competition_cue_weighting_validation.png"
OUTPUT_PDF = ROOT / "results" / "competition_cue_weighting_validation.pdf"

N_POOL = 1000
M_POOL = 100
N1 = N2 = 100
M1 = M2 = 50
N_MODULES = 1
S_RATIO = 1.0
LEAKAGE_RANDOM_DRAWS = N_POOL * M_POOL * M_POOL
SURVIVAL_THRESHOLD = 1e-5


def reconstruct_uptake_matrices(seed):
    """Recreate the exact u matrices used for a saved simulation seed."""
    rng = np.random.default_rng(int(seed))
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    rng.bit_generator.advance(LEAKAGE_RANDOM_DRAWS)

    species1 = rng.choice(N_POOL, N1, replace=False)
    resources1 = rng.choice(M_POOL, M1, replace=False)
    resources2 = param.choose_resources_for_second_community(
        M_POOL, M1, M2, resources1, rng
    )
    remaining_species = np.setdiff1d(np.arange(N_POOL), species1)
    species2 = rng.choice(remaining_species, N2, replace=False)

    u1 = u_pool[np.ix_(species1, resources1)]
    u2 = u_pool[np.ix_(species2, resources2)]
    u3 = u_pool[np.ix_(np.concatenate([species1, species2]), resources1)]
    return {1: u1, 2: u2, 3: u3}


def compute_validation_table(species_data, competition_data):
    competition_lookup = competition_data.set_index(["Seed", "Community"])
    rows = []

    for seed, seed_data in species_data.groupby("Seed", sort=True):
        uptake = reconstruct_uptake_matrices(seed)
        for community in (1, 2, 3):
            group = seed_data[seed_data["Community"].astype(int) == community].sort_values("Species_ID")
            u = uptake[community]
            abundance = np.maximum(group["Abundance"].to_numpy(dtype=float), 0.0)
            epsilon = group["Species_CUE"].to_numpy(dtype=float)
            if len(group) != u.shape[0]:
                raise ValueError(f"seed {seed}, community {community}: species count mismatch")

            survivors = abundance > SURVIVAL_THRESHOLD
            ui0 = np.sum(u, axis=1)
            # Extinct species remain in the initial pool but receive zero final flux weight.
            z = np.where(survivors, abundance * ui0, 0.0)
            mean_z = float(np.mean(z))
            epsilon_pool = float(np.mean(epsilon))
            covariance = float(np.mean((z - mean_z) * (epsilon - epsilon_pool)))
            selection_differential = covariance / mean_z
            e_flux = float(np.sum(z * epsilon) / np.sum(z))
            identity_error = e_flux - epsilon_pool - selection_differential
            rank = spearmanr(z, epsilon)

            key = (int(seed), community)
            competition = float(
                competition_lookup.loc[key, "Heterospecific_Competition_Pressure"]
            )
            rows.append(
                {
                    "Seed": int(seed),
                    "Community": community,
                    "Competition": competition,
                    "E_flux": e_flux,
                    "E_saved_biomass_weighted": float(group["Community_CUE_surv"].iloc[0]),
                    "Epsilon_pool": epsilon_pool,
                    "E_flux_minus_pool": e_flux - epsilon_pool,
                    "Cov_z_epsilon": covariance,
                    "Mean_z": mean_z,
                    "Cov_over_mean_z": selection_differential,
                    "Identity_error": identity_error,
                    "Spearman_z_epsilon": rank.statistic,
                    "Spearman_z_epsilon_p": rank.pvalue,
                    "N_survivors": int(np.sum(survivors)),
                }
            )
    return pd.DataFrame(rows)


def correlation_row(community, group, response, pathway):
    valid = group[["Competition", response]].dropna()
    x = valid["Competition"].to_numpy()
    y = valid[response].to_numpy()
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    fit = linregress(x, y)
    return {
        "Community": community,
        "Pathway": pathway,
        "Response": response,
        "n": len(valid),
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.rvalue**2,
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }


def compute_stats(table):
    rows = []
    responses = {
        "Cov_over_mean_z": "competition -> covariance contribution",
        "E_flux": "competition -> E_flux",
        "Epsilon_pool": "competition -> initial-pool mean CUE",
        "Spearman_z_epsilon": "competition -> rank association of weight and CUE",
    }
    for community, group in table.groupby("Community", sort=True):
        for response, pathway in responses.items():
            rows.append(correlation_row(int(community), group, response, pathway))
    return pd.DataFrame(rows)


def plot_selection_path(table, stats):
    labels = {1: "Community 1", 2: "Community 2", 3: "Coalesced community"}
    colors = {1: "#3B82B4", 2: "#E58B3A", 3: "#439D75"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), sharey=True)

    for ax, community in zip(axes, (1, 2, 3)):
        group = table[table["Community"] == community]
        stat = stats[
            (stats["Community"] == community)
            & (stats["Response"] == "Cov_over_mean_z")
        ].iloc[0]
        x = group["Competition"].to_numpy()
        y = group["Cov_over_mean_z"].to_numpy()
        ax.scatter(x, y, s=34, alpha=0.72, color=colors[community], edgecolor="white", linewidth=0.45)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(line_x, stat["intercept"] + stat["slope"] * line_x, color="#2F2F2F", lw=1.5)
        ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
        ax.text(
            0.04,
            0.96,
            rf"$r$ = {stat['pearson_r']:.2f}" + "\n" + rf"$p$ = {stat['pearson_p']:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        ax.set_title(labels[community], fontsize=12)
        ax.set_xlabel(r"Heterospecific competition pressure, $\mathcal{C}_{intra}$", fontsize=11)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"CUE selection differential, $\mathrm{Cov}(z,\epsilon)/\bar z$", fontsize=11)
    fig.suptitle("Does competition increase the flux weight of high-CUE species?", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    species_data = pd.read_csv(COMMUNITY_DATA)
    competition_data = pd.read_csv(COMPETITION_DATA)
    table = compute_validation_table(species_data, competition_data)
    stats = compute_stats(table)

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DATA, index=False)
    stats.to_csv(OUTPUT_STATS, index=False)
    plot_selection_path(table, stats)

    print(f"maximum covariance identity error: {table['Identity_error'].abs().max():.3e}")
    print(
        stats[stats["Response"].isin(["Cov_over_mean_z", "E_flux", "Epsilon_pool"])][
            ["Community", "Response", "pearson_r", "pearson_p", "r_squared"]
        ].to_string(index=False)
    )
    print(f"Saved {OUTPUT_DATA}")
    print(f"Saved {OUTPUT_STATS}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
