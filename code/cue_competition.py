

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import param as p

N_SPECIES = 100
N_RESOURCES = 50
LEAKAGE_MIN = 0.0
LEAKAGE_MAX = 0.5
S_RATIO_MAX = 1.0
N_SIMULATIONS = 50

MAINTENANCE = np.full(N_SPECIES, 0.2)
RHO = np.full(N_RESOURCES, 0.6)
OMEGA = np.full(N_RESOURCES, 0.1)
C0 = np.full(N_SPECIES, 0.1)
R0 = np.full(N_RESOURCES, 1.0)
T_SPAN = (0, 1e8)

SURVIVAL_THRESHOLD = 1e-5
OUTPUT_CSV = "species_competition_cue_abundance.csv"
OUTPUT_FIG = "figure/species_competition_cue_abundance.png"

# =============================================================================
# 随机社区参数采样
# =============================================================================
def sample_community_parameters(rng):
    n_modules = int(rng.integers(1, N_RESOURCES + 1))
    s_ratio = float(rng.uniform(1.0, S_RATIO_MAX))
    leakage = float(rng.uniform(LEAKAGE_MIN, LEAKAGE_MAX))
    return n_modules, s_ratio, leakage

# =============================================================================
# CUE 计算（均衡态）
# =============================================================================
def compute_equilibrium_cue(sol, uptake, leakage, maintenance):
    """Return equilibrium community CUE, species CUE, and final biomass."""
    final_biomass = sol.y[:N_SPECIES, -1]
    final_resources = sol.y[N_SPECIES:, -1]

    total_uptake = np.sum(uptake * final_resources[None, :], axis=1)
    net_uptake = (1 - leakage) * total_uptake - maintenance
    species_cue = net_uptake / (total_uptake + 1e-12)

    survivor_mask = final_biomass > SURVIVAL_THRESHOLD
    if not np.any(survivor_mask):
        return np.nan, species_cue, final_biomass

    weights = final_biomass[survivor_mask]
    community_cue = np.sum(weights * species_cue[survivor_mask]) / np.sum(weights)
    return float(community_cue), species_cue, final_biomass


# =============================================================================
# 单次随机模拟
# =============================================================================
def run_one(seed):
    rng = np.random.default_rng(seed)
    n_modules, s_ratio, leakage = sample_community_parameters(rng)

    try:
        uptake = p.modular_uptake(N_SPECIES, N_RESOURCES, n_modules, s_ratio, rng)
        leakage_tensor = p.generate_l_tensor(N_SPECIES, N_RESOURCES, n_modules, s_ratio, leakage, uptake, rng)
    except Exception as e:
        return None, f"param gen failed: {e}"

    try:
        sol = p.solve_micrm(
            N_SPECIES, N_RESOURCES, uptake, leakage_tensor, MAINTENANCE,
            lambda_alpha=leakage,
            rho=RHO, omega=OMEGA,
            C0=C0, R0=R0,
            t_span=T_SPAN
        )
    except Exception as e:
        return None, f"solve failed: {e}"

    if sol.y.shape[1] == 0:
        return None, "empty solution"

    community_cue, species_cue, final_biomass = compute_equilibrium_cue(sol, uptake, leakage, MAINTENANCE)
    n_survivors = int(np.sum(final_biomass > SURVIVAL_THRESHOLD))
    species_competition = p.species_level_competition(uptake)

    records = []
    for species_idx in range(N_SPECIES):
        records.append({
            "seed": int(seed),
            "species_id": species_idx,
            "leakage": float(leakage),
            "n_modules": int(n_modules),
            "s_ratio": float(s_ratio),
            "community_cue": community_cue,
            "n_survivors": n_survivors,
            "species_competition": float(species_competition[species_idx]),
            "species_cue": float(species_cue[species_idx]),
            "final_abundance": float(final_biomass[species_idx]),
            "survived": bool(final_biomass[species_idx] > SURVIVAL_THRESHOLD),
        })

    return records, None

# =============================================================================
# 主流程
# =============================================================================
def main():
    species_records = []
    base_seed = 42

    for sim_idx in range(N_SIMULATIONS):
        seed = base_seed + sim_idx
        result, err = run_one(seed)
        if result is None:
            print(f"[skip] sim={sim_idx} seed={seed} reason: {err}")
            continue
        species_records.extend(result)

    print(f"valid simulations: {len(species_records) // N_SPECIES}/{N_SIMULATIONS}")

    df = pd.DataFrame(species_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV} ({len(df)} rows)")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(df["species_competition"], df["species_cue"], s=18, alpha=0.45, color='steelblue')
    ax.set_xlabel("Species competition")
    ax.set_ylabel("Species CUE")
    ax.set_title("Species CUE vs competition")

    ax = axes[1]
    ax.scatter(df["species_competition"], df["final_abundance"], s=18, alpha=0.45, color='darkorange')
    ax.set_xlabel("Species competition")
    ax.set_ylabel("Final abundance")
    ax.set_title("Species abundance vs competition")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
