from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
import param

# Random seed and simulation parameters
BASE_SEED = 50
N_SIMULATIONS = 100

# Exported file names
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RARE_FILE = os.path.join(DATA_DIR, "rare.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "rare_summary.csv")

# Species pool and resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1.0
LEAKAGE_RATE = 0.2

# Community parameters
N1, M1 = 100, 50
N2, M2 = 100, 50

# Physiological parameters
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
T_SPAN = (0, 50000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Dilution rates for invasion experiment
DILUTION_RATES = [0.01,0.1]


def compute_community_cue_eflux(species_cue, C_final, u, R0, survivor_idx):
    """Community CUE with Eflux weights: C_i* * U_i^0 over survivors."""
    idx = np.asarray(survivor_idx, dtype=int)
    if idx.size == 0:
        return np.nan
    Ui0 = np.sum(u * R0[None, :], axis=1)
    weights = C_final[idx] * Ui0[idx]
    return param.safe_weighted_average(species_cue[idx], weights)


def simulate(args):
    """Simulate rare species invasion with specified dilution rate."""
    seed, dilution_rate = args
    rng = np.random.default_rng(seed)

    # Generate species and resource pools
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)
    sol1 = param.solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN)

    resource_indices2 = param.choose_resources_for_second_community(M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2 = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2 = np.full(N2, C0_VALUE)
    R0_2 = np.full(M2, R0_VALUE)

    sol2 = param.solve_micrm(N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN)

    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

    N3 = N1 + N2
    M3 = len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    
    # Apply dilution to invader community
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1] * dilution_rate])
    R0_3 = np.full(M3, R0_VALUE)

    sol3 = param.solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN)

    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_final3 = sol3.y[:N3, -1]

    # CUE calculations
    species_CUE1 = param.compute_species_CUE(u1, R0_1, LEAKAGE_RATE, MAINTENANCE_COST)
    species_CUE2 = param.compute_species_CUE(u2, R0_2, LEAKAGE_RATE, MAINTENANCE_COST)
    species_CUE3 = param.compute_species_CUE(u3, R0_3, LEAKAGE_RATE, MAINTENANCE_COST)

    # Survivors
    survivors1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    community_CUE1 = compute_community_cue_eflux(species_CUE1, C_final1, u1, R0_1, survivors1)
    community_CUE2 = compute_community_cue_eflux(species_CUE2, C_final2, u2, R0_2, survivors2)
    community_CUE3 = compute_community_cue_eflux(species_CUE3, C_final3, u3, R0_3, survivors3)

    # Competition metrics
    competition_comm1 = param.community_level_competition(u1, C_final1, np.sum(u1 * R0_1[None, :], axis=1), survivors1)
    competition_comm2 = param.community_level_competition(u2, C_final2, np.sum(u2 * R0_2[None, :], axis=1), survivors2)
    competition_comm3 = param.community_level_competition(u3, C_final3, np.sum(u3 * R0_3[None, :], axis=1), survivors3)

    competition_species1 = param.species_level_competition(u1)
    competition_species2 = param.species_level_competition(u2)
    competition_species3 = param.species_level_competition(u3)

    competition_dot1 = param.species_level_competition_dot(u1)
    competition_dot2 = param.species_level_competition_dot(u2)
    competition_dot3 = param.species_level_competition_dot(u3)

    # Facilitation metrics
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    L_eff3 = param.calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(L_eff1, axis=1)
    facilitation2 = np.mean(L_eff2, axis=1)
    facilitation3 = np.mean(L_eff3, axis=1)

    # Uptake variance
    uptake_var1 = param.compute_uptake_variance(u1)
    uptake_var2 = param.compute_uptake_variance(u2)
    uptake_var3 = param.compute_uptake_variance(u3)

    # Resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    # Total abundance
    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    # Dominance in merged community
    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community1" if origin1_in_coalesced > origin2_in_coalesced else "Community2"

    # Exp
    species_data = []

    # Community 1 data
    for i in range(N1):
        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 1,
            "Species_ID": i + 1,
            "Origin": "Comm1",
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Abundance": C_final1[i],
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i],
            "Species_Index": int(species_indices1[i])
        })

    # Community 2 data
    for i in range(N2):
        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 2,
            "Species_ID": i + 1,
            "Origin": "Comm2",
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Abundance": C_final2[i],
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i],
            "Species_Index": int(species_indices2[i])
        })

    # Community 3 data
    for i in range(N3):
        origin = "Comm1" if i < N1 else "Comm2"
        species_index = int(species_indices1[i]) if i < N1 else int(species_indices2[i - N1])
        
        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 3,
            "Species_ID": i + 1,
            "Origin": origin,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Abundance": C_final3[i],
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i],
            "Species_Index": species_index
        })

    return species_data


def main():
    """Main function to run rare species invasion simulations."""
    # Generate random seeds
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    # Create parameter combinations
    param_list = [(seed, dr) for seed in seeds for dr in DILUTION_RATES]

    print(f"Starting rare species invasion simulations...")
    print(f"  - Number of seeds: {len(seeds)}")
    print(f"  - Dilution rates: {DILUTION_RATES}")
    print(f"  - Total simulations: {len(param_list)}")
    print(f"  - CPU cores: {cpu_count()}")

    # Run parallel simulations
    with Pool(cpu_count()) as pool:
        all_data_nested = pool.map(simulate, param_list)

    # Flatten results
    all_data = [
        row
        for result in all_data_nested
        if result
        for row in result
    ]

    # Save detailed results
    df = pd.DataFrame(all_data)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RARE_FILE, index=False)

    print(f"\nSimulation completed!")



if __name__ == "__main__":
    main()
