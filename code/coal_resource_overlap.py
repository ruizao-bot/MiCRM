import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import param

# Random seed and simulation parameters
BASE_SEED = 50
N_SIMULATIONS = 50

# Exported file names
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COAL_RESOURCE_FILE = os.path.join(DATA_DIR, "coal_resource.csv")

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
T_SPAN = (0, 100000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Resource overlap ratios for experiment
OVERLAP_RATIOS = [0.25, 0.5, 0.75]


def simulate_overlap(args):
    """Simulate community coalescence with specified resource overlap ratio."""
    seed, overlap_ratio = args
    rng = np.random.default_rng(seed)

    # Generate species and resource pools
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # Calculate shared and unique resources
    overlap_n = int(M1 * overlap_ratio)
    unique_n = M1 - overlap_n
    
    all_resources = np.arange(M_POOL)
    overlap_resources = rng.choice(all_resources, overlap_n, replace=False)
    remain_resources = np.setdiff1d(all_resources, overlap_resources)
    res1_unique = rng.choice(remain_resources, unique_n, replace=False)
    res2_unique = rng.choice(np.setdiff1d(remain_resources, res1_unique), unique_n, replace=False)
    
    resource_indices1 = np.concatenate([overlap_resources, res1_unique])
    resource_indices2 = np.concatenate([overlap_resources, res2_unique])
    
    # Community 1
    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    
    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)
    
    sol1 = param.solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN)

    # Community 2
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

    # Community 3
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = np.union1d(resource_indices1, resource_indices2)
    N3 = N1 + N2
    M3 = len(resource_indices3)
    
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    
    # Restrict species to only utilize resources from their original community
    for i in range(N3):
        for j, res in enumerate(resource_indices3):
            if i < N1:  # Species from community 1
                if res not in resource_indices1:
                    u3[i, j] = 0.0
            else:  # Species from community 2
                if res not in resource_indices2:
                    u3[i, j] = 0.0
    
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M3, R0_VALUE)
    
    sol3 = param.solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN)

    # Extract Community Metrics
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
    
    # Community CUE (survivors only)
    community_CUE1 = param.safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2 = param.safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3 = param.safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])
    
    # Competition metrics
    competition_comm1 = param.community_level_competition(u1)
    competition_comm2 = param.community_level_competition(u2)
    competition_comm3 = param.community_level_competition(u3)
    
    # Facilitation metrics
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    L_eff3 = param.calculate_effective_leakage(u3, l3)
    
    facilitation1 = np.mean(np.sum(L_eff1, axis=1))
    facilitation2 = np.mean(np.sum(L_eff2, axis=1))
    facilitation3 = np.mean(np.sum(L_eff3, axis=1))
    
    # Resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])
    
    # Total abundance
    total_abundance_1 = np.sum(C_final1)
    total_abundance_2 = np.sum(C_final2)
    total_abundance_3 = np.sum(C_final3)
    
    # Dominance in merged community
    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community1" if origin1_in_coalesced > origin2_in_coalesced else "Community2"

    # Bray-Curtis similarity between merged and parent communities
    # Parent 1 composition vector (N1 + N2): [sol1 final, zeros for parent 2 species]
    parent1_vec = np.concatenate([C_final1, np.zeros(N2)])
    parent2_vec = np.concatenate([np.zeros(N1), C_final2])
    coalesced_vec = C_final3

    # Bray-Curtis dissimilarity = sum(|x - y|) / sum(x + y)
    bc_diss_3vs1 = np.sum(np.abs(coalesced_vec - parent1_vec)) / np.sum(coalesced_vec + parent1_vec) if np.sum(coalesced_vec + parent1_vec) > 0 else 1.0
    bc_diss_3vs2 = np.sum(np.abs(coalesced_vec - parent2_vec)) / np.sum(coalesced_vec + parent2_vec) if np.sum(coalesced_vec + parent2_vec) > 0 else 1.0
    sim_3vs1 = 1 - bc_diss_3vs1
    sim_3vs2 = 1 - bc_diss_3vs2

    return {
        "Seed": seed,
        "Overlap": overlap_ratio,
        "CUE1": community_CUE1,
        "CUE2": community_CUE2,
        "CUE3": community_CUE3,
        "Num_Survivors1": len(survivors1),
        "Num_Survivors2": len(survivors2),
        "Num_Survivors3": len(survivors3),
        "Competition1": competition_comm1,
        "Competition2": competition_comm2,
        "Competition3": competition_comm3,
        "Facilitation1": facilitation1,
        "Facilitation2": facilitation2,
        "Facilitation3": facilitation3,
        "Depletion1": depletion1,
        "Depletion2": depletion2,
        "Depletion3": depletion3,
        "Total_Abundance_1": total_abundance_1,
        "Total_Abundance_2": total_abundance_2,
        "Total_Abundance_3": total_abundance_3,
        "Dominant_Community": dominant,
        "Similarity_3vs1": sim_3vs1,
        "Similarity_3vs2": sim_3vs2
    }

def main():
    """Main function to run resource overlap coalescence simulations."""
    # Generate random seeds
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    # Create parameter combinations
    args_list = [(seed, overlap) for seed in seeds for overlap in OVERLAP_RATIOS]

    print(f"Starting resource overlap coalescence simulations...")
    print(f"  - Number of seeds: {len(seeds)}")
    print(f"  - Overlap ratios: {OVERLAP_RATIOS}")
    print(f"  - Total simulations: {len(args_list)}")
    print(f"  - CPU cores: {cpu_count()}")

    # Run parallel simulations
    with Pool(cpu_count()) as pool:
        results = pool.map(simulate_overlap, args_list)

    # Save detailed results
    df = pd.DataFrame(results)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(COAL_RESOURCE_FILE, index=False)

    # Generate summary statistics
    print(f"\nSimulation completed!")
    print(f"  - Results saved to: {COAL_RESOURCE_FILE}")
    print(f"  - Total records: {len(df)}")

if __name__ == "__main__":
    main()
