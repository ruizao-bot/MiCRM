import numpy as np
import pandas as pd
import os, sys
from multiprocessing import Pool, cpu_count
import param

code_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(code_path, os.pardir))
data_dir = os.path.join(project_root, "data")

# Main function for resource overlap experiment
def simulate_overlap(args):
    seed, overlap_ratio = args
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 100, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio, rng)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool, rng)
    rho_pool, omega_pool = np.full(M_pool, 0.6), np.full(M_pool, 0.1)
    t_span = (0, 100000)
    
    # Fix the number of resources for each community M1=M2=50, allocate shared and unique resources based on overlap_ratio
    M1 = M2 = 50
    overlap_n = int(M1 * overlap_ratio)  # Number of shared resources
    unique_n = M1 - overlap_n  # Number of unique resources per community
    
    all_resources = np.arange(M_pool)
    overlap_resources = np.random.choice(all_resources, overlap_n, replace=False)
    remain_resources = np.setdiff1d(all_resources, overlap_resources)
    res1_unique = np.random.choice(remain_resources, unique_n, replace=False)
    res2_unique = np.random.choice(np.setdiff1d(remain_resources, res1_unique), unique_n, replace=False)
    resource_indices1 = np.concatenate([overlap_resources, res1_unique])
    resource_indices2 = np.concatenate([overlap_resources, res2_unique])
    
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
    species_indices2 = np.random.choice(remaining_species, N2, replace=False)
    
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    m1 = 0.2
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = rho_pool[resource_indices1], omega_pool[resource_indices1]
    C0_1 = np.full(N1, 0.01)
    R0_1 = np.full(M1, 1.0)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    m2 = 0.2
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = rho_pool[resource_indices2], omega_pool[resource_indices2]
    C0_2 = np.full(N2, 0.01)
    R0_2 = np.full(M2, 1.0)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)

    # Merge communities
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = np.union1d(resource_indices1, resource_indices2)
    N3, M3 = N1 + N2, len(resource_indices3)
    
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    
    # Restrict species to only utilize resources from their original community
    # Species from community 1 (first N1): can only utilize resources in resource_indices1
    # Species from community 2 (last N2): can only utilize resources in resource_indices2
    for i in range(N3):
        for j, res in enumerate(resource_indices3):
            if i < N1:  # Species from community 1
                if res not in resource_indices1:
                    u3[i, j] = 0.0
            else:  # Species from community 2
                if res not in resource_indices2:
                    u3[i, j] = 0.0
    
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    
    # All resources use the same initial abundance and supply rate
    R0_3 = np.full(M3, 1.0)  # R0 = 1
    rho3 = rho_pool[resource_indices3]  # rho = 0.6 (inherited from resource pool)
    
    m3 = 0.2
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)

    # Calculate CUE based on survivors only
    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_final3 = sol3.y[:N3, -1]
    
    # Compute species CUE
    species_CUE1 = param.compute_species_CUE(u1, R0_1, lambda_alpha1, m1)
    species_CUE2 = param.compute_species_CUE(u2, R0_2, lambda_alpha2, m2)
    species_CUE3 = param.compute_species_CUE(u3, R0_3, lambda_alpha3, m3)
    
    # Get survivors (abundance > 1e-5)
    survivors1 = np.where(C_final1 > 1e-5)[0]
    survivors2 = np.where(C_final2 > 1e-5)[0]
    survivors3 = np.where(C_final3 > 1e-5)[0]
    
    # Calculate community CUE based on survivors only
    community_CUE1 = param.safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2 = param.safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3 = param.safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])
    
    total_1, total_2 = np.sum(C_final3[:N1]), np.sum(C_final3[N1:])
    dominant = "Community 1" if total_1 > total_2 else "Community 2"

    # Compute Bray-Curtis similarity between coalesced and each parent
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
        "Dominant_Community": dominant,
        "Total_Abundance_1": total_1,
        "Total_Abundance_2": total_2,
        "Sim_3vs1": sim_3vs1,
        "Sim_3vs2": sim_3vs2
    }

if __name__ == "__main__":
    seeds_file = os.path.join(code_path, 'seeds.txt')
    with open(seeds_file, 'r') as f:
        seeds = [int(line.strip()) for line in f]
    overlap_list = [0.25, 0.5, 0.75]
    args_list = [(seed, overlap) for seed in seeds for overlap in overlap_list]
    with Pool(cpu_count()) as pool:
        results = pool.map(simulate_overlap, args_list)
    df = pd.DataFrame(results)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "coal_resource.csv"), index=False)
