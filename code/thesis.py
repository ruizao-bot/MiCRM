from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os, sys
code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)

# Project root and data directory (absolute paths)
project_root = os.path.abspath(os.path.join(code_path, os.pardir))
data_dir = os.path.join(project_root, "data")

import param
def compute_uptake_variance(u, N):
    """Return a list of uptake variances for each species."""
    return [np.var(u[i, :]) for i in range(N)]

def simulate(seed):
    np.random.seed(seed)
    
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 200, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50
    
    # Maintenance cost parameters: z_α = χ₀(1 + ε_α)(1 - l_α)∑_j c_αj
    χ0 = 0.2  # baseline maintenance cost coefficient
    ε_pool = np.random.uniform(0, 0.1, N_pool)

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)
    rho_pool, omega_pool = np.full(M_pool, 0.6), np.full(M_pool, 0.1)
    t_span = (0, 100000)  # Simulation time span
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    # Normalize u1 rows to sum to 1
    u1 = u1 / u1.sum(axis=1, keepdims=True)
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    
    # Calculate m1 using formula: z_α = χ₀(1 + ε_α)(1 - λ)∑_j c_αj
    ε1 = ε_pool[species_indices1]
    m1 = np.full(N1,0.2)#χ0 * (1 + ε1) * (1 - λ) * np.sum(u1, axis=1)
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = rho_pool[resource_indices1], omega_pool[resource_indices1]

    if M1 > M2:
        resource_indices2 = np.random.choice(resource_indices1, M2, replace=False)
    elif M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = np.random.choice(remaining_resources, M2 - M1, replace=False)
        resource_indices2 = np.concatenate([resource_indices1, additional_resources])
    else:
        resource_indices2 = resource_indices1.copy()
    remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
    species_indices2 = np.random.choice(remaining_species, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    # Normalize u2 rows to sum to 1
    u2 = u2 / u2.sum(axis=1, keepdims=True)
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    
    # Calculate m2 using formula
    ε2 = ε_pool[species_indices2]
    m2 = np.full(N2,0.2)#χ0 * (1 + ε2) * (1 - λ) * np.sum(u2, axis=1)
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = rho_pool[resource_indices2], omega_pool[resource_indices2]
    C0_1 = np.full(N1, 0.01) 
    C0_2 = np.full(N2, 0.01) 
    #R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    R0_1 = np.random.lognormal(mean=0.0, sigma=1.0, size=M1)
    R0_2 = np.random.lognormal(mean=0.0, sigma=1.0, size=M2)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    # Normalize u3 rows to sum to 1
    u3 = u3 / u3.sum(axis=1, keepdims=True)
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    
    # Calculate m3 using formula
    ε3 = ε_pool[species_indices3]
   
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    rho3 = rho_pool[resource_indices3]
    N3, M3 = N1 + N2, len(resource_indices3)

    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    #R0_3 = np.full(M1, 1)#sol1.y[N1:, -1] + sol2.y[N2:, -1]
    R0_3 = (R0_1 + R0_2) / 2
    m3 = np.full(N3,0.2) #χ0 * (1 + ε3) * (1 - λ) * np.sum(u3, axis=1)
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    # calculate community facilitation
    # Community 1
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    C_feed1 = param.calculate_community_feedback(L_eff1, u1)
    print(f"Community 1 - Community feedback (C_feed): {C_feed1:.6f}")

    # Community 2
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    C_feed2 = param.calculate_community_feedback(L_eff2, u2)
    print(f"Community 2 - Community feedback (C_feed): {C_feed2:.6f}")

    # Community 3 (merged)
    L_eff3 = param.calculate_effective_leakage(u3, l3)
    C_feed3 = param.calculate_community_feedback(L_eff3, u3)
    print(f"Community 3 - Community feedback (C_feed): {C_feed3:.6f}")
    # 计算 CUE
    community_CUE1, species_CUE1 = param.compute_CUE(sol1, N1, u1, R0_1, lambda_alpha1, m1)
    community_CUE2, species_CUE2 = param.compute_CUE(sol2, N2, u2, R0_2, lambda_alpha2, m2)
    community_CUE3, species_CUE3 = param.compute_CUE(sol3, N3, u3, R0_3, lambda_alpha3, m3)

    C_final1, C_final2, C_final3 = sol1.y[:N1, -1], sol2.y[:N2, -1], sol3.y[:N3, -1]
    total_1, total_2 = np.sum(C_final3[:N1]), np.sum(C_final3[N1:])
    dominant = "Community 1" if total_1 > total_2 else "Community 2"

    # Calculate resource depletion (sum of residual resources at equilibrium)
    R_final1 = sol1.y[N1:, -1]  # Resource abundances at equilibrium for community 1
    R_final2 = sol2.y[N2:, -1]  # Resource abundances at equilibrium for community 2
    R_final3 = sol3.y[N3:, -1]  # Resource abundances at equilibrium for community 3
    
    depletion1 = np.sum(R_final1)
    depletion2 = np.sum(R_final2)
    depletion3 = np.sum(R_final3)

    # For each community, filter surviving species and calculate average competition among them
    survivors1 = np.where(C_final1 > 1e-5)[0]
    survivors2 = np.where(C_final2 > 1e-5)[0]
    survivors3 = np.where(C_final3 > 1e-5)[0]

    # Calculate niche overlap (cosine similarity) for all and for survivors
    competition1 = param.community_level_competition(u1)
    competition2 = param.community_level_competition(u2)
    competition3 = param.community_level_competition(u3)


    # Calculate community CUE for survivors
    community_CUE1_surv = (np.sum(C_final1[survivors1] * species_CUE1[survivors1]) / np.sum(C_final1[survivors1]))
    community_CUE2_surv = (np.sum(C_final2[survivors2] * species_CUE2[survivors2]) / np.sum(C_final2[survivors2]))
    community_CUE3_surv = (np.sum(C_final3[survivors3] * species_CUE3[survivors3]) / np.sum(C_final3[survivors3]))

    species_data = []
    for i in range(len(species_CUE1)):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Community_CUE_surv": community_CUE1_surv,
            "Abundance": C_final1[i],
            "Total_Abundance": total_1,
            "Dominant_Community": dominant,
            "Competition":  competition1,
            "Facilitation": C_feed1,
            "Depletion": depletion1
        })

    for i in range(len(species_CUE2)):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Abundance": C_final2[i],
            "Community_CUE": community_CUE2,
            "Community_CUE_surv": community_CUE2_surv,
            "Abundance": C_final2[i],
            "Total_Abundance": total_2,
            "Dominant_Community": dominant,
            "Competition":  competition2,
            "Facilitation": C_feed2,
            "Depletion": depletion2
        })

    for i in range(len(species_CUE3)):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Abundance": C_final3[i],
            "Community_CUE": community_CUE3,
            "Community_CUE_surv": community_CUE3_surv,
            "Abundance": C_final3[i],
            "Total_Abundance": total_1 + total_2,
            "Dominant_Community": dominant,
            "Competition":  competition3,
            "Facilitation": C_feed3,
            "Depletion": depletion3
        })

    # Calculate uptake variance for each species in each community
    uptake_var1 = compute_uptake_variance(u1, N1)
    uptake_var2 = compute_uptake_variance(u2, N2)
    uptake_var3 = compute_uptake_variance(u3, N3)

    # Add uptake variance to species_data
    for i in range(len(species_CUE1)):
        species_data[i]["UptakeVar"] = uptake_var1[i]
    for i in range(len(species_CUE2)):
        idx = len(species_CUE1) + i
        species_data[idx]["UptakeVar"] = uptake_var2[i]
    for i in range(len(species_CUE3)):
        idx = len(species_CUE1) + len(species_CUE2) + i
        species_data[idx]["UptakeVar"] = uptake_var3[i]
    return species_data

if __name__ == "__main__":
    seeds_file = os.path.join(code_path, 'seeds.txt')
    with open(seeds_file, 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_species_data_nested = pool.map(simulate, seeds)
    all_species_data = [row for seed_result in all_species_data_nested if seed_result for row in seed_result]
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(all_species_data)
    df.to_csv(os.path.join(data_dir, "coal.csv"), index=False)
