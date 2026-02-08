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
    
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 100, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)
    t_span = (0, 100000)  # Simulation time span
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    m1 = 0.2
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = np.full(M1, 0.6), np.full(M1, 0.1)

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
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    m2 = 0.2
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = np.full(M2, 0.6), np.full(M2, 0.1)
    C0_1 = np.full(N1, 0.01) 
    C0_2 = np.full(N2, 0.01) 
    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    # R0_1 = np.random.lognormal(mean=0.0, sigma=1.0, size=M1)
    # R0_2 = np.random.lognormal(mean=0.0, sigma=1.0, size=M2)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    
    # Extract resource concentrations at t = 20/omega
    t_target1 = 20.0 / np.mean(omega1)
    idx1 = np.argmin(np.abs(sol1.t - t_target1))
    R_at_t1 = sol1.y[N1:, idx1]
    
    t_target2 = 20.0 / np.mean(omega2)
    idx2 = np.argmin(np.abs(sol2.t - t_target2))
    R_at_t2 = sol2.y[N2:, idx2]
    
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    N3, M3 = N1 + N2, len(resource_indices3)
    omega3 = np.full(M3, 0.1)
    rho3 = np.full(M3, 0.6)

    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M1, 1)#sol1.y[N1:, -1] + sol2.y[N2:, -1]
    m3 = 0.2
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    
    # Extract resource concentration at t = 20/omega for community 3
    t_target3 = 20.0 / np.mean(omega3)
    idx3 = np.argmin(np.abs(sol3.t - t_target3))
    R_at_t3 = sol3.y[N3:, idx3]
    

    # Calculate facilitation metrics (use L_eff mean per species)
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    Facilitation1 = np.mean(L_eff1, axis=1)

    L_eff2 = param.calculate_effective_leakage(u2, l2)
    Facilitation2 = np.mean(L_eff2, axis=1)

    L_eff3 = param.calculate_effective_leakage(u3, l3)
    Facilitation3 = np.mean(L_eff3, axis=1)
    
    # Calculate species CUE using resource concentrations at t = 20/omega
    # But use equilibrium abundance (final time point) for community CUE weighting
    _, species_CUE1 = param.compute_CUE(sol1, N1, u1, R_at_t1, lambda_alpha1, m1)
    _, species_CUE2 = param.compute_CUE(sol2, N2, u2, R_at_t2, lambda_alpha2, m2)
    _, species_CUE3 = param.compute_CUE(sol3, N3, u3, R_at_t3, lambda_alpha3, m3)

    C_final1, C_final2, C_final3 = sol1.y[:N1, -1], sol2.y[:N2, -1], sol3.y[:N3, -1]
    # Use equilibrium abundance for community CUE
    community_CUE1 = np.sum(C_final1 * species_CUE1) / np.sum(C_final1)
    community_CUE2 = np.sum(C_final2 * species_CUE2) / np.sum(C_final2)
    community_CUE3 = np.sum(C_final3 * species_CUE3) / np.sum(C_final3)

    C_final1, C_final2, C_final3 = sol1.y[:N1, -1], sol2.y[:N2, -1], sol3.y[:N3, -1]
    total_1, total_2 = np.sum(C_final3[:N1]), np.sum(C_final3[N1:])
    dominant = "Community 1" if total_1 > total_2 else "Community 2"

    # Calculate resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    # Get survivors and calculate competition
    # Get survivors and calculate competition
    survivors1 = np.where(C_final1 > 1e-5)[0]
    survivors2 = np.where(C_final2 > 1e-5)[0]
    survivors3 = np.where(C_final3 > 1e-5)[0]

    competition1 = param.community_level_competition(u1)
    competition2 = param.community_level_competition(u2)
    competition3 = param.community_level_competition(u3)

    species_competition1 = param.species_level_competition(u1)
    species_competition2 = param.species_level_competition(u2)
    species_competition3 = param.species_level_competition(u3)

    # Calculate community CUE for survivors
    community_CUE1_surv = np.sum(C_final1[survivors1] * species_CUE1[survivors1]) / np.sum(C_final1[survivors1])
    community_CUE2_surv = np.sum(C_final2[survivors2] * species_CUE2[survivors2]) / np.sum(C_final2[survivors2])
    community_CUE3_surv = np.sum(C_final3[survivors3] * species_CUE3[survivors3]) / np.sum(C_final3[survivors3])

    # Calculate uptake variance
    uptake_var1 = compute_uptake_variance(u1, N1)
    uptake_var2 = compute_uptake_variance(u2, N2)
    uptake_var3 = compute_uptake_variance(u3, N3)

    # Build species data
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
            "Competition": competition1,
            "Species_Competition": species_competition1[i],
            "Facilitation": Facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i]
        })

    for i in range(len(species_CUE2)):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Community_CUE_surv": community_CUE2_surv,
            "Abundance": C_final2[i],
            "Total_Abundance": total_2,
            "Dominant_Community": dominant,
            "Competition": competition2,
            "Species_Competition": species_competition2[i],
            "Facilitation": Facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i]
        })

    for i in range(len(species_CUE3)):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Community_CUE_surv": community_CUE3_surv,
            "Abundance": C_final3[i],
            "Total_Abundance": total_1 + total_2,
            "Dominant_Community": dominant,
            "Competition": competition3,
            "Species_Competition": species_competition3[i],
            "Facilitation": Facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i]
        })

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
