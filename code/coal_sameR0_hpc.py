from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os, sys
code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)
import param
import CUE
def simulate(seed):
    np.random.seed(seed)
    
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 200, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50
    m1, m2 = np.full(N1, 0.2), np.full(N2, 0.2)

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
    rho_pool, omega_pool = np.full(M_pool, 0.6), np.full(M_pool, 0.1)
    t_span = (0, 3000)  # Simulation time span
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
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
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = rho_pool[resource_indices2], omega_pool[resource_indices2]
    C0_1 = np.full(N1, 0.01)  # 群落 1 的初始种群密度
    C0_2 = np.full(N2, 0.01)  # 群落 2 的初始种群密度
    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    rho3 = rho_pool[resource_indices3]
    N3, M3 = N1 + N2, len(resource_indices3)

    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M1, 1)#sol1.y[N1:, -1] + sol2.y[N2:, -1]

    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)

    # 计算 CUE
    community_CUE1, species_CUE1 = CUE.compute_CUE(sol1, N1, u1, R0_1, l1, m1)
    community_CUE2, species_CUE2 = CUE.compute_CUE(sol2, N2, u2, R0_2, l2, m2)
    community_CUE3, species_CUE3 = CUE.compute_CUE(sol3, N3, u3, R0_3, l3, m3)

    C_final1, C_final2, C_final3 = sol1.y[:N1, -1], sol2.y[:N2, -1], sol3.y[:N3, -1]
    total_1, total_2 = np.sum(C_final3[:N1]), np.sum(C_final3[N1:])
    dominant = "Community 1" if total_1 > total_2 else "Community 2"

    rel1 = C_final1 / np.sum(C_final1)
    rel2 = C_final2 / np.sum(C_final2)
    rel3 = C_final3 / np.sum(C_final3)

    species_data = []
    for i in range(len(species_CUE1)):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Abundance": C_final1[i],
            "Total_Abundance": total_1,
            "Dominant_Community": dominant
        })

    # 群落 2
    for i in range(len(species_CUE2)):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Abundance": C_final2[i],
            "Community_CUE": community_CUE2,
            "Total_Abundance": total_2,
            "Dominant_Community": dominant
        })

    # 合并后群落（群落 3）
    for i in range(len(species_CUE3)):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Abundance": C_final3[i],
            "Community_CUE": community_CUE3,
            "Total_Abundance": total_1 + total_2,  # 这里按你实际需要写
            "Dominant_Community": dominant
        })

    return species_data

if __name__ == "__main__":
    with open('seeds.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_species_data_nested = pool.map(simulate, seeds)
    all_species_data = [row for seed_result in all_species_data_nested if seed_result for row in seed_result]
    df = pd.DataFrame(all_species_data)
    df.to_csv("../data/coal_sameR0_lsoda.csv", index=False)