
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
    C0_1, C0_2 = np.full(N1, 0.01), np.full(N2, 0.01)

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
    rho_pool, omega_pool = np.full(M_pool, 0.6), np.full(M_pool, 0.1)

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

    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2)

    C_hat1, R_hat1 = sol1.y[:N1, -1], sol1.y[N1:, -1]
    C_hat2, R_hat2 = sol2.y[:N2, -1], sol2.y[N2:, -1]
    alpha1, r1 = param.compute_alpha_r(C_hat1, R_hat1, N1, M1, u1, l1, m1, lambda_alpha1, omega1)
    alpha2, r2 = param.compute_alpha_r(C_hat2, R_hat2, N2, M2, u2, l2, m2, lambda_alpha2, omega2)
    sol_elv1 = param.solve_elv(alpha1, r1, C0_1)
    sol_elv2 = param.solve_elv(alpha2, r2, C0_2)

    R0_3 = R_hat1 + R_hat2
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    rho3, omega3 = rho_pool[resource_indices3], omega_pool[resource_indices3]
    N3 = N1 + N2
    C0_3 = np.zeros(N3)
    C0_3[:N1], C0_3[N1:] = C_hat1, C_hat2

    sol3 = param.solve_micrm(N3, len(resource_indices3), u3, l3, m3, lambda_alpha3, rho3, omega3)
    C_hat3, R_hat3 = sol3.y[:N3, -1], sol3.y[N3:, -1]
    alpha3, r3 = param.compute_alpha_r(C_hat3, R_hat3, N3, len(resource_indices3), u3, l3, m3, lambda_alpha3, omega3)
    sol_elv3 = param.solve_elv(alpha3, r3, C0_3)

    species_data = []
    for idx, (sol, N, u, R0, l, m, label) in enumerate(zip(
        [sol1, sol2, sol3], [N1, N2, N3], [u1, u2, u3], [R_hat1, R_hat2, R0_3], [l1, l2, l3], [m1, m2, m3], [1, 2, 3]
    )):
        community_CUE, species_CUE = CUE.compute_community_CUE2(sol, N, u, R0, l, m)
        C_final = sol.y[:N, -1]
        for i in range(N):
            species_data.append({
                "Seed": seed,
                "Community": label,
                "Species_ID": i + 1,
                "Species_CUE": species_CUE[i],
                "Community_CUE": community_CUE,
                "Abundance": C_final[i]
            })
    return species_data

if __name__ == "__main__":
    with open('seeds.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_species_data_nested = pool.map(simulate, seeds)
    all_species_data = [row for seed_result in all_species_data_nested if seed_result for row in seed_result]
    df = pd.DataFrame(all_species_data)
    df.to_csv("elv_parallel_output.csv", index=False)
