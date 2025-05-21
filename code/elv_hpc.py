from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
code_path = os.path.dirname(os.path.abspath(__file__))
import param
import CUE
import pandas as pd
def simulate(seed):
    np.random.seed(seed)
    N_pool = 1000  # Species pool size
    M_pool = 200    # Resource pool size
    λ = 0.2        # Total leakage rate
    N_modules = 1  # Number of modules
    s_ratio = 1.0 # Modularity ratio
    N1 = 100
    M1 = 50
    m1 = np.full(N1, 0.2)  # maintaining cost rate
    N2 = 100
    M2 = 50
    m2 =  np.full(N2, 0.2)#truncnorm.rvs((0 - 1) / 0.01, np.inf, loc=1, scale=0.01, size=N2)
    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    # Generate uptake matrix and leakage tensor for the species pool
    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
    # Set rho and omega for the resource pool
    rho_pool = np.full(M_pool, 0.6)
    omega_pool = np.full(M_pool, 0.1)
    # Community 1
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    #u1 = 2.5 * u1 / u1.sum(axis=1, keepdims=True)
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    lambda_alpha1 = np.full(M1, λ)
    rho1 = rho_pool[resource_indices1]
    omega1 = omega_pool[resource_indices1]
    # Community 2
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
    #u2 = 2.5 * u2 / u2.sum(axis=1, keepdims=True)
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    lambda_alpha2 = np.full(M2, λ)
    rho2 = rho_pool[resource_indices2]
    omega2 = omega_pool[resource_indices2]
    C0_1 = np.full(N1, 0.01)
    C0_2 = np.full(N2, 0.01)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2)
    # 对群落 1：
    C_hat1 = sol1.y[:N1, -1]
    R_hat1 = sol1.y[N1:, -1]
    alpha1, r1 = param.compute_alpha_r(C_hat1, R_hat1, N1, M1,u1, l1, m1, lambda_alpha1, omega1)
    sol_elv1 = param.solve_elv(alpha1, r1,C0_1 )

    # 对群落 2：
    C_hat2 = sol2.y[:N2, -1]
    R_hat2 = sol2.y[N2:, -1]
    alpha2, r2 = param.compute_alpha_r(C_hat2, R_hat2,N2, M2, u2, l2, m2, lambda_alpha2, omega2)
    sol_elv2 = param.solve_elv(alpha2, r2, C0_2)



    # Merge into Community 3
    R0_3 = sol1.y[N1:, -1] + sol2.y[N2:, -1]
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    N3 = N1 + N2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    # u3 = 2.5 * u3 / u3.sum(axis=1, keepdims=True)
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    rho3 = rho_pool[resource_indices3]
    omega3 = omega_pool[resource_indices3]
    C0_3 = np.zeros(N1 + N2)
    C0_3[:N1]   = C_hat1
    C0_3[N1:]   = C_hat2


    M3 = len(resource_indices3)

    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3)
    C_hat3 = sol3.y[:N3, -1]
    R_hat3 = sol3.y[N3:, -1]

    alpha3, r3 = param.compute_alpha_r(C_hat3, R_hat3, N3, M3, u3, l3, m3, lambda_alpha3, omega3)
    sol_elv3 = param.solve_elv(alpha3, r3, C0_3 )

    community_CUE1, species_CUE1 = CUE.compute_community_CUE2(sol1, N1, u1, R0_1, l1, m1)
    community_CUE2, species_CUE2 = CUE.compute_community_CUE2(sol2, N2, u2, R0_2, l2, m2)
    community_CUE3, species_CUE3 = CUE.compute_community_CUE2(sol3, N3, u3, R0_3, l3, m3)
    results = []
    result_entry = {"Seed": seed}
    sol_list = [sol1, sol2, sol3] 
    N_list = [N1, N2, N3] 
    u_list = [u1, u2, u3]
    R0_list = [R0_1, R0_2, R0_3]
    l_list = [l1, l2, l3]
    m_list = [m1, m2, m3]
    M_list = [M1, M2, M3]
    alpha_list = [alpha1, alpha2, alpha3]
    r_list = [r1, r2, r3]
    C_final_list = []
    num_communities = len(sol_list) 
    for i in range(num_communities):

        C_final = np.array(sol_list[i].y[:, -1])
        C_final_list.append(C_final)

        _, species_CUE = CUE.compute_community_CUE2(
            sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
        )

        for j, (r_val, cue_val, c_val, alpha_val) in enumerate(zip(r_list[i], species_CUE, C_final, alpha_list[i])):
            result_entry[f"r_Comm{i+1}_Sp{j+1}"] = r_val
            result_entry[f"CUE_Comm{i+1}_Sp{j+1}"] = cue_val
            result_entry[f"Cfinal_Comm{i+1}_Sp{j+1}"] = c_val
            result_entry[f"alpha_Comm{i+1}_Sp{j+1}"] = alpha_val
    results.append(result_entry)
    return results

if __name__ == "__main__":
    with open('seeds.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_results_nested = pool.map(simulate, seeds)
    all_results = [row for seed_result in all_results_nested if seed_result for row in seed_result]
    df = pd.DataFrame(all_results)
    df.to_csv("../data/elv_hpc.csv", index=False)
