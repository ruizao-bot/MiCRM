from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
# import statsmodels.api as sm

code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)
import param
import CUE

def simulate(args):
    seed, dilution_rate = args
    np.random.seed(seed)
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 200, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50
    # maintenance cost baseline and per-species epsilon pool
    chi0 = 0.2
    epsilon_pool = np.random.uniform(0, 0.1, N_pool)

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)
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
    C0_1 = np.full(N1, 0.01)  # 群落 1 的初始种群密度
    C0_2 = np.full(N2, 0.01)  # 群落 2 的初始种群密度
    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    # Solve for community 1 and 2
    t_span = (0, 3000)
    # compute maintenance costs per community using param.compute_maintenance
    eps1 = epsilon_pool[species_indices1]
    m1 = param.compute_maintenance(chi0, eps1, λ, u1)
    eps2 = epsilon_pool[species_indices2]
    m2 = param.compute_maintenance(chi0, eps2, λ, u2)

    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    ce1 = sol1.y[:N1, -1]
    ce2 = sol2.y[:N2, -1]
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]

    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    # maintenance for merged community
    eps3 = epsilon_pool[species_indices3]
    m3 = param.compute_maintenance(chi0, eps3, λ, u3)
    lambda_alpha3 = np.full(len(resource_indices3), λ)

    omega3 = omega_pool[resource_indices3]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    rho3 = rho_pool[resource_indices3]
    C0_3 = np.concatenate([ce1, ce2*0.01])
    R0_3 = np.full(M1, 1) 

    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)

    # --- Community 1 (resident) results ---
    _, species_CUE1 = CUE.compute_CUE(sol1, N1, u1, R0_1, λ, m1)
    C_final1 = sol1.y[:N1, -1]
    comm1_data = []
    for i in range(N1):
        comm1_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": "Comm1",
            "Species_ID": i + 1,
            "Origin": "Comm1",
            "CUE": species_CUE1[i],
            "C_final": C_final1[i],
            "Species_Index": int(species_indices1[i])  # store original index
        })

    # --- Community 2 (invader) results ---
    _, species_CUE2 = CUE.compute_CUE(sol2, N2, u2, R0_2, λ, m2)
    C_final2 = sol2.y[:N2, -1]
    comm2_data = []
    for i in range(N2):
        comm2_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": "Comm2",
            "Species_ID": i + 1,
            "Origin": "Comm2",
            "CUE": species_CUE2[i],
            "C_final": C_final2[i],
            "Species_Index": int(species_indices2[i])  # store original index
        })

    # --- Community 3 (merged) results ---
    ce1 = sol1.y[:N1, -1]
    ce2 = sol2.y[:N2, -1]
    C0_3 = np.concatenate([ce1, ce2 * dilution_rate])
    R0_3 = np.full(M1, 1) 
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    _, species_CUE3 = CUE.compute_CUE(sol3, N3, u3, R0_3, λ, m3)
    C_final3 = sol3.y[:N3, -1]
    comm3_data = []
    for i in range(N3):
        origin = "Comm1" if i < N1 else "Comm2"
        # Use the correct species index from the original pools
        if i < N1:
            species_index = int(species_indices1[i])
        else:
            species_index = int(species_indices2[i - N1])
        comm3_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": "Comm3",
            "Species_ID": i + 1,
            "Origin": origin,
            "CUE": species_CUE3[i],
            "C_final": C_final3[i],
            "Species_Index": species_index  # store original index
        })

    return comm1_data + comm2_data + comm3_data

if __name__ == "__main__":
    with open('seeds.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f][:10]

    dilution_rates = [0.01, 0.05, 0.1]
    param_list = [(seed, dr) for seed in seeds for dr in dilution_rates]

    with Pool(cpu_count()) as pool:
        all_data_nested = pool.map(simulate, param_list)

    all_data = [row for result in all_data_nested if result for row in result]
    df = pd.DataFrame(all_data)
    df.to_csv("data/rare.csv", index=False)

