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

def simulate(seed):
    np.random.seed(seed)
    
    # Parameters from coalescence_merge.py
    λ = 0.2
    N_modules = 5
    s_ratio = 10
    N1 = 100
    M = 50
    m1 = np.full(N1, 0.2)
    N2 = 100
    m2 = np.full(N2, 0.2)
    t_span = (0, 100000)
    SURV_THRESH = 1e-5

    # --- Community 1 Setup & Simulation ---
    u1 = param.modular_uptake(N1, M, N_modules, s_ratio)
    l1 = param.generate_l_tensor(N1, M, N_modules, s_ratio, λ)
    lambda_alpha1 = np.full(M, λ)
    rho1 = np.full(M, 0.6)
    omega1 = np.full(M, 0.1)
    C0_1 = np.full(N1, 0.01)
    R0_1 = np.full(M, 1.0)
    sol1 = param.solve_micrm(N1, M, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    ce1 = sol1.y[:N1, -1]

    # --- Community 2 Setup & Simulation ---
    u2 = param.modular_uptake(N2, M, N_modules, s_ratio)
    l2 = param.generate_l_tensor(N2, M, N_modules, s_ratio, λ)
    lambda_alpha2 = np.full(M, λ)
    rho2 = np.full(M, 0.6)
    omega2 = np.full(M, 0.1)
    C0_2 = np.full(N2, 0.01)
    R0_2 = np.full(M, 1.0)
    sol2 = param.solve_micrm(N2, M, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    ce2 = sol2.y[:N2, -1]

    # --- Merged Community 3 Setup & Simulation ---
    u3 = np.vstack([u1, u2])
    l3 = np.concatenate([l1, l2], axis=0)
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(M, λ)
    rho3 = np.full(M, 0.6) # As per coalescence_merge.py
    omega3 = np.full(M, 0.1) # As per coalescence_merge.py
    N3 = N1 + N2
    
    # Initial conditions for merged community
    C0_3 = np.concatenate([ce1, ce2])
    R0_3 = np.full(M, 1.0)
    
    sol3 = param.solve_micrm(N3, M, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    ce3 = sol3.y[:N3, -1]

    # --- Data Collection ---
    # Survival counts
    n_surv1 = int(np.sum(ce1 > SURV_THRESH))
    n_surv2 = int(np.sum(ce2 > SURV_THRESH))
    n_surv3 = int(np.sum(ce3 > SURV_THRESH))

    # Survival rates
    surv_rate1 = n_surv1 / float(N1) if N1 > 0 else 0.0
    surv_rate2 = n_surv2 / float(N2) if N2 > 0 else 0.0
    surv_rate3 = n_surv3 / float(N3) if N3 > 0 else 0.0

    # Store results in a dictionary
    result = {
        "Seed": seed,
        "N_modules": N_modules,
        "s_ratio": s_ratio,
        "lambda": λ,
        "N1": N1, "M": M, "N2": N2,
        "n_surv1": n_surv1,
        "n_surv2": n_surv2,
        "n_surv3": n_surv3,
        "surv_rate1": surv_rate1,
        "surv_rate2": surv_rate2,
        "surv_rate3": surv_rate3,
    }
    
    return result

if __name__ == "__main__":
    seeds_file = os.path.join(code_path, 'seeds.txt')
    with open(seeds_file, 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        results = pool.map(simulate, seeds)
    
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(data_dir, "coal_merge_hpc.csv"), index=False)

    print(f"Simulation complete. Results saved to {os.path.join(data_dir, 'coal_merge_hpc.csv')}")
