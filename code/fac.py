import numpy as np
import os
import sys

# Ensure the repository's `code` directory (where this file lives) is on sys.path
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)

import param
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Parameter settings
np.random.seed(37)
N_pool = 200  # Species pool size
M_pool = 200   # Resource pool size

# Coalescence parameterization
λ_max = 0.2
λ_min = 0.01
s_ratio_max = 10

N1 = 100
M1 = 100
N2 = 100
M2 = 100

# Time span for simulation
t_span = (0, 5000)
SURV_THRESH = 1e-5

results_list = []

# Fixed parameters
λ = 0.2
s_ratio = 10
N_modules = M1
print(f"λ={λ}, s_ratio={s_ratio}, N_modules={N_modules}")

# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)

# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.6)
omega_pool = np.full(M_pool, 0.1)

# Define matching percentages: 20 gradients from 100% to 0%
matching_percentages = np.linspace(1.0, 0.0, 20)

def build_resource_indices(species_indices, M, match_pct):
    """Build resource indices based on species indices and match percentage."""
    if match_pct == 1.0:
        return species_indices[:M].copy()
    elif match_pct == 0.0:
        remaining = np.setdiff1d(np.arange(M_pool), species_indices)
        return np.random.choice(remaining, M, replace=False)
    else:
        n_match = int(M * match_pct)
        matched = np.random.choice(species_indices, n_match, replace=False)
        remaining = np.setdiff1d(np.arange(M_pool), species_indices)
        different = np.random.choice(remaining, M - n_match, replace=False)
        resource_indices = np.concatenate([matched, different])
        np.random.shuffle(resource_indices)
        return resource_indices

for match_pct in matching_percentages:
    print(f"\n=== Processing match percentage: {match_pct*100:.0f}% ===")
    
    # Randomly select species indices for both communities
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    species_indices2 = np.random.choice(N_pool, N2, replace=False)
    
    # Build resource indices based on species indices and match percentage
    resource_indices1 = build_resource_indices(species_indices1, M1, match_pct)
    resource_indices2 = build_resource_indices(species_indices2, M2, match_pct)
    
    # Print overlap statistics
    overlap1 = len(np.intersect1d(species_indices1, resource_indices1))
    overlap2 = len(np.intersect1d(species_indices2, resource_indices2))
    print(f"  Community1 - Species-Resource overlap: {overlap1} / {min(N1, M1)}")
    print(f"  Community2 - Species-Resource overlap: {overlap2} / {min(N2, M2)}")
    
    # Community 1
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = param.generate_l_tensor(N1, M1, N_modules, s_ratio, λ, u1)
    lambda_alpha1 = np.full(M1, λ)
    rho1 = rho_pool[resource_indices1]
    omega1 = omega_pool[resource_indices1]
    
    # Community 2
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = param.generate_l_tensor(N2, M2, N_modules, s_ratio, λ, u2)
    lambda_alpha2 = np.full(M2, λ)
    rho2 = rho_pool[resource_indices2]
    omega2 = omega_pool[resource_indices2]
    
    # Simulate Community 1
    C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
    C0_2 = np.full(N2, 0.01) 
    R0_1 = np.full(M1, 1)
    R0_2 = np.full(M2, 1)
    # Compute maintenance costs for community 1
    m1 = 0.2
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
    
    # Simulate Community 2
    # Compute maintenance costs for community 2
    m2 = 0.2
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    ce2 = sol2.y[:N2, -1]
    
    # Merge into Community 3
    # Use union of resource indices
    resource_indices3 = np.unique(np.concatenate([resource_indices1, resource_indices2]))
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    
    # Get unique species (in case of overlap)
    unique_species_indices3, unique_idx = np.unique(species_indices3, return_index=True)
    
    # Build uptake matrix for merged community
    u3 = u_pool[np.ix_(unique_species_indices3, resource_indices3)]
    
    # Compute maintenance costs for merged community
    m3 = 0.2
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    N3 = len(unique_species_indices3)
    M3 = len(resource_indices3)
    rho3 = rho_pool[resource_indices3]
    
    # Initial conditions for merged community
    # Map equilibrium abundances to merged community
    C0_3 = np.full(N3, 0.01)
    for i, species_idx in enumerate(unique_species_indices3):
        if species_idx in species_indices1:
            idx_in_comm1 = np.where(species_indices1 == species_idx)[0][0]
            C0_3[i] = ce1[idx_in_comm1]
        elif species_idx in species_indices2:
            idx_in_comm2 = np.where(species_indices2 == species_idx)[0][0]
            C0_3[i] = ce2[idx_in_comm2]
    
    R0_3 = np.full(M3, 1)
    l3 = param.generate_l_tensor(N3, M3, N_modules, s_ratio, λ, u3)
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    ce3 = sol3.y[:N3, -1]
    
    # Calculate C_feed for each community
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    C_feed1 = param.calculate_community_feedback(L_eff1, u1)
    L_eff1_mean = float(np.mean(L_eff1))

    L_eff2 = param.calculate_effective_leakage(u2, l2)
    C_feed2 = param.calculate_community_feedback(L_eff2, u2)
    L_eff2_mean = float(np.mean(L_eff2))

    L_eff3 = param.calculate_effective_leakage(u3, l3)
    C_feed3 = param.calculate_community_feedback(L_eff3, u3)
    L_eff3_mean = float(np.mean(L_eff3))
    print(f"  C_feed1={C_feed1:.3f}, C_feed2={C_feed2:.3f}, C_feed3={C_feed3:.3f}")
    
    # Calculate CUE for each community
    community_CUE1, species_CUE1 = param.compute_CUE(sol1, N1, u1, R0_1, λ, m1)
    community_CUE2, species_CUE2 = param.compute_CUE(sol2, N2, u2, R0_2, λ, m2)
    community_CUE3, species_CUE3 = param.compute_CUE(sol3, N3, u3, R0_3, λ, m3)
    
    # Calculate community-level competition for each community
    competition1 = param.community_level_competition(u1)
    competition2 = param.community_level_competition(u2)
    competition3 = param.community_level_competition(u3)
    
    # Counts
    n_surv1 = int(np.sum(ce1 > SURV_THRESH))
    n_surv2 = int(np.sum(ce2 > SURV_THRESH))
    n_surv3 = int(np.sum(ce3 > SURV_THRESH))
    
    results_list.append({
        'match_pct': float(match_pct),
        'N_modules': int(N_modules),
        'Richness1': n_surv1,
        'Richness2': n_surv2,
        'Richness3': n_surv3,
        'C_feed1': C_feed1,
        'C_feed2': C_feed2,
        'C_feed3': C_feed3,
        'L_eff1': L_eff1_mean,
        'L_eff2': L_eff2_mean,
        'L_eff3': L_eff3_mean,
        'CUE1': community_CUE1,
        'CUE2': community_CUE2,
        'CUE3': community_CUE3,
        'Competition1': competition1,
        'Competition2': competition2,
        'Competition3': competition3,
    })


# --- Save Results ---
df = pd.DataFrame(results_list)

# Save the data
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Save data to CSV
data_file = os.path.join(results_dir, 'fac.csv')
df.to_csv(data_file, index=False)
print(f"\nData saved to {data_file}")
print(f"Total simulations: {len(df)}")
