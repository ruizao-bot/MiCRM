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
N_pool = 1000  # Species pool size
M_pool = 200   # Resource pool size

# Coalescence parameterization tied to b (per coalescence.py)
λ_max = 0.3
λ_min = 0.0
s_ratio_max = 10

N1 = 100
M1 = 50
m1 = np.full(N1, 0.2)
N2 = 100
M2 = 50
m2 = np.full(N2, 0.2)

# Time span for simulation
t_span = (0, 5000)
SURV_THRESH = 1e-5

results_list = []

# Loop b from 0 to 1 (inclusive)
for b in np.linspace(0.1, 1.0, 51):

    # derive λ, s_ratio, N_modules from b (as in coalescence.py)
    λ = λ_min + (λ_max - λ_min) * (1 - b)
    s_ratio = 1 + (s_ratio_max - 1) * (1 - b)
    N_modules = round(max(1, M1 * (1 - b**2)))
    print(f"b={b:.3f}, s_ratio={s_ratio:.3f}, N_modules={N_modules}")

    # Generate uptake matrix and leakage tensor for the species pool
    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)
    
    # Set rho and omega for the resource pool
    rho_pool = np.full(M_pool, 0.6)
    omega_pool = np.full(M_pool, 0.1)
    
    # Community 1
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = param.generate_l_tensor(N1, M1, N_modules, s_ratio, λ, u1)
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
    l2 = param.generate_l_tensor(N2, M2, N_modules, s_ratio, λ, u2)
    lambda_alpha2 = np.full(M2, λ)
    rho2 = rho_pool[resource_indices2]
    omega2 = omega_pool[resource_indices2]
    
    # Simulate Community 1
    C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
    C0_2 = np.full(N1, 0.01) 
    R0_1 = np.full(M1, 1)
    R0_2 = np.full(M2, 1)
    
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
    
    # Simulate Community 2
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    ce2 = sol2.y[:N2, -1]
    
    # Merge into Community 3
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    rho3 = rho_pool[resource_indices3]
    C0_3 = np.concatenate([ce1, ce2])
    R0_3 = np.full(M1, 1)
    l3 = param.generate_l_tensor(N3, M3, N_modules, s_ratio, λ, u3)
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    ce3 = sol3.y[:N3, -1]
    # Calculate C_feed for each community
    L_eff1 = param.calculate_effective_leakage(u1,l1)
    C_feed1 = param.calculate_community_feedback(L_eff1, u1)

    L_eff2 = param.calculate_effective_leakage(u2,l2)
    C_feed2 = param.calculate_community_feedback(L_eff2, u2)

    L_eff3 = param.calculate_effective_leakage(u3,l3)
    C_feed3 = param.calculate_community_feedback(L_eff3, u3)
    print(f"  C_feed1={C_feed1:.3f}, C_feed2={C_feed2:.3f}, C_feed3={C_feed3:.3f}")
    
    # Calculate CUE for each community
    community_CUE1, species_CUE1 = param.compute_CUE(sol1, N1, u1, R0_1, λ, m1)
    community_CUE2, species_CUE2 = param.compute_CUE(sol2, N2, u2, R0_2, λ, m2)
    community_CUE3, species_CUE3 = param.compute_CUE(sol3, N3, u3, R0_3, λ, m3)
    
    # Calculate community-level competition for each community
    competition1 = param.community_level_competition(u1)
    competition2 = param.community_level_competition(u2)
    competition3 = param.community_level_competition(u3)
    
    # counts
    n_surv1 = int(np.sum(ce1 > SURV_THRESH))
    n_surv2 = int(np.sum(ce2 > SURV_THRESH))
    n_surv3 = int(np.sum(ce3 > SURV_THRESH))
    
    results_list.append({
        'b': float(b),
        'N_modules': int(N_modules),
        'Richness1': n_surv1,
        'Richness2': n_surv2,
        'Richness3': n_surv3,
        'C_feed1': C_feed1,
        'C_feed2': C_feed2,
        'C_feed3': C_feed3,
        'CUE1': community_CUE1,
        'CUE2': community_CUE2,
        'CUE3': community_CUE3,
        'Competition1': competition1,
        'Competition2': competition2,
        'Competition3': competition3,
    })


# --- Plotting ---
df = pd.DataFrame(results_list)

# Create subplots for multiple analyses
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Richness vs b
axes[0, 0].plot(df['b'], df['Richness1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 0].plot(df['b'], df['Richness2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 0].plot(df['b'], df['Richness3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 0].set_title('Community Richness vs. Degree of looseness')
axes[0, 0].set_xlabel('Degree of looseness')
axes[0, 0].set_ylabel('Richness')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: CUE vs b
axes[0, 1].plot(df['b'], df['CUE1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 1].plot(df['b'], df['CUE2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 1].plot(df['b'], df['CUE3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 1].set_title('Community CUE vs. Degree of looseness')
axes[0, 1].set_xlabel('Degree of looseness')
axes[0, 1].set_ylabel('CUE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Competition vs b
axes[0, 2].plot(df['b'], df['Competition1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 2].plot(df['b'], df['Competition2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 2].plot(df['b'], df['Competition3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 2].set_title('Community Competition vs. Degree of looseness')
axes[0, 2].set_xlabel('Degree of looseness')
axes[0, 2].set_ylabel('Competition')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Plot 4: C_feed vs b
axes[1, 0].plot(df['b'], df['C_feed1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[1, 0].plot(df['b'], df['C_feed2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[1, 0].plot(df['b'], df['C_feed3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[1, 0].set_title('Facilitation vs. Degree of looseness')
axes[1, 0].set_xlabel('Degree of looseness')
axes[1, 0].set_ylabel('Facilitation')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 5: CUE vs C_feed correlation
axes[1, 1].scatter(df['C_feed1'], df['CUE1'], alpha=0.6, label='Community 1', color='red', s=50)
axes[1, 1].scatter(df['C_feed2'], df['CUE2'], alpha=0.6, label='Community 2', color='green', s=50)
axes[1, 1].scatter(df['C_feed3'], df['CUE3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue', s=50)
axes[1, 1].set_title('CUE vs Facilitation')
axes[1, 1].set_xlabel('Facilitation')
axes[1, 1].set_ylabel('CUE')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Plot 6: CUE vs Competition
axes[1, 2].scatter(df['Competition1'], df['CUE1'], alpha=0.6, label='Community 1', color='red', s=50)
axes[1, 2].scatter(df['Competition2'], df['CUE2'], alpha=0.6, label='Community 2', color='green', s=50)
axes[1, 2].scatter(df['Competition3'], df['CUE3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue', s=50)
axes[1, 2].set_title('CUE vs Community Competition')
axes[1, 2].set_xlabel('Competition')
axes[1, 2].set_ylabel('CUE')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()

# Save the plots
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)
plot_file = os.path.join(results_dir, 'b_analysis_0.2.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')

print(f"Analysis plots saved to {plot_file}")

# Save data to CSV for further analysis
data_file = os.path.join(results_dir, 'b_analysis_data_0.2.csv')
df.to_csv(data_file, index=False)
print(f"Data saved to {data_file}")