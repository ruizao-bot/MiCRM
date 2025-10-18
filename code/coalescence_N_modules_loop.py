import numpy as np
import os
import sys

# Ensure the repository's `code` directory (where this file lives) is on sys.path
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)

import param
import pandas as pd
import matplotlib.pyplot as plt

# Parameter settings
np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 200   # Resource pool size

# Coalescence parameterization tied to b (per coalescence.py)
λ_max = 1
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
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
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
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    rho3 = rho_pool[resource_indices3]
    C0_3 = np.concatenate([ce1, ce2])
    R0_3 = np.full(M1, 1)
    
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    ce3 = sol3.y[:N3, -1]
    # Calculate C_feed for each community
    L_eff1 = param.calculate_effective_leakage(u1,l1)
    C_feed1 = param.calculate_community_feedback(L_eff1, u1)

    L_eff2 = param.calculate_effective_leakage(u2,l2)
    C_feed2 = param.calculate_community_feedback(L_eff2, u2)

    L_eff3 = param.calculate_effective_leakage(u3,l3)
    C_feed3 = param.calculate_community_feedback(L_eff3, u3)
    
    # Calculate CUE for each community
    community_CUE1, species_CUE1 = param.compute_CUE(sol1, N1, u1, R0_1, l1, m1)
    community_CUE2, species_CUE2 = param.compute_CUE(sol2, N2, u2, R0_2, l2, m2)
    community_CUE3, species_CUE3 = param.compute_CUE(sol3, N3, u3, R0_3, l3, m3)
    
    # Calculate niche overlap for each community
    niche_overlap1 = param.average_cosine_similarity(u1)
    niche_overlap2 = param.average_cosine_similarity(u2)
    niche_overlap3 = param.average_cosine_similarity(u3)
    
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
        'Niche_overlap1': niche_overlap1,
        'Niche_overlap2': niche_overlap2,
        'Niche_overlap3': niche_overlap3,
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

# Plot 3: Niche Overlap vs b
axes[0, 2].plot(df['b'], df['Niche_overlap1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 2].plot(df['b'], df['Niche_overlap2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 2].plot(df['b'], df['Niche_overlap3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 2].set_title('Niche Overlap vs. Degree of looseness')
axes[0, 2].set_xlabel('Degree of looseness')
axes[0, 2].set_ylabel('Niche Overlap')
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

# Plot 6: CUE vs Niche Overlap
axes[1, 2].scatter(df['Niche_overlap1'], df['CUE1'], alpha=0.6, label='Community 1', color='red', s=50)
axes[1, 2].scatter(df['Niche_overlap2'], df['CUE2'], alpha=0.6, label='Community 2', color='green', s=50)
axes[1, 2].scatter(df['Niche_overlap3'], df['CUE3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue', s=50)
axes[1, 2].set_title('CUE vs Niche Overlap')
axes[1, 2].set_xlabel('Niche Overlap')
axes[1, 2].set_ylabel('CUE')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()

# Save the plots
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'b_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Analysis plots saved to {os.path.join(results_dir, 'b_analysis.png')}")

# Calculate correlations
correlation_cue_feed1 = df['CUE1'].corr(df['C_feed1'])
correlation_cue_feed2 = df['CUE2'].corr(df['C_feed2'])
correlation_cue_feed3 = df['CUE3'].corr(df['C_feed3'])

correlation_niche_cue1 = df['Niche_overlap1'].corr(df['CUE1'])
correlation_niche_cue2 = df['Niche_overlap2'].corr(df['CUE2'])
correlation_niche_cue3 = df['Niche_overlap3'].corr(df['CUE3'])

correlation_niche_feed1 = df['Niche_overlap1'].corr(df['C_feed1'])
correlation_niche_feed2 = df['Niche_overlap2'].corr(df['C_feed2'])
correlation_niche_feed3 = df['Niche_overlap3'].corr(df['C_feed3'])

print(f"\nCUE vs C_feed correlations:")
print(f"  Community 1: {correlation_cue_feed1:.4f}")
print(f"  Community 2: {correlation_cue_feed2:.4f}")
print(f"  Community 3: {correlation_cue_feed3:.4f}")

print(f"\nNiche Overlap vs CUE correlations:")
print(f"  Community 1: {correlation_niche_cue1:.4f}")
print(f"  Community 2: {correlation_niche_cue2:.4f}")
print(f"  Community 3: {correlation_niche_cue3:.4f}")

print(f"\nNiche Overlap vs C_feed correlations:")
print(f"  Community 1: {correlation_niche_feed1:.4f}")
print(f"  Community 2: {correlation_niche_feed2:.4f}")
print(f"  Community 3: {correlation_niche_feed3:.4f}")

# Plotting C_feed
plt.figure(figsize=(12, 8))

# Plot for Community 1
plt.plot(df['b'], df['C_feed1'], marker='o', linestyle='-', label='Community 1')

# Plot for Community 2
plt.plot(df['b'], df['C_feed2'], marker='o', linestyle='-', label='Community 2')

# Plot for Community 3
plt.plot(df['b'], df['C_feed3'], marker='o', linestyle='-', label='Community 3 (Coalesced)')

plt.title('Facilitation vs. Guild Structure')
plt.xlabel('Guild Structure')
plt.ylabel('Facilitation')
plt.legend()
plt.grid(True)

# Save the C_feed plot
plt.savefig(os.path.join(results_dir, 'cfeed_vs_b.png'))
print(f"Facilitation plot saved to {os.path.join(results_dir, 'cfeed_vs_b.png')}")

# Additional CUE-specific plot
plt.figure(figsize=(12, 8))

# Plot CUE for all communities
plt.plot(df['b'], df['CUE1'], marker='o', linestyle='-', label='Community 1', color='red', linewidth=2)
plt.plot(df['b'], df['CUE2'], marker='s', linestyle='-', label='Community 2', color='green', linewidth=2)
plt.plot(df['b'], df['CUE3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue', linewidth=2)

plt.title('Community CUE vs. Guild Structure', fontsize=16)
plt.xlabel('Guild Structure', fontsize=14)
plt.ylabel('CUE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Save the CUE plot
plt.savefig(os.path.join(results_dir, 'CUE_vs_b.png'), dpi=300, bbox_inches='tight')
print(f"CUE plot saved to {os.path.join(results_dir, 'CUE_vs_b.png')}")

# --- CUE vs Niche Overlap plot ---
plt.figure(figsize=(12, 8))
plt.scatter(df['Niche_overlap1'], df['CUE1'], alpha=0.7, label='Community 1', color='red', s=80)
plt.scatter(df['Niche_overlap2'], df['CUE2'], alpha=0.7, label='Community 2', color='green', s=80)
plt.scatter(df['Niche_overlap3'], df['CUE3'], alpha=0.7, label='Community 3 (Coalesced)', color='blue', s=80)

# Add trend lines
from scipy import stats
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df['Niche_overlap1'], df['CUE1'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df['Niche_overlap2'], df['CUE2'])
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df['Niche_overlap3'], df['CUE3'])
x_range = np.linspace(min(df['Niche_overlap1'].min(), df['Niche_overlap2'].min(), df['Niche_overlap3'].min()),
                      max(df['Niche_overlap1'].max(), df['Niche_overlap2'].max(), df['Niche_overlap3'].max()), 100)
plt.plot(x_range, slope1 * x_range + intercept1, '--', color='red', alpha=0.8, linewidth=2)
plt.plot(x_range, slope2 * x_range + intercept2, '--', color='green', alpha=0.8, linewidth=2)
plt.plot(x_range, slope3 * x_range + intercept3, '--', color='blue', alpha=0.8, linewidth=2)

plt.title('CUE vs Niche Overlap', fontsize=16)
plt.xlabel('Niche Overlap', fontsize=14)
plt.ylabel('CUE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation coefficients to the plot
plt.text(0.05, 0.95, f'Community 1: r = {r_value1:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='red')
plt.text(0.05, 0.90, f'Community 2: r = {r_value2:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='green')
plt.text(0.05, 0.85, f'Community 3: r = {r_value3:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='blue')

plt.savefig(os.path.join(results_dir, 'CUE_vs_Niche_overlap.png'), dpi=300, bbox_inches='tight')
print(f"CUE vs Niche Overlap plot saved to {os.path.join(results_dir, 'CUE_vs_Niche_overlap.png')}")

# Histogram of facilitation values to inspect distribution across guild structures
plt.figure(figsize=(12, 8))
bins = max(10, int(np.sqrt(len(df))))
plt.hist(df['C_feed1'], bins=bins, alpha=0.6, label='Community 1', color='red', density=True)
plt.hist(df['C_feed2'], bins=bins, alpha=0.6, label='Community 2', color='green', density=True)
plt.hist(df['C_feed3'], bins=bins, alpha=0.6, label='Community 3 (Coalesced)', color='blue', density=True)
plt.title('Distribution of Facilitation (C_feed)', fontsize=16)
plt.xlabel('Facilitation (C_feed)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

dist_path = os.path.join(results_dir, 'C_feed_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"C_feed distribution plot saved to {dist_path}")

# Save data to CSV for further analysis
data_file = os.path.join(results_dir, 'b_analysis_data.csv')
df.to_csv(data_file, index=False)
print(f"Data saved to {data_file}")