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
M_pool = 200    # Resource pool size
λ = 0.2        # Total leakage rate
s_ratio = 2   # Modularity ratio
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

# Loop through N_modules from 1 to 50
for N_modules in range(1, 51):

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
        'N_modules': N_modules,
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

# Plot 1: Richness vs N_modules
axes[0, 0].plot(df['N_modules'], df['Richness1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 0].plot(df['N_modules'], df['Richness2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 0].plot(df['N_modules'], df['Richness3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 0].set_title('Community Richness vs. Number of Modules')
axes[0, 0].set_xlabel('Number of Modules (N_modules)')
axes[0, 0].set_ylabel('Richness')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: CUE vs N_modules
axes[0, 1].plot(df['N_modules'], df['CUE1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 1].plot(df['N_modules'], df['CUE2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 1].plot(df['N_modules'], df['CUE3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 1].set_title('Community CUE vs. Number of Modules')
axes[0, 1].set_xlabel('Number of Modules (N_modules)')
axes[0, 1].set_ylabel('CUE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Niche Overlap vs N_modules
axes[0, 2].plot(df['N_modules'], df['Niche_overlap1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 2].plot(df['N_modules'], df['Niche_overlap2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 2].plot(df['N_modules'], df['Niche_overlap3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 2].set_title('Niche Overlap vs. Number of Modules')
axes[0, 2].set_xlabel('Number of Modules (N_modules)')
axes[0, 2].set_ylabel('Niche Overlap')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Plot 4: C_feed vs N_modules
axes[1, 0].plot(df['N_modules'], df['C_feed1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[1, 0].plot(df['N_modules'], df['C_feed2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[1, 0].plot(df['N_modules'], df['C_feed3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[1, 0].set_title('Community Feedback vs. Number of Modules')
axes[1, 0].set_xlabel('Number of Modules (N_modules)')
axes[1, 0].set_ylabel('C_feed')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 5: CUE vs C_feed correlation
axes[1, 1].scatter(df['C_feed1'], df['CUE1'], alpha=0.6, label='Community 1', color='red', s=50)
axes[1, 1].scatter(df['C_feed2'], df['CUE2'], alpha=0.6, label='Community 2', color='green', s=50)
axes[1, 1].scatter(df['C_feed3'], df['CUE3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue', s=50)
axes[1, 1].set_title('CUE vs Community Feedback')
axes[1, 1].set_xlabel('C_feed')
axes[1, 1].set_ylabel('CUE')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Plot 6: Niche Overlap vs C_feed correlation
axes[1, 2].scatter(df['Niche_overlap1'], df['C_feed1'], alpha=0.6, label='Community 1', color='red', s=50)
axes[1, 2].scatter(df['Niche_overlap2'], df['C_feed2'], alpha=0.6, label='Community 2', color='green', s=50)
axes[1, 2].scatter(df['Niche_overlap3'], df['C_feed3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue', s=50)
axes[1, 2].set_title('Niche Overlap vs C_feed')
axes[1, 2].set_xlabel('Niche Overlap')
axes[1, 2].set_ylabel('C_feed')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()

# Save the plots
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'N_modules_analysis_s2.png'), dpi=300, bbox_inches='tight')

print(f"Analysis plots saved to {os.path.join(results_dir, 'N_modules_analysis_s2.png')}")

# Print summary statistics
print("\n--- Summary Statistics ---")
print(f"Average CUE values:")
print(f"  Community 1: {df['CUE1'].mean():.4f} ± {df['CUE1'].std():.4f}")
print(f"  Community 2: {df['CUE2'].mean():.4f} ± {df['CUE2'].std():.4f}")
print(f"  Community 3: {df['CUE3'].mean():.4f} ± {df['CUE3'].std():.4f}")

print(f"\nAverage C_feed values:")
print(f"  Community 1: {df['C_feed1'].mean():.4f} ± {df['C_feed1'].std():.4f}")
print(f"  Community 2: {df['C_feed2'].mean():.4f} ± {df['C_feed2'].std():.4f}")
print(f"  Community 3: {df['C_feed3'].mean():.4f} ± {df['C_feed3'].std():.4f}")

print(f"\nAverage Niche Overlap values:")
print(f"  Community 1: {df['Niche_overlap1'].mean():.4f} ± {df['Niche_overlap1'].std():.4f}")
print(f"  Community 2: {df['Niche_overlap2'].mean():.4f} ± {df['Niche_overlap2'].std():.4f}")
print(f"  Community 3: {df['Niche_overlap3'].mean():.4f} ± {df['Niche_overlap3'].std():.4f}")

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
plt.plot(df['N_modules'], df['C_feed1'], marker='o', linestyle='-', label='Community 1')

# Plot for Community 2
plt.plot(df['N_modules'], df['C_feed2'], marker='o', linestyle='-', label='Community 2')

# Plot for Community 3
plt.plot(df['N_modules'], df['C_feed3'], marker='o', linestyle='-', label='Community 3 (Coalesced)')

plt.title('Community Feedback (C_feed) vs. Number of Modules')
plt.xlabel('Number of Modules (N_modules)')
plt.ylabel('Community Feedback (C_feed)')
plt.legend()
plt.grid(True)

# Save the C_feed plot
plt.savefig(os.path.join(results_dir, 'cfeed_vs_N_modules.png'))
print(f"C_feed plot saved to {os.path.join(results_dir, 'cfeed_vs_N_modules.png')}")

# Additional CUE-specific plot
plt.figure(figsize=(12, 8))

# Plot CUE for all communities
plt.plot(df['N_modules'], df['CUE1'], marker='o', linestyle='-', label='Community 1', color='red', linewidth=2)
plt.plot(df['N_modules'], df['CUE2'], marker='s', linestyle='-', label='Community 2', color='green', linewidth=2)
plt.plot(df['N_modules'], df['CUE3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue', linewidth=2)

plt.title('Community CUE vs. Number of Modules', fontsize=16)
plt.xlabel('Number of Modules (N_modules)', fontsize=14)
plt.ylabel('CUE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Save the CUE plot
plt.savefig(os.path.join(results_dir, 'CUE_vs_N_modules.png'), dpi=300, bbox_inches='tight')
print(f"CUE plot saved to {os.path.join(results_dir, 'CUE_vs_N_modules.png')}")

# Additional Niche Overlap vs C_feed plot
plt.figure(figsize=(12, 8))

# Plot Niche Overlap vs C_feed for all communities
plt.scatter(df['Niche_overlap1'], df['C_feed1'], alpha=0.7, label='Community 1', color='red', s=80)
plt.scatter(df['Niche_overlap2'], df['C_feed2'], alpha=0.7, label='Community 2', color='green', s=80)
plt.scatter(df['Niche_overlap3'], df['C_feed3'], alpha=0.7, label='Community 3 (Coalesced)', color='blue', s=80)

# Add trend lines
from scipy import stats
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df['Niche_overlap1'], df['C_feed1'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df['Niche_overlap2'], df['C_feed2'])
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df['Niche_overlap3'], df['C_feed3'])

x_range = np.linspace(df['Niche_overlap1'].min(), df['Niche_overlap1'].max(), 100)
plt.plot(x_range, slope1 * x_range + intercept1, '--', color='red', alpha=0.8, linewidth=2)
plt.plot(x_range, slope2 * x_range + intercept2, '--', color='green', alpha=0.8, linewidth=2)
plt.plot(x_range, slope3 * x_range + intercept3, '--', color='blue', alpha=0.8, linewidth=2)

plt.title('Niche Overlap vs Community Feedback', fontsize=16)
plt.xlabel('Niche Overlap', fontsize=14)
plt.ylabel('C_feed', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation coefficients to the plot
plt.text(0.05, 0.95, f'Community 1: r = {correlation_niche_feed1:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='red')
plt.text(0.05, 0.90, f'Community 2: r = {correlation_niche_feed2:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='green')
plt.text(0.05, 0.85, f'Community 3: r = {correlation_niche_feed3:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, color='blue')

# Save the Niche Overlap vs C_feed plot
plt.savefig(os.path.join(results_dir, 'Niche_overlap_vs_C_feed.png'), dpi=300, bbox_inches='tight')
print(f"Niche Overlap vs C_feed plot saved to {os.path.join(results_dir, 'Niche_overlap_vs_C_feed.png')}")

# Save data to CSV for further analysis
data_file = os.path.join(results_dir, 'N_modules_analysis_data_s2.csv')
df.to_csv(data_file, index=False)
print(f"Data saved to {data_file}")