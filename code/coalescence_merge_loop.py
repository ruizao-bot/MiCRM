import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Ensure the repository's `code` directory (where this file lives) is on sys.path
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)

# Project root and data directory (absolute paths)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

import param

# --- Simulation Parameters ---
# Fixed parameters
np.random.seed(37)
λ = 0.2        # Total leakage rate
s_ratio = 10   # Modularity ratio
N1 = 100
M = 50
m1 = np.full(N1, 0.2)
N2 = 100
m2 = np.full(N2, 0.2)
t_span = (0, 100000)
SURV_THRESH = 1e-5

# List to store results from each N_modules iteration
results_list = []

# --- Main Loop ---
# Loop through N_modules from 1 to 50
for N_modules in range(1, 51):
    print(f"Running simulation for N_modules = {N_modules}...")

    # --- Community Setup ---
    # Community 1
    u1 = param.modular_uptake(N1, M, N_modules, s_ratio)
    l1 = param.generate_l_tensor(N1, M, N_modules, s_ratio, λ)
    lambda_alpha1 = np.full(M, λ)
    rho1 = np.full(M, 1)
    omega1 = np.full(M, 1)

    # Community 2
    u2 = param.modular_uptake(N2, M, N_modules, s_ratio)
    l2 = param.generate_l_tensor(N2, M, N_modules, s_ratio, λ)
    lambda_alpha2 = np.full(M, λ)
    rho2 = np.full(M, 1)
    omega2 = np.full(M, 1)

    # --- Simulation ---
    # Initial conditions
    C0_1 = np.full(N1, 0.01)
    R0_1 = np.full(M, 1)
    C0_2 = np.full(N2, 0.01)
    R0_2 = np.full(M, 1)

    # Simulate Community 1
    sol1 = param.solve_micrm(N1, M, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    ce1 = sol1.y[:N1, -1]

    # Simulate Community 2
    sol2 = param.solve_micrm(N2, M, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    ce2 = sol2.y[:N2, -1]

    # --- Merge and Simulate Community 3 ---
    u3 = np.vstack([u1, u2])
    l3 = np.concatenate([l1, l2], axis=0)
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(M, λ)
    omega3 = np.full(M, 0.6)
    rho3 = np.full(M, 0.1)
    N3 = N1 + N2
    
    C0_3 = np.concatenate([ce1, ce2])
    R0_3 = np.full(M, 1)

    sol3 = param.solve_micrm(N3, M, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    ce3 = sol3.y[:N3, -1]

    # --- Analysis ---
    # Calculate survivor counts
    n_surv1 = int(np.sum(ce1 > SURV_THRESH))
    n_surv2 = int(np.sum(ce2 > SURV_THRESH))
    n_surv3 = int(np.sum(ce3 > SURV_THRESH))

    # Calculate C_feed for each community
    L_eff1 = param.calculate_effective_leakage(u1, l1)
    C_feed1 = param.calculate_community_feedback(L_eff1, u1)

    L_eff2 = param.calculate_effective_leakage(u2, l2)
    C_feed2 = param.calculate_community_feedback(L_eff2, u2)

    L_eff3 = param.calculate_effective_leakage(u3, l3)
    C_feed3 = param.calculate_community_feedback(L_eff3, u3)
    
    # Calculate CUE for each community
    community_CUE1, _ = param.compute_CUE(sol1, N1, u1, R0_1, l1, m1)
    community_CUE2, _ = param.compute_CUE(sol2, N2, u2, R0_2, l2, m2)
    community_CUE3, _ = param.compute_CUE(sol3, N3, u3, R0_3, l3, m3)
    
    # Calculate niche overlap for each community
    niche_overlap1 = param.average_cosine_similarity(u1)
    niche_overlap2 = param.average_cosine_similarity(u2)
    niche_overlap3 = param.average_cosine_similarity(u3)

    # Store results
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


# --- Save Results ---
# Convert list of dictionaries to a DataFrame
results_df = pd.DataFrame(results_list)

# Define paths
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)
csv_output_path = os.path.join(results_dir, "coalescence_merge_loop_data_100.csv")
plot_output_path = os.path.join(results_dir, "coalescence_merge_loop_analysis_100.png")

# Save DataFrame to a CSV file
results_df.to_csv(csv_output_path, index=False)
print(f"\nSimulations complete. Data saved to {csv_output_path}")

# --- Plotting ---
df = results_df

# Create subplots for multiple analyses
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Coalescence Merge Analysis vs. Number of Modules', fontsize=16)

# Plot 1: Richness vs N_modules
axes[0, 0].plot(df['N_modules'], df['Richness1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 0].plot(df['N_modules'], df['Richness2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 0].plot(df['N_modules'], df['Richness3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 0].set_title('Community Richness')
axes[0, 0].set_xlabel('Number of Modules')
axes[0, 0].set_ylabel('Richness')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: CUE vs N_modules
axes[0, 1].plot(df['N_modules'], df['CUE1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 1].plot(df['N_modules'], df['CUE2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 1].plot(df['N_modules'], df['CUE3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 1].set_title('Community CUE')
axes[0, 1].set_xlabel('Number of Modules')
axes[0, 1].set_ylabel('CUE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Niche Overlap vs N_modules
axes[0, 2].plot(df['N_modules'], df['Niche_overlap1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[0, 2].plot(df['N_modules'], df['Niche_overlap2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[0, 2].plot(df['N_modules'], df['Niche_overlap3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[0, 2].set_title('Niche Overlap')
axes[0, 2].set_xlabel('Number of Modules')
axes[0, 2].set_ylabel('Niche Overlap')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Plot 4: C_feed vs N_modules
axes[1, 0].plot(df['N_modules'], df['C_feed1'], marker='o', linestyle='-', label='Community 1', color='red')
axes[1, 0].plot(df['N_modules'], df['C_feed2'], marker='s', linestyle='-', label='Community 2', color='green')
axes[1, 0].plot(df['N_modules'], df['C_feed3'], marker='^', linestyle='-', label='Community 3 (Coalesced)', color='blue')
axes[1, 0].set_title('Community Feedback')
axes[1, 0].set_xlabel('Number of Modules')
axes[1, 0].set_ylabel('C_feed')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 5: CUE vs C_feed correlation
axes[1, 1].scatter(df['C_feed1'], df['CUE1'], alpha=0.6, label='Community 1', color='red')
axes[1, 1].scatter(df['C_feed2'], df['CUE2'], alpha=0.6, label='Community 2', color='green')
axes[1, 1].scatter(df['C_feed3'], df['CUE3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue')
axes[1, 1].set_title('CUE vs Community Feedback')
axes[1, 1].set_xlabel('C_feed')
axes[1, 1].set_ylabel('CUE')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Plot 6: Niche Overlap vs C_feed correlation
axes[1, 2].scatter(df['Niche_overlap1'], df['C_feed1'], alpha=0.6, label='Community 1', color='red')
axes[1, 2].scatter(df['Niche_overlap2'], df['C_feed2'], alpha=0.6, label='Community 2', color='green')
axes[1, 2].scatter(df['Niche_overlap3'], df['C_feed3'], alpha=0.6, label='Community 3 (Coalesced)', color='blue')
axes[1, 2].set_title('Niche Overlap vs C_feed')
axes[1, 2].set_xlabel('Niche Overlap')
axes[1, 2].set_ylabel('C_feed')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
print(f"Analysis plots saved to {plot_output_path}")

# --- Summary Statistics and Correlations ---
print("\n--- Summary Statistics ---")
print(f"Average CUE values:\n  Comm 1: {df['CUE1'].mean():.4f} ± {df['CUE1'].std():.4f}\n  Comm 2: {df['CUE2'].mean():.4f} ± {df['CUE2'].std():.4f}\n  Comm 3: {df['CUE3'].mean():.4f} ± {df['CUE3'].std():.4f}")
print(f"\nAverage C_feed values:\n  Comm 1: {df['C_feed1'].mean():.4f} ± {df['C_feed1'].std():.4f}\n  Comm 2: {df['C_feed2'].mean():.4f} ± {df['C_feed2'].std():.4f}\n  Comm 3: {df['C_feed3'].mean():.4f} ± {df['C_feed3'].std():.4f}")
print(f"\nAverage Niche Overlap values:\n  Comm 1: {df['Niche_overlap1'].mean():.4f} ± {df['Niche_overlap1'].std():.4f}\n  Comm 2: {df['Niche_overlap2'].mean():.4f} ± {df['Niche_overlap2'].std():.4f}\n  Comm 3: {df['Niche_overlap3'].mean():.4f} ± {df['Niche_overlap3'].std():.4f}")

# Calculate correlations
correlations = {
    'CUE_vs_C_feed': [df['CUE1'].corr(df['C_feed1']), df['CUE2'].corr(df['C_feed2']), df['CUE3'].corr(df['C_feed3'])],
    'Niche_vs_CUE': [df['Niche_overlap1'].corr(df['CUE1']), df['Niche_overlap2'].corr(df['CUE2']), df['Niche_overlap3'].corr(df['CUE3'])],
    'Niche_vs_C_feed': [df['Niche_overlap1'].corr(df['C_feed1']), df['Niche_overlap2'].corr(df['C_feed2']), df['Niche_overlap3'].corr(df['C_feed3'])]
}

print(f"\nCUE vs C_feed correlations:\n  Comm 1: {correlations['CUE_vs_C_feed'][0]:.4f}\n  Comm 2: {correlations['CUE_vs_C_feed'][1]:.4f}\n  Comm 3: {correlations['CUE_vs_C_feed'][2]:.4f}")
print(f"\nNiche Overlap vs CUE correlations:\n  Comm 1: {correlations['Niche_vs_CUE'][0]:.4f}\n  Comm 2: {correlations['Niche_vs_CUE'][1]:.4f}\n  Comm 3: {correlations['Niche_vs_CUE'][2]:.4f}")
print(f"\nNiche Overlap vs C_feed correlations:\n  Comm 1: {correlations['Niche_vs_C_feed'][0]:.4f}\n  Comm 2: {correlations['Niche_vs_C_feed'][1]:.4f}\n  Comm 3: {correlations['Niche_vs_C_feed'][2]:.4f}")
