import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the repository's `code` directory is on sys.path
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)

# Project root and results directory (absolute paths)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

import param

# --- Simulation Parameters ---
# Fixed parameters
np.random.seed(37)
N = 100         # Number of species
M = 50          # Number of resources
N_modules = 10   # Number of modules
λ = 0.2         # Total leakage rate

# List to store results
results = []

# --- Main Loop ---
# Loop through s_ratio from 1 to 20
s_ratio_range = range(1, 21)
for s_ratio in s_ratio_range:
    print(f"Processing s_ratio = {s_ratio}...")

    # 1. Generate uptake and leakage matrices for the current s_ratio
    u = param.modular_uptake(N, M, N_modules, s_ratio)
    l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ)

    # 2. Calculate the effective leakage matrix and average niche overlap
    L_eff = param.calculate_effective_leakage(u, l)
    avg_niche_overlap = param.average_cosine_similarity(u)

    # 3. Calculate the correlation between pairwise niche overlap and pairwise C_feed
    niche_overlap_pairs = []
    c_feed_pairs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Pairwise Niche Overlap (u_i vs u_j)
            u_i = u[i, :]
            u_j = u[j, :]
            norm_u_i = np.linalg.norm(u_i)
            norm_u_j = np.linalg.norm(u_j)
            if norm_u_i > 0 and norm_u_j > 0:
                niche_overlap_sim = np.dot(u_i, u_j) / (norm_u_i * norm_u_j)
                niche_overlap_pairs.append(niche_overlap_sim)

            # Pairwise C_feed (L_eff_i vs u_j)
            L_eff_i = L_eff[i, :]
            norm_L_eff_i = np.linalg.norm(L_eff_i)
            if norm_L_eff_i > 0 and norm_u_j > 0:
                c_feed_sim = np.dot(L_eff_i, u_j) / (norm_L_eff_i * norm_u_j)
                c_feed_pairs.append(c_feed_sim)

    # Ensure lists are of the same length before calculating correlation
    if len(niche_overlap_pairs) == len(c_feed_pairs) and len(niche_overlap_pairs) > 1:
        correlation_coef = np.corrcoef(niche_overlap_pairs, c_feed_pairs)[0, 1]
    else:
        correlation_coef = np.nan

    # 4. Store the results for this s_ratio
    results.append({
        's_ratio': s_ratio,
        'avg_niche_overlap': avg_niche_overlap,
        'niche_cfeed_correlation': correlation_coef
    })

# --- Plotting ---
# Convert results to a DataFrame for easy plotting
df_results = pd.DataFrame(results)

# Create a combined plot for the correlation and niche overlap
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Niche Overlap vs C_feed correlation
ax1.plot(df_results['s_ratio'], df_results['niche_cfeed_correlation'], marker='s', linestyle='--', color='r', label='Correlation (Niche Overlap vs C_feed)')
ax1.set_xlabel('Modularity Ratio (s_ratio)')
ax1.set_ylabel('Pearson Correlation Coef.', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(True)
ax1.set_xticks(s_ratio_range)

# Create a second y-axis for average niche overlap
ax2 = ax1.twinx()
ax2.plot(df_results['s_ratio'], df_results['avg_niche_overlap'], marker='^', linestyle=':', color='g', label='Avg. Niche Overlap')
ax2.set_ylabel('Avg. Niche Overlap', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Title and combined legend
fig.suptitle('Effect of Modularity (s_ratio) on Niche Overlap and C_feed Correlation', fontsize=16)
# Collect all handles and labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# Save the plot
output_path = os.path.join(results_dir, "s_ratio_correlations_combined.png")
plt.savefig(output_path, bbox_inches='tight')

print(f"\nAnalysis complete. Plot saved to {output_path}")
