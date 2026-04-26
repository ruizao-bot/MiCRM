"""
Mass Balance Verification for MiCRM Communities at Equilibrium

At equilibrium, all fluxes must balance:
1. For each resource: ρ = ω*R + consumption - leakage_input
2. For each species: uptake*(1-λ) = maintenance + growth
3. Total carbon input = Total carbon output
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set working directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Simulation parameters (from main.py)
N1, M1 = 100, 50
N2, M2 = 100, 50
LEAKAGE_RATE = 0.2
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
SURVIVAL_THRESHOLD = 1e-5

# Configure matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def set_base_theme(ax, grid=False):
    """Apply base theme to matplotlib axes"""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)
    ax.tick_params(axis="both", width=0.4, colors="black", pad=4)
    if grid:
        ax.grid(True, which="major", color="#E5E5E5", linewidth=0.35)
    else:
        ax.grid(False)

# Palette
pal_rgb = {"1": "#E74C3C", "2": "#2ECC71", "3": "#3498DB"}
community_labels = {"1": "Parent 1", "2": "Parent 2", "3": "Coalesced"}

print("Loading data and computing mass balance...")

# Load data
df = pd.read_csv("data/coal.csv")

# Analyze one seed per community as example
example_seed = df['Seed'].iloc[0]

results = []

for comm in [1, 2, 3]:
    df_comm = df[(df['Seed'] == example_seed) & (df['Community'] == comm)]
    
    if len(df_comm) == 0:
        continue
    
    # Get community parameters
    if comm == 1:
        N, M = N1, M1
    elif comm == 2:
        N, M = N2, M2
    else:
        N, M = N1 + N2, max(M1, M2)
    
    # Extract equilibrium state
    C_final = df_comm['Abundance'].values  # (N,)
    total_biomass = df_comm['Total_Abundance'].iloc[0]
    R_final_sum = df_comm['Depletion'].iloc[0]  # Sum of all R
    R_final_avg = R_final_sum / M  # Average resource concentration
    
    # Get species-level data
    species_CUE = df_comm['Species_CUE'].values
    
    # Identify survivors
    survivors = C_final > SURVIVAL_THRESHOLD
    n_survivors = np.sum(survivors)
    
    # ========================================================================
    # 1. CARBON INPUT (per unit time at equilibrium)
    # ========================================================================
    # Resource supply flux
    carbon_input_supply = M * RHO_VALUE
    
    # ========================================================================
    # 2. CARBON OUTPUT (per unit time at equilibrium)
    # ========================================================================
    # Resource dilution/turnover
    carbon_output_dilution = OMEGA_VALUE * R_final_sum
    
    # Maintenance metabolism (respiration)
    carbon_output_maintenance = MAINTENANCE_COST * total_biomass
    
    # Direct loss during uptake (λ fraction is lost, not recycled via leakage)
    # At equilibrium: uptake ≈ maintenance / (1-λ) for survivors
    # Total uptake = Σ_i Σ_k (C_i * u_ik * R_k)
    # Lost fraction = λ * Total uptake
    # Since we don't have individual R_k, we estimate from equilibrium condition
    total_uptake_est = (MAINTENANCE_COST * np.sum(C_final[survivors])) / (1 - LEAKAGE_RATE)
    carbon_output_uptake_loss = LEAKAGE_RATE * total_uptake_est
    
    # Total output
    carbon_output_total = carbon_output_dilution + carbon_output_maintenance + carbon_output_uptake_loss
    
    # ========================================================================
    # 3. MASS BALANCE CHECK
    # ========================================================================
    balance = carbon_input_supply - carbon_output_total
    balance_percent = 100 * balance / carbon_input_supply
    
    # ========================================================================
    # 4. EFFECTIVE CUE CHECK
    # ========================================================================
    # At equilibrium, dC/dt = 0, so:
    # C * (uptake*(1-λ) - m) = 0
    # For surviving species: uptake*(1-λ) = m
    # 
    # Total uptake rate = Total maintenance / (1-λ) for survivors
    # Effective system CUE = (Net growth + Maintenance) / Total uptake
    #                      = Maintenance / Total uptake  (since net growth = 0)
    
    # Observed CUE (from data)
    community_CUE_obs = df_comm['Community_CUE_surv'].iloc[0]
    
    # Expected CUE at equilibrium (for survivors)
    # CUE = (uptake*(1-λ) - m) / uptake = 0 for perfect equilibrium
    # But measured CUE is based on reference R0=1, not R_final
    
    # ========================================================================
    # 5. CARBON STOCKS vs FLOWS
    # ========================================================================
    total_carbon_stock = total_biomass + R_final_sum
    turnover_time_resources = R_final_sum / (OMEGA_VALUE * R_final_sum + 1e-12)
    
    results.append({
        'Community': comm,
        'N_species': N,
        'N_survivors': n_survivors,
        'M_resources': M,
        'Total_Biomass': total_biomass,
        'Total_Resources': R_final_sum,
        'Avg_Resource': R_final_avg,
        'Carbon_Input': carbon_input_supply,
        'Carbon_Output': carbon_output_total,
        'Output_Dilution': carbon_output_dilution,
        'Output_Maintenance': carbon_output_maintenance,
        'Output_Uptake_Loss': carbon_output_uptake_loss,
        'Balance': balance,
        'Balance_Percent': balance_percent,
        'Community_CUE': community_CUE_obs,
        'Total_Uptake_Est': total_uptake_est
    })

# Summary across all seeds
all_seeds_results = []

for seed in df['Seed'].unique():
    for comm in [1, 2, 3]:
        df_comm = df[(df['Seed'] == seed) & (df['Community'] == comm)]
        
        if len(df_comm) == 0:
            continue
        
        if comm == 1:
            M = M1
        elif comm == 2:
            M = M2
        else:
            M = max(M1, M2)
        
        total_biomass = df_comm['Total_Abundance'].iloc[0]
        R_final_sum = df_comm['Depletion'].iloc[0]
        C_final = df_comm['Abundance'].values
        survivors = C_final > SURVIVAL_THRESHOLD
        
        carbon_input = M * RHO_VALUE
        total_uptake_est = (MAINTENANCE_COST * np.sum(C_final[survivors])) / (1 - LEAKAGE_RATE)
        carbon_output = (OMEGA_VALUE * R_final_sum + 
                        MAINTENANCE_COST * total_biomass + 
                        LEAKAGE_RATE * total_uptake_est)
        balance = carbon_input - carbon_output
        balance_percent = 100 * balance / carbon_input
        
        all_seeds_results.append({
            'Seed': seed,
            'Community': comm,
            'Carbon_Input': carbon_input,
            'Carbon_Output': carbon_output,
            'Balance': balance,
            'Balance_Percent': balance_percent
        })

summary_df = pd.DataFrame(all_seeds_results)
results_df = pd.DataFrame(results)

# Save detailed results
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/mass_balance_check.csv', index=False)

# ============================================================================
# CREATE VISUALIZATION: Depletion vs Total Abundance per community
# ============================================================================
os.makedirs('figure', exist_ok=True)

# Get one row per (Seed, Community) since Depletion and Total_Abundance are community-level
plot_df = df.drop_duplicates(subset=['Seed', 'Community'])[['Seed', 'Community', 'Depletion', 'Total_Abundance']]

fig, ax = plt.subplots(figsize=(6, 5))
set_base_theme(ax, grid=True)

for comm in [1, 2, 3]:
    data = plot_df[plot_df['Community'] == comm]
    ax.scatter(data['Depletion'], data['Total_Abundance'],
               s=40, alpha=0.7, color=pal_rgb[str(comm)],
               edgecolors='black', linewidths=0.3,
               label=community_labels[str(comm)])

    # Linear fit
    slope, intercept, r, p, _ = stats.linregress(data['Depletion'], data['Total_Abundance'])
    x_fit = np.linspace(data['Depletion'].min(), data['Depletion'].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color=pal_rgb[str(comm)], linewidth=1.5, linestyle='--')
    print(f"Community {comm} ({community_labels[str(comm)]}): slope = {slope:.4f}, R² = {r**2:.4f}")

ax.set_xlabel('Depletion (ΣR final)')
ax.set_ylabel('Total Abundance (ΣC final)')
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('figure/depletion_vs_abundance.png', dpi=300, bbox_inches='tight')
plt.savefig('figure/depletion_vs_abundance.pdf', bbox_inches='tight')
plt.close()
