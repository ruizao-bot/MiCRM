import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(42)

# Set working directory to project root (parent of code folder)
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Palette and theme setup
pal_rgb = {"1": "#E74C3C", "2": "#2ECC71", "3": "#3498DB"}
community_labels = {"1": "Parent 1", "2": "Parent 2", "3": "Coalesced"}

# Configure matplotlib style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['xtick.major.width'] = 0.3
plt.rcParams['ytick.major.width'] = 0.3
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

def set_base_theme(ax):
    """Apply base theme to matplotlib axes"""
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['right'].set_linewidth(0.3)
    ax.spines['top'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.grid(False)
    ax.tick_params(width=0.3, length=4)

# Load data
print("Loading data...")
df = pd.read_csv("data/coal.csv")

# Get community-level metrics (one value per seed per community)
print("Processing community-level metrics...")
community_metrics = df.groupby(['Seed', 'Community']).agg({
    'feasibility': 'first',  # Same for all species in a community
    'Leading_Eigenvalue': 'first',  # Same for all species in a community
    'N_Survivors': 'first'
}).reset_index()

# Convert Community to string for plotting
community_metrics['Community'] = community_metrics['Community'].astype(str)

community_metrics_nonzero = community_metrics[community_metrics['feasibility'] > 0].copy()

print(f"Total samples: {len(community_metrics)}")
print(f"Non-zero feasibility samples: {len(community_metrics_nonzero)}")
print(f"Percentage with non-zero feasibility: {100*len(community_metrics_nonzero)/len(community_metrics):.1f}%")

# Create output directory
os.makedirs('figure', exist_ok=True)

# ============================================================================
# Figure: Feasibility and Stability Comparison (Box plots)
# ============================================================================
print("Creating Feasibility and Stability comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: Feasibility comparison (non-zero values only)
ax = axes[0]
positions = [1, 2, 3]
for position, comm in zip(positions, ['1', '2', '3']):
    values = community_metrics_nonzero[
        community_metrics_nonzero['Community'] == comm
    ]['feasibility'].values
    if len(values) == 0:
        continue
    jitter = RNG.uniform(-0.12, 0.12, size=len(values))
    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        s=36,
        alpha=0.8,
        color=pal_rgb[comm],
        edgecolors='black',
        linewidths=0.3,
    )

ax.set_xticks(positions)
ax.set_xticklabels([community_labels[c] for c in ['1', '2', '3']])
ax.set_ylabel('Feasibility', fontweight='bold')
ax.set_xlabel('Community', fontweight='bold')
ax.set_title('(A) Feasibility Comparison', loc='left')
ax.set_yscale('log')
set_base_theme(ax)

# Panel B: Stability (Leading Eigenvalue) comparison
ax = axes[1]
for position, comm in zip(positions, ['1', '2', '3']):
    values = community_metrics[
        community_metrics['Community'] == comm
    ]['Leading_Eigenvalue'].values
    if len(values) == 0:
        continue
    jitter = RNG.uniform(-0.12, 0.12, size=len(values))
    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        s=24,
        alpha=0.75,
        color=pal_rgb[comm],
        edgecolors='black',
        linewidths=0.3,
    )

ax.set_xticks(positions)
ax.set_xticklabels([community_labels[c] for c in ['1', '2', '3']])
ax.set_ylabel('Leading Eigenvalue', fontweight='bold')
ax.set_xlabel('Community', fontweight='bold')
ax.set_title('(B) Stability Comparison', loc='left')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
set_base_theme(ax)

plt.tight_layout()
plt.savefig('figure/feasibility_stability_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure/feasibility_stability_comparison.pdf', bbox_inches='tight')
print("Saved: figure/feasibility_stability_comparison.png/pdf")
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for comm in ['1', '2', '3']:
    data_all = community_metrics[community_metrics['Community'] == comm]
    data_nonzero = community_metrics_nonzero[community_metrics_nonzero['Community'] == comm]
    print(f"\n{community_labels[comm]}:")
    print(f"  Total samples: {len(data_all)}, Non-zero: {len(data_nonzero)} ({100*len(data_nonzero)/len(data_all):.1f}%)")
    print(f"  Feasibility (all): {data_all['feasibility'].mean():.4e} ± {data_all['feasibility'].std():.4e}")
    if len(data_nonzero) > 0:
        print(f"  Feasibility (non-zero): {data_nonzero['feasibility'].mean():.4e} ± {data_nonzero['feasibility'].std():.4e}")
    print(f"  Leading Eigenvalue: {data_all['Leading_Eigenvalue'].mean():.6f} ± {data_all['Leading_Eigenvalue'].std():.6f}")
    print(f"  N_Survivors: {data_all['N_Survivors'].mean():.2f} ± {data_all['N_Survivors'].std():.2f}")

print("\n" + "="*60)
print("Figure saved to figure/ directory")
print("="*60)
