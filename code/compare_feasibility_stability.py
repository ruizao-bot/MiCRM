import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
pal_rgb = {
    "1": "#D8A39A",
    "2": "#A8C3A6",
    "3": "#9FB7CC"
}
community_labels = {"1": "Parent 1", "2": "Parent 2", "3": "Coalesced"}

# Configure matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "legend.frameon": False,
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
        ax.grid(True, which="minor", color="#F2F2F2", linewidth=0.2)
    else:
        ax.grid(False)

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
ax.set_ylabel('Feasibility Scale Index')
ax.set_xlabel('Community')
ax.set_title('(A)', loc='left')
ax.ticklabel_format(axis='y', style='plain', useOffset=False)
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
ax.set_ylabel('Leading Eigenvalue')
ax.set_xlabel('Community')
ax.set_title('(B)', loc='left')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
set_base_theme(ax)

plt.tight_layout()
plt.savefig('figure/feasibility_stability_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure/feasibility_stability_comparison.pdf', bbox_inches='tight')
print("Saved: figure/feasibility_stability_comparison.png/pdf")
plt.close()

