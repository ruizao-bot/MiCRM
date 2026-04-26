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
    'Community_CUE': 'first',
    'feasibility': 'first',
    'Leading_Eigenvalue': 'first',
    'N_Survivors': 'first'
}).reset_index()

# Convert Community to string for plotting
community_metrics['Community'] = community_metrics['Community'].astype(str)

print(f"Total samples: {len(community_metrics)}")

# Create output directory
os.makedirs('figure', exist_ok=True)

# ============================================================================
# Figure 1: CUE vs Feasibility (3 panels, one per community)
# ============================================================================
print("Creating CUE vs Feasibility plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, comm in enumerate(['1', '2', '3']):
    ax = axes[i]
    data = community_metrics[community_metrics['Community'] == comm]
    
    # Scatter plot
    ax.scatter(
        data['Community_CUE'],
        data['feasibility'],
        s=50,
        alpha=0.7,
        color=pal_rgb[comm],
        edgecolors='black',
        linewidths=0.5,
    )
    
    # Calculate correlation
    if len(data) > 1:
        corr = np.corrcoef(data['Community_CUE'], data['feasibility'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=ax.transAxes, 
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlabel('Community CUE', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Feasibility Scale Index', fontweight='bold')
    ax.set_title(f'({chr(65+i)}) {community_labels[comm]}', loc='left', fontweight='bold')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    set_base_theme(ax)

plt.tight_layout()
plt.savefig('figure/cue_vs_feasibility.png', dpi=300, bbox_inches='tight')
plt.savefig('figure/cue_vs_feasibility.pdf', bbox_inches='tight')
print("Saved: figure/cue_vs_feasibility.png/pdf")
plt.close()

# ============================================================================
# Figure 2: CUE vs Stability (3 panels, one per community)
# ============================================================================
print("Creating CUE vs Stability plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, comm in enumerate(['1', '2', '3']):
    ax = axes[i]
    data = community_metrics[community_metrics['Community'] == comm]
    
    # Scatter plot
    ax.scatter(
        data['Community_CUE'],
        data['Leading_Eigenvalue'],
        s=50,
        alpha=0.7,
        color=pal_rgb[comm],
        edgecolors='black',
        linewidths=0.5,
    )
    
    # Calculate correlation
    if len(data) > 1:
        corr = np.corrcoef(data['Community_CUE'], data['Leading_Eigenvalue'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=ax.transAxes, 
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Community CUE', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Leading Eigenvalue (Stability)', fontweight='bold')
    ax.set_title(f'({chr(65+i)}) {community_labels[comm]}', loc='left', fontweight='bold')
    set_base_theme(ax)

plt.tight_layout()
plt.savefig('figure/cue_vs_stability.png', dpi=300, bbox_inches='tight')
plt.savefig('figure/cue_vs_stability.pdf', bbox_inches='tight')
print("Saved: figure/cue_vs_stability.png/pdf")
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*60)
print("CORRELATION SUMMARY")
print("="*60)

for comm in ['1', '2', '3']:
    data = community_metrics[community_metrics['Community'] == comm]
    print(f"\n{community_labels[comm]}:")
    
    if len(data) > 1:
        corr_feas = np.corrcoef(data['Community_CUE'], data['feasibility'])[0, 1]
        corr_stab = np.corrcoef(data['Community_CUE'], data['Leading_Eigenvalue'])[0, 1]
        
        print(f"  CUE vs Feasibility: r = {corr_feas:.4f}")
        print(f"  CUE vs Stability:   r = {corr_stab:.4f}")
        print(f"  Mean CUE: {data['Community_CUE'].mean():.4f} ± {data['Community_CUE'].std():.4f}")
        print(f"  Mean Feasibility: {data['feasibility'].mean():.4e} ± {data['feasibility'].std():.4e}")
        print(f"  Mean Stability: {data['Leading_Eigenvalue'].mean():.6f} ± {data['Leading_Eigenvalue'].std():.6f}")

print("\n" + "="*60)
print("Figures saved to figure/ directory")
print("="*60)
