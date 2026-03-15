import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
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
df = pd.read_csv("coal.csv")
df['Status'] = df['Abundance'].apply(lambda x: "Extinction" if x < 1e-5 else "Survival")
df_surv = df[df['Abundance'] > 1e-5].copy()

df_surv['log10_Abundance'] = np.log10(df_surv['Abundance'])
df['log10_Abundance'] = np.log10(df['Abundance'])

# FIGURE 1: CUE vs Similarity with Dominance
df_mut = df_surv.copy()
df_mut['Global_Species_ID'] = df_mut.apply(
    lambda x: x['Species_ID'] + 100 if x['Community'] == 2 else x['Species_ID'], axis=1
)

# Calculate Bray-Curtis dissimilarity
bray_results = []

for s in df_mut['Seed'].unique():
    df_seed = df_mut[df_mut['Seed'] == s]
    if not all(c in df_seed['Community'].values for c in [1, 2, 3]):
        continue
    
    # Create community matrix
    comm_mat = df_seed.pivot_table(
        index='Community', 
        columns='Global_Species_ID', 
        values='Abundance', 
        fill_value=0
    )
    
    if len(comm_mat) != 3:
        continue
    
    # Calculate Bray-Curtis dissimilarity
    bc_matrix = squareform(pdist(comm_mat.values, metric='braycurtis'))
    
    # Map community indices
    comm_idx = {comm: idx for idx, comm in enumerate(comm_mat.index)}
    d31 = bc_matrix[comm_idx[3], comm_idx[1]]
    d32 = bc_matrix[comm_idx[3], comm_idx[2]]
    
    cue1 = df_seed[df_seed['Community'] == 1]['Community_CUE'].iloc[0]
    cue2 = df_seed[df_seed['Community'] == 2]['Community_CUE'].iloc[0]
    
    bray_results.append({
        'Seed': s,
        'Bray_3vs1': d31,
        'Bray_3vs2': d32,
        'CUE_1': cue1,
        'CUE_2': cue2,
        'Sim_3vs1': 1 - d31,
        'Sim_3vs2': 1 - d32
    })

bray_df = pd.DataFrame(bray_results)

# Get dominant community info
df_comm = df[df['Community'].isin([1, 2])].groupby(['Seed', 'Community']).agg(
    Community_CUE=('Community_CUE', 'first'),
    Dominant_Community=('Dominant_Community', 'first')
).reset_index()

# Calculate differences
df_diff = bray_df.copy()
df_diff['CUE_Diff'] = df_diff['CUE_1'] - df_diff['CUE_2']
df_diff['Sim_Diff'] = df_diff['Sim_3vs1'] - df_diff['Sim_3vs2']

# Merge with dominant community
dom_info = df_comm[df_comm['Community'] == 1][['Seed', 'Dominant_Community']]
df_diff = df_diff.merge(dom_info, on='Seed', how='left')

df_diff['DomGroup'] = df_diff['Dominant_Community'].map({
    'Community 1': 'Parent 1',
    'Community 2': 'Parent 2'
})

dom_colors = {
    'Parent 1': pal_rgb['1'],
    'Parent 2': pal_rgb['2'],
}

fig, ax = plt.subplots(figsize=(8.27, 4.72))

for group, color in dom_colors.items():
    df_g = df_diff[df_diff['DomGroup'] == group]
    ax.scatter(df_g['CUE_Diff'], df_g['Sim_Diff'], 
              color=color, alpha=0.7, s=25, label=group)

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('ΔCUE (Parent 1 - Parent 2)')
ax.set_ylabel('ΔSimilarity (Parent 1 - Parent 2)')
ax.set_ylim(-1.05, 1.05)
ax.legend(title='Dominant', loc='best', frameon=True)
set_base_theme(ax)

plt.tight_layout()
plt.savefig('results/dom_sim.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/dom_sim.pdf")
plt.close()

# FIGURE 1B: CUE vs Dominance under Different Resource Overlap
df_resource = pd.read_csv('data/coal_resource.csv')
df_resource['Overlap'] = df_resource['Overlap'].astype(str)

# Compute differences
df_diff_resource = df_resource.copy()
df_diff_resource['CUE_diff'] = df_diff_resource['CUE1'] - df_diff_resource['CUE2']
df_diff_resource['Sim_diff'] = df_diff_resource['Similarity_3vs1'] - df_diff_resource['Similarity_3vs2']

# Create CUE bins
n_bins = 5
df_diff_resource['CUE_bin'] = pd.cut(df_diff_resource['CUE_diff'], bins=n_bins)
df_diff_resource['CUE_mid'] = df_diff_resource['CUE_bin'].apply(lambda x: x.mid)

overlap_colors = {"0.25": "#E74C3C", "0.5": "#F39C12", "0.75": "#3498DB"}

fig, ax = plt.subplots(figsize=(8.27, 4.72))

# Prepare data for boxplot
overlap_vals = sorted(df_diff_resource['Overlap'].unique())
bin_vals = sorted(df_diff_resource['CUE_bin'].unique(), key=lambda x: x.mid)

positions = []
data_list = []
colors_list = []

for i, bin_val in enumerate(bin_vals):
    for j, overlap in enumerate(overlap_vals):
        data = df_diff_resource[
            (df_diff_resource['CUE_bin'] == bin_val) & 
            (df_diff_resource['Overlap'] == overlap)
        ]['Sim_diff']
        
        if len(data) > 0:
            pos = i * (len(overlap_vals) + 0.5) + j
            positions.append(pos)
            data_list.append(data.values)
            colors_list.append(overlap_colors[overlap])

# Create boxplot
bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                boxprops=dict(alpha=0.7),
                medianprops=dict(color='black', linewidth=1),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker='o', markersize=3, alpha=0.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_ylabel('ΔSimilarity (Parent 1 - Parent 2)')
ax.set_xlabel('ΔCUE (Parent 1 - Parent 2)')

# Set x-axis labels
bin_centers = [i * (len(overlap_vals) + 0.5) + 1 for i in range(len(bin_vals))]
ax.set_xticks(bin_centers)
ax.set_xticklabels([f"{b.left:.2f} to {b.right:.2f}" for b in bin_vals], 
                    rotation=45, ha='right', fontsize=8)

# Create legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=overlap_colors[o], alpha=0.7, label=o) 
                   for o in overlap_vals]
ax.legend(handles=legend_elements, title='Resource Overlap', loc='best')

set_base_theme(ax)
plt.tight_layout()
plt.savefig('results/cue_dominance_overlap.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/cue_dominance_overlap.pdf")
plt.close()

# FIGURE 2: Species CUE vs Abundance
# Set uniform y-axis range
y_min = df_surv['log10_Abundance'].min()
y_max = df_surv['log10_Abundance'].max()

# Create combined plot
fig, axes = plt.subplots(3, 2, figsize=(8.27, 7.87), 
                         gridspec_kw={'width_ratios': [1.2, 3], 'hspace': 0.3})

for idx, comm in enumerate(["1", "2", "3"]):
    df_i = df_surv[df_surv['Community'] == int(comm)]
    color = pal_rgb[comm]
    
    # Histogram (left)
    ax_hist = axes[idx, 0]
    ax_hist.hist(df_i['log10_Abundance'], bins=50, orientation='horizontal',
                 color=color, alpha=0.5, edgecolor=color)
    ax_hist.set_ylim(y_min, y_max)
    ax_hist.invert_xaxis()
    ax_hist.set_ylabel('Frequency' if idx == 1 else '')
    ax_hist.set_yticks([])
    set_base_theme(ax_hist)
    
    # Scatter plot (right)
    ax_main = axes[idx, 1]
    ax_main.scatter(df_i['Species_CUE'], df_i['log10_Abundance'],
                   color=color, alpha=0.5, s=20)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel('Species-level CUE' if idx == 2 else '')
    ax_main.set_ylabel('log10(Abundance)')
    set_base_theme(ax_main)

plt.tight_layout()
plt.savefig('results/cue_abund.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/cue_abund.pdf")
plt.close()


# FIGURE 3: Rare Species Invasion
df_rare = pd.read_csv("data/rare.csv")
df_rare['survival'] = df_rare['Abundance'].apply(lambda x: "Survived" if x > 1e-5 else "Extinct")
df_rare_filt = df_rare[df_rare['DilutionRate'].isin([0.01, 0.1])].copy()

# Create CUE bins
n_bins = 20
df_rare_filt['CUE_bin'] = pd.cut(df_rare_filt['Species_CUE'], bins=n_bins)
df_rare_filt['CUE_mid'] = df_rare_filt['CUE_bin'].apply(lambda x: x.mid)

# Count survival/extinction
df_rare_bar = df_rare_filt.groupby(['DilutionRate', 'CUE_bin', 'survival']).size().reset_index(name='count')
df_rare_bar['CUE_mid'] = df_rare_bar['CUE_bin'].apply(lambda x: x.mid)

dilution_labels = {0.01: "Rarity Level = 0.01", 0.1: "Rarity Level = 0.1"}

fig, axes = plt.subplots(2, 1, figsize=(8.27, 5.51), sharex=True)

for idx, dilution in enumerate([0.01, 0.1]):
    ax = axes[idx]
    df_d = df_rare_bar[df_rare_bar['DilutionRate'] == dilution]
    
    # Sort by CUE_mid
    bins_sorted = sorted(df_d['CUE_bin'].unique(), key=lambda x: x.mid)
    
    survived = []
    extinct = []
    
    for bin_val in bins_sorted:
        df_bin = df_d[df_d['CUE_bin'] == bin_val]
        surv = df_bin[df_bin['survival'] == 'Survived']['count'].sum()
        ext = df_bin[df_bin['survival'] == 'Extinct']['count'].sum()
        survived.append(surv)
        extinct.append(ext)
    
    x_pos = np.arange(len(bins_sorted))
    ax.bar(x_pos, extinct, color='#E74C3C', alpha=0.8, label='Extinct')
    ax.bar(x_pos, survived, bottom=extinct, color='#2ECC71', alpha=0.8, label='Survived')
    
    ax.set_ylabel('Species Count')
    ax.text(0.02, 0.95, dilution_labels[dilution], 
            transform=ax.transAxes, va='top', fontsize=12)
    
    if idx == 0:
        ax.legend(title='Outcome', loc='upper right')
    
    if idx == 1:
        ax.set_xlabel('Species-level CUE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{b.left:.2f}" for b in bins_sorted], 
                          rotation=45, ha='right', fontsize=8)
    
    set_base_theme(ax)

plt.tight_layout()
plt.savefig('results/rare_survival.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/rare_survival.pdf")
plt.close()

# FIGURE 4: Facilitation vs Community CUE
df_comm_fac = df_surv.groupby(['Seed', 'Community']).agg(
    Community_CUE=('Community_CUE', 'first'),
    Facilitation=('Facilitation', 'mean')
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(8.27, 4.72), sharey=True)

for idx, comm in enumerate([1, 2, 3]):
    df_i = df_comm_fac[df_comm_fac['Community'] == comm]
    ax = axes[idx]
    ax.scatter(df_i['Facilitation'], df_i['Community_CUE'],
              color=pal_rgb[str(comm)], alpha=0.7, s=15)
    ax.set_title(community_labels[str(comm)], fontsize=12)
    
    # Format x-axis
    x_vals = ax.get_xticks()
    ax.set_xticklabels([f'{x*1e3:.2f}' for x in x_vals], fontsize=10)
    ax.set_xlabel('Facilitation (×10$^{-3}$)')
    
    if idx == 0:
        ax.set_ylabel('Community CUE')
    
    ax.grid(True, alpha=0.2, linewidth=0.3)
    set_base_theme(ax)

plt.tight_layout()
plt.savefig('results/Facilitation_vs_communityCUE.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/Facilitation_vs_communityCUE.pdf")
plt.close()

# FIGURE 5: CUE vs Depletion
df_depletion = df.groupby(['Seed', 'Community']).agg(
    Community_CUE=('Community_CUE', 'first'),
    Niche_Overlap=('Competition', 'first'),
    Depletion=('Depletion', 'first')
).reset_index()

fig = plt.figure(figsize=(8.27, 3.94))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

# Left: Depletion scatter
ax1 = fig.add_subplot(gs[0])
shapes = {1: 'o', 2: '^', 3: 's'}
for comm in [1, 2, 3]:
    df_i = df_depletion[df_depletion['Community'] == comm]
    ax1.scatter(df_i['Community_CUE'], df_i['Depletion'],
               color=pal_rgb[str(comm)], marker=shapes[comm], 
               alpha=0.7, s=30, label=community_labels[str(comm)])
ax1.set_xlabel('Community CUE')
ax1.set_ylabel('Sum of Resource Residual')
ax1.legend(title='Community', loc='best')
set_base_theme(ax1)

# Right: CUE boxplot
ax2 = fig.add_subplot(gs[1])
box_data = [df_depletion[df_depletion['Community'] == c]['Community_CUE'].values 
            for c in [1, 2, 3]]
bp = ax2.boxplot(box_data, positions=[1, 2, 3], widths=0.6, patch_artist=True,
                 boxprops=dict(alpha=0.6),
                 medianprops=dict(color='black', linewidth=1))

for patch, comm in zip(bp['boxes'], [1, 2, 3]):
    patch.set_facecolor(pal_rgb[str(comm)])

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['P1', 'P2', 'Coal.'])
ax2.set_ylabel('Community CUE')
set_base_theme(ax2)

plt.tight_layout()
plt.savefig('results/Residual_vs_CUE.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/Residual_vs_CUE.pdf")
plt.close()


# FIGURE SI: Competition vs Community CUE (Supplementary)
df_comm_agg = df_surv.groupby(['Seed', 'Community', 'Competition', 'Community_CUE', 'Facilitation']).agg(
    Species_CUE_Var=('Species_CUE', 'var')
).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(8.27, 7.09), sharex=False)

for idx, comm in enumerate([1, 2, 3]):
    df_i = df_comm_agg[df_comm_agg['Community'] == comm]
    ax = axes[idx]
    ax.scatter(df_i['Competition'], df_i['Community_CUE'],
              color=pal_rgb[str(comm)], alpha=0.4, s=15)
    ax.set_ylabel('Community CUE')
    ax.text(1.02, 0.5, community_labels[str(comm)], 
            transform=ax.transAxes, va='center', rotation=-90, fontsize=12)
    
    # Format x-axis to show *10^-3
    x_vals = ax.get_xticks()
    ax.set_xticklabels([f'{x*1e3:.2f}' for x in x_vals])
    
    ax.grid(True, alpha=0.3, linewidth=0.3, color='grey')
    set_base_theme(ax)
    
    if idx == 2:
        ax.set_xlabel('Competition (×10$^{-3}$)')

plt.tight_layout()
plt.savefig('results/Competition_vs_communityCUE.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/Competition_vs_communityCUE.pdf")
plt.close()

# Species competition plot
fig, axes = plt.subplots(3, 1, figsize=(8.27, 7.09), sharex=False)

for idx, comm in enumerate([1, 2, 3]):
    df_i = df_surv[df_surv['Community'] == comm]
    ax = axes[idx]
    ax.scatter(df_i['Species_Competition_Dot'], df_i['Species_CUE'],
              color=pal_rgb[str(comm)], alpha=0.6, s=15)
    ax.set_ylabel('Species CUE')
    ax.text(1.02, 0.5, community_labels[str(comm)], 
            transform=ax.transAxes, va='center', rotation=-90, fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.3, color='grey')
    set_base_theme(ax)
    
    if idx == 2:
        ax.set_xlabel('Species Competition')

plt.tight_layout()
plt.savefig('results/species_competition.pdf', dpi=600, bbox_inches='tight')
print("Saved: results/species_competition.pdf")
plt.close()

print("\n=== All figures generated successfully ===")
