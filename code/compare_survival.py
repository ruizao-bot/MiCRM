import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the species-level data from coal_feas.csv
df_raw = pd.read_csv('data/coal.csv')

# Define survival threshold (same as in main.py)
SURVIVAL_THRESHOLD = 1e-5

print(f"Loaded {len(df_raw)} species records")
print(f"Unique seeds: {df_raw['Seed'].nunique()}")
print(f"Communities: {sorted(df_raw['Community'].unique())}")
print()

# Calculate survival counts for each (Seed, Community) combination
# A species survives if its Abundance > SURVIVAL_THRESHOLD
df_raw['Survived'] = df_raw['Abundance'] > SURVIVAL_THRESHOLD

# Group by Seed and Community, count survivors
survival_counts = df_raw.groupby(['Seed', 'Community'])['Survived'].sum().reset_index()
survival_counts.columns = ['Seed', 'Community', 'N_Survivors']

# Pivot to get separate columns for each community
df_pivot = survival_counts.pivot(index='Seed', columns='Community', values='N_Survivors')
df_pivot.columns = [f'n_surv{int(c)}' for c in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# Ensure all three communities are present
for i in [1, 2, 3]:
    col_name = f'n_surv{i}'
    if col_name not in df_pivot.columns:
        df_pivot[col_name] = 0

df = df_pivot

# Calculate summary statistics for each community
communities = ['n_surv1', 'n_surv2', 'n_surv3']
community_names = ['Community 1', 'Community 2', 'Community 3']

print("=" * 70)
print("SURVIVAL RATE COMPARISON ACROSS COMMUNITIES")
print("=" * 70)
print()

# Summary statistics
print("Summary Statistics:")
print("-" * 70)
for i, col in enumerate(communities):
    print(f"\n{community_names[i]}:")
    print(f"  Mean:     {df[col].mean():.2f}")
    print(f"  Median:   {df[col].median():.2f}")
    print(f"  Std Dev:  {df[col].std():.2f}")
    print(f"  Min:      {df[col].min()}")
    print(f"  Max:      {df[col].max()}")
    print(f"  Total:    {df[col].sum()}")

print("\n" + "=" * 70)
print()

# Statistical tests
print("Statistical Comparisons:")
print("-" * 70)

# T-tests between communities
print("\nPairwise t-tests:")
t_stat_12, p_val_12 = stats.ttest_ind(df['n_surv1'], df['n_surv2'])
print(f"Community 1 vs Community 2: t={t_stat_12:.4f}, p={p_val_12:.4f}")

t_stat_13, p_val_13 = stats.ttest_ind(df['n_surv1'], df['n_surv3'])
print(f"Community 1 vs Community 3: t={t_stat_13:.4f}, p={p_val_13:.4f}")

t_stat_23, p_val_23 = stats.ttest_ind(df['n_surv2'], df['n_surv3'])
print(f"Community 2 vs Community 3: t={t_stat_23:.4f}, p={p_val_23:.4f}")

# ANOVA
f_stat, p_val_anova = stats.f_oneway(df['n_surv1'], df['n_surv2'], df['n_surv3'])
print(f"\nOne-way ANOVA: F={f_stat:.4f}, p={p_val_anova:.4f}")

print("\n" + "=" * 70)

# Print correlation matrix
print("\nCorrelation between communities:")
print("-" * 70)
corr_matrix = df[communities].corr()
print(corr_matrix)

print("\n" + "=" * 70)

# Create a combined boxplot + scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
community_names = ['Community 1', 'Community 2', 'Community 3']

# Calculate means for labeling
means = [df[col].mean() for col in communities]

# Prepare data for boxplot
data_for_boxplot = [df[col] for col in communities]

# x positions for the three communities
x_positions = [1, 2, 3]

# Create boxplot
bp = ax.boxplot(data_for_boxplot, positions=x_positions, widths=0.5,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6),
                medianprops=dict(color='black', linewidth=2),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)

# Overlay scatter points for individual simulations
np.random.seed(42)  # For consistent jittering
for i, col in enumerate(communities):
    # Add small random jitter to x-position for visibility
    x_jitter = x_positions[i] + np.random.normal(0, 0.08, size=len(df))
    ax.scatter(x_jitter, df[col], alpha=0.6, s=50, color=colors[i], 
              edgecolor='black', linewidth=0.5, zorder=3)

# Formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(community_names, fontsize=12)
ax.set_ylabel('Number of Survivors', fontsize=13, fontweight='bold')
ax.set_xlabel('Community', fontsize=13, fontweight='bold')
ax.set_title('Survival Comparison Across Communities', fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_xlim(0.5, 3.5)

# Add mean value labels above the boxplots
for i, m in enumerate(means):
    y_max = df[communities[i]].max()
    ax.text(x_positions[i], y_max + 1, f'μ = {m:.1f}', 
           ha='center', fontsize=11, fontweight='bold')


plt.tight_layout()

# Create figure directory if it doesn't exist
import os
os.makedirs('figure', exist_ok=True)

plt.savefig('figure/survival_comparison.png', dpi=300, bbox_inches='tight')
print("\nFigure saved to: figure/survival_comparison.png")
plt.show()
