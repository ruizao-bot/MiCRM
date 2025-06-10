"""
Refactored network analysis code for efficiency and reduced redundancy
Author: Jiayi (refactored by ChatGPT)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ------------------------
# Utility Functions
# ------------------------

def fix_alpha_string(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].replace("\n", " ")
        s = ",".join(s.split())
        return "[" + s + "]"
    return s

def extract_active_species(row, comm_prefix, species_ids, threshold=1e-5):
    cfinal = np.array([row.get(f"Cfinal_{comm_prefix}_Sp{sp}", 0) for sp in species_ids])
    return cfinal > threshold

def get_common_data(df, comm_prefix, threshold=1e-5):
    alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}")]
    species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])
    cue_cols = [f"CUE_{comm_prefix}_Sp{sp}" for sp in species_ids]
    cfinal_cols = [f"Cfinal_{comm_prefix}_Sp{sp}" for sp in species_ids]
    alpha_cols = [f"alpha_{comm_prefix}_Sp{sp}" for sp in species_ids]

    for idx, row in df.iterrows():
        try:
            cue = np.array([row[col] for col in cue_cols])
            cfinal = np.array([row[col] for col in cfinal_cols])
            alpha_matrix = np.stack([
                np.array(ast.literal_eval(fix_alpha_string(row[col]))) for col in alpha_cols
            ])

            active = cfinal > threshold
            if np.sum(active) < 2:
                continue

            yield idx, species_ids, cue[active], cfinal[active], alpha_matrix[active][:, active]
        except Exception as e:
            print(f"Failed on row {idx} ({comm_prefix}): {e}")


# ------------------------
# Functional Summary
# ------------------------

def summarize_CUE_interaction(df, comm_prefix):
    results = []
    for idx, _, cue, cfinal, alpha in get_common_data(df, comm_prefix):
        comm_cue = np.sum(cue * cfinal) / np.sum(cfinal)
        upper = np.triu(np.abs(alpha), k=1)
        mean_strength = np.sum(upper) / np.count_nonzero(upper)
        results.append({"seed": idx, "community": comm_prefix,
                        "community_CUE": comm_cue, "interaction_strength": mean_strength})
    return pd.DataFrame(results)

def summarize_CUE_pathlength(df, comm_prefix):
    results = []
    for idx, _, cue, cfinal, alpha in get_common_data(df, comm_prefix):
        comm_cue = np.sum(cue * cfinal) / np.sum(cfinal)

        G = nx.Graph()
        for i in range(len(cue)):
            G.add_node(i)
        for i in range(len(cue)):
            for j in range(i + 1, len(cue)):
                strength = (abs(alpha[i][j]) + abs(alpha[j][i])) / 2
                if strength > 0:
                    G.add_edge(i, j, weight=strength)

        G_inv = G.copy()
        for u, v, d in G_inv.edges(data=True):
            d['weight'] = 1.0 / (d['weight'] + 1e-8)

        try:
            avg_path = nx.average_shortest_path_length(
                G_inv if nx.is_connected(G_inv) else G_inv.subgraph(max(nx.connected_components(G_inv), key=len)),
                weight='weight')
            results.append({"seed": idx, "community": comm_prefix,
                            "community_CUE": comm_cue, "avg_path_length": avg_path})
        except Exception as e:
            print(f"Path error on row {idx} ({comm_prefix}): {e}")

    return pd.DataFrame(results)


# ------------------------
# Plotting Functions
# ------------------------

def plot_scatter_with_spearman(df_summary, x, y, group_col="community", group_labels=None, group_colors=None, xlabel=None, ylabel=None, title=None):
    communities = df_summary[group_col].unique()
    labels = group_labels if group_labels else communities
    colors = group_colors if group_colors else sns.color_palette("Set1", len(communities))

    pvals, rhos = [], []
    for comm in communities:
        sub = df_summary[df_summary[group_col] == comm]
        rho, pval = spearmanr(sub[x], sub[y])
        pvals.append(pval)
        rhos.append(rho)

    _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    fig, axes = plt.subplots(1, len(communities), figsize=(5 * len(communities), 4), sharey=True)
    if len(communities) == 1:
        axes = [axes]

    for ax, comm, label, color, rho, pval_corr in zip(axes, communities, labels, colors, rhos, pvals_corrected):
        sub = df_summary[df_summary[group_col] == comm]
        sns.regplot(data=sub, x=x, y=y,
                    scatter_kws={"color": color, "alpha": 0.6},
                    line_kws={"color": color, "linestyle": "--"}, ax=ax)
        ax.set_title(f"{label}\nρ = {rho:.2f}, adj. p = {pval_corr:.3f}")
        ax.set_xlabel(xlabel if xlabel else x)
        if ax == axes[0]:
            ax.set_ylabel(ylabel if ylabel else y)

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ------------------------
# Main Workflow
# ------------------------

def run_analysis_and_plot(df):
    communities = ["Comm1", "Comm2", "Comm3"]
    labels = ["Community 1", "Community 2", "Community 3"]
    colors = ["red", "green", "blue"]

    # --- CUE vs Interaction Strength ---
    interaction_results = pd.concat([summarize_CUE_interaction(df, comm) for comm in communities], ignore_index=True)
    plot_scatter_with_spearman(
        interaction_results,
        x="community_CUE",
        y="interaction_strength",
        group_labels=labels,
        group_colors=colors,
        xlabel="Community CUE",
        ylabel="Interaction Strength",
        title="CUE vs Interaction Strength"
    )

    # --- CUE vs Path Length ---
    path_results = pd.concat([summarize_CUE_pathlength(df, comm) for comm in communities], ignore_index=True)
    plot_scatter_with_spearman(
        path_results,
        x="community_CUE",
        y="avg_path_length",
        group_labels=labels,
        group_colors=colors,
        xlabel="Community CUE",
        ylabel="Average Path Length",
        title="CUE vs Average Path Length"
    )
df = pd.read_csv("../data/elv_hpc.csv")
run_analysis_and_plot(df)
comms = ["Comm1", "Comm2", "Comm3"]
for i, comm in enumerate(comms):
    plot_scatter_with_spearman(
        df_summary=summarize_CUE_interaction(df, comms),
        x="community_CUE",
        y="interaction_strength",
        group_col="community",
        group_labels=["Community 1", "Community 2", "Community 3"],
        group_colors=["red", "green", "blue"],
        xlabel="Community CUE",
        ylabel="Interaction Strength",
        title="CUE vs Interaction Strength by Community"
    )