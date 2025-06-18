import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
from scipy.stats import lognorm, gamma
from statsmodels.nonparametric.smoothers_lowess import lowess

# ========== Utility ==========

def fix_alpha_string(s):
    """Fix alpha string for ast.literal_eval."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].replace("\n", " ")
        s = ",".join(s.split())
        return "[" + s + "]"
    return s

# ========== Network Construction ==========

def build_network(df, community, seed, quantile=0.7):
    """Build a network for a given community and seed, filtering by Cfinal > 1e-10."""
    df_sub = df[(df["Seed"] == seed) & (df["community_id"] == community) & (df["Cfinal"] > 1e-10)].copy()
    if df_sub.empty or len(df_sub) < 2:
        return None

    species_ids = df_sub["species_id"].tolist()
    cues = df_sub["CUE"].values
    alpha_matrix = []
    for alpha_str in df_sub["alpha"]:
        alpha_vec = np.array(ast.literal_eval(fix_alpha_string(alpha_str)))
        survivor_indices = [int(sid[2:]) - 1 for sid in species_ids]  # Sp1 -> 0, Sp2 -> 1, ...
        filtered_vec = alpha_vec[survivor_indices]
        alpha_matrix.append(filtered_vec)
    alpha_matrix = np.array(alpha_matrix)

    # Compute pairwise average interaction weights
    all_weights = [
        (abs(alpha_matrix[i, j]) + abs(alpha_matrix[j, i])) / 2
        for i in range(len(species_ids)) for j in range(len(species_ids)) if i < j
    ]
    if len(all_weights) == 0:
        return None
    threshold = np.quantile(all_weights, quantile)

    # Build network
    G = nx.Graph()
    for i, sp in enumerate(species_ids):
        G.add_node(sp, cue=cues[i])
    for i in range(len(species_ids)):
        for j in range(i + 1, len(species_ids)):
            w = (abs(alpha_matrix[i, j]) + abs(alpha_matrix[j, i])) / 2
            if w > threshold:
                G.add_edge(species_ids[i], species_ids[j], weight=w)
    return G

# ========== Analysis Functions ==========

def compute_bidirectional_strengths(df, community, seed_range):
    """Compute average bidirectional interaction strength for each species across seeds."""
    all_strengths = []
    for seed in seed_range:
        G = build_network(df, community, seed)
        if G is None:
            continue
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            strengths = []
            for nbr in neighbors:
                w = G[node][nbr]['weight']
                strengths.append(w)
            if strengths:
                all_strengths.append(np.mean(strengths))
    return np.array(all_strengths)

def compute_degree_distribution(df, community, seed_range):
    """Compute degree distribution for a community across seeds."""
    all_degrees = []
    for seed in seed_range:
        G = build_network(df, community, seed)
        if G is None:
            continue
        all_degrees.extend([d for _, d in G.degree()])
    return np.array(all_degrees)

def compute_cue_and_degree(df, community, seed_range):
    """Get CUE and degree for all species in a community across seeds."""
    cues, degrees = [], []
    for seed in seed_range:
        G = build_network(df, community, seed)
        if G is None:
            continue
        for node in G.nodes:
            cues.append(G.nodes[node]['cue'])
            degrees.append(G.degree(node))
    return np.array(cues), np.array(degrees)

# ========== Plotting Functions ==========

def plot_network(G, title):
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]['cue'] for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap='viridis', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_weights, edge_cmap=plt.cm.coolwarm, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights), width=2, ax=ax)
    sm_nodes = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm_nodes.set_array([])
    plt.colorbar(sm_nodes, ax=ax, shrink=0.7, label="CUE")
    sm_edges = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm_edges.set_array([])
    plt.colorbar(sm_edges, ax=ax, shrink=0.7, label="Interaction Strength")
    ax.set_title(title)
    ax.axis("off")
    plt.show()

def plot_degree_hist(degrees, community, color):
    counts, bins = np.histogram(degrees, bins=30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mask = counts > 0
    plt.plot(bin_centers[mask], np.log10(counts[mask]), marker='o', linestyle='-', color=color)
    plt.xlabel("Degree")
    plt.ylabel("log10(Frequency)")
    plt.title(f"{community}")

def plot_strength_hist(strengths, community, color):
    counts, bins = np.histogram(strengths, bins=30, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, np.log10(counts), marker='o', linestyle='-', color=color, label="Empirical")
    # Lognormal fit
    shape, loc, scale = lognorm.fit(strengths, floc=0)
    x = np.linspace(min(strengths), max(strengths), 200)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, np.log10(pdf), linestyle='--', color='black', label="Lognormal fit")
    plt.xlabel("Interaction Strength")
    plt.ylabel("log10(Density)")
    plt.title(f"{community}")
    plt.legend()

def plot_cue_vs_degree(cues, degrees, community, color):
    plt.scatter(cues, degrees, color=color, s=10, alpha=0.5, label="Species")
    # Gamma fit for degree
    if len(degrees) > 0 and np.all(degrees > 0):
        shape, loc, scale = gamma.fit(degrees, floc=0)
        x = np.linspace(min(cues), max(cues), 200)
        # For each cue value, plot the gamma PDF at that cue (not meaningful)
        # Instead, plot the gamma PDF over the degree axis as a secondary axis
        deg_x = np.linspace(min(degrees), max(degrees), 200)
        gamma_pdf = gamma.pdf(deg_x, shape, loc=loc, scale=scale)
        # Scale PDF to match scatter plot scale for visualization
        gamma_pdf_scaled = gamma_pdf * max(degrees) / max(gamma_pdf)
        plt.plot([min(cues)]*len(deg_x), deg_x, alpha=0)  # dummy for axis
        plt.plot([max(cues)]*len(deg_x), deg_x, alpha=0)  # dummy for axis
        plt.plot([min(cues)+(max(cues)-min(cues))*0.95]*len(deg_x), deg_x, alpha=0)  # dummy for axis
        plt.plot([min(cues)+(max(cues)-min(cues))*0.05]*len(deg_x), deg_x, alpha=0)  # dummy for axis
        plt.plot(np.full_like(deg_x, min(cues)), deg_x, alpha=0)  # dummy for axis
        plt.plot(np.full_like(deg_x, max(cues)), deg_x, alpha=0)  # dummy for axis
        plt.plot(np.full_like(deg_x, min(cues)+(max(cues)-min(cues))*0.95), deg_x, alpha=0)  # dummy for axis
        plt.plot(np.full_like(deg_x, min(cues)+(max(cues)-min(cues))*0.05), deg_x, alpha=0)  # dummy for axis
        plt.plot(np.linspace(min(cues), max(cues), len(deg_x)), gamma_pdf_scaled, color="black", lw=2, linestyle="--", label="Gamma fit (degree)")
    plt.xlabel("CUE")
    plt.ylabel("Degree")
    plt.title(f"{community}")
    plt.legend()

def plot_gengamma_fit(result, comm_prefix, color):
    if result is None:
        print(f"Cannot plot {comm_prefix}: fit failed.")
        return

    a, c_, d_, loc_, scale_ = result["params"]
    cue = result["cue"]
    degree = result["degree"]
    cue_sorted = result["cue_sorted"]
    fitted_y = result["fitted_y"]
    peak_cue = result["peak_cue"]
    r2 = result["r2"]

    print(f"\n--- {comm_prefix} ---")
    print(f"a = {a:.3f}, c = {c_:.3f}, d = {d_:.3f}, loc = {loc_:.3f}, scale = {scale_:.3f}")
    print(f"Peak at CUE = {peak_cue:.3f}, Degree = {result['peak_degree']:.3f}")
    print(f"R² = {r2:.3f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(cue, degree, alpha=0.5, color=color, label="Data")
    plt.plot(cue_sorted, fitted_y, color=color, label=f"Gamma fit (R²={r2:.2f})")
    plt.axvline(peak_cue, color='gray', linestyle='--', label=f"Peak CUE = {peak_cue:.3f}")
    plt.xlabel("CUE")
    plt.ylabel("Degree Centrality")
    plt.title(f"{comm_prefix}: Generalized Gamma Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cue_vs_strength(cues, strengths, community, color):
    plt.scatter(cues, strengths, alpha=0.3, s=5, color=color, label="Species")
    if len(cues) >= 10:
        smoothed = lowess(strengths, cues, frac=0.3, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.5, label="LOWESS fit")
    plt.xlabel("CUE")
    plt.ylabel("Interaction Strength")
    plt.title(f"{community}")
    plt.legend()

# ========== Main Analysis ==========

if __name__ == "__main__":
    df = pd.read_csv("../data/elv_hpc_sameR0.csv")
    communities = ["Comm1", "Comm2", "Comm3"]
    colors = ["red", "green", "blue"]
    seed_range = range(51, 101)

    # Plot network for a single seed
    for comm, color in zip(communities, colors):
        G = build_network(df, comm, seed=52)
        if G is not None:
            plot_network(G, f"Network of {comm} (Seed 52)")

    # Degree distribution
    plt.figure(figsize=(15, 4))
    for i, (comm, color) in enumerate(zip(communities, colors)):
        degrees = compute_degree_distribution(df, comm, seed_range)
        plt.subplot(1, 3, i+1)
        plot_degree_hist(degrees, comm, color)
    plt.tight_layout()
    plt.show()

    # Interaction strength distribution
    plt.figure(figsize=(15, 4))
    for i, (comm, color) in enumerate(zip(communities, colors)):
        strengths = compute_bidirectional_strengths(df, comm, seed_range)
        plt.subplot(1, 3, i+1)
        plot_strength_hist(strengths, comm, color)
    plt.tight_layout()
    plt.show()

    # CUE vs Degree
    plt.figure(figsize=(15, 4))
    for i, (comm, color) in enumerate(zip(communities, colors)):
        cues, degrees = compute_cue_and_degree(df, comm, seed_range)
        plt.subplot(1, 3, i+1)
        plot_cue_vs_degree(cues, degrees, comm, color)
    plt.tight_layout()
    plt.show()
    # CUE vs Interaction Strength
    plt.figure(figsize=(15, 4))
    for i, (comm, color) in enumerate(zip(communities, colors)):
        strengths = compute_bidirectional_strengths(df, comm, seed_range)
        cues, _ = compute_cue_and_degree(df, comm, seed_range)
        min_len = min(len(cues), len(strengths))
        cues = cues[:min_len]
        strengths = strengths[:min_len]
        plt.subplot(1, 3, i+1)
        plot_cue_vs_strength(cues, strengths, comm, color)
    plt.tight_layout()
    plt.show()
    # %%
from statsmodels.nonparametric.smoothers_lowess import lowess

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, comm in enumerate(communities):
    cues, degrees = compute_cue_and_degree(df, comm, seed_range)
    mask = degrees > 0.01
    cues = cues[mask]
    degrees = degrees[mask]

    if len(cues) < 5:
        print(f"Not enough data for {comm}")
        continue

    # LOWESS fit
    smoothed = lowess(degrees, cues, frac=0.3, return_sorted=True)
    ax = axes[i]
    ax.scatter(cues, degrees, alpha=0.5, color=colors[i], s=10, label="Species")
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='black', lw=2, label="LOWESS fit")
    ax.set_title(f"{comm}")
    ax.set_xlabel("CUE")
    if i == 0:
        ax.set_ylabel("Degree")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()