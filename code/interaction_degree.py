import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
import matplotlib as mpl 
from scipy.stats import lognorm, gamma
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gengamma
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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
    df_sub = df[(df["Seed"] == seed) & (df["community_id"] == community) & (df["Cfinal"] > 1e-5)].copy()
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

def get_all_replicate_data(df, comm, seed_range):
    cues, degrees = [], []
    for seed in seed_range:
        G = build_network(df, comm, seed)
        if G is None:
            continue
        for node in G.nodes:
            cues.append(G.nodes[node]['cue'])
            degrees.append(G.degree(node))
    return np.array(cues), np.array(degrees)

# Generalized gamma regression function
def gengamma_regression(x, a, d, p, loc, scale):
    # a: amplitude, d: shape, p: shape, loc: location, scale: scale
    return a * gengamma.pdf(x, a=d, c=p, loc=loc, scale=scale)

# Main function to create the combined plot
if __name__ == "__main__":
    # Set font size to 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Colors for each community
    pal_rgb = {
        "Community 1": "#E74C3C",   # red
        "Community 2": "#2ECC71",   # green
        "Community 3": "#3498DB"    # blue
    }
    communities = ["Community 1", "Community 2", "Community 3"]
    colors = [pal_rgb[c] for c in communities]
    
    # Load data
    df = pd.read_csv("../data/elv_hpc_sameR0.csv")
    seed_range = range(51, 101)
    
    # A4 size: width 21 cm, height 14 cm (good for 2-row plot)
    cm_to_inch = lambda cm: cm / 2.54
    fig_width_in = cm_to_inch(21)
    fig_height_in = cm_to_inch(14)
    
    # Create figure with 2 rows, 3 columns, increase spacing between panels
    fig, axes = plt.subplots(2, 3, figsize=(fig_width_in, fig_height_in), 
                            gridspec_kw={'hspace': 0.45, 'wspace': 0.3})
    
    # Define x-axis limits for each community to ensure alignment
    x_limits = {}
    # Map old community IDs in the dataframe to new names
    comm_map = {"Comm1": "Community 1", "Comm2": "Community 2", "Comm3": "Community 3"}
    df["community_id"] = df["community_id"].replace(comm_map)
    for i, comm in enumerate(communities):
        # Get data for degree centrality
        cue_degree, degree = get_all_replicate_data(df, comm, range(51, 101))
        mask_degree = degree > 0.01
        cue_degree = cue_degree[mask_degree]
        # Get data for interaction strength
        cue_strength = compute_cue_and_degree(df, comm, seed_range)[0]
        strengths = compute_bidirectional_strengths(df, comm, seed_range)
        min_len = min(len(cue_strength), len(strengths))
        cue_strength = cue_strength[:min_len]
        # Combine all CUE values to determine common x-axis limits
        all_cues = np.concatenate((cue_degree, cue_strength))
        min_cue = max(0.1, np.min(all_cues))
        max_cue = min(0.3, np.max(all_cues) * 1.05)
        x_limits[comm] = (min_cue, max_cue)
    
    # Row 1: Degree Centrality vs CUE
    for i, comm in enumerate(communities):
        ax = axes[0, i]
        
        # Get data
        cue, degree = get_all_replicate_data(df, comm, range(51, 101))
        mask = degree > 0.01
        cue = cue[mask]
        degree = degree[mask]
    
        # Scatter plot
        ax.scatter(cue, degree, alpha=0.7, edgecolors='none', color=colors[i], s=3)
        
        # Fit gamma distribution
        try:
            p0 = [np.mean(degree), 1.0, 2.0, np.min(cue), max(np.std(cue), 0.01)]
            bounds = ([0, 0.01, 0.01, np.min(cue)-0.01, 1e-4],
                    [10*np.max(degree), 10, 10, np.max(cue)+0.01, 1.0])
            
            params, _ = curve_fit(gengamma_regression, cue, degree, p0=p0, bounds=bounds, maxfev=10000)
            
            cue_sorted = np.sort(cue)
            fitted_y = gengamma_regression(cue_sorted, *params)
            
            # Plot the fit line
            ax.plot(cue_sorted, fitted_y, color='black', lw=1)
            
            # Peak line
            peak_idx = np.argmax(fitted_y)
            peak_cue = cue_sorted[peak_idx]
            ax.axvline(peak_cue, color='gray', linestyle='--', lw=1)
        except Exception as e:
            print(f"Degree fit failed for {comm}: {e}")
        
        ax.set_title(f"{comm}")
        ax.set_xlabel("CUE")
        
        # Set consistent x-axis limits based on combined data
        ax.set_xlim(x_limits[comm])
        
        # Only show y-label for the first subplot
        if i == 0:
            ax.set_ylabel("Degree Centrality")
    
    # Row 2: Interaction Strength vs CUE
    
    # First collect all strength values to determine common y-axis limits
    all_strengths = []
    for comm in communities:
        cues = compute_cue_and_degree(df, comm, seed_range)[0]
        strengths = compute_bidirectional_strengths(df, comm, seed_range)
        min_len = min(len(cues), len(strengths))
        all_strengths.extend(strengths[:min_len])
    
    # Calculate appropriate y limits with a small buffer (5% of range)
    strength_min = np.min(all_strengths)
    strength_max = np.max(all_strengths)
    strength_range = strength_max - strength_min
    y_buffer = strength_range * 0.05  # 5% buffer
    y_limits = (strength_min - y_buffer, strength_max + y_buffer)
    
    for i, comm in enumerate(communities):
        ax = axes[1, i]
        
        # Get data
        cues = compute_cue_and_degree(df, comm, seed_range)[0]
        strengths = compute_bidirectional_strengths(df, comm, seed_range)
        min_len = min(len(cues), len(strengths))
        cues = cues[:min_len]
        strengths = strengths[:min_len]
    
        # Scatter plot
        ax.scatter(
            cues, strengths,
            s=3, marker='o',
            c=[pal_rgb[comm]],
            edgecolors='none', linewidths=0,
            alpha=0.7, rasterized=False
        )
    
        # LOWESS fit
        if len(cues) >= 10:
            try:
                smoothed = lowess(strengths, cues, frac=0.3, return_sorted=True)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1)
            except Exception as e:
                print(f"LOWESS fit failed for {comm}: {e}")
    
        ax.set_xlabel("CUE")
        # Set consistent x-axis limits to match the plot above
        ax.set_xlim(x_limits[comm])
        # Set appropriate y-axis limits for interaction strength
        ax.set_ylim(y_limits)
        
        # Use scientific notation for y-axis
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # Only show y-label for the first subplot
        if i == 0:
            ax.set_ylabel("Interaction Strength")
    
    plt.tight_layout()
    plt.savefig("../results/CUE_combined_fs14.png", dpi=600, bbox_inches="tight")
    plt.show()
    # Also calculate and print the parameter table
    results = []
    for comm in communities:
        cues, degrees = compute_cue_and_degree(df, comm, seed_range)
        mask = degrees > 0.01
        cue, degree = cues[mask], degrees[mask]
        if len(cue) < 5:
            print(f"Not enough data for {comm}")
            continue
        # Gamma fitting
        try:
            p0 = [np.mean(degree), 1.0, 2.0, np.min(cue), max(np.std(cue), 0.01)]
            bounds = ([0, 0.01, 0.01, np.min(cue)-0.01, 1e-4],
                    [10*np.max(degree), 10, 10, np.max(cue)+0.01, 1.0])
            params, _ = curve_fit(
                gengamma_regression, cue, degree,
                p0=p0, bounds=bounds, maxfev=10000
            )
            # Save params = [A, k, c, x0, lam]
            results.append(
                dict(community=comm,
                    A=params[0],
                    k=params[1],
                    c=params[2],
                    x0=params[3],
                    lam=params[4])
            )
        except Exception as e:
            print(f"Fit failed for {comm}: {e}")
    param_df = pd.DataFrame(results)
    print(param_df)


