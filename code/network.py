import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.stats import spearmanr, linregress
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import multipletests


# ======================= Build network =======================
def fix_alpha_string(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].replace("\n", " ")
        s = ",".join(s.split())
        return "[" + s + "]"
    return s

def build_network(df, comm_prefix, seed=None, quantile=0.7, cfinal_threshold=1e-6):
    # Get all alpha columns for the given community
    alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}")]
    # Extract all species IDs (from column names like 'alpha_CommX_SpY')
    all_species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])
    
    # Select the row corresponding to the specified seed; if not provided, sample a random row
    if seed is not None:
        row = df[df["Seed"] == seed]
        if row.empty:
            return None  # No row exists with the specified seed
        random_row = row.iloc[0]
    else:
        random_row = df.sample(n=1, random_state=np.random.randint(1e6)).iloc[0]
    
    # Filter species based on Cfinal threshold
    # Only include species whose Cfinal value is greater than cfinal_threshold
    species_ids = []
    for sp in all_species_ids:
        # Use .get() to return 0 if the column is missing
        cfinal_value = random_row.get(f"Cfinal_{comm_prefix}_Sp{sp}", 0)
        if cfinal_value > cfinal_threshold:
            species_ids.append(sp)
    
    # Extract CUE values for the filtered species
    cue_vals = {sp: random_row[f"CUE_{comm_prefix}_Sp{sp}"] for sp in species_ids}
    
    # Extract alpha vectors for each filtered species
    alpha_matrix = {}
    for sp in species_ids:
        raw_string = fix_alpha_string(random_row[f"alpha_{comm_prefix}_Sp{sp}"])
        alpha_array = np.array(ast.literal_eval(raw_string))
        alpha_matrix[sp] = alpha_array

    # Compute pairwise average interaction weights between species pairs
    all_weights = [
        (abs(alpha_matrix[i][species_ids.index(j)]) + abs(alpha_matrix[j][species_ids.index(i)])) / 2
        for i in species_ids for j in species_ids if i < j
    ]
    # Use the specified quantile as threshold to keep only strong interactions
    threshold = np.quantile(all_weights, quantile)

    # Create the network graph and add nodes with their CUE attribute
    G = nx.Graph()
    for sp in species_ids:
        G.add_node(sp, cue=cue_vals[sp])
    
    # Add edges between species pairs if their interaction weight exceeds the threshold
    for i in species_ids:
        for j in species_ids:
            if i < j:
                w = (abs(alpha_matrix[i][species_ids.index(j)]) + abs(alpha_matrix[j][species_ids.index(i)])) / 2
                if w > threshold:
                    G.add_edge(i, j, weight=w)

    return G



# ======================= visualization =======================
def plot_network(G, title):
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]['cue'] for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, cmap='viridis', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_weights, edge_cmap=plt.cm.coolwarm, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights), width=2, ax=ax)

    sm_nodes = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm_nodes.set_array([])
    plt.colorbar(sm_nodes, ax=ax, shrink=0.7, label="Mean CUE")

    sm_edges = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm_edges.set_array([])
    plt.colorbar(sm_edges, ax=ax, shrink=0.7, label="Interaction Strength")

    ax.set_title(title)
    ax.axis("off")
    plt.show()

def plot_CUE_vs_interaction(df):
    comms = ["Comm1", "Comm2", "Comm3"]
    labels = ["Community 1", "Community 2", "Community 3"]
    colors = ["red", "green", "blue"]
    all_results = [summarize_community_CUE_interaction(df, comm) for comm in comms]
    df_summary = pd.concat(all_results, ignore_index=True)

    pvals, rhos = [], []
    for comm in comms:
        df_comm = df_summary[df_summary["community"] == comm]
        rho, pval = spearmanr(df_comm["community_CUE"], df_comm["interaction_strength"])
        pvals.append(pval)
        rhos.append(rho)

    _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, comm, label, color, rho, pval_corr in zip(axes, comms, labels, colors, rhos, pvals_corrected):
        df_comm = df_summary[df_summary["community"] == comm]
        sns.regplot(
            data=df_comm, x="community_CUE", y="interaction_strength",
            scatter_kws={"color": color, "alpha": 0.6},
            line_kws={"color": color, "linestyle": "--"}, ax=ax
        )
        ax.set_title(f"{label}\nρ = {rho:.2f}, adj. p = {pval_corr:.3f}")
        ax.set_xlabel("Community CUE")
        if ax == axes[0]:
            ax.set_ylabel("Interaction Strength")
    plt.tight_layout()
    plt.show()




def plot_CUE_vs_pathlength(df):
    comms = ["Comm1", "Comm2", "Comm3"]
    labels = ["Community 1", "Community 2", "Community 3"]
    colors = ["red", "green", "blue"]

    all_results = [summarize_community_CUE_pathlength(df, comm) for comm in comms]
    df_summary = pd.concat(all_results, ignore_index=True)

    pvals, rhos = [], []
    for comm in comms:
        df_comm = df_summary[df_summary["community"] == comm]
        rho, pval = spearmanr(df_comm["community_CUE"], df_comm["avg_path_length"])
        pvals.append(pval)
        rhos.append(rho)

    _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, comm, label, color, rho, pval_corr in zip(axes, comms, labels, colors, rhos, pvals_corrected):
        df_comm = df_summary[df_summary["community"] == comm]
        sns.regplot(
            data=df_comm, x="community_CUE", y="avg_path_length",
            scatter_kws={"color": color, "alpha": 0.6},
            line_kws={"color": color, "linestyle": "--"}, ax=ax
        )
        ax.set_title(f"{label}\nρ = {rho:.2f}, adj. p = {pval_corr:.3f}")
        ax.set_xlabel("Community CUE")
        if ax == axes[0]:
            ax.set_ylabel("Average Path Length")
    plt.tight_layout()
    plt.show()

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

def plot_degree_distribution(G, log_scale=False):
    degrees = [d for _, d in G.degree()]
    
    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    
    if log_scale:
        plt.yscale("log")
        plt.title("Degree Distribution (Log Scale)")

    plt.tight_layout()
    plt.show()
# ======================= Analysis =======================
from scipy.stats import gengamma
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def compute_inverse_weight_betweenness(G):
    """
   calculate inverse weight betweenness centrality
    """
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        strength = d.get('weight', 0)
        distance = 1.0 / (strength + 1e-10)
        d['weight'] = distance
    return nx.betweenness_centrality(G_inv, weight='weight', normalized=True)
def compute_all_betweenness(df, comm_prefix, seed_range):
    """
    For each replicate (seed), compute betweenness centrality for all nodes
    Returns a DataFrame: rows=seeds, columns=species
    """
    betweenness_data = []

    for seed in seed_range:
        G = build_network(df, comm_prefix, seed=seed)
        if G is None:
            continue

        bc_dict = compute_inverse_weight_betweenness(G)
        row = {"Seed": seed}
        for node, bc in bc_dict.items():
            row[f"Sp{node}"] = bc
        betweenness_data.append(row)

    return pd.DataFrame(betweenness_data)

def summarize_community_CUE_interaction(df, comm_prefix):
    alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}")]
    species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])
    cue_cols = [f"CUE_{comm_prefix}_Sp{sp}" for sp in species_ids]
    cfinal_cols = [f"Cfinal_{comm_prefix}_Sp{sp}" for sp in species_ids]
    alpha_cols = [f"alpha_{comm_prefix}_Sp{sp}" for sp in species_ids]

    result_rows = []
    for idx, row in df.iterrows():
        try:
            cue = np.array([row[col] for col in cue_cols])
            cfinal = np.array([row[col] for col in cfinal_cols])
            alpha_matrix = np.stack([np.array(ast.literal_eval(fix_alpha_string(row[col]))) for col in alpha_cols])
            total_abundance = np.sum(cfinal)
            comm_cue = np.sum(cue * cfinal) / total_abundance if total_abundance > 0 else np.nan
            upper = np.triu(np.abs(alpha_matrix), k=1)
            n_pairs = np.count_nonzero(upper)
            mean_strength = np.sum(upper) / n_pairs if n_pairs > 0 else np.nan
            result_rows.append({"seed": idx, "community": comm_prefix, "community_CUE": comm_cue, "interaction_strength": mean_strength})
        except Exception as e:
            print(f"Failed on row {idx} ({comm_prefix}): {e}")
    return pd.DataFrame(result_rows)

def summarize_community_CUE_pathlength(df, comm_prefix):
    alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}")]
    species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])
    cue_cols = [f"CUE_{comm_prefix}_Sp{sp}" for sp in species_ids]
    cfinal_cols = [f"Cfinal_{comm_prefix}_Sp{sp}" for sp in species_ids]
    alpha_cols = [f"alpha_{comm_prefix}_Sp{sp}" for sp in species_ids]

    result_rows = []
    for idx, row in df.iterrows():
        try:
            cue = np.array([row[col] for col in cue_cols])
            cfinal = np.array([row[col] for col in cfinal_cols])
            alpha_matrix = np.stack([np.array(ast.literal_eval(fix_alpha_string(row[col]))) for col in alpha_cols])
            total_abundance = np.sum(cfinal)
            comm_cue = np.sum(cue * cfinal) / total_abundance if total_abundance > 0 else np.nan

            G = nx.Graph()
            for i in range(len(species_ids)):
                G.add_node(i)
            for i in range(len(species_ids)):
                for j in range(i + 1, len(species_ids)):
                    strength = (abs(alpha_matrix[i][j]) + abs(alpha_matrix[j][i])) / 2
                    if strength > 0:
                        G.add_edge(i, j, weight=strength)

            # 倒数加权路径长度
            G_inv = G.copy()
            for u, v, d in G_inv.edges(data=True):
                d['weight'] = 1.0 / (d['weight'] + 1e-8)
            if nx.is_connected(G_inv):
                avg_path = nx.average_shortest_path_length(G_inv, weight='weight')
            else:
                largest = max(nx.connected_components(G_inv), key=len)
                subgraph = G_inv.subgraph(largest)
                avg_path = nx.average_shortest_path_length(subgraph, weight='weight')

            result_rows.append({
                "seed": idx,
                "community": comm_prefix,
                "community_CUE": comm_cue,
                "avg_path_length": avg_path
            })
        except Exception as e:
            print(f"Failed on row {idx} ({comm_prefix}): {e}")
    return pd.DataFrame(result_rows)

# degree & CUE

def gengamma_regression(x, a, c, d, loc, scale):
    return a * gengamma.pdf(x, a=c, c=d, loc=loc, scale=scale)

def analyze_gengamma_fit(df, comm_prefix, seed=None):
    G = build_network(df, comm_prefix, seed=seed)
    if G is None:
        return None

    cue = []
    degree = []
    for node in G.nodes:
        d = nx.degree_centrality(G)[node]
        c = G.nodes[node]["cue"]
        if d > 0:
            cue.append(c)
            degree.append(d)

    cue = np.array(cue)
    degree = np.array(degree)

    mask = degree > 0.01
    cue = cue[mask]
    degree = degree[mask]

    if len(degree) < 5 or np.all(degree < 1e-5):
        return None

    for attempt in range(3):
        try:
            scale_guess = max(np.std(cue), 0.01)
            p0 = [np.mean(degree), 1.0 + 0.5 * attempt, 2.0, np.min(cue), scale_guess]
            bounds = (
                [0, 0.01, 0.01, np.min(cue) - 0.01, 1e-4],
                [10 * np.max(degree), 10, 10, np.max(cue) + 0.01, 1.0]
            )
            params, _ = curve_fit(gengamma_regression, cue, degree,
                                  p0=p0, bounds=bounds, maxfev=10000)
            break
        except Exception:
            return None

    a, c_, d_, loc_, scale_ = params
    cue_sorted = np.sort(cue)
    fitted_y = gengamma_regression(cue_sorted, a, c_, d_, loc_, scale_)
    y_pred_all = gengamma_regression(cue, a, c_, d_, loc_, scale_)
    r2 = r2_score(degree, y_pred_all)

    if r2 < 0:
        return None

    peak_idx = np.argmax(fitted_y)
    peak_cue = cue_sorted[peak_idx]
    peak_degree = fitted_y[peak_idx]

    return {
        "cue": cue,
        "degree": degree,
        "cue_sorted": cue_sorted,
        "fitted_y": fitted_y,
        "params": (a, c_, d_, loc_, scale_),
        "r2": r2,
        "peak_cue": peak_cue,
        "peak_degree": peak_degree
    }

def get_all_replicate_data(df, comm_prefix, seed_range):
    cues = []
    degrees = []

    for seed in seed_range:
        G = build_network(df, comm_prefix, seed=seed)
        if G is None:
            continue
        for node in G.nodes:
            d = nx.degree_centrality(G)[node]
            c = G.nodes[node]["cue"]
            if d > 0:
                cues.append(c)
                degrees.append(d)

    return np.array(cues), np.array(degrees)


# ======================= Usage =======================
df = pd.read_csv("../data/elv_hpc.csv")
comms = ["Comm1", "Comm2", "Comm3"]
labels = ["Community 1", "Community 2", "Community 3"]
colors = ["red", "green", "blue"]
seed = 52

for i, comm in enumerate(comms):
    G = build_network(df, comm, seed=seed)
    plot_network(G, f"Network of {comm}")
    result = analyze_gengamma_fit(df, comm, seed=seed)
    plot_gengamma_fit(result, f"{comm} (seed {seed})", colors[i])

# plot_CUE_vs_interaction(df)
# plot_CUE_vs_pathlength(df)

# %%
# plot degree&CUE for all replicates
colors = ["red", "green", "blue"]
communities = ["Comm1", "Comm2", "Comm3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    # 收集所有 replicate 的数据
    cue, degree = get_all_replicate_data(df, comm, range(51, 101))
    mask = degree > 0.01
    cue = cue[mask]
    degree = degree[mask]

    # 拟合 generalized gamma
    p0 = [np.mean(degree), 1.0, 2.0, np.min(cue), max(np.std(cue), 0.01)]
    bounds = ([0, 0.01, 0.01, np.min(cue)-0.01, 1e-4],
              [10*np.max(degree), 10, 10, np.max(cue)+0.01, 1.0])
    try:
        params, _ = curve_fit(gengamma_regression, cue, degree, p0=p0, bounds=bounds)
    except:
        print(f"Fit failed for {comm}")
        continue

    cue_sorted = np.sort(cue)
    fitted_y = gengamma_regression(cue_sorted, *params)
    y_pred_all = gengamma_regression(cue, *params)
    r2 = r2_score(degree, y_pred_all)

    # Peak
    peak_idx = np.argmax(fitted_y)
    peak_cue = cue_sorted[peak_idx]
    peak_degree = fitted_y[peak_idx]

    # 绘图
    ax = axes[i]
    ax.scatter(cue, degree, alpha=0.5, color=colors[i], s=10, label="Data points")
    ax.plot(cue_sorted, fitted_y, color='black', label="Gamma Fit")
    ax.axvline(peak_cue, color='gray', linestyle='--', label=f"Peak CUE = {peak_cue:.3f}")
    ax.set_title(f"{comm} (R² = {r2:.2f})")
    ax.set_xlabel("CUE")
    if i == 0:
        ax.set_ylabel("Degree Centrality")
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# %%
# degree distribution
colors = ["red", "green", "blue"]
communities = ["Comm1", "Comm2", "Comm3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    all_degrees = []

    for seed in range(51, 101):
        G = build_network(df, comm, seed=seed)
        if G is None:
            continue

        degrees = [d for _, d in G.degree()]
        all_degrees.extend(degrees)

    all_degrees = np.array(all_degrees)

    # Compute histogram
    counts, bins = np.histogram(all_degrees, bins=30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Mask out zero counts for log10
    mask = counts > 0
    log_counts = np.log10(counts[mask])
    bin_centers = bin_centers[mask]

    # Plot
    ax = axes[i]
    ax.plot(bin_centers, log_counts, marker='o', linestyle='-', color=colors[i])
    ax.set_xlabel("Degree")
    ax.set_title(f"{comm}")

axes[0].set_ylabel("log10(Frequency)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# %%
# Betweenness
from collections import defaultdict

all_betweenness = defaultdict(list)
for comm in ["Comm1", "Comm2", "Comm3"]:
    for seed in range(51, 101):
        G = build_network(df, comm, seed=seed)
        if G is None:
            continue
        bws = compute_inverse_weight_betweenness(G)
        all_betweenness[comm].extend(list(bws.values()))

colors = ["red", "green", "blue"]
communities = ["Comm1", "Comm2", "Comm3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    values = np.array(all_betweenness[comm])
    bins = 30
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    ax = axes[i]
    ax.plot(bin_centers, counts, marker='o', linestyle='-', color=colors[i])
    ax.set_xlabel("Betweenness")
    ax.set_title(comm)
    ax.set_yscale("log")
    if i == 0:
        ax.set_ylabel("Frequency (log scale)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# %%
# interaction strength for each species
def compute_bidirectional_strengths(df, comm_prefix, seed_range, cfinal_threshold=1e-5):
    all_strengths = []

    for seed in seed_range:
        row = df[df["Seed"] == seed]
        if row.empty:
            continue
        row = row.iloc[0]

        # 获取所有 alpha 列和对应物种 ID
        alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}_Sp")]
        all_species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])

        # 根据 Cfinal 过滤保留的物种
        species_ids = []
        for sp in all_species_ids:
            cval = row.get(f"Cfinal_{comm_prefix}_Sp{sp}", 0)
            if cval > cfinal_threshold:
                species_ids.append(sp)

        S = len(species_ids)
        if S < 2:
            continue  # 物种太少跳过

        # 构建 alpha 矩阵（只包含筛选后物种之间的相互作用）
        alpha_matrix = np.zeros((S, S))
        for i, sp_i in enumerate(species_ids):
            try:
                raw = row[f"alpha_{comm_prefix}_Sp{sp_i}"]
                if isinstance(raw, str):
                    raw = fix_alpha_string(raw)
                    alpha_vec = ast.literal_eval(raw)
                else:
                    alpha_vec = raw

                # 获取该向量中对应筛选物种的位置
                selected_indices = [all_species_ids.index(sp_j) for sp_j in species_ids]
                filtered_vec = np.array(alpha_vec)[selected_indices]

                alpha_matrix[i, :] = filtered_vec
            except Exception as e:
                print(f"[Error] Seed={seed}, Sp{sp_i}: {e}")
                alpha_matrix[i, :] = np.nan

        # 计算每个物种的平均 interaction strength
        for i in range(S):
            if np.isnan(alpha_matrix[i, :]).any():
                continue
            total = 0
            count = 0
            for j in range(S):
                if i != j and not (np.isnan(alpha_matrix[i, j]) or np.isnan(alpha_matrix[j, i])):
                    a_ij = alpha_matrix[i, j]
                    a_ji = alpha_matrix[j, i]
                    total += (abs(a_ij) + abs(a_ji)) / 2
                    count += 1
            if count > 0:
                all_strengths.append(total / count)

    return np.array(all_strengths)


# %%
# lognormal fit
from scipy.stats import lognorm
colors = ["red", "green", "blue"]
communities = ["Comm1", "Comm2", "Comm3"]
seed_range = range(51, 101)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    # 获取所有 interaction strengths（你已有该函数）
    strength_vals = compute_bidirectional_strengths(df, comm, seed_range)
    strength_vals = np.array(strength_vals)
    strength_vals = strength_vals[~np.isnan(strength_vals)]  # 去除 nan

    # 拟合 lognormal 分布（scipy 里是按 shape=s, loc, scale 拟合的）
    shape, loc, scale = lognorm.fit(strength_vals, floc=0)  # 通常 fix loc=0 更稳

    # 拟合曲线数据
    x = np.linspace(min(strength_vals), max(strength_vals), 200)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    hist_counts, bin_edges = np.histogram(strength_vals, bins=30, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 画图
    ax = axes[i]
    ax.plot(bin_centers, np.log10(hist_counts), marker='o', linestyle='-', color=colors[i], label="Empirical")
    ax.plot(x, np.log10(pdf), linestyle='--', color='black', label="Lognormal fit")
    ax.set_xlabel("Interaction Strength")
    ax.set_title(f"{comm}")
    if i == 0:
        ax.set_ylabel("log10(Density)")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
# # lognorm Cramér–von Mises test
from scipy.stats import lognorm
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

def cramervonmises_statistic(empirical_cdf, theoretical_cdf):
    """
    Compute Cramér–von Mises statistic: ∑ (F_emp(x) - F_model(x))² over all x
    """
    diffs = empirical_cdf - theoretical_cdf
    return np.mean(diffs**2)

def evaluate_lognorm_cvm(strength_vals):
    strength_vals = np.sort(strength_vals)
    ecdf = ECDF(strength_vals)
    x = strength_vals
    F_emp = ecdf(x)

    # Fit lognormal
    shape, loc, scale = lognorm.fit(strength_vals, floc=0)
    F_model = lognorm.cdf(x, shape, loc=loc, scale=scale)

    # Compute Cramér–von Mises statistic
    cvm_stat = cramervonmises_statistic(F_emp, F_model)
    return cvm_stat, shape, loc, scale

for comm in ["Comm1", "Comm2", "Comm3"]:
    strength_vals = compute_bidirectional_strengths(df, comm, range(51, 101))
    strength_vals = np.array(strength_vals)
    strength_vals = strength_vals[~np.isnan(strength_vals)]

    cvm_stat, shape, loc, scale = evaluate_lognorm_cvm(strength_vals)
    print(f"{comm}: Cramér–von Mises statistic = {cvm_stat:.5f}")

# %%
# edge strength distribution in a net
from scipy import stats
colors = ["red", "green", "blue"]
communities = ["Comm1", "Comm2", "Comm3"]
seed_range = range(51, 101)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    all_weights = []

    for seed in seed_range:
        G = build_network(df, comm, seed=seed)
        if G is None:
            continue
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        all_weights.extend(weights)

    all_weights = np.array(all_weights)

    # 拟合 Gamma 分布（floc=0 更稳定）
    params_gamma = stats.gamma.fit(all_weights, floc=0)
    shape, loc, scale = params_gamma
    print(f"{comm} Gamma params: shape={shape:.4f}, loc={loc:.1e}, scale={scale:.1e}")

    # 拟合曲线
    x = np.linspace(min(all_weights), max(all_weights), 300)
    pdf_gamma = stats.gamma.pdf(x, shape, loc, scale)

    # 画图：数据直方图 + 拟合曲线
    counts, bins = np.histogram(all_weights, bins=30, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax = axes[i]
    ax.plot(bin_centers, counts, 'o', color=colors[i], label='Empirical')
    ax.plot(x, pdf_gamma, linestyle='--', color=colors[i], label='Gamma fit')
    ax.set_title(comm)
    ax.set_xlabel("Edge Interaction Strength")
    if i == 0:
        ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# species level average interaction strength and degree

communities = ["Comm1", "Comm2", "Comm3"]
colors = ["red", "green", "blue"]
seed_range = range(51, 101)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    # 获取 interaction strength（每个物种）
    strengths = compute_bidirectional_strengths(df, comm, seed_range)

    # 获取 degree（每个物种）
    degrees = []
    for seed in seed_range:
        G = build_network(df, comm, seed=seed)
        if G is None:
            continue
        degrees.extend([d for _, d in G.degree()])

    degrees = np.array(degrees)
    strengths = np.array(strengths)

    # 对齐数量
    min_len = min(len(degrees), len(strengths))
    degrees = degrees[:min_len]
    strengths = strengths[:min_len]

    # 作图
    ax = axes[i]
    ax.scatter(degrees, strengths, alpha=0.5, color=colors[i], s=10)
    ax.set_xlabel("Degree")
    ax.set_title(comm)
    if i == 0:
        ax.set_ylabel("Interaction Strength")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%
# CUE and interaction strength (species level)
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


communities = ["Comm1", "Comm2", "Comm3"]
colors = ["red", "green", "blue"]
seed_range = range(51, 101)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, comm in enumerate(communities):
    # 获取 interaction strength（每个物种）
    strengths = compute_bidirectional_strengths(df, comm, seed_range)

    # 获取 CUE（每个物种）
    cues = []
    for seed in seed_range:
        row = df[df["Seed"] == seed]
        if row.empty:
            continue
        row = row.iloc[0]
        cue_cols = [col for col in df.columns if col.startswith(f"CUE_{comm}_Sp")]
        for col in cue_cols:
            sp_id = col.split("_Sp")[-1]
            cfinal_col = f"Cfinal_{comm}_Sp{sp_id}"
            cfinal = row.get(cfinal_col, 0)
            if cfinal > 1e-5:
                cue = row.get(col, np.nan)
                cues.append(cue)

    strengths = np.array(strengths)
    cues = np.array(cues)

    # 对齐长度
    min_len = min(len(cues), len(strengths))
    cues = cues[:min_len]
    strengths = strengths[:min_len]

    # 去除非法值（如非正数）
    valid = strengths > 0
    strengths = strengths[valid]
    cues = cues[valid]

    # 绘制散点图
    ax = axes[i]
    ax.scatter(cues, strengths, alpha=0.3, s=5, color=colors[i], label="Species")

    # 使用 LOWESS 平滑拟合
    if len(cues) >= 10:
        smoothed = lowess(strengths, cues, frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.5, label="LOWESS fit")


    ax.set_xlabel("CUE")
    ax.set_title(comm)
    if i == 0:
        ax.set_ylabel("Interaction Strength")
    ax.legend()

plt.suptitle("CUE vs Interaction Strength (Species Level)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
