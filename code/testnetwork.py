import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
import matplotlib as mpl 
from scipy.stats import lognorm, gamma
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

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

def plot_cue_vs_strength(cues, strengths, community, color, show_ylabel=False):
    plt.scatter(cues, strengths, alpha=0.3, s=5, color=color, label="Species")
    if len(cues) >= 10:
        smoothed = lowess(strengths, cues, frac=0.3, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.5, label="LOWESS fit")
    plt.xlabel("CUE")

    ax = plt.gca()
    if show_ylabel:
        plt.ylabel("Interaction Strength")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False)

    plt.title(f"{community}")
   # plt.legend()


# ========== Main Analysis ==========
# %%
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
if __name__ == "__main__":
    df = pd.read_csv("../data/elv_hpc_sameR0.csv")
    communities = ["Comm1", "Comm2", "Comm3"]
    pal_rgb = {"Comm1": "#E74C3C", "Comm2": "#2ECC71", "Comm3": "#3498DB"}
    colors = [pal_rgb[c] for c in communities]
    seed_range = range(51, 101)

    # Plot network for a single seed
    for comm in communities:
        color = pal_rgb[comm]
        G = build_network(df, comm, seed=52)
        if G is not None:
            plot_network(G, f"Network of {comm} (Seed 52)")

    # # Degree distribution
    # plt.figure(figsize=(15, 4))
    # for i, comm in enumerate(communities):
    #     color = pal_rgb[comm]
    #     degrees = compute_degree_distribution(df, comm, seed_range)
    #     plt.subplot(1, 3, i+1)
    #     plot_degree_hist(degrees, comm, color)
    # plt.tight_layout()
    # plt.show()

    # # Interaction strength distribution
    # plt.figure(figsize=(15, 4))
    # for i, comm in enumerate(communities):
    #     color = pal_rgb[comm]
    #     strengths = compute_bidirectional_strengths(df, comm, seed_range)
    #     plt.subplot(1, 3, i+1)
    #     plot_strength_hist(strengths, comm, color)
    # plt.tight_layout()
    # plt.show()

    # # CUE vs Degree
    # plt.figure(figsize=(15, 4))
    # for i, comm in enumerate(communities):
    #     color = pal_rgb[comm]
    #     cues, degrees = compute_cue_and_degree(df, comm, seed_range)
    #     plt.subplot(1, 3, i+1)
    #     plot_cue_vs_degree(cues, degrees, comm, color)
    # plt.tight_layout()
    # plt.show()

    # CUE vs Interaction Strength
    # %%
cm_to_inch = lambda cm: cm / 2.54
fig_width_in = cm_to_inch(21)
fig_height_in = fig_width_in * 0.3

plt.figure(figsize=(fig_width_in, fig_height_in))


for i, comm in enumerate(["Comm1", "Comm2", "Comm3"]):
    plt.subplot(1, 3, i + 1)
    cues = compute_cue_and_degree(df, comm, seed_range)[0]
    strengths = compute_bidirectional_strengths(df, comm, seed_range)
    min_len = min(len(cues), len(strengths))
    cues = cues[:min_len]
    strengths = strengths[:min_len]

    show_ylabel = (i == 0)  # 只对第一个子图显示 y 轴标题
    plot_cue_vs_strength(cues, strengths, comm, color=pal_rgb[comm], show_ylabel=show_ylabel)

plt.tight_layout()
plt.savefig("../results/CUE_interaction.png", dpi=600, bbox_inches="tight")
plt.show()

    # # --- Community-level CUE vs Average Degree (with pal_rgb color) ---
    # avg_degrees = []
    # comm_cues = []
    # comm_ids = []
    # seed_ids = []

    # for comm in communities:
    #     for seed in seed_range:
    #         G = build_network(df, comm, seed)
    #         if G is None:
    #             continue
    #         degrees = [G.degree(n) for n in G.nodes()]
    #         if len(degrees) == 0:
    #             continue
    #         avg_degree = np.mean(degrees)
    #         cues = [G.nodes[n]['cue'] for n in G.nodes()]
    #         comm_cue = np.mean(cues)
    #         avg_degrees.append(avg_degree)
    #         comm_cues.append(comm_cue)
    #         comm_ids.append(comm)
    #         seed_ids.append(seed)

    # df_plot = pd.DataFrame({
    #     "Community": comm_ids,
    #     "Seed": seed_ids,
    #     "AvgDegree": avg_degrees,
    #     "CommunityCUE": comm_cues
    # })

    # plt.figure(figsize=(7, 5))
    # for comm in communities:
    #     color = pal_rgb[comm]
    #     sub = df_plot[df_plot["Community"] == comm]
    #     plt.scatter(sub["AvgDegree"], sub["CommunityCUE"], color=color, label=comm, alpha=0.7)
    #     # Linear regression for each community
    #     if len(sub) > 1:
    #         X = sm.add_constant(sub["AvgDegree"])
    #         model = sm.OLS(sub["CommunityCUE"], X).fit()
    #         x_pred = np.linspace(sub["AvgDegree"].min(), sub["AvgDegree"].max(), 100)
    #         y_pred = model.predict(sm.add_constant(x_pred))
    #         plt.plot(x_pred, y_pred, color=color, linestyle="--", label=f"{comm} fit (R²={model.rsquared:.2f})")
    # plt.xlabel("Average Degree")
    # plt.ylabel("Community-level CUE")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
# %%
    # # --- Community-level CUE vs Average Interaction Strength (with pal_rgb color) ---
    # avg_strengths = []
    # comm_cues = []
    # comm_ids = []
    # seed_ids = []

    # for comm in communities:
    #     for seed in seed_range:
    #         G = build_network(df, comm, seed)
    #         if G is None:
    #             continue
    #         strengths = []
    #         for node in G.nodes:
    #             neighbors = list(G.neighbors(node))
    #             if not neighbors:
    #                 continue
    #             node_strengths = [G[node][nbr]['weight'] for nbr in neighbors]
    #             strengths.append(np.mean(node_strengths))
    #         if len(strengths) == 0:
    #             continue
    #         avg_strength = np.mean(strengths)
    #         cues = [G.nodes[n]['cue'] for n in G.nodes()]
    #         comm_cue = np.mean(cues)
    #         avg_strengths.append(avg_strength)
    #         comm_cues.append(comm_cue)
    #         comm_ids.append(comm)
    #         seed_ids.append(seed)

    # df_plot_strength = pd.DataFrame({
    #     "Community": comm_ids,
    #     "Seed": seed_ids,
    #     "AvgStrength": avg_strengths,
    #     "CommunityCUE": comm_cues
    # })

    # plt.figure(figsize=(7, 5))
    # for comm in communities:
    #     color = pal_rgb[comm]
    #     sub = df_plot_strength[df_plot_strength["Community"] == comm]
    #     plt.scatter(sub["AvgStrength"], sub["CommunityCUE"], color=color, label=comm, alpha=0.7)
    #     # Linear regression for each community
    #     if len(sub) > 1:
    #         X = sm.add_constant(sub["AvgStrength"])
    #         model = sm.OLS(sub["CommunityCUE"], X).fit()
    #         x_pred = np.linspace(sub["AvgStrength"].min(), sub["AvgStrength"].max(), 100)
    #         y_pred = model.predict(sm.add_constant(x_pred))
    #         plt.plot(x_pred, y_pred, color=color, linestyle="--", label=f"{comm} fit (R²={model.rsquared:.2f})")
    # plt.xlabel("Average Interaction Strength")
    # plt.ylabel("Community-level CUE")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
# # %%
# import matplotlib
# import matplotlib.pyplot as plt

# # 设置字体和字号，与高斯拟合图一致
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12

# # A4 宽度尺寸
# cm_to_inch = lambda cm: cm / 2.54
# fig_width_in = cm_to_inch(21)
# fig_height_in = fig_width_in * 0.35

# fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=True)

# for i, comm in enumerate(["Comm1", "Comm2", "Comm3"]):
#     cues = compute_cue_and_degree(df, comm, seed_range)[0]
#     strengths = compute_bidirectional_strengths(df, comm, seed_range)
#     min_len = min(len(cues), len(strengths))
#     cues = cues[:min_len]
#     strengths = strengths[:min_len]

#     show_ylabel = (i == 0)
#     plt.sca(axes[i])
#     plot_cue_vs_strength(cues, strengths, comm, color=pal_rgb[comm], show_ylabel=show_ylabel)

# plt.tight_layout(rect=[0, 0, 1, 0.92])
# plt.show()
# %%
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score         
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'serif'

# 设置字号
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

# A4 宽度尺寸
cm_to_inch = lambda cm: cm / 2.54
fig_width_in = cm_to_inch(21)
fig_height_in = fig_width_in * 0.3

fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=True)

# -------- 高斯函数 --------
def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B


for i, comm in enumerate(communities):
    cues, degrees = compute_cue_and_degree(df, comm, seed_range)
    mask = degrees > 0.01
    cues, degrees = cues[mask], degrees[mask]

    if len(cues) < 5:          # 数据太少时跳过
        print(f"Not enough data for {comm}")
        continue

    # -------- 高斯拟合 --------
    # 初始猜值：A=振幅≈(max-min), mu≈峰位置, sigma≈半宽, B≈底
    A0   = degrees.max() - degrees.min()
    mu0  = cues[np.argmax(degrees)]
    sig0 = (cues.max() - cues.min()) / 6      # 经验：6σ≈全宽
    B0   = degrees.min()
    p0   = [A0, mu0, sig0, B0]

    # 约束：A>0, sigma>0
    bounds = ([0, cues.min(), 1e-4, 0],
              [np.inf, cues.max(), np.inf, np.inf])

    try:
        popt, _ = curve_fit(gaussian, cues, degrees,
                            p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        print(f"Gaussian fit failed for {comm}")
        continue

    # 生成平滑曲线
    xfit = np.linspace(cues.min(), cues.max(), 200)
    yfit = gaussian(xfit, *popt)

    # 拟合优度 R²
    ypred_all = gaussian(cues, *popt)
    r2 = r2_score(degrees, ypred_all)

    # 峰值位置
    peak_cue    = popt[1]
    peak_degree = gaussian(peak_cue, *popt)

    # -------- 绘图 --------
    ax = axes[i]
    ax.scatter(cues, degrees, alpha=0.3,
               color=colors[i], s=2, label="Data points")
    ax.plot(xfit, yfit, color='black', lw=1, label="Gaussian fit")
    ax.axvline(peak_cue, color='gray', ls='--',
               label=f"Peak CUE = {peak_cue:.3f}")
    #ax.set_title(f"{comm} (R² = {r2:.2f})")
    ax.set_xlabel("CUE")
    if i == 0:
        ax.set_ylabel("Degree")
    #ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("../results/CUE_degree_guassian.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
# %% test fitness
# from scipy import stats
# from sklearn.model_selection import ShuffleSplit
# from sklearn.metrics   import mean_squared_error

# #……（popt, pcov 已由 curve_fit 得到）……
# n = len(degrees); k = len(popt)
# y_hat = gaussian(cues, *popt)
# resid = degrees - y_hat
# RSS   = np.sum(resid**2)

# # R², RMSE
# R2   = r2_score(degrees, y_hat)
# RMSE = np.sqrt(RSS / n)

# # AIC, BIC
# AIC = n*np.log(RSS/n) + 2*k
# BIC = n*np.log(RSS/n) + k*np.log(n)

# # F-test vs 常数模型
# RSS_null = np.sum((degrees - degrees.mean())**2)
# df_full  = n - k
# df_null  = n - 1
# F  = ((RSS_null - RSS) / (df_null - df_full)) / (RSS / df_full)
# pF = 1 - stats.f.cdf(F, df_null - df_full, df_full)

# print(f"R²={R2:.3f}, RMSE={RMSE:.3f}, AIC={AIC:.1f}, BIC={BIC:.1f}, F={F:.2f}, p={pF:.3g}")

# # k-fold CV (e.g. 5-fold, 100 repeats)
# rs = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# mse_cv = []
# for train, test in rs.split(cues):
#     popt_i, _ = curve_fit(gaussian, cues[train], degrees[train], p0=popt, bounds=bounds)
#     y_pred = gaussian(cues[test], *popt_i)
#     mse_cv.append(mean_squared_error(degrees[test], y_pred))
# print("CV-MSE=", np.mean(mse_cv))

# %%
# # #### Community CUE as a function of interaction strength
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming df, communities, colors, and seed_range are already defined

# avg_strengths = []
# comm_cues = []
# comm_ids = []
# seed_ids = []

# for comm in communities:
#     for seed in seed_range:
#         G = build_network(df, comm, seed)
#         if G is None:
#             continue
#         # Average interaction strength for this network
#         strengths = []
#         for node in G.nodes:
#             neighbors = list(G.neighbors(node))
#             if not neighbors:
#                 continue
#             node_strengths = [G[node][nbr]['weight'] for nbr in neighbors]
#             strengths.append(np.mean(node_strengths))
#         if len(strengths) == 0:
#             continue
#         avg_strength = np.mean(strengths)
#         # Community-level CUE (mean of all node CUEs)
#         cues = [G.nodes[n]['cue'] for n in G.nodes()]
#         comm_cue = np.mean(cues)
#         avg_strengths.append(avg_strength)
#         comm_cues.append(comm_cue)
#         comm_ids.append(comm)
#         seed_ids.append(seed)

# # Create DataFrame for plotting
# df_plot = pd.DataFrame({
#     "Community": comm_ids,
#     "Seed": seed_ids,
#     "AvgStrength": avg_strengths,
#     "CommunityCUE": comm_cues
# })

# plt.figure(figsize=(7, 5))
# for comm, color in zip(communities, colors):
#     sub = df_plot[df_plot["Community"] == comm]
#     plt.scatter(sub["AvgStrength"], sub["CommunityCUE"], color=color, label=comm, alpha=0.7)
# plt.xlabel("Average Interaction Strength")
# plt.ylabel("Community-level CUE")
# plt.legend()
# plt.tight_layout()
# plt.show()


# %%
