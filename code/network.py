import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast

def fix_alpha_string(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].replace("\n", " ")
        s = ",".join(s.split())
        return "[" + s + "]"
    return s

def build_network(df, comm_prefix, quantile = 0.8):
    alpha_cols = [col for col in df.columns if col.startswith(f"alpha_{comm_prefix}")]
    species_ids = sorted([int(col.split("_Sp")[-1]) for col in alpha_cols])

    cue_cols = [f"CUE_{comm_prefix}_Sp{sp}" for sp in species_ids]
    cue_df = df[cue_cols]
    mean_cue = cue_df.mean(axis=0)
    mean_cue.index = species_ids

    alpha_matrix = {}
    for sp in species_ids:
        raw = df[f"alpha_{comm_prefix}_Sp{sp}"].apply(fix_alpha_string)
        alpha_array = np.stack(raw.apply(ast.literal_eval).to_numpy())
        alpha_matrix[sp] = alpha_array.mean(axis=0)
    all_weights = []
    for i in range(len(species_ids)):
        for j in range(i+1, len(species_ids)):
            w = (abs(alpha_matrix[species_ids[i]][j]) + abs(alpha_matrix[species_ids[j]][i])) / 2
            all_weights.append(w)
    threshold = np.quantile(all_weights, quantile)

    G = nx.Graph()
    for sp in species_ids:
        G.add_node(sp, cue=mean_cue[sp])

    for i, sp_i in enumerate(species_ids):
        for j, sp_j in enumerate(species_ids):
            if i < j:
                w = (abs(alpha_matrix[sp_i][j]) + abs(alpha_matrix[sp_j][i]) )/ 2
                if w > threshold:
                    G.add_edge(sp_i, sp_j, weight=w)

    return G

def plot_network(G, title):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable, get_cmap

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]['cue'] for n in G.nodes()]

    # 边权值和颜色归一化
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights)
    min_w = min(edge_weights)

    norm = Normalize(vmin=min_w, vmax=max_w)
    cmap_edges = get_cmap("coolwarm")
    cmap_nodes = get_cmap("viridis")

    edge_colors = [cmap_edges(norm(w)) for w in edge_weights]
    edge_widths = [1 + w / max_w for w in edge_weights]

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 节点和边绘图
    nodes = nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, cmap=cmap_nodes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.9, ax=ax)

    # 添加 colorbar for node cue
    sm_nodes = ScalarMappable(cmap=cmap_nodes)
    sm_nodes.set_array(node_colors)
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.7)
    cbar_nodes.set_label("Mean CUE")

    # 添加 colorbar for edge weight (interaction strength)
    sm_edges = ScalarMappable(norm=norm, cmap=cmap_edges)
    sm_edges.set_array(edge_weights)
    cbar_edges = plt.colorbar(sm_edges, ax=ax, shrink=0.7, orientation='vertical')
    cbar_edges.set_label("Interaction Strength")

    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.show()





df = pd.read_csv("../data/elv_hpc.csv")

for comm in ["Comm1", "Comm2", "Comm3"]:
    G = build_network(df, comm)
    plot_network(G, f"Species Interaction Network for {comm}")
