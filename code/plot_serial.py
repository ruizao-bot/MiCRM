# ====================================================================================================
# ======================================== settings =========================================
# ====================================================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

SURVIVAL_THRESHOLD = 1e-5

pal_rgb = {
    "1": "#D8A39A",
    "2": "#A8C3A6",
    "3": "#9FB7CC"
}

community_labels = {
    "1": "Community 1",
    "2": "Community 2",
    "3": "Coalesced Community"
}

xlabels = {
    "1": "Species-level CUE of Community 1",
    "2": "Species-level CUE of Community 2",
    "3": "Species-level CUE of Coalesced Community"
}

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

def style_ax(ax, grid=False):
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

def first_unique(series):
    vals = pd.Series(series).dropna().unique()
    return vals[0] if len(vals) > 0 else np.nan

# ── Load serial transfer data ──────────────────────────────────────────────────
df = pd.read_csv("data/coal_serial.csv")
df = df.rename(columns={"Species_Competition_Dot": "Species_Competition2"})
df["Community"] = df["Community"].astype(str)
df["Species_ID"] = pd.to_numeric(df["Species_ID"], errors="coerce")
df["Species_CUE"] = pd.to_numeric(df["Species_CUE"], errors="coerce")
df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")

# Use Community_CUE as the survivor-weighted CUE (serial transfer equivalent)
df["Community_CUE_surv"] = df["Community_CUE"]

df_surv = df[df["Abundance"] > SURVIVAL_THRESHOLD].copy()
df_surv["log10_Abundance"] = np.log10(df_surv["Abundance"])


# ====================================================================================================
# ================================ Species CUE vs Abundance (no theory curve) ====================
# ====================================================================================================

y_min = np.nanmin(df_surv["log10_Abundance"])
y_max = np.nanmax(df_surv["log10_Abundance"])

fig = plt.figure(figsize=(8.6, 12))
gs = fig.add_gridspec(
    3, 2,
    width_ratios=[1.45, 3.55],
    hspace=0.38,
    wspace=0.06
)

for i, comm in enumerate(["1", "2", "3"]):
    dat_surv = df_surv[df_surv["Community"] == comm].copy()
    dat_surv = dat_surv[
        np.isfinite(dat_surv["Species_CUE"]) &
        np.isfinite(dat_surv["log10_Abundance"])
    ]

    ax_hist = fig.add_subplot(gs[i, 0])
    ax_main = fig.add_subplot(gs[i, 1], sharey=ax_hist)

    ax_hist.hist(
        dat_surv["log10_Abundance"].dropna(),
        bins=50,
        orientation="horizontal",
        color=pal_rgb[comm],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.3
    )
    ax_hist.set_ylim(y_min, y_max)
    ax_hist.invert_xaxis()
    ax_hist.set_xlabel("Density")
    ax_hist.set_ylabel(r"Abundance ($\log_{10}$ scale)", labelpad=10)
    style_ax(ax_hist, grid=False)

    ax_main.scatter(
        dat_surv["Species_CUE"],
        dat_surv["log10_Abundance"],
        s=42,
        alpha=0.55,
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel(xlabels[comm])
    ax_main.set_ylabel("")
    ax_main.tick_params(axis="y", left=False, labelleft=False)
    style_ax(ax_main, grid=False)

plt.tight_layout()
plt.savefig("figure/serial_cue_abundance.pdf", bbox_inches="tight")
plt.show()


# ====================================================================================================
# ====================== ΔCUE vs ΔSimilarity with Dominance (serial transfer) ====================
# ====================================================================================================

df_mut = df_surv.copy()
df_mut["Global_Species_ID"] = np.where(
    df_mut["Community"] == "2",
    df_mut["Species_ID"] + 100,
    df_mut["Species_ID"]
)

bray_rows = []

for s in sorted(df_mut["Seed"].unique()):
    df_seed = df_mut[df_mut["Seed"] == s].copy()

    comm_mat = (
        df_seed[["Community", "Global_Species_ID", "Abundance"]]
        .pivot_table(
            index="Community",
            columns="Global_Species_ID",
            values="Abundance",
            aggfunc="sum",
            fill_value=0
        )
        .reindex(["1", "2", "3"])
        .fillna(0)
    )

    bc = squareform(pdist(comm_mat.values, metric="braycurtis"))

    d31 = bc[2, 0]
    d32 = bc[2, 1]

    cue1 = first_unique(df_seed.loc[df_seed["Community"] == "1", "Community_CUE_surv"])
    cue2 = first_unique(df_seed.loc[df_seed["Community"] == "2", "Community_CUE_surv"])

    bray_rows.append({
        "Seed": s,
        "Bray_3vs1": d31,
        "Bray_3vs2": d32,
        "CUE_1": cue1,
        "CUE_2": cue2,
        "Sim_3vs1": 1 - d31,
        "Sim_3vs2": 1 - d32
    })

bray_results = pd.DataFrame(bray_rows)

df_comm = (
    df[df["Community"].isin(["1", "2"])]
    .groupby(["Seed", "Community"], as_index=False)
    .agg(
        Community_CUE_surv=("Community_CUE_surv", first_unique),
        Dominant_Community=("Dominant_Community", first_unique)
    )
)

df_diff = bray_results.copy()
df_diff["CUE_Diff"] = df_diff["CUE_1"] - df_diff["CUE_2"]
df_diff["Sim_Diff"] = df_diff["Sim_3vs1"] - df_diff["Sim_3vs2"]

dom_seed = (
    df_comm[df_comm["Community"] == "1"][["Seed", "Dominant_Community"]]
    .drop_duplicates()
    .copy()
)

df_diff = df_diff.merge(dom_seed, on="Seed", how="left")
df_diff["DomGroup"] = np.select(
    [
        df_diff["Dominant_Community"] == "Community 1",
        df_diff["Dominant_Community"] == "Community 2"
    ],
    ["Community 1", "Community 2"],
    default="Neither"
)

dom_colors = {
    "Community 1": pal_rgb["1"],
    "Community 2": pal_rgb["2"]
}

fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

for grp in ["Community 1", "Community 2"]:
    dat = df_diff[df_diff["DomGroup"] == grp]
    ax1.scatter(
        dat["CUE_Diff"],
        dat["Sim_Diff"],
        s=100,
        alpha=0.65,
        facecolors=dom_colors[grp],
        edgecolors="black",
        linewidths=0.5,
        label=grp,
        zorder=3
    )

ax1.axhline(0, linestyle="--", color="black", linewidth=0.7)
ax1.set_xlabel(r"Community CUE difference")
ax1.set_ylabel(r"Similarity difference")
ax1.legend(
    title="Dominant community",
    loc="lower right",
    frameon=True,
    edgecolor="black",
    framealpha=0.6
)
style_ax(ax1, grid=False)

plt.tight_layout()
plt.savefig("figure/serial_combine.pdf", bbox_inches="tight")
plt.show()
