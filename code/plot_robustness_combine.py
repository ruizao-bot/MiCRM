"""
Combined robustness figure:
  Panel A: ΔCUE vs ΔSimilarity for serial transfer (from plot_serial.py)
  Panel B: Prediction accuracy vs ρ₃ with 95% CI (from plot_rho_sweep.py)

Output: figure/robustness_combine.pdf
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import stats

warnings.filterwarnings("ignore")

SURVIVAL_THRESHOLD = 1e-5

pal_rgb = {"1": "#D8A39A", "2": "#A8C3A6"}

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
    "ps.fonttype": 42,
})


def style_ax(ax, linewidth=0.4):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(linewidth)
    ax.tick_params(axis="both", width=linewidth, colors="black", pad=4)
    ax.grid(False)


def first_unique(series):
    vals = pd.Series(series).dropna().unique()
    return vals[0] if len(vals) > 0 else np.nan


# ── Panel A data: serial transfer ─────────────────────────────────────────────
df_s = pd.read_csv("data/coal_serial.csv")
df_s = df_s[df_s["Seed"].isin(sorted(df_s["Seed"].unique())[:50])]
df_s["Community"] = df_s["Community"].astype(str)
df_s["Community_CUE_surv"] = df_s["Community_CUE"]

df_surv_s = df_s[df_s["Abundance"] > SURVIVAL_THRESHOLD].copy()
df_surv_s["Global_Species_ID"] = np.where(
    df_surv_s["Community"] == "2",
    df_surv_s["Species_ID"] + 100,
    df_surv_s["Species_ID"]
)

bray_rows = []
for s in sorted(df_surv_s["Seed"].unique()):
    df_seed = df_surv_s[df_surv_s["Seed"] == s].copy()
    comm_mat = (
        df_seed[["Community", "Global_Species_ID", "Abundance"]]
        .pivot_table(index="Community", columns="Global_Species_ID",
                     values="Abundance", aggfunc="sum", fill_value=0)
        .reindex(["1", "2", "3"]).fillna(0)
    )
    bc = squareform(pdist(comm_mat.values, metric="braycurtis"))
    cue1 = first_unique(df_seed.loc[df_seed["Community"] == "1", "Community_CUE_surv"])
    cue2 = first_unique(df_seed.loc[df_seed["Community"] == "2", "Community_CUE_surv"])
    bray_rows.append({
        "Seed": s, "CUE_1": cue1, "CUE_2": cue2,
        "Sim_3vs1": 1 - bc[2, 0], "Sim_3vs2": 1 - bc[2, 1],
    })

bray_df = pd.DataFrame(bray_rows)
bray_df["CUE_Diff"] = bray_df["CUE_1"] - bray_df["CUE_2"]
bray_df["Sim_Diff"] = bray_df["Sim_3vs1"] - bray_df["Sim_3vs2"]

dom_seed = (
    df_s[df_s["Community"] == "1"]
    .groupby("Seed", as_index=False)
    .agg(Dominant_Community=("Dominant_Community", first_unique))
)
df_diff = bray_df.merge(dom_seed, on="Seed", how="left")
df_diff["DomGroup"] = np.select(
    [df_diff["Dominant_Community"] == "Community 1",
     df_diff["Dominant_Community"] == "Community 2"],
    ["Community 1", "Community 2"],
    default="Neither"
)


# ── Panel B data: ρ₃ sweep accuracy ──────────────────────────────────────────
def compute_accuracy_ci(df):
    df = df.copy()
    df["Community"] = df["Community"].astype(str)
    df_surv = df[df["Abundance"] > SURVIVAL_THRESHOLD].copy()
    df_surv["Global_Species_ID"] = np.where(
        df_surv["Community"] == "2",
        df_surv["Species_ID"] + 100,
        df_surv["Species_ID"]
    )
    cue_df = (
        df[df["Community"].isin(["1", "2"])]
        .groupby(["Seed", "Community"], as_index=False)
        .agg(cue=("Community_CUE_surv", first_unique))
    )
    cue_pivot = cue_df.pivot_table(index="Seed", columns="Community", values="cue")

    hits = []
    for s in sorted(df_surv["Seed"].unique()):
        sd = df_surv[df_surv["Seed"] == s]
        cm = (sd.pivot_table(index="Community", columns="Global_Species_ID",
                              values="Abundance", aggfunc="sum", fill_value=0)
                .reindex(["1", "2", "3"]).fillna(0))
        if cm.shape[0] < 3:
            continue
        bc = squareform(pdist(cm.values, metric="braycurtis"))
        sim_diff = (1 - bc[2, 0]) - (1 - bc[2, 1])
        if s not in cue_pivot.index:
            continue
        row = cue_pivot.loc[s]
        cue_diff = row.get("1", np.nan) - row.get("2", np.nan)
        if np.isnan(cue_diff):
            continue
        hits.append(int(np.sign(cue_diff) == np.sign(sim_diff)))

    n, k = len(hits), sum(hits)
    if n == 0:
        return np.nan, np.nan, np.nan
    z = stats.norm.ppf(0.975)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return p_hat, max(0.0, centre - half), min(1.0, centre + half)


df_rho = pd.read_csv("data/coal_rho_sweep.csv")
df_rho = df_rho[df_rho["Seed"].isin(sorted(df_rho["Seed"].unique())[:50])]
rho_values = sorted(df_rho["Rho3"].unique())

accs, ci_lo, ci_hi = [], [], []
for rho3 in rho_values:
    acc, lo, hi = compute_accuracy_ci(df_rho[df_rho["Rho3"] == rho3])
    accs.append(acc)
    ci_lo.append(lo)
    ci_hi.append(hi)

accs  = np.array(accs)
err_lo = accs - np.array(ci_lo)
err_hi = np.array(ci_hi) - accs


# ── Combined figure ───────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel A ───────────────────────────────────────────────────────────────────
for grp in ["Community 1", "Community 2"]:
    dat = df_diff[df_diff["DomGroup"] == grp]
    ax_a.scatter(
        dat["CUE_Diff"], dat["Sim_Diff"],
        s=100, alpha=0.65,
        facecolors=pal_rgb[grp[-1]],
        edgecolors="black", linewidths=0.5,
        label=grp, zorder=3
    )
ax_a.axhline(0, linestyle="--", color="black", linewidth=0.7)
ax_a.set_xlabel("Community CUE difference")
ax_a.set_ylabel("Similarity difference")
ax_a.legend(title="Dominant community", loc="lower right",
            frameon=True, edgecolor="black", framealpha=0.6)
style_ax(ax_a)

# ── Panel B ───────────────────────────────────────────────────────────────────
x = np.arange(len(rho_values))
ax_b.bar(x, accs * 100, color="#9FB7CC", edgecolor="none", width=0.55, zorder=3)
ax_b.errorbar(x, accs * 100,
              yerr=[err_lo * 100, err_hi * 100],
              fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)
ax_b.axhline(80, linestyle="--", color="gray", linewidth=0.8, zorder=2)
ax_b.set_xticks(x)
ax_b.set_xticklabels([str(r) for r in rho_values])
ax_b.set_xlabel(r"Resource inflow rate of coalesced community")
ax_b.set_ylabel("Prediction accuracy (%)")
ax_b.set_ylim(0, 105)
ax_b.set_yticks([0, 20, 40, 60, 80, 100])
for xi, acc, hi in zip(x, accs, err_hi):
    ax_b.text(xi, (acc + hi) * 100 + 1, f"{acc:.0%}", ha="center", va="bottom", fontsize=12)
style_ax(ax_b, linewidth=0.8)

# 在图框外添加 (A) (B) 标注（不加粗，仅括号）
fig.text(0.01, 0.98, '(A)', ha='left', va='top', fontsize=15)
fig.text(0.51, 0.98, '(B)', ha='left', va='top', fontsize=15)

plt.tight_layout()
plt.savefig("figure/robustness_combine.pdf", bbox_inches="tight")
plt.show()
