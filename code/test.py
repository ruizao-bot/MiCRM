# 把每个物种的 actual_CUE 按 "该 seed 的存活物种数" 上色
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/coal.csv")
df_c1 = df[df["Community"] == 1]

# 每个 seed 的存活物种数（你已经存了 N_Survivors）
plt.figure(figsize=(10, 6))
sc = plt.scatter(df_c1["actual_CUE"], df_c1["Abundance"],
                 c=df_c1["N_Survivors"], cmap="viridis",
                 alpha=0.5, s=10)
plt.colorbar(sc, label="N_Survivors (per seed)")
plt.yscale("log")
plt.xlabel("actual_CUE")
plt.ylabel("Abundance")
plt.show()

# 你的 main.py 已经存了 depletion = sum(R_final)
df_seed = df_c1.groupby("Seed").agg(
    cue_mean=("actual_CUE", "mean"),
    depletion=("Depletion", "first"),
    n_surv=("N_Survivors", "first"),
)
plt.scatter(df_seed["cue_mean"], df_seed["depletion"])
plt.xlabel("Seed-mean actual_CUE")
plt.ylabel("Depletion (sum of R at end)")
plt.show()