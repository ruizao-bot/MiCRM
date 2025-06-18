# --- Analysis: CUE vs Survival Probability ---
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/rare_invade_hpc.csv")
# Scatter plot
plt.figure(figsize=(6,4))
plt.scatter(df["CUE"], df["Survival"], alpha=0.3)
plt.xlabel("CUE")
plt.ylabel("Survival (1=Yes, 0=No)")
plt.title("CUE vs. Survival (Rare Invasion, Invaders Only)")
plt.tight_layout()
plt.show()

# Logistic regression
X = sm.add_constant(df["CUE"])
y = df["Survival"]
model = sm.Logit(y, X).fit(disp=0)
print(model.summary())
odds_ratio = np.exp(model.params["CUE"])
print(f"Odds ratio for CUE: {odds_ratio:.2f}")

# Plot predicted probability
cue_range = np.linspace(df["CUE"].min(), df["CUE"].max(), 100)
X_pred = sm.add_constant(cue_range)
pred_prob = model.predict(X_pred)
plt.figure(figsize=(6,4))
plt.plot(cue_range, pred_prob, label="Predicted Survival Probability")
plt.xlabel("CUE")
plt.ylabel("Probability of Survival")
plt.title("Effect of CUE on Invader Survival (Rare Invasion)")
plt.legend()
plt.tight_layout()
plt.show()