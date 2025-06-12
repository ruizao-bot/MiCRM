from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import statsmodels.api as sm

code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)
import param
import CUE

def simulate(seed):
    np.random.seed(seed)
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 200, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50
    m1, m2 = np.full(N1, 0.2), np.full(N2, 0.2)

    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
    rho_pool, omega_pool = np.full(M_pool, 0.6), np.full(M_pool, 0.1)

    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = rho_pool[resource_indices1], omega_pool[resource_indices1]

    if M1 > M2:
        resource_indices2 = np.random.choice(resource_indices1, M2, replace=False)
    elif M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = np.random.choice(remaining_resources, M2 - M1, replace=False)
        resource_indices2 = np.concatenate([resource_indices1, additional_resources])
    else:
        resource_indices2 = resource_indices1.copy()
    remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
    species_indices2 = np.random.choice(remaining_species, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = rho_pool[resource_indices2], omega_pool[resource_indices2]

    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, np.full(N1, 0.01), R0_1)
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, np.full(N2, 0.01), R0_2)

    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    omega3 = omega_pool[resource_indices3]
    rho3 = rho_pool[resource_indices3]
    N3, M3 = N1 + N2, len(resource_indices3)

    # Rare invasion: invader starts rare
    ce1 = sol1.y[:N1, -1]
    ce2 = sol2.y[:N2, -1]
    re1 = sol1.y[N1:, -1]
    re2 = sol2.y[N2:, -1]
    C0_3 = np.concatenate([ce1, ce2 * 0.001])
    R0_3 = re1 + re2

    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3)

    # CUE and survival for invaders in merged community
    _, species_CUE3 = CUE.compute_CUE(sol3, N3, u3, R0_3, l3, m3)
    C_final3 = sol3.y[:N3, -1]
    invader_data = []
    for i in range(N1, N3):  # Only invader species
        invader_data.append({
            "Seed": seed,
            "Species_ID": i - N1 + 1,
            "CUE": species_CUE3[i],
            "Survival": int(C_final3[i] >= 1e-5),
            "C_final": C_final3[i]
        })
    return invader_data

if __name__ == "__main__":
    # You can generate seeds as needed
    with open('seeds.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_invader_data_nested = pool.map(simulate, seeds)
    all_invader_data = [row for seed_result in all_invader_data_nested if seed_result for row in seed_result]
    df = pd.DataFrame(all_invader_data)
    df.to_csv("../data/rare_invade_hpc.csv", index=False)

    # --- Analysis: CUE vs Survival Probability ---
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np

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