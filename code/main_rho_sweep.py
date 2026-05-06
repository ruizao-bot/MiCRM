"""
Robustness check: sweep three ρ values for Community 3.
Saves a single CSV data/coal_rho_sweep.csv with a Rho3 column.

Rho3 values tested: 0.3, 0.6 (baseline), 1.2
All other parameters are identical to the baseline simulation.
"""

from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
import param

# ── Simulation parameters (same as baseline main.py) ─────────────────────────
BASE_SEED        = 37
N_SIMULATIONS    = 50

COAL_FILE        = "data/coal_rho_sweep.csv"

N_POOL, M_POOL   = 1000, 100
N_MODULES        = 1
S_RATIO          = 1
LEAKAGE_RATE     = 0.2

N1, M1 = 100, 50
N2, M2 = 100, 50

MAINTENANCE_COST = 0.2
RHO_VALUE        = 0.6          # ρ for Community 1 & 2 (fixed)
OMEGA_VALUE      = 0.1
T_SPAN           = (0, 100000)
C0_VALUE         = 0.01
R0_VALUE         = 1

SURVIVAL_THRESHOLD = 1e-5
EV_THRESHOLD       = 0.00
INTE_CUE_N_SAVE_POINTS = 40

# ── Three ρ values to sweep for Community 3 ───────────────────────────────────
RHO3_VALUES = [0.3, 0.6, 1.2, 2.4]


# ── CUE helper ────────────────────────────────────────────────────────────────
def _compute_eta(l):
    return 1.0 - np.sum(l, axis=2)

def _ensure_m_vec(m, N):
    if np.ndim(m) == 0:
        return np.full(N, float(m))
    return np.asarray(m, dtype=float)

def compute_Gi0_Ui0_eps(u, l, R0, m):
    N = u.shape[0]
    eta = _compute_eta(l)
    m_vec = _ensure_m_vec(m, N)
    Ui0 = np.sum(u * R0[None, :], axis=1)
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)
    eps = (Gi0 - m_vec) / (Ui0 + 1e-12)
    return eta, Gi0, Ui0, eps


# ── Core simulation: solve C1/C2 once, then sweep all rho3 values ───────────────
def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # Community 1
    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1  = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1  = np.full(N1, C0_VALUE)
    R0_1  = np.full(M1, R0_VALUE)

    # Community 2
    resource_indices2 = param.choose_resources_for_second_community(
        M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2  = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2  = np.full(N2, C0_VALUE)
    R0_2  = np.full(M2, R0_VALUE)

    # Solve parent communities
    sol1 = param.solve_micrm(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN,
        n_save_points=INTE_CUE_N_SAVE_POINTS)
    sol2 = param.solve_micrm(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN,
        n_save_points=INTE_CUE_N_SAVE_POINTS)

    # Stability check
    lambda_vec1 = np.full(N1, LEAKAGE_RATE)
    lambda_vec2 = np.full(N2, LEAKAGE_RATE)
    J1 = param.MiCRM_jac(N1, M1, u1, l1, MAINTENANCE_COST, rho1, omega1, lambda_vec1, sol1)
    J2 = param.MiCRM_jac(N2, M2, u2, l2, MAINTENANCE_COST, rho2, omega2, lambda_vec2, sol2)
    ev1 = param.leading_eigenvalue(J1)
    ev2 = param.leading_eigenvalue(J2)
    if not (np.isfinite(ev1) and ev1 < EV_THRESHOLD and
            np.isfinite(ev2) and ev2 < EV_THRESHOLD):
        return None

    # Pre-compute parent community quantities (shared across all rho3 values)
    C_final1 = np.maximum(sol1.y[:N1, -1], 0.0)
    C_final2 = np.maximum(sol2.y[:N2, -1], 0.0)
    surv1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    surv2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    _, _, _, species_CUE1 = compute_Gi0_Ui0_eps(u1, l1, R0_1, MAINTENANCE_COST)
    _, _, _, species_CUE2 = compute_Gi0_Ui0_eps(u2, l2, R0_2, MAINTENANCE_COST)
    comm_CUE1 = param.safe_weighted_average(species_CUE1[surv1], C_final1[surv1])
    comm_CUE2 = param.safe_weighted_average(species_CUE2[surv2], C_final2[surv2])

    # Community 3 setup (indices, u3, l3 shared; only rho3 changes)
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3  = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3  = np.full(M3, R0_VALUE)
    _, _, _, species_CUE3_base = compute_Gi0_Ui0_eps(u3, l3, R0_3, MAINTENANCE_COST)

    # Sweep rho3 values — only coalescence ODE changes
    rows = []
    for rho3_value in RHO3_VALUES:
        rho3 = np.full(M3, rho3_value)

        sol3 = param.solve_micrm(
            N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN,
            n_save_points=INTE_CUE_N_SAVE_POINTS)

        C_final3 = np.maximum(sol3.y[:N3, -1], 0.0)
        surv3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]
        comm_CUE3 = param.safe_weighted_average(species_CUE3_base[surv3], C_final3[surv3])

        origin1 = np.sum(C_final3[:N1])
        origin2 = np.sum(C_final3[N1:])
        dominant = "Community 1" if origin1 > origin2 else "Community 2"

        for comm_id, (N_c, species_CUE, comm_CUE, C_final, n_surv) in enumerate([
            (N1, species_CUE1, comm_CUE1, C_final1, len(surv1)),
            (N2, species_CUE2, comm_CUE2, C_final2, len(surv2)),
            (N3, species_CUE3_base, comm_CUE3, C_final3, len(surv3)),
        ], start=1):
            for i in range(N_c):
                rows.append({
                    "Seed":              seed,
                    "Rho3":              rho3_value,
                    "Community":         comm_id,
                    "Species_ID":        i + 1,
                    "Species_CUE":       species_CUE[i],
                    "Community_CUE_surv": comm_CUE,
                    "Abundance":         C_final[i],
                    "Dominant_Community": dominant,
                    "N_Survivors":       n_surv,
                })

    return rows


def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    print(f"Running {N_SIMULATIONS} seeds × {len(RHO3_VALUES)} ρ values (C1/C2 solved once per seed)...")

    with Pool(cpu_count()) as pool:
        results = pool.map(simulate, seeds)

    all_rows = [row for res in results if res for row in res]

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(COAL_FILE, index=False)
    print(f"Saved: {COAL_FILE}  ({len(df)} rows, {df['Seed'].nunique()} seeds)")

    # Quick accuracy summary
    from scipy.spatial.distance import pdist, squareform

    def first_unique(s):
        vals = pd.Series(s).dropna().unique()
        return vals[0] if len(vals) > 0 else np.nan

    print("\nPrediction accuracy by ρ₃:")
    for rho3 in RHO3_VALUES:
        sub = df[df["Rho3"] == rho3].copy()
        sub["Community"] = sub["Community"].astype(str)
        df_surv = sub[sub["Abundance"] > SURVIVAL_THRESHOLD].copy()
        df_surv["Global_Species_ID"] = np.where(
            df_surv["Community"] == "2",
            df_surv["Species_ID"] + 100,
            df_surv["Species_ID"]
        )
        cue_df = (
            sub[sub["Community"].isin(["1", "2"])]
            .groupby(["Seed", "Community"], as_index=False)
            .agg(cue=("Community_CUE_surv", first_unique))
        )
        cue_pivot = cue_df.pivot_table(index="Seed", columns="Community", values="cue")

        correct = 0; total = 0
        for s in df_surv["Seed"].unique():
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
            if np.sign(cue_diff) == np.sign(sim_diff):
                correct += 1
            total += 1

        print(f"  ρ₃ = {rho3:.1f}:  {correct}/{total} = {correct/total:.1%}" if total else f"  ρ₃ = {rho3:.1f}: no data")


if __name__ == "__main__":
    main()
