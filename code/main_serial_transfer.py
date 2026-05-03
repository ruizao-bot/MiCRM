"""
main_serial_transfer.py
=======================
Community coalescence simulation using **serial transfer** instead of chemostat.

Chemostat:  dR/dt = rho - omega*R - consumption + leakage   (continuous supply)
Serial transfer:
  Within each cycle  dR/dt = -consumption + leakage          (batch, no supply)
  At transfer:       C  →  D * C
                     R  →  D * R_end + (1 - D) * R_feed      (dilute + replenish)

Steady state is detected when the relative change in community composition
between consecutive cycles falls below CONVERGENCE_TOL.
"""

from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
import param

# ─── Random seed & bookkeeping ────────────────────────────────────────────────
BASE_SEED         = 37
N_SIMULATIONS     = 50

# ─── Output files ─────────────────────────────────────────────────────────────
DATA_DIR          = "data"
COAL_FILE         = os.path.join(DATA_DIR, "coal_serial.csv")

# ─── Species / resource pool ──────────────────────────────────────────────────
N_POOL, M_POOL    = 1000, 100
N_MODULES         = 1
S_RATIO           = 1.0
LEAKAGE_RATE      = 0.2

# ─── Community size ───────────────────────────────────────────────────────────
N1, M1            = 100, 50
N2, M2            = 100, 50

# ─── Physiology ───────────────────────────────────────────────────────────────
MAINTENANCE_COST  = 0.2

# ─── Serial-transfer protocol ─────────────────────────────────────────────────
T_CYCLE           = 24.0      # duration of one growth cycle (same time units as main.py)
DILUTION_FACTOR   = 0.01      # fraction transferred each cycle (100× dilution)
R_FEED_VALUE      = 1.0       # resource concentration in fresh medium
N_CYCLES_MAX      = 1000      # safety cap on number of cycles
CONVERGENCE_TOL   = 1e-3      # relative L2 change in C between cycles to declare steady-state
N_CHECK_CYCLES    = 5         # number of consecutive converged cycles required

# ─── Initial conditions ───────────────────────────────────────────────────────
C0_VALUE          = 0.01
R0_VALUE          = 1.0

# ─── Survival threshold ───────────────────────────────────────────────────────
SURVIVAL_THRESHOLD = 1e-5


# =============================================================================
# Serial-transfer ODE utilities
# =============================================================================

def _solve_batch_cycle(N, M, u, l, m, lambda_alpha, C0, R0, T_CYCLE,
                       tol_ode=1e-8, method="BDF", n_t_eval=None):
    """Integrate one growth cycle with no external supply (rho=0, omega=0).

    Set n_t_eval to a positive integer to get a dense trajectory for CUE
    integration; leave as None for cheap end-point-only runs.
    """
    def ode(t, y):
        C = np.maximum(y[:N], 0.0)
        R = np.maximum(y[N:], 0.0)
        anab_flux  = u * (R * (1.0 - lambda_alpha))   # (N, M)  retained fraction
        dCdt       = C * (np.sum(anab_flux, axis=1) - m)
        consumption = np.sum(C[:, None] * u * R, axis=0)  # (M,)
        leakage     = np.einsum("i,j,ij,ijk->k", C, R, u, l)
        dRdt        = -consumption + leakage
        return np.concatenate([dCdt, dRdt])

    t_eval = np.linspace(0.0, T_CYCLE, int(n_t_eval)) if n_t_eval else None
    Y0  = np.concatenate([C0, R0])
    sol = solve_ivp(
        ode,
        (0.0, T_CYCLE),
        Y0,
        t_eval=t_eval,
        method=method,
        rtol=tol_ode,
        atol=tol_ode * 1e-3,
        dense_output=False,
    )
    return sol


def _transfer_step(C_end, R_end, R_feed, D):
    """Apply serial-transfer dilution and media replenishment."""
    C_new = D * C_end
    R_new = D * R_end + (1.0 - D) * R_feed
    return C_new, R_new


N_T_EVAL_DENSE = 200   # time points saved in the final (CUE) cycle


def run_serial_transfer(N, M, u, l, m, C0, R_feed,
                        T_cycle=T_CYCLE,
                        dilution=DILUTION_FACTOR,
                        n_cycles_max=N_CYCLES_MAX,
                        conv_tol=CONVERGENCE_TOL,
                        n_check=N_CHECK_CYCLES):
    """Run serial-transfer until composition converges.

    All intermediate cycles use cheap end-point-only integration.
    Once converged, one final cycle is re-run with a dense t_eval so that
    the returned trajectory can be used directly for CUE integration
    without any further ODE calls.

    Returns
    -------
    C_ss      : (N,) steady-state biomass at end of convergence cycle
    last_sol  : dense-trajectory OdeSolution of the final cycle
    n_cycles  : number of cycles run
    """
    lambda_alpha = np.full(M, LEAKAGE_RATE)
    C = C0.copy()
    R = np.full(M, R_feed)

    converged_streak = 0
    C_prev = C.copy()
    C_end  = C.copy()
    R_end  = R.copy()

    for cycle in range(n_cycles_max):
        # Cheap run: no dense output needed during convergence phase
        sol = _solve_batch_cycle(N, M, u, l, m, lambda_alpha, C, R, T_cycle)

        C_end = np.maximum(sol.y[:N, -1], 0.0)
        R_end = np.maximum(sol.y[N:,  -1], 0.0)

        norm_prev = np.linalg.norm(C_prev)
        rel_change = (
            np.linalg.norm(C_end - C_prev) / norm_prev
            if norm_prev > 0 else np.inf
        )

        if rel_change < conv_tol:
            converged_streak += 1
        else:
            converged_streak = 0

        C_prev = C_end.copy()

        if converged_streak >= n_check:
            # Re-run this final cycle with dense time points for CUE
            C_start = _transfer_step(C_end, R_end, R_feed, dilution)[0]
            R_start = _transfer_step(C_end, R_end, R_feed, dilution)[1]
            last_sol = _solve_batch_cycle(
                N, M, u, l, m, lambda_alpha, C_start, R_start, T_cycle,
                n_t_eval=N_T_EVAL_DENSE
            )
            return C_end, last_sol, cycle + 1

        C, R = _transfer_step(C_end, R_end, R_feed, dilution)

    # Not converged – still return a dense final cycle
    last_sol = _solve_batch_cycle(
        N, M, u, l, m, lambda_alpha, C, R, T_cycle,
        n_t_eval=N_T_EVAL_DENSE
    )
    return C_end, last_sol, n_cycles_max


# =============================================================================
# CUE computation (same formulas as main.py, applied to batch-cycle trajectory)
# =============================================================================

def _compute_eta(l):
    return 1.0 - np.sum(l, axis=2)


def intrinsic_cue(u, l, R0, m):
    """Species-level intrinsic CUE at reference resource level R0."""
    eta = _compute_eta(l)
    Ui0 = np.sum(u * R0[None, :],         axis=1)
    Gi0 = np.sum(u * eta * R0[None, :],   axis=1)
    return (Gi0 - m) / (Ui0 + 1e-12)


def _flux_rates(u, l, R_t, m, N):
    eta         = 1.0 - np.sum(l, axis=2)
    uptake_pb_t = u @ R_t
    anab_gross_t = (u * eta) @ R_t
    m_vec = np.full(N, float(m)) if np.ndim(m) == 0 else np.asarray(m, dtype=float)
    anab_pb_t = anab_gross_t - m_vec[:, None]
    return uptake_pb_t, anab_pb_t


def compute_actual_cue_cycle(u, l, sol, N, m):
    """Biomass-weighted time-integral CUE using the pre-computed dense trajectory."""
    if sol is None or sol.t is None or len(sol.t) < 2:
        return np.full(N, np.nan)

    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)

    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N)
    total_uptake = np.trapezoid(C_t * uptake_pb_t, x=t, axis=1)
    total_anab   = np.trapezoid(C_t * anab_pb_t,   x=t, axis=1)
    return np.divide(
        total_anab, total_uptake,
        out=np.full(N, np.nan, dtype=float),
        where=np.abs(total_uptake) > 1e-12
    )


def compute_actual_community_cue_cycle(u, l, sol, N, m, survivor_idx=None):
    """Community-level CUE using the pre-computed dense trajectory."""
    if sol is None or sol.t is None or len(sol.t) < 2:
        return np.nan

    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)

    if survivor_idx is not None:
        idx = np.asarray(survivor_idx, dtype=int)
        if idx.size == 0:
            return np.nan
        C_t  = C_t[idx, :]
        u    = u[idx, :]
        l    = l[idx, :, :]
        m    = np.asarray(m)[idx] if np.ndim(m) > 0 else m
        N_eff = idx.size
    else:
        N_eff = N

    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N_eff)
    total_uptake = np.sum(np.trapezoid(C_t * uptake_pb_t, x=t, axis=1))
    total_anab   = np.sum(np.trapezoid(C_t * anab_pb_t,   x=t, axis=1))
    if np.abs(total_uptake) < 1e-12:
        return np.nan
    return total_anab / total_uptake


# =============================================================================
# Per-seed simulation
# =============================================================================

def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # ── Community 1 ───────────────────────────────────────────────────────────
    species_indices1  = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    C0_1    = np.full(N1, C0_VALUE)
    R_feed1 = np.full(M1, R_FEED_VALUE)
    R0_1    = np.full(M1, R0_VALUE)

    # ── Community 2 ───────────────────────────────────────────────────────────
    resource_indices2 = param.choose_resources_for_second_community(
        M_POOL, M1, M2, resource_indices1, rng
    )
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2  = rng.choice(remaining_species, N2, replace=False)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    C0_2    = np.full(N2, C0_VALUE)
    R_feed2 = np.full(M2, R_FEED_VALUE)
    R0_2    = np.full(M2, R0_VALUE)

    # ── Run serial transfer for communities 1 & 2 ─────────────────────────────
    C_ss1, last_sol1, cyc1 = run_serial_transfer(
        N1, M1, u1, l1, MAINTENANCE_COST, C0_1, R_FEED_VALUE
    )
    C_ss2, last_sol2, cyc2 = run_serial_transfer(
        N2, M2, u2, l2, MAINTENANCE_COST, C0_2, R_FEED_VALUE
    )

    C_final1 = np.maximum(C_ss1, 0.0)
    C_final2 = np.maximum(C_ss2, 0.0)

    # ── Coalesced community (Community 3) ─────────────────────────────────────
    species_indices3  = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

    N3 = N1 + N2
    M3 = len(resource_indices3)

    # Initial condition for coalescence: post-steady-state biomasses, fresh resources
    C0_3    = np.concatenate([C_final1, C_final2])
    R_feed3 = np.full(M3, R_FEED_VALUE)

    C_ss3, last_sol3, cyc3 = run_serial_transfer(
        N3, M3, u3, l3, MAINTENANCE_COST, C0_3, R_FEED_VALUE
    )

    C_final3 = np.maximum(C_ss3, 0.0)

    # ── Survivors ─────────────────────────────────────────────────────────────
    surv1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    surv2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    surv3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    # ── CUE ───────────────────────────────────────────────────────────────────
    species_CUE1 = intrinsic_cue(u1, l1, R0_1, MAINTENANCE_COST)
    species_CUE2 = intrinsic_cue(u2, l2, R0_2, MAINTENANCE_COST)
    R0_3 = np.full(M3, R0_VALUE)
    species_CUE3 = intrinsic_cue(u3, l3, R0_3, MAINTENANCE_COST)

    actual_CUE1 = compute_actual_cue_cycle(u1, l1, last_sol1, N1, MAINTENANCE_COST)
    actual_CUE2 = compute_actual_cue_cycle(u2, l2, last_sol2, N2, MAINTENANCE_COST)
    actual_CUE3 = compute_actual_cue_cycle(u3, l3, last_sol3, N3, MAINTENANCE_COST)

    actual_comm_CUE1 = compute_actual_community_cue_cycle(
        u1, l1, last_sol1, N1, MAINTENANCE_COST, survivor_idx=surv1
    )
    actual_comm_CUE2 = compute_actual_community_cue_cycle(
        u2, l2, last_sol2, N2, MAINTENANCE_COST, survivor_idx=surv2
    )
    actual_comm_CUE3 = compute_actual_community_cue_cycle(
        u3, l3, last_sol3, N3, MAINTENANCE_COST, survivor_idx=surv3
    )

    community_CUE1 = param.safe_weighted_average(species_CUE1[surv1], C_final1[surv1])
    community_CUE2 = param.safe_weighted_average(species_CUE2[surv2], C_final2[surv2])
    community_CUE3 = param.safe_weighted_average(species_CUE3[surv3], C_final3[surv3])

    # ── Other community metrics ────────────────────────────────────────────────
    competition1 = param.community_level_competition(u1)
    competition2 = param.community_level_competition(u2)
    competition3 = param.community_level_competition(u3)

    comp_sp1 = param.species_level_competition(u1)
    comp_sp2 = param.species_level_competition(u2)
    comp_sp3 = param.species_level_competition(u3)

    comp_dot1 = param.species_level_competition_dot(u1)
    comp_dot2 = param.species_level_competition_dot(u2)
    comp_dot3 = param.species_level_competition_dot(u3)

    L_eff1 = param.calculate_effective_leakage(u1, l1)
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    L_eff3 = param.calculate_effective_leakage(u3, l3)
    fac1 = np.mean(L_eff1, axis=1)
    fac2 = np.mean(L_eff2, axis=1)
    fac3 = np.mean(L_eff3, axis=1)

    uvar1 = param.compute_uptake_variance(u1)
    uvar2 = param.compute_uptake_variance(u2)
    uvar3 = param.compute_uptake_variance(u3)

    # Resource depletion = resources remaining at end of last cycle
    dep1 = float(np.sum(np.maximum(last_sol1.y[N1:, -1], 0.0))) if last_sol1 is not None else np.nan
    dep2 = float(np.sum(np.maximum(last_sol2.y[N2:, -1], 0.0))) if last_sol2 is not None else np.nan
    dep3 = float(np.sum(np.maximum(last_sol3.y[N3:, -1], 0.0))) if last_sol3 is not None else np.nan

    total_ab1 = float(np.sum(C_final1))
    total_ab2 = float(np.sum(C_final2))
    total_ab3 = float(np.sum(C_final3))

    origin1_in3 = float(np.sum(C_final3[:N1]))
    origin2_in3 = float(np.sum(C_final3[N1:]))
    dominant = "Community 1" if origin1_in3 > origin2_in3 else "Community 2"

    # ── Assemble per-species rows ──────────────────────────────────────────────
    species_data = []

    for comm_id, (N_c, u_c, actual_c, sp_cue_c, comm_cue_c, act_comm_c,
                  C_c, total_ab_c, comp_c, comp_sp_c, comp_dot_c,
                  fac_c, dep_c, uvar_c, surv_c, cyc_c) in enumerate([
        (N1, u1, actual_CUE1, species_CUE1, community_CUE1, actual_comm_CUE1,
         C_final1, total_ab1, competition1, comp_sp1, comp_dot1,
         fac1, dep1, uvar1, surv1, cyc1),
        (N2, u2, actual_CUE2, species_CUE2, community_CUE2, actual_comm_CUE2,
         C_final2, total_ab2, competition2, comp_sp2, comp_dot2,
         fac2, dep2, uvar2, surv2, cyc2),
        (N3, u3, actual_CUE3, species_CUE3, community_CUE3, actual_comm_CUE3,
         C_final3, total_ab3, competition3, comp_sp3, comp_dot3,
         fac3, dep3, uvar3, surv3, cyc3),
    ], start=1):
        for i in range(N_c):
            species_data.append({
                "Seed":                 seed,
                "Community":            comm_id,
                "Species_ID":           i + 1,
                "Species_CUE":          sp_cue_c[i],
                "actual_CUE":           actual_c[i],
                "actual_community_CUE": act_comm_c,
                "Community_CUE":        comm_cue_c,
                "Abundance":            C_c[i],
                "Total_Abundance":      total_ab_c,
                "Dominant_Community":   dominant,
                "Competition":          comp_c,
                "Species_Competition":  comp_sp_c[i],
                "Species_Competition_Dot": comp_dot_c[i],
                "Facilitation":         fac_c[i],
                "Depletion":            dep_c,
                "UptakeVar":            uvar_c[i],
                "N_Survivors":          len(surv_c),
                "N_Cycles":             cyc_c,
            })

    return species_data


# =============================================================================
# Entry point
# =============================================================================

def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(
        0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32
    ).tolist()

    print(f"Running {N_SIMULATIONS} serial-transfer coalescence simulations "
          f"on {cpu_count()} cores …")
    print(f"  T_CYCLE={T_CYCLE}, D={DILUTION_FACTOR}, "
          f"N_CYCLES_MAX={N_CYCLES_MAX}, CONV_TOL={CONVERGENCE_TOL}")

    with Pool(cpu_count()) as pool:
        all_results = pool.map(simulate, seeds)

    all_species_data = [
        row
        for result in all_results
        if result
        for row in result
    ]

    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(all_species_data)
    df.to_csv(COAL_FILE, index=False)
    print(f"Saved {len(df)} rows → {COAL_FILE}")


if __name__ == "__main__":
    main()
