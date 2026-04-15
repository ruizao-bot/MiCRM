# This script performs the main computations for the paper, including:
# 1) Generating multiple community1/community2 instances for each balance1/balance2
#    (parameterized, with diagnostic metrics)
# 2) Pairwise merging for each (community1, community2) pair, computing post-merge stability (ev)
#    and feasibility probability
# 3) Exporting a CSV file containing full replicate-level details
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
import paramn

# ===================== Parameters =====================
M = 20
λ_min = 0
λ_max = 0.5
s_ratio_max = 50

N1 = 20
N2 = 20

m1 = np.full(N1, 0.1)
m2 = np.full(N2, 0.1)
rho = np.full(M, 5)
omega = np.full(M, 0.2)

# ===================== MiCRM and eLV/feasibility functions =====================
def dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec):
    C = y[:N]
    R = y[N:]
    uptake_sum = np.sum(u * R[None, :], axis=1)          # (N,)
    dCdt = C * ((1 - lambda_vec) * uptake_sum - m)       # (N,)

    dRdt = rho - omega * R
    consumption = np.sum(C[:, None] * u * R, axis=0)     # (M,)
    dRdt -= consumption

    leakage = np.einsum('i,j,ij,ijk->k', C, R, u, l)     # (M,)
    dRdt += leakage

    return np.concatenate([dCdt, dRdt])

def calculate_elv_params(C_hat, R_hat, N, M, u, l_tensor, m, rho, omega, lambda_vec):
    C_u = C_hat[:, None] * u  # (N, M)

    L = np.einsum('ib,iab->ab', C_u, l_tensor, optimize='optimal')  # (M, M)

    diag = omega + np.sum(C_u, axis=0)  # (M,)
    D = np.diag(diag) - L               # (M, M)

    term1 = -u * R_hat  # (N, M)
    term2 = np.einsum('ib,iab->ia', u * R_hat, l_tensor, optimize='optimal')  # (N, M)
    V = term1 + term2  # (N, M)

    partial_R_C = np.linalg.solve(D, V.T)  # (M, N)

    row_scale = (1 - lambda_vec)[:, None]        # (N,1)
    weighted_u = row_scale * u                   # (N,M)

    alpha = weighted_u @ partial_R_C             # (N,N)

    growth_terms = (1 - lambda_vec) * np.sum(u * R_hat, axis=1)
    interaction_terms = alpha @ C_hat
    r = growth_terms - m - interaction_terms

    return alpha, r

def dCdt_elv(t, C, r, alpha):
    return C * (r + alpha @ C)

def make_micrm_equilibrium_event(N, M, u, l, m, rho, omega, lambda_vec, tol=1e-5):
    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec)
        return np.max(np.abs(deriv)) - tol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1
    return equilibrium_event

def make_elv_equilibrium_event(r, alpha, tol=1e-5):
    def event(t, C):
        deriv = dCdt_elv(t, C, r, alpha)
        return np.max(np.abs(deriv)) - tol
    event.terminal = True
    event.direction = -1
    return event

def LV_jac(alpha, C_eq, threshold=1e-5):
    mask = C_eq > threshold
    if not np.any(mask):
        raise ValueError("No feasible species above threshold.")
    C = C_eq[mask]
    A = alpha[np.ix_(mask, mask)]
    return np.diag(C) @ A

def MiCRM_jac(N, M, u, l, m, rho, omega, lambda_vec, sol):
    state = sol.y[:, -1]
    C = state[:N]
    R = state[N:]

    # CC block
    cc_diag = -m + ((1 - lambda_vec)[:, None] * u * R[None, :]).sum(axis=1)
    CC = np.diag(cc_diag)

    # CR block
    CR = C[:, None] * (1 - lambda_vec)[:, None] * u

    # RR block
    P = C[:, None, None] * u[:, :, None] * l
    RR = P.sum(axis=0).T
    diag_val = np.diag(RR)
    sub_diag = (C[:, None] * u).sum(axis=0)
    diag_rr = diag_val - sub_diag - omega
    np.fill_diagonal(RR, diag_rr)

    # RC block
    Q = u * R[None, :]
    Ql = Q[:, :, None] * l
    term2 = Ql.sum(axis=1)
    RC = (term2 - Q).T

    top    = np.hstack([CC, CR])
    bottom = np.hstack([RC, RR])
    return np.vstack([top, bottom])

def leading_eigenvalue(J):
    eigvals = np.linalg.eigvals(J)
    re = np.real(eigvals)
    return float(re[np.argmax(re)])

def leading_hermitian_eigenvalue(J):
    H = (J + J.T) / 2
    eigvals = np.linalg.eigvalsh(H)
    return np.max(eigvals)

# ---- R: mvtnorm ----
import rpy2.robjects as ro
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

def feasibility_prob(alpha, eps=0.0, maxpts=None, abseps=1e-6, releps=0):
    """
    Compute P[X >= 0] using R mvtnorm::pmvnorm (Genz–Bretz algorithm),
    where X ~ N(0, Sigma) and Sigma = (A^-1)(A^-1)^T.
    """
    S = alpha.shape[0]
    if S == 0:
        return 0.0
    A = alpha.copy()
    if eps > 0.0:
        A = A + eps * np.eye(S)
    try:
        X = np.linalg.solve(A, np.eye(S))
    except np.linalg.LinAlgError:
        return 0.0
    Sigma = X @ X.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + 1e-10 * np.eye(S)

    mvtnorm = importr('mvtnorm')
    r = ro.r
    lower = FloatVector([0.0] * S)
    upper = FloatVector([r('Inf')[0]] * S)
    mean = FloatVector([0.0] * S)
    r_sigma = r['matrix'](ro.FloatVector(Sigma.flatten('F')), nrow=S, ncol=S)
    if maxpts is None:
        maxpts = int(max(250000, 10000 * S))
    algo = mvtnorm.GenzBretz(maxpts=maxpts, abseps=abseps, releps=releps)
    res = mvtnorm.pmvnorm(lower=lower, upper=upper, mean=mean,
                          sigma=r_sigma, algorithm=algo)
    return float(res[0])

# ===================== Scan / workflow parameters =====================
balances = np.round(np.arange(0.0, 1.0 + 0.05, 0.05), 2)  # 0, 0.05, ..., 1
n_repeats_target = 20
max_seed_tries   = 200
ev_threshold     = 0.01
thr = 1e-5

# Common initial conditions and integration horizon
C0_1 = np.full(N1, 1)
C0_2 = np.full(N2, 1)
R0   = np.full(M, 1)
t_span = (0, 1e8)

# ===================== balance -> parameter mapping =====================
def build_params_for_balance(balance, N):
    """
    N_modules: monotonic function of balance
    s_ratio:   smaller balance -> larger s_ratio (more "cooperative")
    lambda (scalar for lambda_vec): same monotonic mapping
    """
    N_modules = max(1, int(M * (1 - balance**2)))
    s_ratio   = 1 + (s_ratio_max - 1) * (1 - balance)
    lam       = λ_min + (λ_max - λ_min) * (1 - balance)
    lambda_vec = np.full(N, lam)
    return N_modules, s_ratio, lambda_vec, lam

# ===================== Single community simulation =====================
def simulate_single_community(N, C0, N_modules, s_ratio, lambda_vec, m, seed):
    """
    Returns:
      ok, payload(dict)

    payload contains:
      C_eq, R_eq, alpha, r, ev_micrm, feas_prob, u, l, lambda_vec,
      N_modules, s_ratio
    """
    np.random.seed(seed)
    try:
        # Generate u and l_tensor (from paramn)
        u = paramn.modular_uptake(N, M, N_modules, s_ratio)
        l = paramn.generate_l_tensor(N, M, N_modules, s_ratio, lambda_vec)
    except Exception as e:
        return False, {"reason": f"paramn generation failed: {e}"}

    try:
        Y0 = np.concatenate([C0, R0])
        micrm_event = make_micrm_equilibrium_event(N, M, u, l, m, rho, omega, lambda_vec)
        sol = solve_ivp(lambda t, y: dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec),
                        t_span, Y0, method='BDF', events=micrm_event)
        if sol.y.shape[1] == 0:
            return False, {"reason": "integration returned empty"}

        C_eq = sol.y[:N, -1]
        R_eq = sol.y[N:, -1]
        alpha, r = calculate_elv_params(C_eq, R_eq, N, M, u, l, m, rho, omega, lambda_vec)

        # MiCRM Jacobian and leading real-part eigenvalue
        J_micrm = MiCRM_jac(N, M, u, l, m, rho, omega, lambda_vec, sol)
        ev_micrm = leading_eigenvalue(J_micrm)

        # Feasibility probability computed on the surviving subsystem
        mask = C_eq > thr
        feas_p = feasibility_prob(alpha[np.ix_(mask, mask)], eps=0.0) if np.any(mask) else 0.0

        return True, {
            "C_eq": C_eq, "R_eq": R_eq,
            "alpha": alpha, "r": r,
            "ev_micrm": float(ev_micrm), "feas_prob": float(feas_p),
            "u": u, "l": l, "lambda_vec": lambda_vec,
            "N_modules": int(N_modules), "s_ratio": float(s_ratio)
        }
    except Exception as e:
        return False, {"reason": f"simulate failed: {e}"}

# ===================== Collect n valid replicates per balance =====================
def collect_replicates_for_balance(balance, N, C0, m, base_seed):
    N_modules, s_ratio, lambda_vec, lam = build_params_for_balance(balance, N)
    reps, tries, seed = [], 0, base_seed

    while len(reps) < n_repeats_target and tries < max_seed_tries:
        tries += 1
        ok, payload = simulate_single_community(
            N=N, C0=C0, N_modules=N_modules, s_ratio=s_ratio,
            lambda_vec=lambda_vec, m=m, seed=seed
        )
        seed += 1
        if not ok:
            continue
        if np.isfinite(payload["ev_micrm"]) and (payload["ev_micrm"] < ev_threshold):
            reps.append(payload)

    return {
        "reps": reps,
        "lam": lam,
        "lambda_vec": lambda_vec,
        "N_modules": int(N_modules),
        "s_ratio": float(s_ratio),
        "valid_repeats": len(reps),
        "tries": tries
    }

# ===================== Merge and evaluate (distinct species, shared resources) =====================
def merge_and_measure(rep1, rep2, lambda_vec1, lambda_vec2, N_modules3=None, eps_feas=1e-3):
    try:
        u1, l1, C1_eq = rep1["u"], rep1["l"], rep1["C_eq"]
        u2, l2, C2_eq = rep2["u"], rep2["l"], rep2["C_eq"]
        N1_local, M1 = u1.shape
        N2_local, M2 = u2.shape
        assert M1 == M2, "This implementation assumes both communities have the same number of resources"

        # Species count after merge
        N3 = N1_local + N2_local
        M3 = M1

        # Merge species; keep resource dimension M unchanged
        u3 = np.vstack((u1, u2))
        l3 = np.vstack((l1, l2))
        m3 = np.concatenate((np.full(N1_local, 0.1), np.full(N2_local, 0.1)))
        lambda_vec3 = np.concatenate((lambda_vec1, lambda_vec2))

        # Initial condition: concatenate C, and use global R0 for resources
        C0_3 = np.concatenate([C1_eq, C2_eq])
        Y0_3 = np.concatenate([C0_3, R0])

        rho3   = rho
        omega3 = omega

        micrm_event3 = make_micrm_equilibrium_event(N3, M3, u3, l3, m3, rho3, omega3, lambda_vec3)
        sol3 = solve_ivp(lambda t, y: dCdt_Rdt(t, y, N3, M3, u3, l3, m3, rho3, omega3, lambda_vec3),
                         t_span, Y0_3, method='BDF', events=micrm_event3)
        if sol3.y.shape[1] == 0:
            return False, np.nan, np.nan

        C3_eq = sol3.y[:N3, -1]
        R3_eq = sol3.y[N3:, -1]

        alpha3, r3 = calculate_elv_params(C3_eq, R3_eq, N3, M3, u3, l3, m3, rho3, omega3, lambda_vec3)

        J3 = MiCRM_jac(N3, M3, u3, l3, m3, rho3, omega3, lambda_vec3, sol3)
        ev3 = leading_eigenvalue(J3)

        mask3 = C3_eq > thr
        feas3 = feasibility_prob(alpha3[np.ix_(mask3, mask3)], eps=eps_feas) if np.any(mask3) else 0.0

        return True, float(ev3), float(feas3)

    except Exception:
        return False, np.nan, np.nan

# ===================== Main =====================
def main():
    base_seed = 43

    # Step 1&2: for each balance1, generate 20 qualified replicates of community1
    bank1 = {}
    seed_cursor = base_seed
    for b1 in tqdm(balances, desc="Precomputing community1 (by balance1)"):
        out = collect_replicates_for_balance(
            balance=b1, N=N1, C0=C0_1, m=m1, base_seed=seed_cursor
        )
        bank1[b1] = out
        seed_cursor += max(1, out["tries"])

    # Step 3&4: for each balance2, generate 20 qualified replicates of community2
    bank2 = {}
    for b2 in tqdm(balances, desc="Precomputing community2 (by balance2)"):
        out = collect_replicates_for_balance(
            balance=b2, N=N2, C0=C0_2, m=m2, base_seed=seed_cursor
        )
        bank2[b2] = out
        seed_cursor += max(1, out["tries"])

    # Write all community1/community2 replicates into a long table
    rows = []

    for b1, pack in bank1.items():
        for k, rep in enumerate(pack["reps"]):
            rows.append({
                "community": "community1",
                "replicate": k,
                "competition_cooperation_balance1": float(b1),
                "competition_cooperation_balance2": np.nan,
                "ev": float(rep["ev_micrm"]),
                "feasibility": float(rep["feas_prob"]),
                "N_modules": int(pack["N_modules"]),
                "s_ratio": float(pack["s_ratio"]),
                "lambda_scalar": float(bank1[b1]["lambda_vec"][0])  # identical across species under the same balance
            })

    for b2, pack in bank2.items():
        for k, rep in enumerate(pack["reps"]):
            rows.append({
                "community": "community2",
                "replicate": k,
                "competition_cooperation_balance1": np.nan,
                "competition_cooperation_balance2": float(b2),
                "ev": float(rep["ev_micrm"]),
                "feasibility": float(rep["feas_prob"]),
                "N_modules": int(pack["N_modules"]),
                "s_ratio": float(pack["s_ratio"]),
                "lambda_scalar": float(bank2[b2]["lambda_vec"][0])
            })

    # Step 5&6: for each (b1, b2) pair, do pairwise merges and write each pair as one row (replicate = pair index)
    for b1 in tqdm(balances, desc="Merging across (balance1, balance2)"):
        pack1 = bank1[b1]
        lambda_vec1 = pack1["lambda_vec"]
        reps1 = pack1["reps"]

        for b2 in balances:
            pack2 = bank2[b2]
            lambda_vec2 = pack2["lambda_vec"]
            reps2 = pack2["reps"]

            n_pairs = min(len(reps1), len(reps2), n_repeats_target)

            # For the merged community, use the finer modular resolution
            Nm3 = max(pack1["N_modules"], pack2["N_modules"])

            for i in range(n_pairs):
                ok, ev3, feas3 = merge_and_measure(
                    reps1[i], reps2[i], lambda_vec1, lambda_vec2,
                    N_modules3=Nm3, eps_feas=1e-3
                )
                if ok and np.isfinite(ev3):
                    rows.append({
                        "community": "community3",
                        "replicate": i,
                        "competition_cooperation_balance1": float(b1),
                        "competition_cooperation_balance2": float(b2),
                        "ev": float(ev3),
                        "feasibility": float(feas3),
                        "N_modules": int(Nm3),
                        "s_ratio": np.nan,              # no single s_ratio after concatenation
                        "lambda_scalar": np.nan         # merged lambda is a concatenated vector; keep empty here
                    })

    # Step 7: write CSV (long table: one row per replicate)
    df = pd.DataFrame(rows, columns=[
        "community",
        "replicate",
        "competition_cooperation_balance1",
        "competition_cooperation_balance2",
        "ev",
        "feasibility",
        "N_modules",
        "s_ratio",
        "lambda_scalar",
    ])
    df.to_csv("scan_balances_results.csv", index=False)
    print("all done, results saved to scan_balances_results.csv")

# ===================== Entry point =====================
if __name__ == "__main__":
    main()