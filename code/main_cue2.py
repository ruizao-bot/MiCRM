import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
import paramn

# ===================== Parameters =====================
N1 = 20
N2 = 20
M  = 20

λ_min       = 0.0
λ_max       = 0.5
s_ratio_max = 50

m1    = np.full(N1, 0.1)
m2    = np.full(N2, 0.1)
rho   = np.full(M, 5.0)
omega = np.full(M, 0.2)
R0    = np.ones(M)

balances         = np.round(np.arange(0.0, 1.0 + 0.05, 0.05), 2)
n_repeats_target = 20
max_seed_tries   = 200
ev_threshold     = 0.01
thr              = 1e-5

t_span = (0, 1e8)

# ===================== MiCRM ODE =====================
def dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec):
    C = y[:N]
    R = y[N:]
    uptake_sum = np.sum(u * R[None, :], axis=1)
    dCdt = C * ((1 - lambda_vec) * uptake_sum - m)

    dRdt = rho - omega * R
    consumption = np.sum(C[:, None] * u * R, axis=0)
    dRdt -= consumption
    leakage = np.einsum('i,j,ij,ijk->k', C, R, u, l)
    dRdt += leakage

    return np.concatenate([dCdt, dRdt])

def make_micrm_equilibrium_event(N, M, u, l, m, rho, omega, lambda_vec, tol=1e-5):
    def event(t, y):
        deriv = dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec)
        return np.max(np.abs(deriv)) - tol
    event.terminal  = True
    event.direction = -1
    return event

# ===================== eLV / Jacobian =====================
def calculate_elv_params(C_hat, R_hat, N, M, u, l_tensor, m, rho, omega, lambda_vec):
    C_u  = C_hat[:, None] * u
    L    = np.einsum('ib,iab->ab', C_u, l_tensor, optimize='optimal')
    diag = omega + np.sum(C_u, axis=0)
    D    = np.diag(diag) - L
    term1 = -u * R_hat
    term2 = np.einsum('ib,iab->ia', u * R_hat, l_tensor, optimize='optimal')
    V     = term1 + term2
    partial_R_C = np.linalg.solve(D, V.T)
    alpha = (1 - lambda_vec)[:, None] * u @ partial_R_C
    growth_terms = (1 - lambda_vec) * np.sum(u * R_hat, axis=1)
    r = growth_terms - m - alpha @ C_hat
    return alpha, r

def MiCRM_jac(N, M, u, l, m, rho, omega, lambda_vec, sol):
    state = sol.y[:, -1]
    C = state[:N]
    R = state[N:]
    cc_diag = -m + ((1 - lambda_vec)[:, None] * u * R[None, :]).sum(axis=1)
    CC = np.diag(cc_diag)
    CR = C[:, None] * (1 - lambda_vec)[:, None] * u
    P  = C[:, None, None] * u[:, :, None] * l
    RR = P.sum(axis=0).T
    diag_rr = np.diag(RR) - (C[:, None] * u).sum(axis=0) - omega
    np.fill_diagonal(RR, diag_rr)
    Q    = u * R[None, :]
    RC   = ((Q[:, :, None] * l).sum(axis=1) - Q).T
    return np.vstack([np.hstack([CC, CR]), np.hstack([RC, RR])])

def leading_eigenvalue(J):
    re = np.real(np.linalg.eigvals(J))
    return float(re[np.argmax(re)])

# ---- Feasibility via R mvtnorm ----
import rpy2.robjects as ro
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

def feasibility_prob(alpha, eps=0.0, maxpts=None, abseps=1e-6, releps=0):
    S = alpha.shape[0]
    if S == 0:
        return 0.0
    A = alpha.copy()
    if eps > 0.0:
        A += eps * np.eye(S)
    try:
        X = np.linalg.solve(A, np.eye(S))
    except np.linalg.LinAlgError:
        return 0.0
    Sigma = 0.5 * (X @ X.T + (X @ X.T).T) + 1e-10 * np.eye(S)
    mvtnorm = importr('mvtnorm')
    r       = ro.r
    if maxpts is None:
        maxpts = int(max(250000, 10000 * S))
    algo = mvtnorm.GenzBretz(maxpts=maxpts, abseps=abseps, releps=releps)
    res  = mvtnorm.pmvnorm(
        lower=FloatVector([0.0] * S),
        upper=FloatVector([r('Inf')[0]] * S),
        mean=FloatVector([0.0] * S),
        sigma=r['matrix'](ro.FloatVector(Sigma.flatten('F')), nrow=S, ncol=S),
        algorithm=algo)
    return float(res[0])

# ===================== balance -> structural parameters =====================
def params_for_balance(balance, N):
    N_modules  = max(1, int(M * (1 - balance**2)))
    s_ratio    = 1.0 + (s_ratio_max - 1) * (1 - balance)
    lam        = λ_min + (λ_max - λ_min) * (1 - balance)
    lambda_vec = np.full(N, lam)
    return int(N_modules), float(s_ratio), lambda_vec, float(lam)

# ===================== Simulate one community =====================
def simulate_community(N, C0, m_vec, N_modules, s_ratio, lambda_vec, seed):
    np.random.seed(seed)
    try:
        u = paramn.modular_uptake(N, M, N_modules, s_ratio)
        l = paramn.generate_l_tensor(N, M, N_modules, s_ratio, lambda_vec)
    except Exception as e:
        return False, {"reason": str(e)}

    try:
        Y0    = np.concatenate([C0, R0])
        event = make_micrm_equilibrium_event(N, M, u, l, m_vec, rho, omega, lambda_vec)
        sol   = solve_ivp(
            lambda t, y: dCdt_Rdt(t, y, N, M, u, l, m_vec, rho, omega, lambda_vec),
            t_span, Y0, method='BDF', events=event)
        if sol.y.shape[1] == 0:
            return False, {"reason": "empty integration"}

        C_eq  = sol.y[:N, -1]
        R_eq  = sol.y[N:, -1]
        alpha, r = calculate_elv_params(C_eq, R_eq, N, M, u, l, m_vec, rho, omega, lambda_vec)
        ev    = leading_eigenvalue(MiCRM_jac(N, M, u, l, m_vec, rho, omega, lambda_vec, sol))
        mask  = C_eq > thr
        feas  = feasibility_prob(alpha[np.ix_(mask, mask)]) if np.any(mask) else 0.0

        total_up  = np.sum(u * R0[None, :], axis=1)
        net_up    = (1 - lambda_vec) * total_up - m_vec
        spec_cue  = net_up / (total_up + 1e-12)
        cue = (np.sum(C_eq[mask] * spec_cue[mask]) / (np.sum(C_eq[mask]) + 1e-12))

        return True, {
            "C_eq": C_eq, "R_eq": R_eq,
            "alpha": alpha, "r": r,
            "ev_micrm": ev, "feas_prob": feas, "cue": cue,
            "u": u, "l": l, "lambda_vec": lambda_vec,
            "N_modules": N_modules, "s_ratio": s_ratio,
        }
    except Exception as e:
        return False, {"reason": str(e)}

# ===================== Collect replicates for one balance =====================
def collect_replicates(balance, N, C0, m_vec, base_seed):
    N_modules, s_ratio, lambda_vec, lam = params_for_balance(balance, N)
    reps, tries, seed = [], 0, base_seed

    while len(reps) < n_repeats_target and tries < max_seed_tries:
        tries += 1
        ok, payload = simulate_community(N, C0, m_vec, N_modules, s_ratio, lambda_vec, seed)
        seed += 1
        if not ok:
            continue
        if np.isfinite(payload["ev_micrm"]) and payload["ev_micrm"] < ev_threshold:
            reps.append(payload)

    return {
        "reps": reps,
        "lam": lam, "lambda_vec": lambda_vec,
        "N_modules": N_modules, "s_ratio": s_ratio,
        "valid_repeats": len(reps), "tries": tries,
    }

# ===================== Coalescence by combination =====================
def merge_and_measure(rep1, rep2, eps_feas=1e-3):
    try:
        u1, l1, C1_eq = rep1["u"], rep1["l"], rep1["C_eq"]
        u2, l2, C2_eq = rep2["u"], rep2["l"], rep2["C_eq"]
        N3 = u1.shape[0] + u2.shape[0]

        u3 = np.vstack([u1, u2])
        l3 = np.vstack([l1, l2])
        m3 = np.concatenate([np.full(u1.shape[0], 0.1), np.full(u2.shape[0], 0.1)])
        lambda_vec3 = np.concatenate([rep1["lambda_vec"], rep2["lambda_vec"]])

        Y0_3  = np.concatenate([C1_eq, C2_eq, R0])
        event3 = make_micrm_equilibrium_event(N3, M, u3, l3, m3, rho, omega, lambda_vec3)
        sol3   = solve_ivp(
            lambda t, y: dCdt_Rdt(t, y, N3, M, u3, l3, m3, rho, omega, lambda_vec3),
            t_span, Y0_3, method='BDF', events=event3)
        if sol3.y.shape[1] == 0:
            return False, np.nan, np.nan, np.nan

        C3_eq = sol3.y[:N3, -1]
        R3_eq = sol3.y[N3:, -1]
        alpha3, _ = calculate_elv_params(C3_eq, R3_eq, N3, M, u3, l3, m3, rho, omega, lambda_vec3)
        ev3   = leading_eigenvalue(MiCRM_jac(N3, M, u3, l3, m3, rho, omega, lambda_vec3, sol3))
        mask3 = C3_eq > thr
        feas3 = feasibility_prob(alpha3[np.ix_(mask3, mask3)], eps=eps_feas) if np.any(mask3) else 0.0

        total_up3 = np.sum(u3 * R0[None, :], axis=1)
        net_up3   = (1 - lambda_vec3) * total_up3 - m3
        spec_cue3 = net_up3 / (total_up3 + 1e-12)
        cue_obs   = (np.sum(C3_eq[mask3] * spec_cue3[mask3]) /
                     (np.sum(C3_eq[mask3]) + 1e-12))

        return True, float(ev3), float(feas3), float(cue_obs)
    except Exception:
        return False, np.nan, np.nan, np.nan

# ===================== Main =====================
def main():
    base_seed = 43

    bank1, seed_cursor = {}, base_seed
    for b1 in tqdm(balances, desc="Community 1"):
        out = collect_replicates(b1, N1, np.ones(N1), m1, seed_cursor)
        bank1[b1] = out
        seed_cursor += max(1, out["tries"])

    bank2 = {}
    for b2 in tqdm(balances, desc="Community 2"):
        out = collect_replicates(b2, N2, np.ones(N2), m2, seed_cursor)
        bank2[b2] = out
        seed_cursor += max(1, out["tries"])

    rows = []

    for b1, pack in bank1.items():
        for k, rep in enumerate(pack["reps"]):
            rows.append({
                "community": "community1", "replicate": k,
                "balance1": float(b1), "balance2": np.nan,
                "ev": rep["ev_micrm"], "feasibility": rep["feas_prob"], "cue_obs": rep["cue"],
                "N_modules": pack["N_modules"], "s_ratio": pack["s_ratio"],
                "lambda_scalar": pack["lam"],
            })

    for b2, pack in bank2.items():
        for k, rep in enumerate(pack["reps"]):
            rows.append({
                "community": "community2", "replicate": k,
                "balance1": np.nan, "balance2": float(b2),
                "ev": rep["ev_micrm"], "feasibility": rep["feas_prob"], "cue_obs": rep["cue"],
                "N_modules": pack["N_modules"], "s_ratio": pack["s_ratio"],
                "lambda_scalar": pack["lam"],
            })

    for b1 in tqdm(balances, desc="Merging"):
        reps1 = bank1[b1]["reps"]
        for b2 in balances:
            reps2 = bank2[b2]["reps"]
            n_pairs = min(len(reps1), len(reps2), n_repeats_target)
            for i in range(n_pairs):
                ok, ev3, feas3, cue_obs = merge_and_measure(reps1[i], reps2[i])
                if ok and np.isfinite(ev3):
                    rows.append({
                        "community": "community3", "replicate": i,
                        "balance1": float(b1), "balance2": float(b2),
                        "ev": ev3, "feasibility": feas3, "cue_obs": cue_obs,
                        "N_modules": np.nan, "s_ratio": np.nan, "lambda_scalar": np.nan,
                    })

    pd.DataFrame(rows, columns=[
        "community", "replicate",
        "balance1", "balance2",
        "ev", "feasibility", "cue_obs",
        "N_modules", "s_ratio", "lambda_scalar",
    ]).to_csv("scan_balances_results.csv", index=False)
    print("done → scan_balances_results.csv")

if __name__ == "__main__":
    main()
