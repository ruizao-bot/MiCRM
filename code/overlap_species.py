# This script simulates the case where the two parental communities share a subset of species.
# We coalese them at t=0.
# Results are saved to a CSV file.
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
from contextlib import contextmanager
# ===================== Global parameters =====================
M = 20

N_shared = 5
N_unique = 15
N_parent = N_shared + N_unique              # 20
N_merge  = N_shared + N_unique + N_unique   # 35

balances  = np.round(np.arange(0.0, 1.0 + 0.05, 0.05), 2)
n_repeats = 20 

rho   = np.full(M, 5.0)
omega = np.full(M, 0.2)
m0 = 0.1

λ_min = 0.0
λ_max = 0.5
s_ratio_max = 50.0

thr = 1e-5
t_span = (0, 1e8)
R0 = np.full(M, 1.0)

# ===================== Seed retry / numerical health parameters =====================
MAX_QC_TRIES = 50

# Adaptive ridge: only to ensure numerical stability for logdet/svd (the proxy does not need large regularization)
COND_MAX = 1e10
EPS_REL_INIT = 1e-10
EPS_REL_CAP  = 1e-2
EPS_GROWTH   = 10.0
EPS_REFINE_STEPS = 20

# ===================== Utility: make paramn randomness reproducible =====================
@contextmanager
def with_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)

# ===================== balance -> parameter mapping =====================
def build_params_for_balance(balance, N, M, s_ratio_max, λ_min, λ_max):
    """
    N_modules: changes monotonically with balance
    s_ratio:   smaller balance -> larger s_ratio
    lam:       smaller balance -> larger lam
    """
    N_modules = max(1, int(M * (1 - balance**2)))
    s_ratio   = 1 + (s_ratio_max - 1) * (1 - balance)
    lam       = λ_min + (λ_max - λ_min) * (1 - balance)
    lambda_vec = np.full(N, lam)
    return N_modules, s_ratio, lambda_vec, lam

def cap_modules(N_modules, N, M):
    return int(np.clip(N_modules, 1, min(N, M)))

# ===================== MiCRM dynamics and helper functions =====================
def dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec):
    C = y[:N]
    R = y[N:]
    uptake_sum = np.sum(u * R[None, :], axis=1)
    dCdt = C * ((1 - lambda_vec) * uptake_sum - m)

    dRdt = rho - omega * R
    consumption = np.sum(C[:, None] * u * R, axis=0)
    dRdt -= consumption
    leakage = np.einsum("i,j,ij,ijk->k", C, R, u, l)
    dRdt += leakage
    return np.concatenate([dCdt, dRdt])

def calculate_elv_params(C_hat, R_hat, N, M, u, l_tensor, m, rho, omega, lambda_vec):
    C_u = C_hat[:, None] * u
    L = np.einsum("ib,iab->ab", C_u, l_tensor, optimize="optimal")
    diag = omega + np.sum(C_u, axis=0)
    D = np.diag(diag) - L

    term1 = -u * R_hat
    term2 = np.einsum("ib,iba->ia", u * R_hat, l_tensor, optimize="optimal")
    V = term1 + term2

    partial_R_C = np.linalg.solve(D, V.T)
    weighted_u  = (1 - lambda_vec)[:, None] * u
    alpha = weighted_u @ partial_R_C

    growth_terms = (1 - lambda_vec) * np.sum(u * R_hat, axis=1)
    r_vec = growth_terms - m - alpha @ C_hat
    return alpha, r_vec

def make_micrm_equilibrium_event(N, M, u, l, m, rho, omega, lambda_vec, tol=1e-5):
    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec)
        return np.max(np.abs(deriv)) - tol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1
    return equilibrium_event

def MiCRM_jac(N, M, u, l, m, rho, omega, lambda_vec, sol):
    state = sol.y[:, -1]
    C = state[:N]
    R = state[N:]

    cc_diag = -m + ((1 - lambda_vec)[:, None] * u * R[None, :]).sum(axis=1)
    CC = np.diag(cc_diag)
    CR = C[:, None] * (1 - lambda_vec)[:, None] * u

    P = C[:, None, None] * u[:, :, None] * l
    RR = P.sum(axis=0)
    diag_val = np.diag(RR)
    sub_diag = (C[:, None] * u).sum(axis=0)
    diag_rr = diag_val - sub_diag - omega
    np.fill_diagonal(RR, diag_rr)

    Q = u * R[None, :]
    Ql = Q[:, :, None] * l
    term2 = Ql.sum(axis=1)
    RC = (term2 - Q).T

    top = np.hstack([CC, CR])
    bottom = np.hstack([RC, RR])
    return np.vstack([top, bottom])

def leading_eigenvalue(J):
    eigvals = np.linalg.eigvals(J)
    re = np.real(eigvals)
    return float(re[np.argmax(re)])

# ===================== Diagnostic metrics =====================
def compute_row_cosine_similarities(u):
    N = u.shape[0]
    if N < 2:
        return 0.0
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    sim = u_normalized @ u_normalized.T
    off_diag = sim[~np.eye(N, dtype=bool)]
    return float(np.mean(off_diag)) if off_diag.size else 0.0

def compute_effective_leakage(u, l):
    N, M = u.shape
    if N < 2:
        return 0.0
    total_u_sum = np.sum(u, axis=0)
    vals = []
    for i in range(N):
        avg_u_others = (total_u_sum - u[i]) / (N - 1)
        li = l[i].copy()
        np.fill_diagonal(li, 0.0)
        leak_to_b = li.sum(axis=0)
        vals.append(float(np.sum(leak_to_b * avg_u_others)))
    return float(np.mean(vals)) if vals else 0.0

# ===================== Adaptive ridge: find minimal eps to keep A_reg numerically well-conditioned =====================
def estimate_scale_for_eps(A):
    diag = np.diag(A)
    diag_abs = np.abs(diag[np.isfinite(diag)])
    diag_abs = diag_abs[diag_abs > 0]
    if diag_abs.size > 0:
        scale = float(np.median(diag_abs))
    else:
        a_abs = np.abs(A[np.isfinite(A)])
        a_abs = a_abs[a_abs > 0]
        scale = float(np.median(a_abs)) if a_abs.size > 0 else 1.0
    return max(scale, 1e-12)

def adaptive_ridge(A,
                  cond_max=COND_MAX,
                  eps_rel_init=EPS_REL_INIT,
                  eps_rel_cap=EPS_REL_CAP,
                  growth=EPS_GROWTH,
                  refine_steps=EPS_REFINE_STEPS):
    """
    Find the smallest eps_abs = eps_rel * scale such that:
      cond(A + eps_abs I) <= cond_max
    """
    S = A.shape[0]
    if S == 0:
        return None
    I = np.eye(S)
    scale = estimate_scale_for_eps(A)

    def ok_at(eps_rel):
        eps_abs = eps_rel * scale
        A_reg = A + eps_abs * I
        try:
            cond = float(np.linalg.cond(A_reg))
        except np.linalg.LinAlgError:
            return False, None
        if (not np.isfinite(cond)) or (cond > cond_max):
            return False, None
        return True, {"A_reg": A_reg, "cond": cond, "eps_abs": eps_abs, "eps_rel": eps_rel, "scale": scale}

    eps_rel = eps_rel_init
    prev = None
    ok, met = ok_at(eps_rel)
    while (not ok) and (eps_rel < eps_rel_cap):
        prev = eps_rel
        eps_rel *= growth
        ok, met = ok_at(eps_rel)

    if not ok:
        return None

    lo = prev if prev is not None else eps_rel_init / growth
    lo = max(lo, eps_rel_init / growth)
    hi = eps_rel
    best = met

    for _ in range(refine_steps):
        mid = np.sqrt(lo * hi)
        okm, metm = ok_at(mid)
        if okm:
            hi = mid
            best = metm
        else:
            lo = mid

    best["status"] = "ok"
    return best

# ===================== C2: logdet / spectral proxies + feasibility-domain mapping =====================
def compute_proxies_from_A(A_reg):
    """
    Returns:
      - logdet(A A^T) = 2 * sum(log sigma_i)
      - log_volume = -0.5 * logdet(A A^T)
      - log_sigma_min, log_inv_spec, log_inv_frob
      - sigma_min, sigma_max
    """
    s = np.linalg.svd(A_reg, compute_uv=False)
    smin = float(np.min(s))
    smax = float(np.max(s))

    tiny = 1e-300
    logdet_AAT = float(2.0 * np.sum(np.log(s + tiny)))
    log_volume = float(-0.5 * logdet_AAT)
    log_smin = float(np.log(smin + tiny))
    log_inv_spec = float(-log_smin)

    inv_frob = float(np.sqrt(np.sum(1.0 / (s * s + tiny))))
    log_inv_frob = float(np.log(inv_frob + tiny))

    return {
        "sigma_min": smin,
        "sigma_max": smax,
        "logdet_AAT": logdet_AAT,
        "log_volume": log_volume,
        "log_sigma_min": log_smin,
        "log_inv_spec": log_inv_spec,
        "log_inv_frob": log_inv_frob,
    }


def map_log_volume_to_feasible_proxies(log_volume: float, S: int):
    S = max(int(S), 1)
    lv = float(log_volume)
    feasible_volume_proxy = float(np.exp(np.clip(lv, -700, 700)))
    feasible_scale_per_dim = float(np.exp(np.clip(lv / S, -700, 700)))

    log10_feasible_volume_proxy = float(lv / np.log(10.0))
    log10_feasible_scale_per_dim = float((lv / S) / np.log(10.0))

    return {
        "feasible_volume_proxy": feasible_volume_proxy,
        "feasible_scale_per_dim": feasible_scale_per_dim,
        "log10_feasible_volume_proxy": log10_feasible_volume_proxy,
        "log10_feasible_scale_per_dim": log10_feasible_scale_per_dim,
    }


# ===================== Build parental communities =====================
def build_parent_community(balance, rng, N=20, M=20):
    Nmod, sratio, lambda_vec, lam_scalar = build_params_for_balance(
        balance, N, M, s_ratio_max, λ_min, λ_max
    )
    Nmod = cap_modules(Nmod, N, M)

    seed = int(rng.integers(0, 2**31 - 1))
    with with_numpy_seed(seed):
        u = paramn.modular_uptake(N, M, Nmod, sratio)
        l = paramn.generate_l_tensor(N, M, Nmod, sratio, lambda_vec)

    return u, l, lambda_vec, Nmod, sratio, lam_scalar


# ===================== Single experiment: compute proxies (no pmvnorm feasibility) =====================
def run_one_b1b2_once(b1, b2, rng):
    # 1) Parent community 1 (N=20)
    u1, l1, lam1, Nmod1, sratio1, lam_scalar1 = build_parent_community(b1, rng, N=N_parent, M=M)

    shared_idx  = np.arange(0, N_shared, dtype=int)
    unique1_idx = np.arange(N_parent - N_unique, N_parent, dtype=int)

    u_shared   = u1[shared_idx, :]
    l_shared   = l1[shared_idx, :, :]
    lam_shared = lam1[shared_idx]

    u1_unique   = u1[unique1_idx, :]
    l1_unique   = l1[unique1_idx, :, :]
    lam1_unique = lam1[unique1_idx]

    # 2) Community 2 pool (N=20), take the last N_unique
    u2_pool, l2_pool, lam2_pool, Nmod2, sratio2, lam_scalar2 = build_parent_community(b2, rng, N=N_parent, M=M)
    unique2_idx = np.arange(N_parent - N_unique, N_parent, dtype=int)

    u2_unique   = u2_pool[unique2_idx, :]
    l2_unique   = l2_pool[unique2_idx, :, :]
    lam2_unique = lam2_pool[unique2_idx]

    # Parent community 2 = shared(from 1) + unique2(from b2)
    u2 = np.vstack([u_shared, u2_unique])
    l2 = np.vstack([l_shared, l2_unique])

    # Metrics
    u1_cos = compute_row_cosine_similarities(u1)
    u2_cos = compute_row_cosine_similarities(u2)
    leak1_eff = compute_effective_leakage(u1, l1)
    leak2_eff = compute_effective_leakage(u2, l2)
    u1_cos_unique = compute_row_cosine_similarities(u1_unique)
    u2_cos_unique = compute_row_cosine_similarities(u2_unique)

    # 3) Merged community (N_merge=35)
    u3 = np.vstack([u_shared, u1_unique, u2_unique])
    l3 = np.vstack([l_shared, l1_unique, l2_unique])
    lambda_vec3 = np.concatenate([lam_shared, lam1_unique, lam2_unique])

    m3 = np.full(N_merge, m0)

    # Initial condition: shared species start at 2, unique species start at 1
    C_shared0 = np.ones(N_shared) * 2.0
    C0_3 = np.concatenate([C_shared0, np.ones(N_unique), np.ones(N_unique)])
    Y0_3 = np.concatenate([C0_3, R0])

    micrm_event3 = make_micrm_equilibrium_event(N_merge, M, u3, l3, m3, rho, omega, lambda_vec3)
    sol3 = solve_ivp(
        lambda t, y: dCdt_Rdt(t, y, N_merge, M, u3, l3, m3, rho, omega, lambda_vec3),
        t_span,
        Y0_3,
        method="BDF",
        events=micrm_event3
    )
    if sol3.y.shape[1] == 0:
        return False, None

    C3_eq = sol3.y[:N_merge, -1]
    R3_eq = sol3.y[N_merge:, -1]

    alpha3, _ = calculate_elv_params(C3_eq, R3_eq, N_merge, M, u3, l3, m3, rho, omega, lambda_vec3)
    J3 = MiCRM_jac(N_merge, M, u3, l3, m3, rho, omega, lambda_vec3, sol3)
    ev3 = leading_eigenvalue(J3)

    mask3 = C3_eq > thr
    S_surv = int(np.sum(mask3))
    if S_surv == 0:
        return False, None

    A_surv = alpha3[np.ix_(mask3, mask3)]

    # Adaptive ridge, only to ensure logdet/svd numerical stability
    reg = adaptive_ridge(A_surv)
    if reg is None:
        return False, None

    A_reg = reg["A_reg"]
    proxies = compute_proxies_from_A(A_reg)

    mapped = map_log_volume_to_feasible_proxies(proxies["log_volume"], S_surv)

    payload = {
        "u1_cos": u1_cos,
        "u2_cos": u2_cos,
        "leak1": leak1_eff,
        "leak2": leak2_eff,
        "u1_cos_unique": u1_cos_unique,
        "u2_cos_unique": u2_cos_unique,
        "ev3": ev3,
        "S_surv": S_surv,

        # Regularization info
        "cond_alpha_reg": float(reg["cond"]),
        "eps_abs_used": float(reg["eps_abs"]),
        "eps_rel_used": float(reg["eps_rel"]),
        "alpha_scale": float(reg["scale"]),

        # Proxies
        **proxies,

        # Mapped feasibility-domain proxies
        **mapped,

        # Parent parameters (optional)
        "Nmod1": int(Nmod1), "sratio1": float(sratio1), "lam1": float(lam_scalar1),
        "Nmod2": int(Nmod2), "sratio2": float(sratio2), "lam2": float(lam_scalar2),
    }
    return True, payload


# ===================== Main loop: two-layer loop over b1,b2, with n_repeats per pair =====================
def run_simulation():
    base_seed = 43
    rows = []

    total_pairs = len(balances) ** 2
    pbar = tqdm(total=total_pairs, desc="Scanning (b1,b2) pairs")

    for b1 in balances:
        for b2 in balances:
            for rep in range(n_repeats):
                ok = False
                out = None
                used_try = 0

                seed_base = (
                    base_seed * 10_000_000 +
                    int(round(b1 * 1000)) * 10_000 +
                    int(round(b2 * 1000)) * 100 +
                    rep * 1_000_000
                )

                for ktry in range(MAX_QC_TRIES):
                    rng = np.random.default_rng(seed_base + ktry)
                    ok, out = run_one_b1b2_once(b1, b2, rng)
                    if ok:
                        used_try = ktry + 1
                        break

                if not ok:
                    continue

                rows.append({
                    "b1": float(b1),
                    "b2": float(b2),
                    "repeat": int(rep),
                    "qc_tries": int(used_try),

                    # Parent parameters
                    "parent1_N_modules": out["Nmod1"],
                    "parent1_s_ratio": out["sratio1"],
                    "parent1_lam": out["lam1"],
                    "parent2_pool_N_modules": out["Nmod2"],
                    "parent2_pool_s_ratio": out["sratio2"],
                    "parent2_pool_lam": out["lam2"],

                    # Overlap / leakage metrics
                    "u1_avg_cosine_similarity": out["u1_cos"],
                    "u1_unique_cosine_similarity": out["u1_cos_unique"],
                    "u2_avg_cosine_similarity": out["u2_cos"],
                    "u2_unique_cosine_similarity": out["u2_cos_unique"],
                    "community1_effective_leakage": out["leak1"],
                    "community2_effective_leakage": out["leak2"],

                    # Stability and survivors
                    "community3_stability_ev": out["ev3"],
                    "S_surv": out["S_surv"],

                    # Regularization diagnostics
                    "cond_alpha_reg": out["cond_alpha_reg"],
                    "eps_abs_used": out["eps_abs_used"],
                    "eps_rel_used": out["eps_rel_used"],
                    "alpha_scale": out["alpha_scale"],

                    # C2 proxies
                    "proxy_logdet_AAT": out["logdet_AAT"],
                    "proxy_log_volume": out["log_volume"],
                    "proxy_log_sigma_min": out["log_sigma_min"],
                    "proxy_log_inv_spec": out["log_inv_spec"],
                    "proxy_log_inv_frob": out["log_inv_frob"],
                    "sigma_min": out["sigma_min"],
                    "sigma_max": out["sigma_max"],

                    # Mapped feasibility-domain proxies
                    "feasible_volume_proxy": out["feasible_volume_proxy"],
                    "feasible_scale_per_dim": out["feasible_scale_per_dim"],
                    "log10_feasible_volume_proxy": out["log10_feasible_volume_proxy"],
                    "log10_feasible_scale_per_dim": out["log10_feasible_scale_per_dim"],
                })

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    out_name = "shared_species.csv"
    df.to_csv(out_name, index=False)
    print(f"Saved: {out_name}")
    print(f"Total rows saved: {len(df)}")
    return df


if __name__ == "__main__":
    print("Starting simulation...")
    print(f"N_shared={N_shared}, N_unique={N_unique}, N_merge={N_merge}")
    print(f"b1/b2 combinations: {len(balances)} × {len(balances)} = {len(balances) ** 2}")
    print(f"Repeats per (b1,b2): {n_repeats}")

    df = run_simulation()
    print("Simulation completed!")