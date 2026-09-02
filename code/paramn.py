# This file generates the U matrix and L matrix required by MiCRM,
# supporting modular structure and tunable cooperation strength.
import numpy as np

def _split_modules_det(total, K):
    """
    Deterministically split [0..total-1] into K consecutive modules,
    making them as even as possible.
    Using deterministic rather than random remainder allocation
    reduces noise across different balance values.
    """
    assert K >= 1
    base = total // K
    rem  = total - base * K
    sizes = [base + 1] * rem + [base] * (K - rem)
    groups, s = [], 0
    for sz in sizes:
        groups.append(list(range(s, s + sz)))
        s += sz
    return groups  # list[list[int]]

def _coop_strength_from_sratio(s_ratio):
    """
    Map s_ratio (>= 1) to cooperation strength gamma in [0,1):
      s_ratio = 1   -> gamma = 0   (most competitive)
      s_ratio -> inf -> gamma -> 1 (most cooperative)
    """
    s_ratio = max(1.0, float(s_ratio))
    return 1.0 - 1.0 / s_ratio  # monotonic, smooth, no upper-bound dependence

# ------------------ uptake ------------------
def modular_uptake(N, M, N_modules, s_ratio, kappa=10.0, main_bias_scale=5.0):
    """
    Generate an (N x M) uptake matrix with row sum = 1 (Dirichlet sampling).
    - Compatible with the old function signature
    - Larger s_ratio -> more cooperative (more specialized, lower overlap)
    - s_ratio = 1    -> most competitive (more uniform, higher overlap)

    Key parameters:
      kappa           : Dirichlet concentration coefficient;
                        larger values give lower variance (smoother curves)
      main_bias_scale : baseline multiplier for the main-resource bias strength
    """
    assert 1 <= N_modules <= min(N, M), "N_modules must be <= min(N,M)"
    gamma = _coop_strength_from_sratio(s_ratio)   # cooperation strength in [0,1)
    res_modules = _split_modules_det(M, N_modules)
    cons_modules = _split_modules_det(N, N_modules)

    U = np.zeros((N, M), dtype=float)

    # "Module bias" + "main resource bias":
    # stronger cooperation means stronger specialization within a module
    for k, consumers in enumerate(cons_modules):
        Rk = np.array(res_modules[k], dtype=int)
        mk = len(Rk)
        # Baseline alpha: enhanced inside the module, weak outside the module
        alpha_base = (1.0 - gamma) * np.ones(M, dtype=float)
        alpha_base[Rk] += gamma * (M / max(mk, 1))

        # Each consumer additionally selects one "main resource" within the module,
        # with an extra bias to reduce overlap among rows
        # A deterministic round-robin assignment is used here;
        # randomness still comes from the Dirichlet sampling step
        for idx, i in enumerate(consumers):
            alpha_i = alpha_base.copy()
            if mk > 0:
                main_res = Rk[idx % mk]
                # Main-resource bias strength increases monotonically with gamma
                bias = main_bias_scale * M * gamma
                alpha_i[main_res] += bias
            # Noise reduction: scale everything by kappa
            alpha_i = np.maximum(alpha_i, 1e-8) * float(kappa)
            U[i, :] = np.random.dirichlet(alpha_i)

    # Numerical safety: renormalize rows
    # Dirichlet already sums to 1, but this is an extra safeguard
    U /= U.sum(axis=1, keepdims=True)
    return U

# ------------------ leakage (per-species MxM) ------------------
def modular_leakage(M, N_modules, s_ratio, λ, mode="ring",
                    phi_min=0.05, phi_max=0.6, add_noise=0.0):
    assert 1 <= N_modules <= M, "N_modules must be <= M"
    gamma = _coop_strength_from_sratio(s_ratio)
    phi = float(phi_min + (phi_max - phi_min) * gamma)

    res_modules = _split_modules_det(M, N_modules)
    L = np.zeros((M, M), dtype=float)

    for k, Rk in enumerate(res_modules):
        Rk = np.array(Rk, dtype=int)
        mk = len(Rk)
        if mk == 0:
            continue

        # Next module
        next_k = (k + 1) % N_modules
        Rnext = np.array(res_modules[next_k], dtype=int)
        mn = len(Rnext)

        # Target distribution within the current module: w_in
        # diagonal is largest, and off-diagonal entries increase as gamma increases
        v_coop_diag = 2.0 / (mk + 1.0)
        v_coop_off  = 1.0 / (mk + 1.0)

        for idx, r in enumerate(Rk):
            # Place the "diagonal" at the within-module position idx for this row
            w_in = np.full(mk, gamma * v_coop_off, dtype=float)
            w_in[idx] = (1 - gamma) * 1.0 + gamma * v_coop_diag
            # Since both v_comp and v_coop are already normalized,
            # w_in is naturally normalized as well

            # Within-module allocation: (1 - phi)
            L[r, Rk] = (1.0 - phi) * w_in

            # Across-module allocation: phi (evenly distributed in ring mode)
            if mn > 0:
                L[r, Rnext] += phi * (1.0 / mn)

    # Numerical safety and row scaling to λ
    if add_noise > 0:
        L += add_noise * np.random.rand(M, M)
    L = np.maximum(L, 1e-12)
    L /= L.sum(axis=1, keepdims=True)
    L *= float(λ)
    return L


# ------------------ stack N species ------------------
def generate_l_tensor(N, M, N_modules, s_ratio, lambda_vec,
                      mode="ring", phi_min=0.05, phi_max=0.6):
    """
    Generate an (M x M) leakage matrix L_i for each species i,
    where each row sums to lambda_vec[i].
    Returns a tensor of shape (N, M, M).
    """
    assert len(lambda_vec) == N, "Length of lambda_vec must equal the number of species N"
    l_tensor = np.zeros((N, M, M), dtype=float)
    for i in range(N):
        l_tensor[i] = modular_leakage(
            M, N_modules, s_ratio, λ=float(lambda_vec[i]),
            mode=mode, phi_min=phi_min, phi_max=phi_max
        )
    return l_tensor