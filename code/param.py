import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def modular_uptake(N, M, N_modules, s_ratio):
    assert N_modules <= M and N_modules <= N, "N_modules must be less than or equal to both M and N"

    # Baseline calculations
    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    # Get module sizes for M
    diffR = np.full(N_modules, sR, dtype=int)
    diffR[np.random.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    # Get module sizes for N
    diffC = np.full(N_modules, sC, dtype=int)
    diffC[np.random.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))]

    # Preallocate u matrix
    u = np.random.rand(N, M)

    # Apply scaling
    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio
        
    # Normalize each row
    for i in range(N):
        u[i, :] /= np.sum(u[i, :])
    return u


def modular_leakage(M, N_modules, s_ratio, λ):
    assert N_modules <= M, "N_modules must be less than or equal to M"

    # Baseline
    sR = M // N_modules
    dR = M - (N_modules * sR)

    # Get module sizes and add to make to M
    diffR = np.full(N_modules, sR, dtype=int)
    diffR[np.random.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    l = np.random.rand(M, M)

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    for i in range(M):
        l[i, :] = λ * l[i, :] / np.sum(l[i, :])

    return l


def generate_l_tensor(N, M, N_modules, s_ratio, λ):
    l_tensor = np.array([modular_leakage(M, N_modules, s_ratio, λ) for _ in range(N)])
    return l_tensor


def solve_micrm(N, M, u, l, m, lambda_alpha, rho, omega, 
                C0, R0, t_span=None, t_eval=None):
    """
    Solve the MiCRM (Microbial Consumer Resource Model) ODE system.

    Parameters:
        N, M             - Number of consumers and resources
        u                - Uptake matrix (N x M)
        l                - Leakage tensor (N x M x M)
        m                - Maintenance costs (N,)
        lambda_alpha     - Leakage fraction per resource (M,)
        rho, omega       - Resource input and decay rates (M,)
        C0, R0           - Optional initial conditions for C and R
        t_span           - Time span for integration (tuple); default is (0, 600)
        t_eval           - Time points to evaluate the solution (array)

    Returns:
        sol              - Solution object from scipy.integrate.solve_ivp
    """
    if t_span is None:
        t_span = (0, 600)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)
    if C0 is None:
        C0 = np.full(N, 0.01)
    if R0 is None:
        R0 = np.full(M, 1.0)

    Y0 = np.concatenate([C0, R0])

    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]
        dCdt = np.zeros(N)
        dRdt = np.zeros(M)

        # Consumer dynamics
        for i in range(N):
            dCdt[i] = sum(C[i] * R[α] * u[i, α] * (1 - lambda_alpha[α]) for α in range(M)) - C[i] * m[i]

        # Resource dynamics
        for α in range(M):
            dRdt[α] = rho[α] - R[α] * omega[α]
            dRdt[α] -= sum(C[i] * R[α] * u[i, α] for i in range(N))
            dRdt[α] += sum(sum(C[i] * R[β] * u[i, β] * l[i, β, α] for β in range(M)) for i in range(N))

        return np.concatenate([dCdt, dRdt])

    return solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval, method="BDF")


def compute_alpha_r(C_hat, R_hat, N, M, u, l, m, lambda_alpha, omega):

    D = np.diag(omega + np.sum(C_hat[:, np.newaxis] * u, axis=0))
    D -= np.einsum('i,ig,iag->ag', C_hat, u * R_hat, l)
    partial_R_C = np.zeros((M, N))
    for j in range(N):
        v_j = -R_hat * u[j] + np.einsum('b,b,ba->a', R_hat, u[j], l[j])
        partial_R_C[:, j] = np.linalg.solve(D, v_j)
    alpha = np.einsum('ia,a,aj->ij', u, 1 - lambda_alpha, partial_R_C)
    r = np.sum(u * (1 - lambda_alpha) * R_hat, axis=1) - m - np.sum(alpha * C_hat, axis=1)
    return alpha, r


def solve_elv(alpha, r, C0, t_span=(0, 600), t_eval=None):
    N = len(C0)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)

    def dCdt_elv(t, C):
        dCdt = np.zeros(N)
        for i in range(N):
            dCdt[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in range(N)))
        return dCdt

    sol = solve_ivp(dCdt_elv, t_span, C0, t_eval=t_eval, method="BDF")
    return sol
