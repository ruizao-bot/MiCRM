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


def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method='LSODA'
):
    """
    Integrate the MiCRM ODEs until equilibrium or t_span is reached.

    Parameters:
        N, M: int
            Number of consumers and resources.
        u, l, m, lambda_alpha, rho, omega: model parameters.
        C0, R0: initial conditions for consumers and resources.
        t_span: tuple
            Time span for integration (default: (0, 1000)).
        t_eval: array or None
            Time points to evaluate solution (default: 300 points in t_span).
        tol: float
            Tolerance for equilibrium detection.
        method: str
            Integration method for solve_ivp.

    Returns:
        sol: OdeResult
            Solution object from scipy.integrate.solve_ivp.
    """
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]
        uptake = u * (R * (1 - lambda_alpha))  # (N, M)
        dCdt = C * (np.sum(uptake, axis=1) - m)
        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)  # (M,)
        dRdt -= consumption
        leakage = np.einsum('i,j,ij,ijk->k', C, R, u, l)
        dRdt += leakage
        return np.concatenate([dCdt, dRdt])

    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y)
        return np.max(np.abs(deriv)) - tol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)
    Y0 = np.concatenate([C0, R0])

    sol = solve_ivp(
        dCdt_Rdt, t_span, Y0, t_eval=t_eval, method=method,
        events=equilibrium_event
    )
    return sol

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


def solve_elv(alpha, r, C0, t_span=(0, 1000), t_eval=None):
    N = len(C0)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)

    def dCdt_elv(t, C):
        dCdt = np.zeros(N)
        for i in range(N):
            dCdt[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in range(N)))
        return dCdt

    sol = solve_ivp(dCdt_elv, t_span, C0, t_eval=t_eval, method="RK45")
    return sol
