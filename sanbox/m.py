import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import param

# -------------------------
# Parameters
N = 5           # consumer 数量
M = 4           # resource 数量
λ = 0.3           # 总 leakage rate
λ_u = np.ones(N)  # 每个 consumer 的 uptake scaling factor (全为1)

N_modules = 2     
s_ratio = 10.0 


# 生成 uptake matrix 和 leakage tensor
u = param.modular_uptake(N, M, N_modules, s_ratio, λ_u)  # uptake matrix, shape (N, M)
lambda_alpha = np.full(M, λ)                             # 每个 resource 的 leakage rate
m = np.linspace(0, 0.8, N) 
rho = np.full(M, 1)           # 资源输入浓度（全为1）
omega = np.full(M, 0.05)      # 资源衰减率

l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ)  # leakage tensor


# ODE system
def dCdt_Rdt(t, y):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]
    
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))
    
    return np.concatenate([dCdt, dRdt])

# Initial values
C0 = np.full(N, 1)  # consumer
R0 = np.full(M, 1)  # resource
Y0 = np.concatenate([C0, R0])

# Time scale
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)

# Solve ODE
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  
    R = sol.y[N:, i]  
    total_uptake = u @ (R0-R)  
    net_uptake = total_uptake * (1 - λ) - m  
    CUE[:, i] = net_uptake / total_uptake  

# 计算最终 CUE 值
final_CUE = CUE[:, -1]

# 线性回归函数
def plot_regression(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label='Data points', color='b')
    
    # 计算线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    
    plt.plot(x_fit, y_fit, color='r', linestyle='dashed', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# 画 u_mean vs. CUE 关系
plot_regression(m, final_CUE, "Maintanence", "Final CUE", "Maintancence vs. CUE")