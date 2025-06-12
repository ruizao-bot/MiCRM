import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
T = 100  # 总传代次数
f = 0.1  # 稀释因子
R_input = 10  # 每次传代补充的资源量
C = np.zeros(T)  # 微生物生物量
R = np.zeros(T)  # 资源浓度

# 初始值
C[0] = 1  # 初始生物量
R[0] = R_input  # 初始资源

# 进行 T 轮传代
for t in range(1, T):
    C[t] = f * C[t-1]  # 细胞生物量的传代
    R[t] = f * R[t-1] + (1 - f) * R_input  # 资源补充

# 绘制传代过程
plt.figure(figsize=(8, 5))
plt.plot(range(T), C, label="Microbial Biomass (C)", color='b')
plt.plot(range(T), R, label="Resource Concentration (R)", color='g')
plt.xlabel("Transfer Cycles")
plt.ylabel("Concentration")
plt.legend()
plt.title("Serial Transfer Simulation")
plt.show()
