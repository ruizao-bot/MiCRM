import numpy as np
import matplotlib.pyplot as plt

a0 = 468.377
a = [-10.908, -94.690, -23.415, 15.252, 6.585, -47.030, -34.384, 6.605]
b = [10.091, -9.844, -4.692, 29.007, -21.353, 9.151, 12.451, 20.043]
x0, y0 = 834.99, 1074.41

theta = np.linspace(0, 2*np.pi, 720)
r = a0 + sum(a[k]*np.cos((k+1)*theta) + b[k]*np.sin((k+1)*theta) for k in range(8))
x = x0 + r*np.cos(theta)
y = y0 + r*np.sin(theta)

plt.plot(x, y)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.show()
