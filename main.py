import matplotlib.pyplot as plt
import numpy as np

def rk4(r, t, h):                    #edited; no need for input f
        """ Runge-Kutta 4 method """
        k1 = h*f(r, t)
        k2 = h*f(r+0.5*k1, t+0.5*h)
        k3 = h*f(r+0.5*k2, t+0.5*h)
        k4 = h*f(r+k3, t+h)
        return (k1 + 2*k2 + 2*k3 + k4)/6

def f(r, t):
        alpha = 1.0
        beta = 0.5
        gamma = 0.5
        sigma = 2.0
        x, y = r[0], r[1]
        fxd = x*(alpha - beta*y)
        fyd = -y*(gamma - sigma*x)
        return np.array([fxd, fyd], float)

# h=0.001                               #edited
# tpoints = np.arange(0, 30, h)         #edited
# xpoints, ypoints  = [], []
# r = np.array([2, 2], float)
# for t in tpoints:
#         xpoints.append(r[0])          #edited
#         ypoints.append(r[1])          #edited
#         r += rk4(r, t, h)             #edited; no need for input f
# plt.plot(tpoints, xpoints)
# plt.plot(tpoints, ypoints)
# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.title("Lotka-Volterra Model")
# plt.savefig("Lotka_Volterra.png")
# plt.show()