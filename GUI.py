import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import math

import numpy as np



def rk4(r, t, h, w_alpha, w_beta, w_gamma, w_delta):
    """ Runge-Kutta 4 method """
    k1 = h * f(r, t, w_alpha, w_beta, w_gamma, w_delta)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k4 = h * f(r + k3, t + h, w_alpha, w_beta, w_gamma, w_delta)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def f(r, t, w_alpha, w_beta, w_gamma, w_delta):
    alpha = w_alpha
    beta = w_beta
    gamma = w_gamma
    delta = w_delta
    x, y = r[0], r[1]
    fxd = x * (alpha - beta * y)
    fyd = -y * (gamma - delta * x)
    return np.array([fxd, fyd], float)

def updateGraph():
    global currentGraph
    plt.clf()
    h = 0.001  # edited
    tpoints = np.arange(0, 100, h)
    xpoints, ypoints = [], []
    r = np.array([10, 5], float)
    for t in tpoints:
        xpoints.append(r[0])
        ypoints.append(r[1])
        r += rk4(r, t, h, w_alpha.get(), w_beta.get(), w_gamma.get(), w_delta.get())
    plt.plot(tpoints, xpoints)
    plt.plot(tpoints, ypoints)
    fig.canvas.draw()

# This defines the Python GUI backend to use for matplotlib
matplotlib.use('TkAgg')
# Initialize an instance of Tk
root = tk.Tk()
# Initialize matplotlib figure for graphing purposes
fig = plt.figure(1)
# Special type of "canvas" to allow for matplotlib graphing
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()

h=0.001
tpoints = np.arange(0, 100, h)
xpoints, ypoints  = [], []
r = np.array([10, 5], float)
for t in tpoints:
        xpoints.append(r[0])
        ypoints.append(r[1])
        r += rk4(r, t, h, 0.3, 0.28, 0.7, 0.3)
plt.plot(tpoints, xpoints)
plt.plot(tpoints, ypoints)

# Add the plot to the tkinter widget
plot_widget.pack()
# Create a tkinter button at the bottom of the window and link it with the updateGraph function

f_top_alpha = tk.Frame(root)
l_1 = tk.Label(f_top_alpha, text="Alpha")
w_alpha = tk.Scale(f_top_alpha, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
w_alpha.set(0.3)
f_top_alpha.pack()
l_1.pack(side=tk.LEFT)
w_alpha.pack(side=tk.LEFT)

f_top_beta= tk.Frame(root)
l_2 = tk.Label(f_top_beta, text="Beta")
w_beta = tk.Scale(f_top_beta, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
w_beta.set(0.28)
f_top_beta.pack()
l_2.pack(side=tk.LEFT)
w_beta.pack(side=tk.LEFT)

f_top_gamma = tk.Frame(root)
l_3 = tk.Label(f_top_gamma, text="Gamma")
w_gamma = tk.Scale(f_top_gamma, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
w_gamma.set(0.7)
f_top_gamma.pack()
l_3.pack(side=tk.LEFT)
w_gamma.pack(side=tk.LEFT)

f_top_delta = tk.Frame(root)
l_4 = tk.Label(f_top_delta, text="Delta")
w_delta= tk.Scale(f_top_delta, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
w_delta.set(0.3)
f_top_delta.pack()
l_4.pack(side=tk.LEFT)
w_delta.pack(side=tk.LEFT)

tk.Button(root,text="Update",command=updateGraph).pack()

root.mainloop()
