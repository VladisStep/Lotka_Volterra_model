import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import math

import numpy as np
from main import rk4


def rk4(r, t, h, w_alpha, w_beta, w_gamma, w_delta):
    """ Runge-Kutta 4 method """
    k1 = h * f(r, t, w_alpha, w_beta, w_gamma, w_delta)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k4 = h * f(r + k3, t + h, w_alpha, w_beta, w_gamma, w_delta)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def f(r, t, w_alpha, w_beta, w_gamma, w_delta):
    alpha = w_alpha/100
    beta = w_beta/100
    gamma = w_gamma/100
    sigma = w_delta/100
    x, y = r[0], r[1]
    fxd = x * (alpha - beta * y)
    fyd = -y * (gamma - sigma * x)
    return np.array([fxd, fyd], float)

def updateGraph():
    global currentGraph
    plt.clf()
    h = 0.001  # edited
    tpoints = np.arange(0, 30, h)  # edited
    xpoints, ypoints = [], []
    r = np.array([2, 2], float)
    for t in tpoints:
        xpoints.append(r[0])  # edited
        ypoints.append(r[1])  # edited
        r += rk4(r, t, h, w_alpha.get(), w_beta.get(), w_gamma.get(), w_delta.get())  # edited; no need for input f
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

h=0.001                               #edited
tpoints = np.arange(0, 30, h)         #edited
xpoints, ypoints  = [], []
r = np.array([2, 2], float)
for t in tpoints:
        xpoints.append(r[0])          #edited
        ypoints.append(r[1])          #edited
        r += rk4(r, t, h, 100, 50, 50, 200)             #edited; no need for input f
plt.plot(tpoints, xpoints)
plt.plot(tpoints, ypoints)

# Add the plot to the tkinter widget
plot_widget.pack()
# Create a tkinter button at the bottom of the window and link it with the updateGraph function
tk.Label(root, text="Alpha").pack()
w_alpha = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
w_alpha.pack()

tk.Label(root, text="Beta").pack()
w_beta = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
w_beta.pack()


tk.Label(root, text="Gamma").pack()
w_gamma = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
w_gamma.pack()

tk.Label(root, text="Delta").pack()
w_delta = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
w_delta.pack()

tk.Button(root,text="Update",command=updateGraph).pack()

root.mainloop()
