import sys
import time

import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt

# TODO: размер графиков при изменении

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


def f(r, t, w_alpha, w_beta, w_gamma, w_delta, c, d):
    alpha = w_alpha
    beta = w_beta
    gamma = w_gamma
    delta = w_delta
    x, y = r[0], r[1]
    fxd = x * (alpha - beta * y) + c * x
    fyd = -y * (gamma - delta * x) + d * y
    return np.array([fxd, fyd], float)


def rk4(r, t, h, w_alpha, w_beta, w_gamma, w_delta, c, d):
    k1 = h * f(r, t, w_alpha, w_beta, w_gamma, w_delta, c, d)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta, c, d)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta, c, d)
    k4 = h * f(r + k3, t + h, w_alpha, w_beta, w_gamma, w_delta, c, d)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


class ApplicationWindow(QtWidgets.QMainWindow):
    _startX = 10
    _startY = 5
    xpoints = []
    ypoints = []

    alpha = 0.3
    beta = 0.28
    gamma = 0.7
    delta = 0.3

    c = 0.05
    d = 0

    h = 0.01



    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        layout.addWidget(dynamic_canvas)

        layoutAnimalsAndLetters = QtWidgets.QHBoxLayout(self._main)

        layoutAnimals = QtWidgets.QHBoxLayout(self._main)
        layoutLetters = QtWidgets.QVBoxLayout(self._main)

        layoutLabels = QtWidgets.QHBoxLayout(self._main)
        self.balanceX = QLabel(self._main)
        self.balanceX.setText("X")
        layoutLabels.addWidget(self.balanceX)
        self.balanceY = QLabel(self._main)
        self.balanceY.setText("Y")
        layoutLabels.addWidget(self.balanceY)
        layout.addLayout(layoutLabels)

        layoutStart = QtWidgets.QVBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("Preys")
        mySlider = QSlider(Qt.Vertical, self)
        mySlider.setMaximum(30)
        mySlider.setValue(10)
        mySlider.valueChanged.connect(self.changeValueX)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutAnimals.addLayout(layoutStart)

        layoutStart = QtWidgets.QVBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("Predatos")
        mySlider = QSlider(Qt.Vertical, self)
        mySlider.setMaximum(30)
        mySlider.setValue(5)
        mySlider.valueChanged.connect(self.changeValueY)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutAnimals.addLayout(layoutStart)

        layoutStart= QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("alpha")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setValue(30)
        mySlider.valueChanged.connect(self.changeValueAlpha)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutLetters.addLayout(layoutStart)

        layoutStart = QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("beta")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setValue(28)
        mySlider.valueChanged.connect(self.changeValueBeta)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutLetters.addLayout(layoutStart)

        layoutStart = QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("gamma")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setValue(70)
        mySlider.valueChanged.connect(self.changeValueGamma)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutLetters.addLayout(layoutStart)

        layoutStart = QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("delta")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setValue(30)
        mySlider.valueChanged.connect(self.changeValueDelta)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layoutLetters.addLayout(layoutStart)

        layoutAnimalsAndLetters.addLayout(layoutAnimals)
        layoutAnimalsAndLetters.addLayout(layoutLetters)

        layout.addLayout(layoutAnimalsAndLetters)

        layoutStart = QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("C")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setMaximum(200)
        mySlider.setValue(int(self.c * 100 + 100))
        mySlider.valueChanged.connect(self.changeValueC)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layout.addLayout(layoutStart)

        layoutStart = QtWidgets.QHBoxLayout(self._main)
        lbl = QLabel(self)
        lbl.setText("D")
        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setMaximum(200)
        mySlider.setValue(int(self.d * 100 + 100))
        mySlider.valueChanged.connect(self.changeValueD)
        layoutStart.addWidget(lbl)
        layoutStart.addWidget(mySlider)
        layout.addLayout(layoutStart)

        self._static_ax = static_canvas.figure.subplots()
        tpoints = np.arange(0, 100, self.h)
        xpoints, ypoints = [], []
        r = np.array([self._startX, self._startY], float)

        for t in tpoints:
            xpoints.append(r[0])
            ypoints.append(r[1])
            r += rk4(r, t, self.h, self.alpha, self.beta, self.delta, self.gamma, self.c, self.d)

        self._line, = self._static_ax.plot(xpoints, ypoints, "k-")

        self._dynamic_ax = dynamic_canvas.figure.subplots()

        tpoints = np.arange(0, 100, self.h)
        xpoints, ypoints = [], []
        r = np.array([self._startX, self._startY], float)
        for t in tpoints:
            xpoints.append(r[0])
            ypoints.append(r[1])
            r += rk4(r, t, self.h, self.alpha, self.beta, self.gamma, self.delta, self.c, self.d)

        self.xpoints = xpoints
        self.ypoints = ypoints


        self._line_x, = self._dynamic_ax.plot(tpoints, xpoints)
        self._line_y, = self._dynamic_ax.plot(tpoints, ypoints)
        self._timer = dynamic_canvas.new_timer(50)

        self._time = 1
        self._timer.add_callback(self._update_canvas)
        self._timer.start()


    def _update_canvas(self):
        tpoints = np.arange(0, 100, self.h)

        for i in range(50):
            self.ypoints.pop(0)
            self.xpoints.pop(0)
            r = np.asarray([self.xpoints[len(self.xpoints) - 1], self.ypoints[len(self.ypoints) - 1]])
            r += rk4(r, 100 + (self._time + i) * self.h, self.h, self.alpha, self.beta, self.gamma, self.delta, self.c, self.d)
            self.xpoints.append(r[0])
            self.ypoints.append(r[1])

        self._time += 5

        self._line_x.set_data(np.arange(0, 100, self.h), self.xpoints)
        self._line_y.set_data(np.arange(0, 100, self.h), self.ypoints)

        self._line_x.figure.canvas.draw()
        self._line_y.figure.canvas.draw()




    def restart(self):

        self._timer.stop()

        tpoints = np.arange(0, 100, self.h)
        self.xpoints, self.ypoints = [], []
        r = np.array([self._startX, self._startY], float)

        for t in tpoints:
            self.xpoints.append(r[0])
            self.ypoints.append(r[1])
            r += rk4(r, t, self.h, self.alpha, self.beta, self.gamma, self.delta, self.c, self.d)

        self._line.set_data(self.xpoints, self.ypoints)
        self._line.figure.canvas.draw()

        self._timer.start()

    def set_balance_starts(self):
        balance_x = (self.gamma - self.d)/self.delta
        self.balanceX.setText("Balanced X: " + str(round(balance_x, 2)))
        balance_y = (self.alpha + self.c)/self.beta
        self.balanceY.setText("Balanced Y: " + str(round(balance_y, 2)))


    def changeValueX(self, value):
        self._startX = value
        self.restart()

    def changeValueY(self, value):
        self._startY = value
        self.restart()

    def changeValueAlpha(self, value):
        self.alpha = float(value)/100
        self.restart()
        self.set_balance_starts()

    def changeValueBeta(self, value):
        self.beta = float(value)/100
        self.restart()
        self.set_balance_starts()

    def changeValueGamma(self, value):
        self.gamma= float(value)/100
        self.restart()
        self.set_balance_starts()

    def changeValueDelta(self, value):
        self.delta = float(value)/100
        self.restart()
        self.set_balance_starts()

    def changeValueC(self, value):
        self.c = float(value - 100) / 100
        print(self.c)
        self.restart()
        self.set_balance_starts()

    def changeValueD(self, value):
        self.d = float(value - 100) / 100
        self.restart()
        self.set_balance_starts()


qapp = QtWidgets.QApplication.instance()
if not qapp:
    qapp = QtWidgets.QApplication(sys.argv)

app = ApplicationWindow()
app.show()
app.activateWindow()
app.raise_()
qapp.exec_()
