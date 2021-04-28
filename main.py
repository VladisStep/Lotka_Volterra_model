import math
import sys
import time

import numpy as np
from PyQt5.QtWidgets import QInputDialog, QMessageBox

import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


def reverse_euler(r, t, h, alpha, beta, gamma, delta):
    x, y = r[0], r[1]
    a = beta * h + gamma * beta * h * h
    b = 1 - alpha * h - beta * h * y + gamma * h - gamma * h * h * alpha - h * delta * x
    c = -y + alpha * h * y
    d = b * b - 4 * a * c
    if d < 0:
        print("Отрицательный дискриминант")
    y1 = (-b + math.sqrt(d))/(2 * a)
    y2 = (-b - math.sqrt(d))/(2 * a)

    y_res = 0

    if y1 >= 0:
        y_res = y1
        # print(y1, y2, a, b, c, d, x, y, r)
    elif y2 >= 0:
        y_res = y2
        # print(y1, y2, a, b, c, d, x, y, r)
    else:
        print("Отрицательные решения обратным методом Эйлера", y1, y2, a, b, c, d, x, y, r)

    x_res = x/(1 - alpha * h + beta * h * y_res)

    return np.array([x_res, y_res])


def reverse_rk2(r, t, h, alpha, beta, gamma, delta):
    x, y = r[0], r[1]
    a = -beta * (h/2)**2 - gamma * beta * (h/2)**3 + delta * beta * x * (h/2)**3 - delta * (h/2)**3 * beta * x
    b = -1 + alpha * (h/2) - beta * (h/2) * y - beta * (h/2)**2 * gamma * y - gamma * (h/2) + \
        gamma * alpha * (h/2)**2 - gamma * (h/2)**2 * beta * y + beta * delta * (h/2)**2 * x * y + \
        delta * x * (h/2) - alpha * (h/2)**2 * delta * x + beta * (h/2)**2 * x * y * delta - \
        delta * (h/2)**2 * y * beta * x + delta * (h/2)**2 * alpha * x - delta * (h/2)**2 * beta * x * y
    c = - gamma * y + alpha * (h/2) * gamma * y - beta * (h/2) * y**2 * gamma + delta * x * y - \
        alpha * (h/2) * delta * x * y + beta * (h/2) * delta * x * y**2 + delta * (h/2) * y * alpha * x - \
        delta * (h/2) * y**2 * beta * x

    ky1, ky2 = quadratic_resolve(a, b, c)
    ky_res = 0
    y_res = 0

    y1 = y + h * ky1
    y2 = y + h * ky2

    if y1 >= 0:
        ky_res = ky1
        y_res = y1
    elif y2 >= 0:
        ky_res = ky2
        y_res = y2
    else:
        print("Оба у отрицательные в обратном рк2")

    kx_res = (alpha * x - beta * x * y - beta * x * (h/2) * ky_res)/(1 - alpha * (h/2) + beta * (h/2) * y + beta * (h/2)**2 * ky_res)
    x_res = x + h * kx_res

    return np.array([x_res, y_res])


def quadratic_resolve(a, b, c):
    d = b * b - 4 * a * c
    if d < 0:
        print("Отрицательный дискриминант")
    x1 = (-b + math.sqrt(d)) / (2 * a)
    x2 = (-b - math.sqrt(d)) / (2 * a)

    return x1, x2


def f_for_Newton(y_for_Newton,a, b, c):

    return a * (y_for_Newton ** 2) + b*y_for_Newton + c

def f_shtrih_for_Newton(y_for_Newton,a, b, c):

    return a * y_for_Newton + b

def Newton_method(r, a, b, c):
    y_0 = r[1]

    for _ in range(100):
        y_0 = y_0 - f_for_Newton(y_0, a, b, c)/f_shtrih_for_Newton(y_0, a, b, c)

    print('Newton: y_0 =', y_0, 'res: f(y_0) =', a*y_0**2 + b*y_0 + c)

    return y_0


def f(r, t, w_alpha, w_beta, w_gamma, w_delta):
    alpha = w_alpha
    beta = w_beta
    gamma = w_gamma
    delta = w_delta
    x, y = r[0], r[1]
    fxd = x * (alpha - beta * y)
    fyd = -y * (gamma - delta * x)
    return np.array([fxd, fyd], float)


def rk4(r, t, h, w_alpha, w_beta, w_gamma, w_delta):
    k1 = h * f(r, t, w_alpha, w_beta, w_gamma, w_delta)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, w_alpha, w_beta, w_gamma, w_delta)
    k4 = h * f(r + k3, t + h, w_alpha, w_beta, w_gamma, w_delta)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk2(r, t, h, w_alpha, w_beta, w_gamma, w_delta):
    k1 = h * f(r, t,  w_alpha, w_beta, w_gamma, w_delta)
    k2 = h * f(r + k1, t + h,  w_alpha, w_beta, w_gamma, w_delta)
    return r + 0.5 * (k1 + k2)


def euler(r, t, h, w_alpha, w_beta, w_gamma, w_delta):
    return f(r, t, w_alpha, w_beta, w_gamma, w_delta) * h



def doDots(r, t_max, h, alpha, beta, gamma, delta, method, name):
    t_points = np.arange(0, t_max, h)
    x_points, y_points = [], []

    for t in t_points:
        x_points.append(r[0])
        y_points.append(r[1])
        r = method(r, t, h, alpha, beta, gamma, delta)

    return x_points, y_points, t_points


def draw_plot(r, t_max, h, alpha, beta, gamma, delta, method, name):

    x_points, y_points, t_points = doDots(r, t_max, h, alpha, beta, gamma, delta, method, name)

    plt.grid()
    plt.plot(t_points, x_points, '-', t_points, y_points, '-')
    plt.savefig('./two_graphs/' + name + '.png')
    plt.clf()

    plt.grid()
    plt.plot(x_points, y_points, '-')
    plt.savefig('./circles/' + name + '.png')
    plt.clf()
    plt.show()


class ApplicationWindow(QtWidgets.QMainWindow):
    startX = 10
    startY = 5
    xpoints = []
    ypoints = []

    alpha = 0.3
    beta = 0.28
    gamma = 0.7
    delta = 0.3

    methods = [rk4, rk2, euler]

    t = 100.0

    h = 0.01


    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # layout.addWidget(static_canvas)
        # # self.addToolBar(NavigationToolbar(static_canvas, self))
        #
        #
        #
        #
        # dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # layout.addWidget(dynamic_canvas)
        # # self.addToolBar(QtCore.Qt.BottomToolBarArea,
        # #                 NavigationToolbar(dynamic_canvas, self))

        layoutLabels= QtWidgets.QHBoxLayout(self._main)

        self.balanceX = QtWidgets.QLabel(self._main)
        layoutLabels.addWidget(self.balanceX)
        self.balanceY = QtWidgets.QLabel(self._main)
        layoutLabels.addWidget(self.balanceY)

        self.set_balance()

        layout.addLayout(layoutLabels)


        layoutPreysAndPredators = QtWidgets.QHBoxLayout(self._main)

        #-Preys----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Preys")
        self.textBoxPreys = QtWidgets.QLineEdit()
        self.textBoxPreys.setText(str(self.startX))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxPreys)
        layoutPreysAndPredators.addLayout(layoutTextBox)

        #-Predators------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Predators")
        self.textBoxPredators = QtWidgets.QLineEdit()
        self.textBoxPredators.setText(str(self.startY))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxPredators)
        layoutPreysAndPredators.addLayout(layoutTextBox)

        layout.addLayout(layoutPreysAndPredators)

        layoutCoef = QtWidgets.QHBoxLayout(self._main)

        #-alpha----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Alpha")
        self.textBoxAlpha = QtWidgets.QLineEdit()
        self.textBoxAlpha.setText(str(self.alpha))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxAlpha)
        layoutCoef.addLayout(layoutTextBox)

        #-beta-----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Beta")
        self.textBoxBeta = QtWidgets.QLineEdit()
        self.textBoxBeta.setText(str(self.beta))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxBeta)
        layoutCoef.addLayout(layoutTextBox)

        #-gamma-----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Gamma")
        self.textBoxGamma = QtWidgets.QLineEdit()
        self.textBoxGamma.setText(str(self.gamma))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxGamma)
        layoutCoef.addLayout(layoutTextBox)

        #-delta-----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Delta")
        self.textBoxDelta = QtWidgets.QLineEdit()
        self.textBoxDelta.setText(str(self.delta))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxDelta)
        layoutCoef.addLayout(layoutTextBox)

        layout.addLayout(layoutCoef)

        layoutTimeAndH = QtWidgets.QHBoxLayout(self._main)

        #-t---------------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Time")
        self.textBoxTime = QtWidgets.QLineEdit()
        self.textBoxTime.setText(str(self.t))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxTime)
        layoutTimeAndH.addLayout(layoutTextBox)

        #-h---------------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("h")
        self.textBoxH = QtWidgets.QLineEdit()
        self.textBoxH.setText(str(self.h))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxH)
        layoutTimeAndH.addLayout(layoutTextBox)

        layout.addLayout(layoutTimeAndH)


        layoutButtons = QtWidgets.QHBoxLayout(self._main)
        self.btn = QtWidgets.QPushButton("Print graphs")
        self.btn.clicked.connect(self.doGraphs)
        layoutButtons.addWidget(self.btn)

        layout.addLayout(layoutButtons)

        # self._static_ax = static_canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self._static_ax.plot(t, np.tan(t), ".")
        #
        # self._dynamic_ax = dynamic_canvas.figure.subplots()
        # t = np.linspace(0, 10, 101)
        # # Set up a Line2D.
        # self._line, = self._dynamic_ax.plot(t, np.sin(t + time.time()))
        # self._timer = dynamic_canvas.new_timer(50)
        # self._timer.add_callback(self._update_canvas)
        # self._timer.start()

    def _update_canvas(self):
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._line.set_data(t, np.sin(t + time.time()))
        self._line.figure.canvas.draw()

    def changeParams(self):
        try:
            self.startX = float(self.textBoxPreys.text())
        except Exception:
            QMessageBox.about(self, 'Error', 'Количиство жертв должно быть числом')
            pass

        try:
            self.startY = float(self.textBoxPredators.text())
        except Exception:
            QMessageBox.about(self, 'Error', 'Количиство хищников должно быть числом')
            pass

        try:
            alphaTest = float(self.textBoxAlpha.text())
            if alphaTest < 0.0 or alphaTest > 1.0:
                QMessageBox.about(self, 'Error', 'Альфа должнa быть в диапазоне от 0 до 1')
            else:
                self.alpha = alphaTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Альфа должнa быть числом')
            pass
        try:
            betaTest = float(self.textBoxBeta.text())
            if betaTest < 0.0 or betaTest > 1.0:
                QMessageBox.about(self, 'Error', 'Бета должнa быть в диапазоне от 0 до 1')
            else:
                self.beta = betaTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Бета должнa быть числом')
            pass
        try:
            gammaTest = float(self.textBoxGamma.text())
            if gammaTest < 0.0 or gammaTest > 1.0:
                QMessageBox.about(self, 'Error', 'Гамма должнa быть в диапазоне от 0 до 1')
            else:
                self.gamma = gammaTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Гамма должнa быть числом')
            pass
        try:
            deltaTest = float(self.textBoxDelta.text())
            if deltaTest < 0.0 or deltaTest > 1.0:
                QMessageBox.about(self, 'Error', 'Дельта должнa быть в диапазоне от 0 до 1')
            else:
                self.delta = deltaTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Дельта должнa быть числом')
            pass

        try:
            timeTest = float(self.textBoxTime.text())
            if timeTest <= 0.0:
                QMessageBox.about(self, 'Error', 'Время должно быть больше 0')
            else:
                self.t = timeTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Время должно быть числом')
            pass
        try:
            hTest = float(self.textBoxH.text())
            if hTest <= 0.0:
                QMessageBox.about(self, 'Error', 'Шаг должен быть больше 0')
            else:
                self.h = hTest
        except Exception:
            QMessageBox.about(self, 'Error', 'Шаг должен  быть числом')
            pass


    def printGraphs(self):
        print('print')

        # self.changeParams()
        # r = np.array([self.startX, self.startY])
        # doDots(r, self.t, self.h, self.alpha, self.beta, self.gamma, self.delta, euler, "euler")

    def set_balance(self):
        balance_x = self.gamma/self.delta
        balance_y = self.alpha/self.beta

        self.balanceX.setText("X = " + str(balance_x))
        self.balanceY.setText("Y = " + str(balance_y))


    def doGraphs(self):
        self.changeParams()
        self.set_balance()


        r = np.array([self.startX, self.startY])
        draw_plot(r, self.t, self.h, self.alpha, self.beta, self.gamma, self.delta, rk4, "rk4")
        r = np.array([self.startX, self.startY])
        draw_plot(r, self.t, self.h, self.alpha, self.beta, self.gamma, self.delta, rk2, "rk2")
        r = np.array([self.startX, self.startY])
        draw_plot(r, self.t, self.h, self.alpha, self.beta, self.gamma, self.delta, euler, "euler")
        r = np.array([self.startX, self.startY])
        draw_plot(r, self.t, self.h, self.alpha, self.beta, self.gamma, self.delta, reverse_euler, "reverse_euler")

        # print(self.startX, self.startY, self.alpha, self.beta, self.gamma, self.delta, self.t)



if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec_()