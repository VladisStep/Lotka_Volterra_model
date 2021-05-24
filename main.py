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

def f(r, params):
    alpha, beta, gamma, delta = params
    x, y = r[0], r[1]
    fxd = x * (alpha - beta * y)
    fyd = -y * (gamma - delta * x)
    return np.array([fxd, fyd], float)




def reverse_euler(r, h, t_points, params, function):

    for t in range(1, len(t_points)):
        r_b = r + function(r, params) * h
        r = r + function(r_b, params) * h

    return r




def reverse_rk2(r, h, t_points, params, function):

    for t in range(1, len(t_points)):

        k1 = (h / 2) * function(r, params)
        k2 = h * function(r + k1, params)
        r_b = r + k2

        k1 = (h / 2) * function(r_b, params)
        k2 = h * function(r_b + k1, params)
        r = r + k2

    return r




def rk4(r, h, t_points, params, function):

    for t in range(1, len(t_points)):
        k1 = h * function(r, params)
        k2 = h * function(r + 0.5 * k1, params)
        k3 = h * function(r + 0.5 * k2, params)
        k4 = h * function(r + k3, params)

        r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return r



def rk2(r, h, t_points, params, function):

    for t in range(1, len(t_points)):
        k1 = (h / 2) * function(r, params)
        k2 = h * function(r + k1, params)
        r = r + k2
    return r




def trapezoid(r, h, t_points, params, function):

    for t in range(1, len(t_points)):
        r_b = r + function(r, params) * h
        r = r + (function(r, params) + function(r_b, params)) * h * 1/2

    return r




def ralston(r, h, t_points, params, function):
    a2 = 0.40

    b31 = (-2889.0 + 1428.0 * 5 ** (1 / 2)) / 1024.0
    b32 = (3785.0 - 1620.0 * 5 ** (1 / 2)) / 1024.0
    b41 = (-3365.0 + 2094.0 * 5 ** (1 / 2)) / 6040.0
    b42 = (-975.0 - 3046.0 * 5 ** (1 / 2)) / 2552.0
    b43 = (467040.0 + 203968.0 * 5 ** (1 / 2)) / 240845.0

    g1 = (263.0 + 24.0 * 5 ** (1 / 2)) / 1812.0
    g2 = (125.0 - 1000.0 * 5 ** (1 / 2)) / 3828.0
    g3 = 1024.0 * (3346.0 + 1623.0 * 5 ** (1 / 2)) / 5924787.0
    g4 = (30.0 - 4.0 * 5 ** (1 / 2)) / 123.0

    h2 = a2 * h

    for t in range(1, len(t_points)):
        k1 = function(r, params)
        k2 = function(r + h2 * k1, params)
        k3 = function(r + h * (b31 * k1 + b32 * k2), params)
        k4 = function(r + h * (b41 * k1 + b42 * k2 + b43 * k3), params)

        r = r + h * (g1 * k1 + g2 * k2 + g3 * k3 + g4 * k4)

    return r




def euler(r, h, t_points, params, function):

    for t in range(1, len(t_points)):

        r = r + function(r, params) * h

    return r




def Adams_Moulton(r, h, t_points, params, function):

    y_0 = r

    y_1 = rk4(y_0, h, [0, 1, 2, 3], params, function)
    if (len(t_points) == 1): return y_1
    y_2 = rk4(y_1, h, [0, 1, 2, 3], params, function)
    if (len(t_points) == 2): return y_2
    y_3 = rk4(y_2, h, [0, 1, 2, 3], params, function)
    if (len(t_points) == 3): return y_3

    f_m2 = function(y_0, params)
    f_m1 = function(y_1, params)
    f_0 = function(y_2, params)
    f_1 = function(y_3, params)
    y_4 = y_3

    for i in range(3, len(t_points) - 1):

        f_m3, f_m2, f_m1, f_0 = f_m2, f_m1, f_0, f_1
        y_4 = y_3 + (h / 24) * (55 * f_0 - 59 * f_m1 + 37 * f_m2 - 9 * f_m3)
        f_1 = function(y_4, params)

        y_4 = y_3 + (h / 24) * (9 * f_1 + 19 * f_0 - 5 * f_m1 + f_m2)
        f_1 = function(y_4, params)
        y_3 = y_4

    return y_4




def Adams_Bashforth(r, h, t_points, params, function):

    y_0 = r
    y_1 = rk4(y_0, h, [0, 1, 2], params, function)
    if (len(t_points) == 1): return y_1
    y_2 = rk4(y_1, h, [0, 1, 2], params, function)
    if (len(t_points) == 2): return y_2

    K1 = function(y_1, params)
    K2 = function(y_0, params)
    y_3 = function(y_0, params)

    for i in range(2, len(t_points)):
        K3 = K2
        K2 = K1
        K1 = function(y_2, params)
        y_3 = y_2 + h * (23 * K1 - 16 * K2 + 5 * K3)/12
        y_2 = y_3

    return y_3




def doDots(r, t_max, h, params, method):
    t_points = np.arange(0, t_max, h)
    x_points, y_points  = [], []

    points = []

    points.append(r)

    for i in range(1, len(t_points)):
        points.append(method(r, h, t_points[0:i], params, f))

        if i % 1000 == 0:
            print(i)

    for p in points:
        x_points.append(p[0])
        y_points.append(p[1])


    return np.array([x_points, y_points])




def get_balance(alpha, beta, gamma, delta):
    balance_x = gamma / delta
    balance_y = alpha / beta
    return balance_x, balance_y




def invariant(points, alpha, beta, gamma, delta):
    x_points = points[0]
    y_points = points[1]
    V = delta * x_points - gamma * np.log(x_points) + beta * y_points - alpha * np.log(y_points)
    return V




def analytical(r, t_points, alpha, beta, gamma, delta):
    x = r[0]
    y = r[1]
    x_balanced, y_balanced = get_balance(alpha, beta, gamma, delta)
    w = math.sqrt(alpha * gamma)
    Ax = x - x_balanced
    Bx = (alpha * x - beta * x * y) / w
    Ay = y - y_balanced
    By = (-gamma * y + delta * x * y) / w
    return np.array(Ax * np.cos(w * t_points) + Bx * np.sin(w * t_points) + x_balanced), \
           np.array(Ay * np.cos(w * t_points) + By * np.sin(w * t_points) + y_balanced)




def draw_plot(t_points, x_points, y_points, name):

    plt.grid()
    graph_title(name)
    plt.plot(t_points, x_points, '-', t_points, y_points, '-')
    plt.legend(['preys', 'predators'])
    plt.savefig('./two_graphs/' + name + '.png')
    plt.clf()

    plt.grid()
    fig, ax = plt.subplots()
    graph_title(name)
    plt.plot(x_points, y_points, '-')
    plt.savefig('./circles/' + name + '.png')
    plt.clf()

    return np.array([x_points, y_points]), t_points




def draw_difference(res1, res2, t_points, name):
    dif = np.fabs(res1 - res2)
    graph_title(name)
    plt.plot(t_points, dif[0], '-', t_points, dif[1], '-')
    plt.legend(['preys', 'predators'])
    plt.savefig('./errors/' + name + '.png')
    plt.clf()
    return dif




def draw_invariant(t_points, v_points, method_name):
    plt.grid()
    graph_title(method_name)
    plt.plot(t_points, v_points, '-')
    plt.savefig('./invariant/' + method_name + '.png')
    plt.clf()




def draw_analytical(t_points, x_points, y_points):
    plt.grid()
    graph_title('analytical')
    plt.plot(t_points, x_points, '-', t_points, y_points, '-')
    plt.legend(['preys', 'predators'])
    plt.savefig('./two_graphs/analytical.png')
    plt.clf()

    plt.grid()
    graph_title('analytical')
    plt.plot(x_points, y_points, '-')
    plt.savefig('./circles/analytical.png')
    plt.clf()




def graph_title(name):
    fig, ax = plt.subplots()
    ax.set_title(name +
                 ' alpha = ' + str(ApplicationWindow.alpha) +
                 ' beta = ' + str(ApplicationWindow.beta) +
                 ' gamma = ' + str(ApplicationWindow.gamma) +
                 ' delta = ' + str(ApplicationWindow.delta) + '\n')




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

        layoutLabels = QtWidgets.QHBoxLayout(self._main)

        self.balanceX = QtWidgets.QLabel(self._main)
        layoutLabels.addWidget(self.balanceX)
        self.balanceY = QtWidgets.QLabel(self._main)
        layoutLabels.addWidget(self.balanceY)

        self.set_balance()

        layout.addLayout(layoutLabels)

        layoutPreysAndPredators = QtWidgets.QHBoxLayout(self._main)

        # -Preys----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Preys")
        self.textBoxPreys = QtWidgets.QLineEdit()
        self.textBoxPreys.setText(str(self.startX))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxPreys)
        layoutPreysAndPredators.addLayout(layoutTextBox)

        # -Predators------------------------------------------------
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

        # -alpha----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Alpha")
        self.textBoxAlpha = QtWidgets.QLineEdit()
        self.textBoxAlpha.setText(str(self.alpha))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxAlpha)
        layoutCoef.addLayout(layoutTextBox)

        # -beta-----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Beta")
        self.textBoxBeta = QtWidgets.QLineEdit()
        self.textBoxBeta.setText(str(self.beta))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxBeta)
        layoutCoef.addLayout(layoutTextBox)

        # -gamma-----------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Gamma")
        self.textBoxGamma = QtWidgets.QLineEdit()
        self.textBoxGamma.setText(str(self.gamma))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxGamma)
        layoutCoef.addLayout(layoutTextBox)

        # -delta-----------------------------------------------------
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

        # -t---------------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Time")
        self.textBoxTime = QtWidgets.QLineEdit()
        self.textBoxTime.setText(str(self.t))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxTime)
        layoutTimeAndH.addLayout(layoutTextBox)

        # -h---------------------------------------------------------
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


    def set_balance(self):
        balance_x, balance_y = get_balance(self.alpha, self.beta, self.gamma, self.delta)
        self.balanceX.setText("X = " + str(balance_x))
        self.balanceY.setText("Y = " + str(balance_y))

    def doGraphs(self):
        self.changeParams()
        self.set_balance()

        methods = [
            [euler, "euler"],
            [reverse_euler, "reverse_euler"],
            [rk2, "rk2"],
            [reverse_rk2, "reverse_rk2"],
            [rk4, "rk4"],
            [ralston, "ralston"],
            [trapezoid, "trapezoid"],
            [Adams_Moulton, "Adams_Moulton"],
            [Adams_Bashforth, "Adams_Bashforth"]
        ]

        t_points = np.arange(0, self.t, self.h)

        # Analytical
        r = np.array([self.startX, self.startY])
        analytical_x_res, analytical_y_res = analytical(r, t_points, self.alpha, self.beta, self.gamma, self.delta)
        draw_plot(t_points, analytical_x_res, analytical_y_res, 'analytical')
        draw_analytical(t_points, analytical_x_res, analytical_y_res)
        analytical_res = np.array([analytical_x_res, analytical_y_res])

        params = [self.alpha, self.beta, self.gamma, self.delta]
        for m in methods:
            print('Start', m[1])
            r = np.array([self.startX, self.startY])
            res = doDots(r, self.t, self.h, params, m[0])
            draw_plot(t_points[0:len(res[1])], res[0], res[1], m[1])
            dif = draw_difference(res, analytical_res, t_points, m[1])

            invariantDots = invariant(res, self.alpha, self.beta, self.gamma, self.delta)
            draw_invariant(t_points, invariantDots, m[1])
            print('End', m[1])

        print("Correct")


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
