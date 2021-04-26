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

        #-t---------------------------------------------------------
        layoutTextBox = QtWidgets.QVBoxLayout(self._main)
        lbl = QtWidgets.QLabel(self)
        lbl.setText("Time")
        self.textBoxTime = QtWidgets.QLineEdit()
        self.textBoxTime.setText(str(self.t))
        layoutTextBox.addWidget(lbl)
        layoutTextBox.addWidget(self.textBoxTime)
        layout.addLayout(layoutTextBox)


        self.btn = QtWidgets.QPushButton("Enter")
        self.btn.clicked.connect(self.doGraphs)
        layout.addWidget(self.btn)

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

    def doGraphs(self):
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