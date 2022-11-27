import matplotlib as plt
plt.rc("font",family='SimHei') # 中文字体
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import math
import cv2
plt.use('Qt5Agg')


class MyFigureCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        y = (3, 2, 5, 6, 4)
        x = (1, 2, 3, 4, 5)
        self.axes.plot(x, y, 'ro--')


class MainWindow2(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        ll = QtWidgets.QVBoxLayout(self.main_widget)
        mc = MyFigureCanvas(self.main_widget, width=5, height=4, dpi=300)
        ll.addWidget(mc)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    ui = MainWindow2()
    ui.setWindowTitle("Test")
    ui.show()
    sys.exit(qApp.exec_())

