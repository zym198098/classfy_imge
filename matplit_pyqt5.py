
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#               2015 Jens H Nielsen
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals
import sys
import os
import random
import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
# 注意这里的宽度和高度的单位是英寸，1英寸=100像素
        fig = Figure(figsize=(width, height), dpi=dpi)#画布
        self.axes = fig.add_subplot(111)#plt
        

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        # FigureCanvas.setSizePolicy(self,
        #                            QtWidgets.QSizePolicy.Expanding,
        #                            QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyMatplotlibFigure(FigureCanvas):
  """
  创建一个画布类，并把画布放到FigureCanvasQTAgg
  """
  def __init__(self, width=10, heigh=10, dpi=100):
    # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        self.figs = Figure(figsize=(width, heigh), dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs) # 在父类种激活self.fig， 

  def mat_plot_drow(self, epoch, loss1,loss2):
    """
    用清除画布刷新的方法绘图
    :return:
    """
    # self.figs.clf() # 清理画布，这里是clf()
    self.axes.cla()   
    self.axes = self.figs.add_subplot(211) # 清理画布后必须重新添加绘图区
    self.axes.patch.set_facecolor("#01386a") # 设置ax区域背景颜色
    self.axes.patch.set_alpha(0.5) # 设置ax区域背景颜色透明度
    self.figs.patch.set_facecolor('#01386a') # 设置绘图区域颜色
    # self.axes.spines['bottom'].set_color('r') # 设置下边界颜色
    # self.axes.spines['top'].set_visible(False) # 顶边界不可见
    # self.axes.spines['right'].set_visible(False) # 右边界不可见
    # # 设置左、下边界在（0，0）处相交
    # self.axes.spines['bottom'].set_position(('data', 0)) # 设置y轴线原点数据为 0
    # self.axes.spines['left'].set_position(('data', 0)) # 设置x轴线原点数据为 0   
    self.axes.set_title("训练loss")
    self.axes.plot(epoch, loss1,marker='o', linewidth=1)
    self.axes.plot(epoch, loss2, marker='o', linewidth=0.5)
    self.axes.legend(['loss1','loss2']) 
    self.axes.set_xticks(epoch)

    self.axes1 = self.figs.add_subplot(212) # 清理画布后必须重新添加绘图区
    self.axes1.patch.set_facecolor("#01386a") # 设置ax区域背景颜色
    self.axes1.patch.set_alpha(0.5) # 设置ax区域背景颜色透明度
    # self.figs.patch.set_facecolor('#01386a') # 设置绘图区域颜色
    # self.axes.spines['bottom'].set_color('r') # 设置下边界颜色
    # self.axes.spines['top'].set_visible(False) # 顶边界不可见
    # self.axes.spines['right'].set_visible(False) # 右边界不可见
    # # 设置左、下边界在（0，0）处相交
    # self.axes.spines['bottom'].set_position(('data', 0)) # 设置y轴线原点数据为 0
    # self.axes.spines['left'].set_position(('data', 0)) # 设置x轴线原点数据为 0   
    self.axes1.set_title("训练loss-1")
    self.axes1.plot(epoch, loss1,marker='o', linewidth=1)
    self.axes1.plot(epoch, loss2, marker='o', linewidth=0.5)
    self.axes1.legend(['train_loss1','val_loss']) 
    self.axes1.set_xticks(epoch)
    self.axes1.draw()
    self.figs.canvas.flush_events() # 画布刷新self.figs.canvas
  def mat_plot_drow(self, epoch, loss1):
        self.axes.cla()   
        print(epoch)
        self.axes1 = self.figs.add_subplot(212) # 清理画布后必须重新添加绘图区
        self.axes1.patch.set_facecolor("#01386a") # 设置ax区域背景颜色
        self.axes1.patch.set_alpha(0.5) # 设置ax区域背景颜色透明度
        # self.figs.patch.set_facecolor('#01386a') # 设置绘图区域颜色
        # self.axes.spines['bottom'].set_color('r') # 设置下边界颜色
        # self.axes.spines['top'].set_visible(False) # 顶边界不可见
        # self.axes.spines['right'].set_visible(False) # 右边界不可见
        # # 设置左、下边界在（0，0）处相交
        # self.axes.spines['bottom'].set_position(('data', 0)) # 设置y轴线原点数据为 0
        # self.axes.spines['left'].set_position(('data', 0)) # 设置x轴线原点数据为 0   
        self.axes1.set_title("训练epoch")
        self.axes1.plot(epoch, loss1,marker='o', linewidth=1)
        self.axes1.legend(['loss1']) 
        self.axes1.set_xticks(epoch)
        self.axes1.draw()
        self.figs.canvas.flush_events() # 画布刷新self.figs.canvas

class Mymetrics_plCanvas(FigureCanvas):
    """Simple canvas with a sine plot."""
    def __init__(self, width=10, heigh=10, dpi=100):
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        self.figs = Figure(figsize=(width, heigh), dpi=dpi)
        super(Mymetrics_plCanvas, self).__init__(self.figs) # 在父类种激活self.fig， 
    def cnf_matrix_plotter(self,classes,df:pd.DataFrame, cmap=plt.cm.Blues):
        """
         标签名称列表，传入pandas混淆矩阵，绘制混淆矩阵图
        """
        cm = confusion_matrix(df['标注类别名称'], df['top-1-预测名称'])#获得混淆矩阵
        
        self.figs.clf() # 清理画布，这里是clf()
        
        self.axes = self.figs.add_subplot(111) # 清理画布后必须重新添加绘图区
        self.axes.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.colorbar() # 色条
        tick_marks = np.arange(len(classes))
        self.axes.set_title('混淆矩阵', fontsize=25)
        self.axes.set_xlabel('预测类别', fontsize=20, c='r')
        self.axes.set_ylabel('真实类别', fontsize=20, c='r')
        # plt.ylabel('真实类别', fontsize=25, c='r')
        # plt.tick_params(labelsize=16) # 设置类别文字大小
        self.axes.tick_params(labelsize=11) # 设置类别文字大小
        # plt.xticks(tick_marks, classes, rotation=90) # 横轴文字旋转
        self.axes.set_xticks(tick_marks)      
        self.axes.set_xticklabels(classes, rotation=90)
        self.axes.set_yticks(tick_marks)
        self.axes.set_yticklabels( classes)
        
        # 写数字
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            self.axes.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="black",
                    fontsize=11)
        self.figs.canvas.draw() # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events() # 画布刷新self.figs.canvas


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)
        main_lay= QtWidgets.QVBoxLayout(self)


        self.main_widget = QtWidgets.QWidget()
        # main_lay.addWidget(self.main_widget)
        # self.label = QtWidgets.QLabel(self.main_widget)
        # self.label.setGeometry(QtCore.QRect(0, 0, 800, 600))
       
        self.main_widget.setLayout(main_lay)
        l = QtWidgets.QHBoxLayout()
        # l_bottom= QtWidgets.QHBoxLayout( )
        
        btn1=QtWidgets.QPushButton("update")
        # l_bottom.addWidget(btn1)
        self.mx=Mymetrics_plCanvas(width=8, heigh=6, dpi=100)
        # l.addWidget(sc)
        # l.addWidget(dc)
        self.loss=MyMatplotlibFigure(width=8,heigh=6,dpi=100)
        l.addWidget(self.mx)
        l.addWidget(self.loss)
        main_lay.addLayout(l)
        main_lay.addWidget(btn1)
        classes,df1=self.matrix_load_file('metrics/idx_to_labels.npy','metrics/测试集预测结果.csv')
        self.mx.cnf_matrix_plotter( classes,df1,cmap='Blues')
        # sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        # dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        
       
        
        

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        

        self.statusBar().showMessage("All hail matplotlib!", 2000)
        btn1.clicked.connect(self.btn_update)
    

    def draw_loss(self,figure:MyMatplotlibFigure):
        t = np.arange(10)  
        t=t+1
        # t=a[1:]
        s1=np.random.rand(10)*10
        s2=np.random.rand(10)
        # t = np.arange(0.0, 5.0, 0.01)
        # s = np.cos(2 * np.pi * t)
        figure.mat_plot_drow(epoch=t,loss1=s1,loss2=s2)
    def matrix_load_file(self,filename_class:str,filename_metrics:str):
        '''
        加载混淆矩阵的类名map文件（numpy格式）
        加载模型预测的结果文件（csv格式）包含:df['标注类别名称'], df['top-1-预测名称'] 两列
        '''
        # idx_to_labels = np.load('metrics/idx_to_labels.npy', allow_pickle=True).item()
        idx_to_labels = np.load(filename_class, allow_pickle=True).item()
        # 获得类别名称
        classes = list(idx_to_labels.values())
        df = pd.read_csv(filename_metrics)
        # df = pd.read_csv('metrics/测试集预测结果.csv')
        return classes,df

    def btn_update(self):
        
        self.draw_loss(self.loss)
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )

if __name__=='__main__':
    qApp = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow()

    aw.setWindowTitle("%s" % progname)
    aw.showMaximized()
    sys.exit(qApp.exec_())
    #qApp.exec_()