# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_UI(object):
    def setupUi(self, UI):
        if not UI.objectName():
            UI.setObjectName(u"UI")
        UI.resize(1000, 651)
        self.horizontalLayout = QHBoxLayout(UI)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(UI)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.btn_picsdir = QPushButton(UI)
        self.btn_picsdir.setObjectName(u"btn_picsdir")

        self.gridLayout.addWidget(self.btn_picsdir, 0, 0, 1, 1)

        self.lineEdit_picdir = QLineEdit(UI)
        self.lineEdit_picdir.setObjectName(u"lineEdit_picdir")

        self.gridLayout.addWidget(self.lineEdit_picdir, 0, 1, 1, 1)

        self.comb_model = QComboBox(UI)
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.setObjectName(u"comb_model")

        self.gridLayout.addWidget(self.comb_model, 2, 1, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.btn_train = QPushButton(UI)
        self.btn_train.setObjectName(u"btn_train")

        self.horizontalLayout_3.addWidget(self.btn_train)


        self.gridLayout.addLayout(self.horizontalLayout_3, 3, 0, 1, 6)

        self.label_4 = QLabel(UI)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)

        self.doubleSpinBox_trainsize = QDoubleSpinBox(UI)
        self.doubleSpinBox_trainsize.setObjectName(u"doubleSpinBox_trainsize")
        self.doubleSpinBox_trainsize.setMinimum(0.100000000000000)
        self.doubleSpinBox_trainsize.setMaximum(1.000000000000000)
        self.doubleSpinBox_trainsize.setSingleStep(0.010000000000000)
        self.doubleSpinBox_trainsize.setValue(0.850000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox_trainsize, 1, 1, 1, 1)

        self.label_5 = QLabel(UI)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 1, 2, 1, 1)

        self.checkBox_multGpu = QCheckBox(UI)
        self.checkBox_multGpu.setObjectName(u"checkBox_multGpu")

        self.gridLayout.addWidget(self.checkBox_multGpu, 1, 4, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_6 = QLabel(UI)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.classes = QSpinBox(UI)
        self.classes.setObjectName(u"classes")
        self.classes.setMinimum(2)
        self.classes.setMaximum(100)

        self.horizontalLayout_4.addWidget(self.classes)


        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 5, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(UI)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.train_benchsize = QSpinBox(UI)
        self.train_benchsize.setObjectName(u"train_benchsize")
        self.train_benchsize.setMinimum(1)
        self.train_benchsize.setMaximum(128)

        self.horizontalLayout_5.addWidget(self.train_benchsize)

        self.label_2 = QLabel(UI)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_5.addWidget(self.label_2)

        self.val_benchsize = QSpinBox(UI)
        self.val_benchsize.setObjectName(u"val_benchsize")
        self.val_benchsize.setMinimum(1)
        self.val_benchsize.setMaximum(256)

        self.horizontalLayout_5.addWidget(self.val_benchsize)


        self.gridLayout.addLayout(self.horizontalLayout_5, 2, 2, 1, 2)

        self.label_7 = QLabel(UI)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 2, 4, 1, 1)

        self.epochs = QSpinBox(UI)
        self.epochs.setObjectName(u"epochs")
        self.epochs.setMinimum(1)
        self.epochs.setMaximum(500)
        self.epochs.setValue(10)

        self.gridLayout.addWidget(self.epochs, 2, 5, 1, 1)

        self.img_size = QSpinBox(UI)
        self.img_size.setObjectName(u"img_size")
        self.img_size.setMinimum(32)
        self.img_size.setMaximum(2048)
        self.img_size.setSingleStep(8)
        self.img_size.setValue(224)

        self.gridLayout.addWidget(self.img_size, 0, 5, 1, 1)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_8 = QLabel(UI)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_7.addWidget(self.label_8)


        self.gridLayout.addLayout(self.horizontalLayout_7, 0, 4, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.btn_traintext = QPushButton(UI)
        self.btn_traintext.setObjectName(u"btn_traintext")

        self.horizontalLayout_6.addWidget(self.btn_traintext)

        self.lineEdit_traintext = QLineEdit(UI)
        self.lineEdit_traintext.setObjectName(u"lineEdit_traintext")

        self.horizontalLayout_6.addWidget(self.lineEdit_traintext)

        self.btn_valtext = QPushButton(UI)
        self.btn_valtext.setObjectName(u"btn_valtext")

        self.horizontalLayout_6.addWidget(self.btn_valtext)

        self.lineEdit_valtext = QLineEdit(UI)
        self.lineEdit_valtext.setObjectName(u"lineEdit_valtext")

        self.horizontalLayout_6.addWidget(self.lineEdit_valtext)


        self.gridLayout.addLayout(self.horizontalLayout_6, 0, 2, 1, 2)

        self.lr = QDoubleSpinBox(UI)
        self.lr.setObjectName(u"lr")
        self.lr.setDecimals(6)
        self.lr.setMinimum(0.000010000000000)
        self.lr.setMaximum(0.010000000000000)
        self.lr.setSingleStep(0.000010000000000)
        self.lr.setValue(0.001000000000000)

        self.gridLayout.addWidget(self.lr, 1, 3, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.textBrowser = QTextBrowser(UI)
        self.textBrowser.setObjectName(u"textBrowser")

        self.horizontalLayout_2.addWidget(self.textBrowser)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.horizontalLayout.addLayout(self.verticalLayout)


        self.retranslateUi(UI)

        QMetaObject.connectSlotsByName(UI)
    # setupUi

    def retranslateUi(self, UI):
        UI.setWindowTitle(QCoreApplication.translate("UI", u"Form", None))
        self.label.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u6a21\u578b", None))
        self.btn_picsdir.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u5206\u7c7b\u56fe\u7247", None))
        self.lineEdit_picdir.setText(QCoreApplication.translate("UI", u"./pics", None))
        self.comb_model.setItemText(0, QCoreApplication.translate("UI", u"restnet50", None))
        self.comb_model.setItemText(1, QCoreApplication.translate("UI", u"restnet101", None))
        self.comb_model.setItemText(2, QCoreApplication.translate("UI", u"restnet152", None))
        self.comb_model.setItemText(3, QCoreApplication.translate("UI", u"effiction4", None))

        self.btn_train.setText(QCoreApplication.translate("UI", u"\u5f00\u59cb\u8bad\u7ec3", None))
        self.label_4.setText(QCoreApplication.translate("UI", u"\u8bad\u7ec3\u56fe\u7247\u5360\u6bd4", None))
        self.label_5.setText(QCoreApplication.translate("UI", u"\u5b66\u4e60\u7387", None))
        self.checkBox_multGpu.setText(QCoreApplication.translate("UI", u"\u591aGPU", None))
        self.label_6.setText(QCoreApplication.translate("UI", u"\u7c7b\u522b\u6570", None))
        self.label_3.setText(QCoreApplication.translate("UI", u"\u8bad\u7ec3\u6279\u6b21\u5927\u5c0f", None))
        self.label_2.setText(QCoreApplication.translate("UI", u"\u9a8c\u8bc1\u6279\u6b21\u5927\u5c0f", None))
        self.label_7.setText(QCoreApplication.translate("UI", u"\u8bad\u7ec3\u6b21\u6570", None))
        self.label_8.setText(QCoreApplication.translate("UI", u"\u56fe\u7247\u5927\u5c0f", None))
        self.btn_traintext.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u8bad\u7ec3\u6587\u4ef6", None))
        self.lineEdit_traintext.setText(QCoreApplication.translate("UI", u"train.text", None))
        self.btn_valtext.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u9a8c\u8bc1\u6587\u4ef6", None))
        self.lineEdit_valtext.setText(QCoreApplication.translate("UI", u"val.text", None))
    # retranslateUi

