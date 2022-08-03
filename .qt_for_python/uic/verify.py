# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'verify.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form_verify(object):
    def setupUi(self, Form_verify):
        if not Form_verify.objectName():
            Form_verify.setObjectName(u"Form_verify")
        Form_verify.resize(939, 683)
        self.verticalLayout = QVBoxLayout(Form_verify)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.comb_modeltype = QComboBox(Form_verify)
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.setObjectName(u"comb_modeltype")

        self.gridLayout.addWidget(self.comb_modeltype, 0, 1, 1, 1)

        self.lineEdit_modelpath = QLineEdit(Form_verify)
        self.lineEdit_modelpath.setObjectName(u"lineEdit_modelpath")

        self.gridLayout.addWidget(self.lineEdit_modelpath, 0, 3, 1, 1)

        self.lineEdit_classnames = QLineEdit(Form_verify)
        self.lineEdit_classnames.setObjectName(u"lineEdit_classnames")

        self.gridLayout.addWidget(self.lineEdit_classnames, 0, 5, 1, 1)

        self.verify_bench_size = QSpinBox(Form_verify)
        self.verify_bench_size.setObjectName(u"verify_bench_size")
        self.verify_bench_size.setMinimum(1)

        self.gridLayout.addWidget(self.verify_bench_size, 1, 3, 1, 1)

        self.btn_picdir = QPushButton(Form_verify)
        self.btn_picdir.setObjectName(u"btn_picdir")

        self.gridLayout.addWidget(self.btn_picdir, 1, 0, 1, 1)

        self.btn_classnames = QPushButton(Form_verify)
        self.btn_classnames.setObjectName(u"btn_classnames")

        self.gridLayout.addWidget(self.btn_classnames, 0, 4, 1, 1)

        self.lineEdit_picdir = QLineEdit(Form_verify)
        self.lineEdit_picdir.setObjectName(u"lineEdit_picdir")

        self.gridLayout.addWidget(self.lineEdit_picdir, 1, 1, 1, 1)

        self.btn_modelpath = QPushButton(Form_verify)
        self.btn_modelpath.setObjectName(u"btn_modelpath")

        self.gridLayout.addWidget(self.btn_modelpath, 0, 2, 1, 1)

        self.label = QLabel(Form_verify)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QLabel(Form_verify)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)

        self.btn_verify = QPushButton(Form_verify)
        self.btn_verify.setObjectName(u"btn_verify")

        self.gridLayout.addWidget(self.btn_verify, 2, 0, 1, 6)


        self.verticalLayout.addLayout(self.gridLayout)

        self.textBrowser = QTextBrowser(Form_verify)
        self.textBrowser.setObjectName(u"textBrowser")

        self.verticalLayout.addWidget(self.textBrowser)


        self.retranslateUi(Form_verify)

        QMetaObject.connectSlotsByName(Form_verify)
    # setupUi

    def retranslateUi(self, Form_verify):
        Form_verify.setWindowTitle(QCoreApplication.translate("Form_verify", u"Form", None))
        self.comb_modeltype.setItemText(0, QCoreApplication.translate("Form_verify", u"restnet50", None))
        self.comb_modeltype.setItemText(1, QCoreApplication.translate("Form_verify", u"restnet101", None))
        self.comb_modeltype.setItemText(2, QCoreApplication.translate("Form_verify", u"restnet152", None))
        self.comb_modeltype.setItemText(3, QCoreApplication.translate("Form_verify", u"effection4", None))

        self.lineEdit_modelpath.setText(QCoreApplication.translate("Form_verify", u"best.pth", None))
        self.lineEdit_classnames.setText(QCoreApplication.translate("Form_verify", u"classnames.txt", None))
        self.btn_picdir.setText(QCoreApplication.translate("Form_verify", u"\u56fe\u7247\u8def\u5f84", None))
        self.btn_classnames.setText(QCoreApplication.translate("Form_verify", u"\u7c7b\u522b\u540d\u79f0\u6587\u4ef6", None))
        self.btn_modelpath.setText(QCoreApplication.translate("Form_verify", u"\u6a21\u578b\u6587\u4ef6\u8def\u5f84", None))
        self.label.setText(QCoreApplication.translate("Form_verify", u"\u6a21\u578b\u7c7b\u522b", None))
        self.label_2.setText(QCoreApplication.translate("Form_verify", u"\u9a8c\u8bc1\u6279\u6b21", None))
        self.btn_verify.setText(QCoreApplication.translate("Form_verify", u"\u9a8c\u8bc1", None))
    # retranslateUi

