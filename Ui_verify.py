# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/zym/pytorch_proj/classfy_imge/verify.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form_verify(object):
    def setupUi(self, Form_verify):
        Form_verify.setObjectName("Form_verify")
        Form_verify.resize(939, 683)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form_verify)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.comb_modeltype = QtWidgets.QComboBox(Form_verify)
        self.comb_modeltype.setObjectName("comb_modeltype")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.comb_modeltype.addItem("")
        self.gridLayout.addWidget(self.comb_modeltype, 0, 1, 1, 1)
        self.lineEdit_modelpath = QtWidgets.QLineEdit(Form_verify)
        self.lineEdit_modelpath.setObjectName("lineEdit_modelpath")
        self.gridLayout.addWidget(self.lineEdit_modelpath, 0, 3, 1, 1)
        self.lineEdit_classnames = QtWidgets.QLineEdit(Form_verify)
        self.lineEdit_classnames.setObjectName("lineEdit_classnames")
        self.gridLayout.addWidget(self.lineEdit_classnames, 0, 5, 1, 1)
        self.verify_bench_size = QtWidgets.QSpinBox(Form_verify)
        self.verify_bench_size.setMinimum(1)
        self.verify_bench_size.setObjectName("verify_bench_size")
        self.gridLayout.addWidget(self.verify_bench_size, 1, 3, 1, 1)
        self.btn_picdir = QtWidgets.QPushButton(Form_verify)
        self.btn_picdir.setObjectName("btn_picdir")
        self.gridLayout.addWidget(self.btn_picdir, 1, 0, 1, 1)
        self.btn_classnames = QtWidgets.QPushButton(Form_verify)
        self.btn_classnames.setObjectName("btn_classnames")
        self.gridLayout.addWidget(self.btn_classnames, 0, 4, 1, 1)
        self.lineEdit_picdir = QtWidgets.QLineEdit(Form_verify)
        self.lineEdit_picdir.setObjectName("lineEdit_picdir")
        self.gridLayout.addWidget(self.lineEdit_picdir, 1, 1, 1, 1)
        self.btn_modelpath = QtWidgets.QPushButton(Form_verify)
        self.btn_modelpath.setObjectName("btn_modelpath")
        self.gridLayout.addWidget(self.btn_modelpath, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(Form_verify)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Form_verify)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)
        self.btn_verify = QtWidgets.QPushButton(Form_verify)
        self.btn_verify.setObjectName("btn_verify")
        self.gridLayout.addWidget(self.btn_verify, 2, 0, 1, 6)
        self.verticalLayout.addLayout(self.gridLayout)
        self.textBrowser = QtWidgets.QTextBrowser(Form_verify)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)

        self.retranslateUi(Form_verify)
        QtCore.QMetaObject.connectSlotsByName(Form_verify)

    def retranslateUi(self, Form_verify):
        _translate = QtCore.QCoreApplication.translate
        Form_verify.setWindowTitle(_translate("Form_verify", "Form"))
        self.comb_modeltype.setItemText(0, _translate("Form_verify", "restnet50"))
        self.comb_modeltype.setItemText(1, _translate("Form_verify", "restnet101"))
        self.comb_modeltype.setItemText(2, _translate("Form_verify", "restnet152"))
        self.comb_modeltype.setItemText(3, _translate("Form_verify", "effection4"))
        self.lineEdit_modelpath.setText(_translate("Form_verify", "best.pth"))
        self.lineEdit_classnames.setText(_translate("Form_verify", "classnames.txt"))
        self.btn_picdir.setText(_translate("Form_verify", "图片路径"))
        self.btn_classnames.setText(_translate("Form_verify", "类别名称文件"))
        self.btn_modelpath.setText(_translate("Form_verify", "模型文件路径"))
        self.label.setText(_translate("Form_verify", "模型类别"))
        self.label_2.setText(_translate("Form_verify", "验证批次"))
        self.btn_verify.setText(_translate("Form_verify", "验证"))