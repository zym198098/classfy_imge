# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpinBox, QTextBrowser, QVBoxLayout,
    QWidget)

class Ui_UI(object):
    def setupUi(self, UI):
        if not UI.objectName():
            UI.setObjectName(u"UI")
        UI.resize(1335, 651)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(UI.sizePolicy().hasHeightForWidth())
        UI.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(UI)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.btn_picsdir = QPushButton(UI)
        self.btn_picsdir.setObjectName(u"btn_picsdir")

        self.horizontalLayout_7.addWidget(self.btn_picsdir)

        self.lineEdit_picdir = QLineEdit(UI)
        self.lineEdit_picdir.setObjectName(u"lineEdit_picdir")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_picdir.sizePolicy().hasHeightForWidth())
        self.lineEdit_picdir.setSizePolicy(sizePolicy1)

        self.horizontalLayout_7.addWidget(self.lineEdit_picdir)

        self.label_4 = QLabel(UI)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_7.addWidget(self.label_4)

        self.doubleSpinBox_trainsize = QDoubleSpinBox(UI)
        self.doubleSpinBox_trainsize.setObjectName(u"doubleSpinBox_trainsize")
        self.doubleSpinBox_trainsize.setMinimum(0.100000000000000)
        self.doubleSpinBox_trainsize.setMaximum(1.000000000000000)
        self.doubleSpinBox_trainsize.setSingleStep(0.010000000000000)
        self.doubleSpinBox_trainsize.setValue(0.850000000000000)

        self.horizontalLayout_7.addWidget(self.doubleSpinBox_trainsize)

        self.label_8 = QLabel(UI)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_7.addWidget(self.label_8)

        self.img_size = QSpinBox(UI)
        self.img_size.setObjectName(u"img_size")
        self.img_size.setMinimum(32)
        self.img_size.setMaximum(2048)
        self.img_size.setSingleStep(8)
        self.img_size.setValue(224)

        self.horizontalLayout_7.addWidget(self.img_size)

        self.checkBox_amp = QCheckBox(UI)
        self.checkBox_amp.setObjectName(u"checkBox_amp")

        self.horizontalLayout_7.addWidget(self.checkBox_amp)

        self.checkBox_multGpu = QCheckBox(UI)
        self.checkBox_multGpu.setObjectName(u"checkBox_multGpu")

        self.horizontalLayout_7.addWidget(self.checkBox_multGpu)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label = QLabel(UI)
        self.label.setObjectName(u"label")

        self.horizontalLayout_4.addWidget(self.label)

        self.comb_model = QComboBox(UI)
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.addItem("")
        self.comb_model.setObjectName(u"comb_model")

        self.horizontalLayout_4.addWidget(self.comb_model)

        self.label_5 = QLabel(UI)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.lr = QDoubleSpinBox(UI)
        self.lr.setObjectName(u"lr")
        self.lr.setDecimals(6)
        self.lr.setMinimum(0.000010000000000)
        self.lr.setMaximum(0.010000000000000)
        self.lr.setSingleStep(0.000010000000000)
        self.lr.setValue(0.001000000000000)

        self.horizontalLayout_4.addWidget(self.lr)

        self.label_6 = QLabel(UI)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.comb_lr_f = QComboBox(UI)
        self.comb_lr_f.addItem("")
        self.comb_lr_f.addItem("")
        self.comb_lr_f.setObjectName(u"comb_lr_f")

        self.horizontalLayout_4.addWidget(self.comb_lr_f)

        self.label_3 = QLabel(UI)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)

        self.train_benchsize = QSpinBox(UI)
        self.train_benchsize.setObjectName(u"train_benchsize")
        self.train_benchsize.setMinimum(1)
        self.train_benchsize.setMaximum(128)

        self.horizontalLayout_4.addWidget(self.train_benchsize)

        self.label_2 = QLabel(UI)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.val_benchsize = QSpinBox(UI)
        self.val_benchsize.setObjectName(u"val_benchsize")
        self.val_benchsize.setMinimum(1)
        self.val_benchsize.setMaximum(256)

        self.horizontalLayout_4.addWidget(self.val_benchsize)

        self.label_7 = QLabel(UI)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_4.addWidget(self.label_7)

        self.epochs = QSpinBox(UI)
        self.epochs.setObjectName(u"epochs")
        self.epochs.setMinimum(1)
        self.epochs.setMaximum(500)
        self.epochs.setValue(10)

        self.horizontalLayout_4.addWidget(self.epochs)


        self.horizontalLayout_3.addLayout(self.horizontalLayout_4)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.btn_train = QPushButton(UI)
        self.btn_train.setObjectName(u"btn_train")

        self.verticalLayout.addWidget(self.btn_train)

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
        self.btn_picsdir.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u5206\u7c7b\u56fe\u7247", None))
        self.lineEdit_picdir.setText(QCoreApplication.translate("UI", u"./pics", None))
        self.label_4.setText(QCoreApplication.translate("UI", u"\u8bad\u7ec3\u56fe\u7247\u5360\u6bd4", None))
        self.label_8.setText(QCoreApplication.translate("UI", u"\u56fe\u7247\u5927\u5c0f", None))
        self.checkBox_amp.setText(QCoreApplication.translate("UI", u"\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3", None))
        self.checkBox_multGpu.setText(QCoreApplication.translate("UI", u"\u591aGPU", None))
        self.label.setText(QCoreApplication.translate("UI", u"\u9009\u62e9\u6a21\u578b", None))
        self.comb_model.setItemText(0, QCoreApplication.translate("UI", u"restnet50", None))
        self.comb_model.setItemText(1, QCoreApplication.translate("UI", u"restnet101", None))
        self.comb_model.setItemText(2, QCoreApplication.translate("UI", u"restnet152", None))
        self.comb_model.setItemText(3, QCoreApplication.translate("UI", u"effiction4", None))
        self.comb_model.setItemText(4, QCoreApplication.translate("UI", u"densenet121", None))
        self.comb_model.setItemText(5, QCoreApplication.translate("UI", u"densenet161", None))
        self.comb_model.setItemText(6, QCoreApplication.translate("UI", u"regnet_x_32gf", None))
        self.comb_model.setItemText(7, QCoreApplication.translate("UI", u"vision_transformer.vit_b_32", None))

        self.label_5.setText(QCoreApplication.translate("UI", u"\u5b66\u4e60\u7387", None))
        self.label_6.setText(QCoreApplication.translate("UI", u"\u5b66\u4e60\u7387\u66f4\u65b0\u65b9\u6cd5", None))
        self.comb_lr_f.setItemText(0, QCoreApplication.translate("UI", u"CosineAnnealingLR", None))
        self.comb_lr_f.setItemText(1, QCoreApplication.translate("UI", u"StepLR", None))

        self.label_3.setText(QCoreApplication.translate("UI", u"\u8bad\u7ec3\u6279\u6b21\u5927\u5c0f", None))
        self.label_2.setText(QCoreApplication.translate("UI", u"\u9a8c\u8bc1\u6279\u6b21\u5927\u5c0f", None))
        self.label_7.setText(QCoreApplication.translate("UI", u"epochs", None))
        self.btn_train.setText(QCoreApplication.translate("UI", u"\u5f00\u59cb\u8bad\u7ec3", None))
    # retranslateUi

