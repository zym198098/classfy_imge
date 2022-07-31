import sys
# from PyQt5.QtWidgets import QApplication, QWidget,QLabel,QPushButton,QCheckBox, QComboBox,QLineEdit
# from PyQt5.QtGui import QFont
# from PyQt5.QtCore import Qt,QThread,pyqtSignal
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from qtpy import PYQT5
# from yaml import emit
# from PyQt5.Qt import QApplication, QWidget, QThread


class MyThread(QThread):
    printtext=pyqtSignal(str)
    def __init__(self,train_para={}):
        super().__init__()
        self.train_para=train_para
        
        
	
	# 开启线程后默认执行
    def run(self):
        train_bench_size=self.train_para["train_bench_size"]
        for i in range(train_bench_size):
            print("执行....%d" % (i + 1))
            text="执行....%d" % (i + 1)
            self.printtext.emit(text)
            time.sleep(1)


class Exchange_of_weather_degree_units(QWidget):

    def __init__(self):
        super().__init__()
        self.setting()
        self.train_para={"train_bench_size":20}
        self.my_thread = MyThread(self.train_para)  # 创建线程
        self.my_thread.printtext.connect(self.print_text)
        self.my_thread.start()  # 开始线程


    def setting(self):
        self.unit = None

        self.choice = QComboBox(self)
        self.choice.addItem('℃')
        self.choice.addItem('℉')
        self.choice.activated[str].connect(self.choice_)
        self.choice.move(50,15)

        self.number = QLineEdit(self)
        self.number.setPlaceholderText('输入转化的数值')
        self.number.move(15,50)

        self.arrowhead = QLabel(self)
        self.arrowhead.setText('——————>')
        self.arrowhead.setFont(QFont('microsoft Yahei', 20))
        self.arrowhead.move(165,20)

        self.result = QLabel(self)
        self.result.setText('                         ')
        self.result.setFont(QFont('microsoft Yahei', 15))
        self.result.move(370, 27.5)

        self.yes = QPushButton('确定',self)
        self.yes.clicked.connect(self.yes_)
        self.yes.move(220,50)

        self.setGeometry(300, 100, 520, 100)
        self.setWindowTitle('摄氏度与华氏度的转换')
        self.show()

    def choice_(self,text):
        self.unit = text

    def yes_(self):
        try:
            if self.unit == '℃':
                result_ = eval(self.number.text()) * 1.8 + 32
                self.result.setText(str(result_) + '℉')

            if self.unit == '℉':
                result_ = round((eval(self.number.text()) - 32) / 1.8,6)
                self.result.setText(str(result_) + '℃')

            else:
                result_ = eval(self.number.text()) * 1.8 + 32
                self.result.setText(str(result_) + '℃')
        except:
            self.result.setText('请输入数字')
    def print_text(self,text):
        self.result.setText(text)

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Ex = Exchange_of_weather_degree_units()
    Ex.show()
    sys.exit(app.exec_())
