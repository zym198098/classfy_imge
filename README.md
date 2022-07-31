# classfy_imge 深度学习图片分类
## pyqt5 环境配置
安装PyQt5以及PyQt5-tools(这里面就有designer了)
```
pip install PyQt5
pip install PyQt5-tools
```
安装 代码提示
```
pip install PyQt5-stubs
```

## pre_process_img.py 
按文件夹名称分类，一个文件架名称为一个类。将该文件夹下的所有文件，归到一个类别。
## 打包程序
### 安装pyinstaller
首先安装pyinstaller，使用安装命令：pip install pyinstaller；
### 打包单个文件
```
pyinstaller -F -w -i xxx.ico main.py --noconsole
```
### 多个文件打包
```
pyinstaller [主文件] -p [其他文件1] -p [其他文件2] --hidden-import [自建模块1] --hidden-import [自建模块2]

pyinstaller -F -w main.py -p enterTest1.py -p enterTest2.py -p test1.py -p test2.py --hidden-import enterTest1 --hidden-import enterTest2 --hidden-import test1 --hidden-import test2
```
打包命令
```
pyinstaller -F -w train.py -p Ui_train.py -p utils.py  
```
