# classfy_imge 深度学习图片分类

## 安装pytorch

torchvision>1.11

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

## 使用

1. 选择分类图片目录
2. 选择模型
3. 勾选预训练
4. 选择训练批次 benchsize*100<训练总样本数；否则会退出训练
5. 选择验证批次
6. 选择学习率
7. 选择epochs
8. 开始训练，从头训练 要删除 当前目录下的 _best.pth 否则会从_best.pth 保存的权重开始训练
9. 继续前次训练；如果更目录下有 _best.pth,程序自动从_best.pth训练

## pre_process_img.py

按文件夹名称分类，一个文件架名称为一个类。将该文件夹下的所有文件，归到一个类别。

## walk.py

目录测试

## train.py

训练主程序。在run目录下保存tensorborad 训练记录文件.训练完成保存:best.pth 、last.pth、*.onnx三个文件。best.pth文件保存了训练权重可用于迁移学学习继续训练。onnx 为动态benchsize。模型命名规则(best为例）：modelname_imgsize_best.pth

## verify.py

批量测试模型，泛化能力。

## val.py

测试单个类别分类效果。如将目录定位到“臭蛋“，运行后会将模型推理结果不为”臭蛋“的图片保存在”erro_classs/臭蛋/错误分类名/**.jpg"目录下，同时保存erro_臭蛋.txt文件

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
