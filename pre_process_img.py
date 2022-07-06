#将鸡蛋分类图片整理到一个文件夹
from copy import copy
import os
import random
import shutil
# from cv2 import split#打乱数据用的

# 百分之60用来当训练集
train_ratio = 0.9

# 用来当测试集
test_ratio = 1-train_ratio

# rootdata = r"data"  #数据的根目录
# /media/zym/软件/pics/jidan/标注_09151051
# rootdata = "class_image/data/test1"  #数据的根目录
rootdata = "/home/zym/下载/egg" 
newroot="/home/zym/下载/egg1/"
train_list, test_list = [],[]#读取里面每一类的类别
data_list = []

#生产train.txt和test.txt
class_flag = -1
classnames={"null":0}
for dir,folder,file in os.walk(rootdata):
    # print(dir)
    # print(folder)
    # for i in range(len(file)):
        
    #     data_list.append(os.path.join(dir,file[i]))

    for i in range(0,int(len(file)*train_ratio)):
        dir1=dir.split('/')
        len1=len(classnames)
        if dir1[-1]in classnames.keys():
            print(dir1[-1])
        else:
            classnames[dir1[-1]]=len1
            if os.path.exists(newroot+dir1[-1])==False:
                os.mkdir(newroot+dir1[-1])
        
        new_filepath=os.path.join(newroot+dir1[-1], file[i])
        old_filepath=os.path.join(dir, file[i])
        shutil.copyfile(old_filepath,new_filepath)
       
        
        

  

    print(classnames)  

