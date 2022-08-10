import os
from pydoc import classname
import random
import platform

def create_classimage_dataset(root_dir="./data",train_ratio=0.8,train_name='train_jidan.txt',
test_name='test_jidan.txt',shuffle:bool=True):

        imgType_list = {'jpg', 'bmp', 'png', 'jpeg'} #支持的图片文件格式   
        # 用来当测试集
        test_ratio = 1-train_ratio
        #生产train.txt和test.txt
        class_flag = -1
        # classnames={"null":0}
        classnames=dict()
        train_lable=[]
        test_label=[]
        train_list, test_list = [],[]#读取里面每一类的类别
        for dir,folder,file in os.walk(root_dir):
            print(dir)
            print(folder)
      
            file_pic=[]
            data_list = []
            
            for i in range(len(file)):
                 jpg=file[i].split(".")[-1]
                 if jpg in imgType_list:
                        file_pic.append(file[i])

            for i in range(0,int(len(file_pic))):
                system_name=platform.system()
                if system_name=="Linux":
                    dir1=dir.split('/')
                elif system_name=="Windows":
                    dir1=dir.split('/')
                len1=len(classnames)
                if dir1[-1]in classnames.keys():
                    # print(dir1[-1])
                    pass
                else:
                    classnames[dir1[-1]]=len1
                    print(dir)
                train_data = os.path.join(dir, file_pic[i])+'\t'+str(classnames[dir1[-1]])+'\n'
                data_list.append(train_data)
                

            

            # print(classnames)
            #打乱图片
            if shuffle:
                random.shuffle(data_list)  
            # 按比例存放在train_liat test_list
           
            len_train=int(len(data_list)*train_ratio)
            len_test=len(data_list)-len_train

            for i in range(len(data_list)):
                    if i<len_train:
                        label=data_list[i]
                        train_lable.append(int(label[-2]))
                        train_list.append(data_list[i])
                    else:
                        label=data_list[i]
                        test_label.append(int(label[-2]))
                        test_list.append(data_list[i])
            class_flag += 1

        print(classnames)
        for it in classnames:
            sum=train_lable.count(classnames[it])
            sum_t=test_label.count(classnames[it])
            print(f"train list classname:{it},sum:{sum}")
            print(f"test list classname:{it},sum:{sum_t}")

        # random.shuffle(train_list)#打乱次序
        # random.shuffle(test_list)
        # # train_name='class_image/train_jidan.txt'
        # # test_name='class_image/test_jidan.txt'
        with open(train_name,'w',encoding='UTF-8') as f:
            for train_img in train_list:
                f.write(str(train_img))

        with open(test_name,'w',encoding='UTF-8') as f:
            for test_img in test_list:
                f.write(test_img)
        a=classnames.values()
        b=classnames.keys()
        c=dict(zip(a,b)) 
        with open("classnames.txt","w",encoding='UTF-8') as f:
            for i ,value1 in c.items():
                print(i)
                print(value1) 
                classs_name=str(i)+":"+str(value1)+'\n'
                f.write(classs_name) 
        # class_names=dict()
        # with open("classnames.txt","r",encoding='UTF-8') as f:
        #     while True:
        #         line=f.readline()
        #         line=line[:-1]

        #         print(line)
        #         if line=="":
        #             break 
        #         else:
        #             class1=line.split(":")
        #             class_names[int(class1[0])]=class1[-1]

        return c
      
def read_classnames(classnames_dir:str)->dict():
        r"""
        根据classnames.txt 返回类名字典
        txt格式如下，一行一个类别
        0:有精蛋
        1:臭蛋
        注意 “:" 为英文字符


        """
        class_names=dict()
        with open("classnames.txt","r",encoding='UTF-8') as f:
            while True:
                line=f.readline()
                line=line[:-1]

                print(line)
                if line=="":
                    break 
                else:
                    class1=line.split(":")
                    class_names[int(class1[0])]=class1[-1]
        return class_names

if __name__=="__main__"  :
    import torchvision.models as models
    import torch
    device='cpu'
    if torch.cuda.is_available():
            device="cuda:0"
    input=torch.randn(1,3,448,448).to(device)
    print(input) 
    model=models.densenet121(pretrained=False,num_classes=5)
    print(model)
    model=model.to(device)
    output=model(input)
    print(output)
   
    model1=models.densenet121(pretrained=False,num_classes=5)
    model2=models.regnet_x_32gf(pretrained=False,num_classes=5)
    
    print(model1)
    print(model2)

    rootdata="/home/zym/下载/egg1" 
    create_classimage_dataset(rootdata)