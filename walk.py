import os
import random

def create_classimage_dataset(root_dir="./data",train_ratio=0.8,train_name='train_jidan.txt',
test_name='test_jidan.txt'):

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
        return c
if __name__=="__main__"  :
    rootdata="/home/zym/下载/egg1" 
    create_classimage_dataset(rootdata)