##把数据传入模型中进行训练
import torch
from PIL import Image

import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
import os
import numpy as np
# 数据归一化与标准化

# 图片标准化
transform_BZ= transforms.Normalize(
    # mean=[0.485, 0.456, 0.406],#imagenet 标准化参数
    # std=[0.229, 0.224, 0.225]
#     [-0.0060067656, -0.006891677, -0.007734876]
# [0.009368785, 0.009223793, 0.009070563]
    mean=[0.5, 0.5, 0.5],# 标准化参数
    std=[0.5, 0.5, 0.5]
    # mean=[0.03371112, 0.03064092, 0.027711032],# 标准化参数
    # std=[0.028724361, 0.028832901, 0.028438283]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True,img_size=224):
        self.imgs = self.get_images(txt_path)
        self.train_flag = train_flag
        self.img_size=(img_size,img_size)
        self.classes=dict()
        # self.targets=[]
        self.class_to_idx={}

        self.train_tf = transforms.Compose([
                transforms.RandomRotation(2,center=(0,0),expand=True),
                transforms.Resize(self.img_size),#将图片压缩成224*224的大小
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                # transforms.ColorJitter(brightness=[0.3,0.5],contrast=[0.3,0.5],saturation=[0.3,0.5]),
                # transforms.ColorJitter(contrast=[1.2,1.3]),
                transforms.ToTensor(),#把图片改为Tensor格式
                transform_BZ#图片标准化的步骤
            ])
        self.val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.Resize(self.img_size),
                # transforms.ColorJitter(contrast=[1.2,1.3]),
                transforms.ToTensor(),
                # transform_BZ#标准化操作
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs = f.readlines()
            # si=len(imgs)
            self.targets=list(map(lambda x:int(x.strip().split('\t')[1]), imgs))
            
            
            imgs = list(map(lambda x:x.strip().split('\t'), imgs))
            
            
        return imgs#返回图片信息

    def padding_black(self, img):#如果尺寸太小可以扩充
        w, h  = img.size
        scale = (self.img_size*1.0)/ max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = self.img_size
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):#返回真正想返回的东西
        img_path, label = self.imgs[index]
        img = Image.open(img_path)#打开图片
        img = img.convert('RGB')#转换为RGB 格式
        # img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_height_and_width(self, idx):
        data_height = self.img_size
        data_width = self.img_size
        return data_height, data_width


class LoadData_csv(Dataset):
    def __init__(self, txt_path, train_flag=True,img_size=664):
        self.cls_map={}
        self.imgs = self.get_images(txt_path)#图片目录
        self.train_flag = train_flag
        self.img_size=(img_size,img_size)
        self.classes=dict()
        self.targets=[]
        

        self.train_tf = transforms.Compose([
                transforms.RandomRotation(2,center=(0,0),expand=True),
                transforms.Resize(self.img_size),#将图片压缩成224*224的大小
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                # transforms.ColorJitter(brightness=[0.3,0.5],contrast=[0.3,0.5],saturation=[0.3,0.5]),
                # transforms.ColorJitter(contrast=[1.2,1.3]),
                transforms.ToTensor(),#把图片改为Tensor格式
                # transform_BZ#图片标准化的步骤
            ])
        self.val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.Resize(self.img_size),
                # transforms.ColorJitter(contrast=[1.2,1.3]),
                transforms.ToTensor(),
                # transform_BZ#标准化操作
            ])

    def get_images(self, txt_path):
#         status = 'open'
# assert status == 'open' ,'The status need open.'

        path_exist=os.path.exists(txt_path)
        assert path_exist==True,txt_path+' file not exist'
        if( path_exist!=True):
            imgs=[]
            return imgs
        # imgs=np.loadtxt(txt_path,dtype=str, delimiter=",",
        #                  skiprows=1)   
        imgs=np.loadtxt(txt_path,dtype=str, delimiter=",")                   
        # print(imgs.shape)
        classes=imgs[:,-1]
        classes.shape
        cls=np.unique(classes)#去重获取类名
        # print(len(cls))
        i=0
        # cls_map={}
        for name in cls:
            self.cls_map[name]=i
            i=i+1
        # print(len(imgs))
        for j in range(0,(len(imgs))):
            name=imgs[j][1]
            imgs[j][1]=self.cls_map[name]
        return imgs#返回图片信息

    def __getitem__(self, index):#返回真正想返回的东西
        img_path, label = self.imgs[index]
        img = Image.open(img_path)#打开图片
        img = img.convert('RGB')#转换为RGB 格式
        # img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs)

   
if __name__ == "__main__":
    train_dataset = LoadData("class_image/train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(image)
        print(label)