from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# %matplotlib inline

import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

classname=dict()
with open("dataset/classnames.txt","r",encoding='UTF-8') as f:
            while True:
                line=f.readline()
                line=line[:-1]

                print(line)
                if line=="":
                    break 
                else:
                    class1=line.split(":")
                    classname[int(class1[0])]=class1[-1]
model = resnet50(pretrained=False,num_classes=len(classname))
model.load_state_dict(torch.load("resnet50_224_best.pth"))
model.eval()
model.to(device)

test_transform  = transforms.Compose([
                transforms.Resize((224,224)),#将图片压缩成224*224的大小
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                # transform_BZ#图片标准化的步骤
            ])
       # 将模型转为验证模式
    # model.eval()
    # print(model)
imgType_list = {'jpg', 'bmp', 'png', 'jpeg'} #支持的图片文件格式
img_path = 'pics/egg_chd.jpg'
img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
print(input_tensor.shape)
targets = classname[0]
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
# from pytorch_grad_cam import GradCAM
print(model)
target_layers = [model.layer4[-1]]
# # Grad-CAM++
# from pytorch_grad_cam import GradCAMPlusPlus
# target_layers = [model.layer4[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
cam_map = cam(input_tensor=input_tensor)[0] # 不加平滑
print(cam_map.shape)
plt.imshow(cam_map)
plt.show()
import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡
plt.imshow(result)
plt.show()
result.save('output/B1.jpg')
