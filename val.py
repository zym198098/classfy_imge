from pickle import TRUE
import time
from matplotlib import image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData
from torch.optim import lr_scheduler
from torchvision.models import alexnet  # 最简单的模型
from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
from torchvision.models import inception_v3     # Inception 系列
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
import time
def get_ResNet(classes,pretrained=True,loadfile = None):
    ResNet=resnet101(pretrained)# 这里自动下载官方的预训练模型
    if loadfile!= None:
        ResNet.load_state_dict(torch.load( loadfile))	#加载本地模型 
        
    # 将所有的参数层进行冻结
    for param in ResNet.parameters():
        param.requires_grad = False
    # 这里打印下全连接层的信息
    print(ResNet.fc)
    x = ResNet.fc.in_features #获取到fc层的输入
    ResNet.fc = nn.Linear(x, classes) # 定义一个新的FC层
    print(ResNet.fc) # 最后再打印一下新的模型
    return ResNet
def padding_black( img):#如果尺寸太小可以扩充
        w, h  = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img
if __name__=='__main__':
    classname={0: '有精蛋', 1: '反蛋', 2: '空位蛋', 3: '臭蛋', 4: '无精蛋'}
        # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # model=models.efficientnet_b4(pretrained=False,num_classes=5)
    model=models.resnet50(pretrained=False,num_classes=6)
    model.load_state_dict(torch.load("jidan_restnet50_best0708.pth"))
    # print(model)
    # model.eval()
    model.to(device)
  
    # model = torch.load("jidan_efficent4_best.pth").to(device)
    
    val_tf  = transforms.Compose([
                transforms.Resize(664),#将图片压缩成224*224的大小
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                # transform_BZ#图片标准化的步骤
            ])
       # 将模型转为验证模式
    # model.eval()
    # print(model)

    img = Image.open("./pics/egg_fd.jpg")#打开图片
    img = img.convert('RGB')#转换为RGB 格式
    img = padding_black(img)
    img =val_tf(img)

    print(img.shape)
    img1=torch.reshape(img,(1,3,664,664))
    img1=img1.to(device)
    with torch.no_grad():
        time_start=time.time()
        pred = model(img1)
        time_end=time.time()
        print("time pred:",time_end-time_start)
        # pred.to("cpu")
        # print(pred.shape)
        result=torch.softmax(pred[0],0)
        result.max()
        print(result)
        num=torch.argmax(result).item()+1
        print("class name}:",classname[num],"Score:",result[num-1].item())
  
        
    # plt.show(img)
    # print("img")