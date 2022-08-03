from pickle import TRUE
import time
from matplotlib import image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData
from torch.optim import lr_scheduler
   # Inception 系列
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
import time
import walk
import utils
def get_ResNet(classes,pretrained=True,loadfile = None):
    ResNet=models.resnet101(pretrained)# 这里自动下载官方的预训练模型
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
    classname=walk.create_classimage_dataset("/home/zym/下载/egg1",0.99,train_name="verfy1.txt",test_name="verfy2.txt",shuffle=False)
    print(classname)
    # classname={0: '反蛋', 1: '无精蛋', 2: '有精蛋', 3: '空位蛋', 4: '臭蛋'}
        # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # model=models.efficientnet_b4(pretrained=False,num_classes=5)
    model=models.densenet121(pretrained=False,num_classes=len(classname))
    model.load_state_dict(torch.load("densenet121_224_best.pth"))
    # print(model)
    model.eval()
    model.to(device)
  
    # model = torch.load("jidan_efficent4_best.pth").to(device)
    
    val_tf  = transforms.Compose([
                transforms.Resize((224,224)),#将图片压缩成224*224的大小
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                # transform_BZ#图片标准化的步骤
            ])
    test_data=utils.LoadData("verfy1.txt",train_flag=False,img_size=224)
    test_dataloader=DataLoader(test_data,batch_size=48,num_workers=3,pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()
    running_vloss = 0.0
    time_startv = time.time()
    correct=0
    size=len(test_dataloader.dataset)
                
    classes_size=len(classname )          
    class_correct = list(0. for i in range(classes_size))#10是类别的个数
    class_total = list(0. for i in range(classes_size))
    vloss_sum=0            
    with torch.no_grad():
                
                for i, vdata in enumerate(test_dataloader):
                        vinputs, vlabels = vdata
                        vinputs,vlabels=vinputs.to(device),vlabels.to(device)
                        voutputs = model(vinputs)
                        _, predicted = torch.max(voutputs, 1)
                        c = (predicted == vlabels).squeeze()#每一个batch的(predicted==labels)
                        for i in range(len(vinputs)):#4是每一个batch的个数
                            label = vlabels[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1

                        vloss = loss_fn(voutputs, vlabels)
                        running_vloss += vloss
                        vloss_sum+=1
                        correct += (voutputs.argmax(1) == vlabels).type(torch.float).sum().item()
                        if (int(correct/size)*100)%10==0:
                            print(f"correct:{correct},/{size}")
                            # train_text=f"correct:{correct},/{size}"
                            # self.printtext.emit(train_text)
    for i in range(classes_size):
                    print('Accuracy of %5s : %3f%%: %d' % (
                    classname[i], 100.0 * class_correct[i] / class_total[i],class_total[i]))#每一个类别的准确率
                    train_text='Accuracy of %5s : %3f%%: %d' % (
                    classname[i], 100.0 * class_correct[i] / class_total[i],class_total[i])
                    # self.textBrowser.append(train_text)
                    # self.printtext.emit(train_text)
                    arc1=100 * class_correct[i] / class_total[i]
                
    correct /= size

    avg_vloss = running_vloss / vloss_sum
    print(' LOSS valid {}  vaild arc {}'.format( avg_vloss,correct*100))
    # train_text='LOSS train: '+str(avg_loss)+'LOSS valid :'+str(avg_vloss)+ 'train arc:'+ str(arc*100)+'vaild arc :'+ str(correct*100)
    # self.printtext.emit(train_text)
    time_endv = time.time()
    print(f"test time: {(time_endv-time_startv)}")
    # train_text=(f"test time: {(time_endv-time_startv)}")
    #             # self.textBrowser.append(train_text)
    # self.printtext.emit(train_text)

               


    # img = Image.open("./pics/egg_kw.jpg")#打开图片
    # img = img.convert('RGB')#转换为RGB 格式
    # img =val_tf(img)
    # print(img.shape)
    # img1=torch.reshape(img,(1,3,224,224))
    # img1=img1.to(device)
    # with torch.no_grad():
    #     time_start=time.time()
    #     pred = model(img1)
    #     time_end=time.time()
    #     print("time pred:",time_end-time_start)
    #     # pred.to("cpu")
    #     # print(pred.shape)
    #     result=torch.softmax(pred[0],0)
    #     result.max()
    #     print(result)
    #     num=torch.argmax(result).item()
    #     print("class name :",classname[num],"Score:",result[num].item())
  
        
    # plt.show(img)
    # print("img")