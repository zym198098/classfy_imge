import time
from matplotlib.pyplot import text
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader
from utils import LoadData
from torch.optim import lr_scheduler
import torchvision
import torchvision.models as models
from datetime import datetime
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import os
import random
# create data text
def create_classimage_dataset(root_dir="./data",train_ratio=0.85,train_name='class_image/train_jidan.txt',
test_name='class_image/test_jidan.txt'):
    # 百分之60用来当训练集
        # train_ratio = 0.9

        # 用来当测试集
        test_ratio = 1-train_ratio

        # rootdata = r"data"  #数据的根目录
        # /media/zym/软件/pics/jidan/标注_09151051
        # rootdata = "class_image/data/test1"  #数据的根目录
        rootdata = root_dir
        train_list, test_list = [],[]#读取里面每一类的类别
        data_list = []

        #生产train.txt和test.txt
        class_flag = -1
        # classnames={"null":0}
        # {'有精蛋': 0, '反蛋': 1, '空位蛋': 2, '臭蛋': 3, '无精蛋': 4}
        # classnames=dict()
        train_lable=[]
        test_label=[]
        classnames={'有精蛋': 0, '反蛋': 1, '空位蛋': 2, '臭蛋': 3, '无精蛋': 4} 
        for dir,folder,file in os.walk(rootdata):
            # print(dir)
            # print(folder)
            for i in range(len(file)):
                
                data_list.append(os.path.join(dir,file[i]))

            for i in range(0,int(len(file)*train_ratio)):
                dir1=dir.split('/')
                len1=len(classnames)
                if dir1[-1]in classnames.keys():
                    # print(dir1[-1])
                    pass
                else:
                    classnames[dir1[-1]]=len1
                train_data = os.path.join(dir, file[i])+'\t'+str(classnames[dir1[-1]])+'\n'
                train_lable.append(dir1[-1])
                train_list.append(train_data)
                

            for i in range(int(len(file) * train_ratio),len(file)):
                dir1=dir.split('/')
                len1=len(classnames)
                if dir1[-1]in classnames.keys():
                        # print(dir1[-1])
                        pass
                else:
                    classnames[dir1[-1]]=len1
                test_data = os.path.join(dir, file[i]) + '\t' +str(classnames[dir1[-1]])+'\n'
                test_label.append(dir1[-1])
                test_list.append(test_data)

            # print(classnames)  

            class_flag += 1

        print(classnames)
        for it in classnames:
            sum=train_lable.count(it)
            sum_t=test_label.count(it)
            print(f"train list classname:{it},sum:{sum}")
            print(f"test list classname:{it},sum:{sum_t}")

        random.shuffle(train_list)#打乱次序
        random.shuffle(test_list)
        # train_name='class_image/train_jidan.txt'
        # test_name='class_image/test_jidan.txt'
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
 
if __name__=='__main__':
    # classes=create_classimage_dataset(root_dir="/media/zym/项目2/鸡蛋",train_ratio=0.8,train_name='train_jidan_1.txt',
    #     test_name='test_jidan_1.txt')
    classes=create_classimage_dataset(root_dir="/home/zym/下载/egg1",train_ratio=0.8,train_name='train_jidan_1.txt',
        test_name='test_jidan_1.txt')
    print(classes)
    # {0: '反蛋', 1: '无精蛋', 2: '有精蛋', 3: '空位蛋', 4: '臭蛋'}#median
    # {'有精蛋': 0, '反蛋': 1, '空位蛋': 2, '臭蛋': 3, '无精蛋': 4} #home
  

    #类别数量
    classes_size=len(classes)
    # 多GPU训练
    # muli_gpu=False
    # batch_size =2#训练批次
    # batch_size_test =20#验证批次
    # ##给训练集和测试集分别创建一个数据集加载器 class_image/test.txt
    # train_data = LoadData("train_jidan.txt", False)
    # valid_data = LoadData("test_jidan.txt", False)
    # train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(dataset=valid_data, num_workers=3, pin_memory=True, batch_size=batch_size_test)

    # list_mean,list_std=getStat(train_data)
    # print(list_mean)
    # print(list_std)
    # fashmins
    # train_dataloader = DataLoader(dataset=training_data, num_workers=4, pin_memory=True, batch_size=256, shuffle=True)
    # test_dataloader = DataLoader(dataset=testing_data, num_workers=2, pin_memory=True, batch_size=800)
   

    
    '''
        随着模型的加深，需要训练的模型参数量增加，相同的训练次数下模型训练准确率起来得更慢
    '''

    # model = alexnet(pretrained=False, num_classes=5).to(device) # 29.3%（不使用模型的预训练参数）

    '''        VGG 系列    '''
    # model = vgg11(pretrained=False, num_classes=5).to(device)   #  23.1%
    # model = vgg13(pretrained=False, num_classes=5).to(device)   # 30.0%
    # model = vgg16(pretrained=False, num_classes=5).to(device)


    '''        ResNet 系列    '''
    # model = resnet18(pretrained=False, num_classes=200).to(device)    # 43.6%
    # model = resnet34(pretrained=False, num_classes=5).to(device)
    # model = resnet50(pretrained= False, num_classes=5).to(device)
    # model = resnet101(pretrained=False, num_classes=10).to(device)   #  26.2%
    # model = resnet152(pretrained=False, num_classes=5).to(device)


    '''        Inception 系列    '''
    # model = inception_v3(pretrained=False, num_classes=5).to(device)
    
    # model = get_ResNet(10,True).to(device)#迁移学习
    # model=models.resnet50(pretrained=False,num_classes=classes_size)
    # def __init__(
    #     self,
    #     image_size: int,
    #     patch_size: int,
    #     num_layers: int,
    #     num_heads: int,
    #     hidden_dim: int,
    #     mlp_dim: int,
    #     dropout: float = 0.0,
    #     attention_dropout: float = 0.0,
    #     num_classes: int = 1000,
    #     representation_size: Optional[int] = None,
    #     norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    #     conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    # ):
    #  return _vision_transformer(
    #     arch="vit_b_32",
    #     patch_size=32,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    #     pretrained=pretrained,
    #     progress=progress,
    #     **kwargs,
    # )
    # model=models.VisionTransformer(image_size=640,patch_size=32,num_layers=12,num_heads=12,hidden_dim=768,mlp_dim=3072,dropout=0.2,num_classes=classes_size+1)
    model=models.efficientnet_b4(pretrained=False,num_classes=classes_size)
    # model=models.resnet50(pretrained=False,num_classes=classes_size+1)
         # 保存训练好的模型
    # model_path='jidan_rest50_last.pth'
    model_name='jidan_efficent4'
    model_path=model_name+'_last.pth'
    model.load_state_dict(torch.load(model_name+'_last.pth') )
 # 多GPU训练
    muli_gpu=False
    # print(model)
        # 如果显卡可用，则用显卡进行训练
    device="cpu"
    if torch.cuda.is_available():
        device="cuda:0"
        if (torch.cuda.device_count()>1) & muli_gpu:
            model=nn.DataParallel(model)
    
  
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # input=torch.randn(1,3,664,664).to(device)
    # y=model(input)
    # print(y)
     
    print(f"Using {device} device")

    batch_size =24#训练批次
    batch_size_test =24#验证批次
    ##给训练集和测试集分别创建一个数据集加载器 class_image/test.txt
    train_data = LoadData("train_jidan_1.txt", False)
    valid_data = LoadData("test_jidan_1.txt", False)
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=3, pin_memory=True, batch_size=batch_size_test)
    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    running_vloss = 0.0
    time_startv = time.time()
    correct=0
    size=len(train_dataloader.dataset)
    torch.cuda.empty_cache()#清理cuda缓存
    model.eval()
    class_correct = list(0. for i in range(classes_size))#10是类别的个数
    class_total = list(0. for i in range(classes_size))
    vloss_sum=0
    with torch.no_grad():
          
            for i, vdata in enumerate(train_dataloader):
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
    for i in range(classes_size):
            print('Accuracy of %5s : %3f%%: %d' % (
            classes[i], 100.0 * class_correct[i] / class_total[i],class_total[i]))#每一个类别的准确率
            arc1=100 * class_correct[i] / class_total[i]
            # writer.add_scalars('value arc classes',{str(classes[i]):arc1},epoch_number + 1)
        
    correct /= size

    avg_vloss = running_vloss / vloss_sum
    print(' LOSS valid {} vaild arc {}'.format( avg_vloss,correct*100))
    time_endv = time.time()
    print(f"test time: {(time_endv-time_startv)}")
        





