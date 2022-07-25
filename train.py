'''
    加载pytorch自带的模型，从头训练自己的数据
'''
from pickle import TRUE
import time
from matplotlib.pyplot import text
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader
from utils import LoadData
from torch.optim import lr_scheduler
# from torchvision.models import alexnet  # 最简单的模型
# from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
# from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
# from torchvision.models import inception_v3     # Inception 系列
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import os
import random
classes="class"
class_correct=78
class_total=100

# training_data = datasets.FashionMNIST(
#     root="dataset",
#     train=True,
#     download=True,
#     transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize(96),
#                 transforms.ToTensor()
# ])
# )

# testing_data = datasets.FashionMNIST(
#     root="dataset",
#     train=False,
#     download=True,
#     transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize(96),
#                transforms.ToTensor()])
# )
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
        classnames=dict()
        train_lable=[]
        test_label=[]
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
    
#train 函数
def train_one_epoch(training_loader,model,epoch_index, tb_writer, loss_fn, optimizer,device="cpu"):
    running_loss = 0.
    last_loss = 0.
    arc=0.
    
    correct = 0.
    total = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs,labels=inputs.to(device),labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    arc=float(correct / total)
    # print(f"traning accuracy:{arc:>3f}")
    # writer.add_scalars("Accuracy", {"Train": arc}, 1)

    return last_loss ,arc
        
def test(dataloader, model,loss_fn,device="cpu"):
    size = len(dataloader.dataset)
    print("test:",size)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            
            # print(pred.size())
            # print(pred)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if (int(correct/size)*100)%10==0:
                print(f"correct:{correct},/{size}")
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_ResNet(classes,pretrained=True,loadfile = None):
    # ResNet=resnet101(pretrained)# 这里自动下载官方的预训练模型 
    # ResNet = resnet152(pretrained)
    ResNet = models.efficientnet_b4(pretrained)
    efficientnet=True

    if loadfile!= None:
        ResNet.load_state_dict(torch.load( loadfile))	#加载本地模型 
        
    # 将所有的参数层进行冻结
    for param in ResNet.parameters():
        param.requires_grad = False
    # 这里打印下全连接层的信息
    if efficientnet==True:
        # print(ResNet)
        print(ResNet.classifier[1])
        x = ResNet.classifier[1].in_features #获取到fc层的输入
        ResNet.classifier[1]= nn.Linear(x, classes) # 定义一个新的FC层
        print(ResNet.classifier[1]) # 最后再打印一下新的模型

    else:
        # print(ResNet)
        print(ResNet.fc)
        x = ResNet.fc.in_features #获取到fc层的输入
        ResNet.fc = nn.Linear(x, classes) # 定义一个新的FC层
        print(ResNet.fc) # 最后再打印一下新的模型
    return ResNet
# 计算均值和方差
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
    # for X, _ in enumerate(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
if __name__=='__main__':

    classes = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
        }
    classes=create_classimage_dataset(root_dir="/home/zym/下载/egg1",train_ratio=0.8,train_name='train_jidan1.txt',
        test_name='test_jidan1.txt')
    print(classes)
    #类别数量
    classes_size=len(classes)
    # 多GPU训练
    muli_gpu=False
    batch_size =3#训练批次
    batch_size_test =24#验证批次
    ##给训练集和测试集分别创建一个数据集加载器 class_image/test.txt
    train_data = LoadData("train_jidan.txt", True)
    valid_data = LoadData("test_jidan.txt", False)
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=3, pin_memory=True, batch_size=batch_size_test)

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

    print(model)
        # 如果显卡可用，则用显卡进行训练
    device="cpu"
    if torch.cuda.is_available():
        device="cuda:0"
        if (torch.cuda.device_count()>1) & muli_gpu:
            model=nn.DataParallel(model)
    
     # 保存训练好的模型
    # model_path='jidan_rest50_last.pth'
    model_name='jidan_efficent4'
    # model_name='jidan_restnet50'
    model_path=model_name+'_last.pth' 
    #从最好的模型开始训练
    model.load_state_dict(torch.load(model_name+'_last.pth') )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input=torch.randn(1,3,664,664).to(device)
    # with torch.no_grad():
    #     model.eval()
    #     y=model(input)
    #     print(y)
     
    print(f"Using {device} device")
    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001)  # 初始学习率
    optimizer = torch.optim.RAdam(model.parameters(),lr=5e-5,weight_decay=0.0001)  # 初始学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)#按批次减小学习率




    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    write_name='runs/egg_'+model_name+'_{}'.format(timestamp)
    # writer = SummaryWriter('runs/egg_rest50_trainer_{}'.format(timestamp))
    writer = SummaryWriter(write_name)
    epoch_number = 0

    EPOCHS = 20

    best_vloss = 2.0

    torch.cuda.empty_cache()#清理cuda缓存


    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        print("___lr:" ,optimizer.state_dict()['param_groups'][0]['lr'] )

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        time_start = time.time()
        avg_loss ,arc= train_one_epoch(train_dataloader,model,epoch_number, writer, loss_fn, optimizer,device)
        time_end = time.time()
        print(f"train time: {(time_end-time_start)}")
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        if lr>0.00001: 
            exp_lr_scheduler.step()

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        time_startv = time.time()
        correct=0
        size=len(test_dataloader.dataset)
        torch.cuda.empty_cache()#清理cuda缓存
        model.eval()
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
        for i in range(classes_size):
            print('Accuracy of %5s : %3f%%: %d' % (
            classes[i], 100.0 * class_correct[i] / class_total[i],class_total[i]))#每一个类别的准确率
            arc1=100 * class_correct[i] / class_total[i]
            writer.add_scalars('value arc classes',{str(classes[i]):arc1},epoch_number + 1)
        
        correct /= size

        avg_vloss = running_vloss / vloss_sum
        print('LOSS train {} LOSS valid {} train arc {} vaild arc {}'.format(avg_loss, avg_vloss,arc*100,correct*100))
        time_endv = time.time()
        print(f"test time: {(time_endv-time_startv)}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('traing arc vs test arc',
                        { 'traing_arc':arc*100,'testing arc' : correct*100},
                        epoch_number + 1)
       
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)
            #baocun
            modelbest=model
            modelbest.eval()
            torch.save(modelbest.state_dict(),model_name+'_best0708.pth')


        epoch_number += 1
        torch.cuda.empty_cache()#清理cuda缓存
    
    
    print("Done!")

    # 保存训练好的模型
    # model_path='jidan_rest50_last.pth'
    # model_path='jidan_efficent4_last.pth'
    # torch.save(model.state_dict(), "model.pth")
    model.eval()
    torch.save(model.state_dict(),model_path)
    #单gpu
    # input=torch.randn(1,3,664,664).to(device)
   
    # # Export the model
    # torch.onnx.export(model,               # model being run
    #                 input,                         # model input (or a tuple for multiple inputs)
    #                 "jidan.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=11,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['output'], # the model's output names
    #                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                                 'output' : {0 : 'batch_size'}})

    # 多GPU
    from collections import OrderedDict
    import torch.onnx 
    import onnx
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (torch.cuda.device_count()>1) & muli_gpu:

    
        state_dict=torch.load(model_path)
        new_state_dict=OrderedDict()
        # print(state_dict)
        for k,v in state_dict.items():
            name=k[7:]
            new_state_dict[name]=v
            # print(k)
            # print(v)

        model1=models.resnet50(pretrained=False,num_classes=5)
        model1.load_state_dict(new_state_dict)
        model1.to(device)
        model1.eval()
        onnx_path=model_path+".onnx"
        # input=torch.randn(1,3,640,640).to(device)
            # Export the model
        torch.onnx.export(model1,               # model being run
                            input,                         # model input (or a tuple for multiple inputs)
                            onnx_path,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=10,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
    else:
        model.eval()
        onnx_path=model_path+".onnx"
        # input=torch.randn(1,3,640,664).to(device)
            # Export the model
        torch.onnx.export(model,               # model being run
                            input,                         # model input (or a tuple for multiple inputs)
                            onnx_path,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=10,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})

    print("Saved PyTorch Model Success!")