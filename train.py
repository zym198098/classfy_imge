'''
    加载pytorch自带的模型，从头训练自己的数据
'''
from concurrent.futures import thread
from posixpath import split
# from pickle import TRUE
import time
from sqlalchemy import false
# from matplotlib.pyplot import text
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader
from utils.utils_egg import LoadData
from torch.optim import lr_scheduler
# from torchvision.models import alexnet  # 最简单的模型
# from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
# from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
# from torchvision.models import inception_v3     # Inception 系列
# import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import os
import random
import imghdr
#加载PYQT5 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
# from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget,QFileDialog
from PyQt5.QtCore import pyqtSignal 
from PyQt5.QtWidgets import QComboBox
from Ui_train import Ui_UI

import sys

from collections import OrderedDict
import torch.onnx 
import onnx
import torchvision
import torch.torch_version as version
import torchvision.version as tversion


class MyThread(QThread):
    printtext=pyqtSignal(str)
    def __init__(self,train_params={}):
        super().__init__()
        self.class_size=80
        self.train_bench_size=1
        self.val_bench_size=1
        self.pic_dir="./pics"
        self.mult_gpu=False
        self.model_type=0
        self.model_type_name="restnet50"
        self.epochs=1
        self.train_percent=0.8
        self.img_size=224
        self.lr=0.01
        self.lr_f=0
        self.train_params=train_params
        self.model_name="restnet50"
        self.train_exit=False
        
	
	# 开启线程后默认执行
    def run(self):
        # self.train_params["train_bench_size"]=self.train_benchsize.value()
        # self.train_params["val_bench_size"]=self.val_benchsize.value()
        # self.train_params["pic_dir"]=self.lineEdit_picdir.text()
        # self.train_params["mult_gpu"]=self.checkBox_multGpu.isChecked()
        # self.train_params["model_type"]=self.comb_model.currentIndex()
        # self.train_params["epochs"]=self.epochs.value()
        # self.train_params["train_percent"]=self.doubleSpinBox_trainsize.value()#训练文件占总图片数的百分比
        # self.train_params["img_size"]=self.img_size.value()#训练图片缩放大小
        # self.train_params["lr"]=self.lr.value()#学习率
        
        if 'train_bench_size' in self.train_params:
            self.train_bench_size=self.train_params.get('train_bench_size')
        
        if 'train_bench_size' in self.train_params:
                self.val_bench_size=self.train_params.get('val_bench_size')
        
        if 'pic_dir' in self.train_params:
                self.pic_dir=self.train_params.get('pic_dir')
        
        if 'mult_gpu' in self.train_params:
                self.mult_gpu=self.train_params.get('mult_gpu')
        
        if 'model_type' in self.train_params:
                self.model_type=self.train_params.get('model_type')
        if 'model_type_name' in self.train_params:
                self.model_type=self.train_params.get('model_type_name')
        
        if 'epochs' in self.train_params:
                self.epochs=self.train_params.get('epochs')
        
        if 'train_percent' in self.train_params:
                self.train_percent=self.train_params.get('train_percent')
        
        if 'img_size' in self.train_params:
                self.img_size=self.train_params.get('img_size')
        
        if 'lr' in self.train_params:
                self.lr=self.train_params.get('lr')
        if "model_name" in self.train_params:
            self.model_name=self.train_params.get('model_name')
        if "amp" in self.train_params:
            self.amp=self.train_params.get('amp')
        if "lr_f" in self.train_params:#学习率调整方法
            self.lr_f=self.train_params.get('lr_f')
        self.btn_train_cleck()
    #根据index 生成模型     
    def get_model(self):
        model_index=self.model_type
        if model_index==0:
            model=models.resnet50(pretrained=False,num_classes=self.class_size)
        elif model_index==1:
            model=models.resnet101(pretrained=False,num_classes=self.class_size)
        elif model_index==2:
                model=models.resnet152(pretrained=False,num_classes=self.class_size)
        elif model_index==3:
                model=models.efficientnet_b4(pretrained=False,num_classes=self.class_size)
        elif model_index==4:
                model=models.densenet121(pretrained=False,num_classes=self.class_size)
            # densenet161
        elif model_index==5:
                model=models.densenet161(pretrained=False,num_classes=self.class_size)    
        elif model_index==6:
                model=models.regnet_x_32gf(pretrained=False,num_classes=self.class_size)
        elif model_index==7:
                model=models.vision_transformer.vit_b_32(pretrained=False,image_size=self.img_size,num_classes=self.class_size)
        # 8 efficientnet_v2_m pytorch 1.12
        elif model_index==8:
                model=models.efficientnet_v2_m(dropout=0.2,pretrained=False,image_size=self.img_size,num_classes=self.class_size)
        elif model_index==9:
                model=models.efficientnet_v2_l(dropout=0.2,pretrained=False,image_size=self.img_size,num_classes=self.class_size)
                
        return model
   #根据名称生成模型 
    def get_model_byname(self):
            model_name=self.model_type
    #         "resnet18",
    # "resnet34",
    # "resnet50",
    # "resnet101",
    # "resnet152",
    # "resnext50_32x4d",
    # "resnext101_32x8d",
    # "resnext101_64x4d",
    # "wide_resnet50_2",
    # "wide_resnet101_2",
    # "densenet121",
    # "densenet161",
    # "densenet169",
    # "densenet201",
    #  "efficientnet_b0",
    # "efficientnet_b1",
    # "efficientnet_b2",
    # "efficientnet_b3",
    # "efficientnet_b4",
    # "efficientnet_b5",
    # "efficientnet_b6",
    # "efficientnet_b7",
    # "efficientnet_v2_s",
    # "efficientnet_v2_m",
    # "efficientnet_v2_l",
    #  "vit_b_16",
    # "vit_b_32",
    # "vit_l_16",
    # "vit_l_32",
    # "vit_h_14",
    # "swin_t",
    # "swin_s",
    # "swin_b",}
    #resnet 10
            if model_name=='resnet18':
                model=models.resnet18(pretrained=False,num_classes=self.class_size)
            elif model_name=='resnet34':
                model=models.resnet34(pretrained=False,num_classes=self.class_size)
            if model_name=='resnet50':
                model=models.resnet50(pretrained=False,num_classes=self.class_size)
            elif model_name=='resnet101':
                model=models.resnet101(pretrained=False,num_classes=self.class_size)
            elif model_name=='resnet152':
                    model=models.resnet152(pretrained=False,num_classes=self.class_size)
            if model_name=='resnext50_32x4d':
                model=models.resnext50_32x4d(pretrained=False,num_classes=self.class_size)
            elif model_name=='resnext101_32x8d':
                model=models.resnext101_32x8d(pretrained=False,num_classes=self.class_size)
            if model_name=='resnext101_64x4d':
                model=models.resnext101_64x4d(pretrained=False,num_classes=self.class_size)
            elif model_name=='wide_resnet50_2':
                model=models.wide_resnet50_2(pretrained=False,num_classes=self.class_size)
            elif model_name=='wide_resnet101_2':
                    model=models.wide_resnet101_2(pretrained=False,num_classes=self.class_size)
                    
            # densenet
            elif model_name=='densenet121':
                    model=models.densenet121(pretrained=False,num_classes=self.class_size)
                # densenet161
            elif model_name=='densenet161':
                    model=models.densenet161(pretrained=False,num_classes=self.class_size)  
            elif model_name=='densenet169':
                    model=models.densenet169(pretrained=False,num_classes=self.class_size)
                # densenet161
            elif model_name=='densenet201':
                    model=models.densenet201(pretrained=False,num_classes=self.class_size)  
# efficientnet
            elif model_name=='efficientnet_b0':
                    model=models.efficientnet_b0(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b1':
                    model=models.efficientnet_b1(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b2':
                    model=models.efficientnet_b2(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b3':
                    model=models.efficientnet_b3(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b4':
                    model=models.efficientnet_b4(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b5':
                    model=models.efficientnet_b5(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b6':
                    model=models.efficientnet_b6(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_b7':
                    model=models.efficientnet_b7(pretrained=False,num_classes=self.class_size)
            elif model_name=='efficientnet_v2_s':
                    model=models.efficientnet_v2_s(dropout=0.2,pretrained=False,image_size=self.img_size,num_classes=self.class_size)   
            elif model_name=='efficientnet_v2_m':
                    model=models.efficientnet_v2_m(dropout=0.2,pretrained=False,image_size=self.img_size,num_classes=self.class_size)
            elif model_name=='efficientnet_v2_l':
                    model=models.efficientnet_v2_l(dropout=0.2,pretrained=False,image_size=self.img_size,num_classes=self.class_size)

            # vision_transformer
            elif model_name=='vit_b_16':
                    model=models.vision_transformer.vit_b_16(pretrained=False,image_size=self.img_size,num_classes=self.class_size)      
            elif model_name=='vit_b_32':
                    model=models.vision_transformer.vit_b_32(pretrained=False,image_size=self.img_size,num_classes=self.class_size)
            elif model_name=='vit_l_16':
                    model=models.vision_transformer.vit_l_16(pretrained=False,image_size=self.img_size,num_classes=self.class_size)      
            elif model_name=='vit_l_32':
                    model=models.vision_transformer.vit_l_32(pretrained=False,image_size=self.img_size,num_classes=self.class_size)
            elif model_name=='vit_h_14':
                    model=models.vision_transformer.vit_h_14(pretrained=False,image_size=self.img_size,num_classes=self.class_size)
#swim

            elif model_name=='swin_t':
                    model=models.swin_transformer.swin_t(image_size=self.img_size,num_classes=self.class_size)
            elif model_name=='swin_s':
                    model=models.swin_transformer.swin_s(image_size=self.img_size,num_classes=self.class_size)
            elif model_name=='swin_b':
                    model=models.swin_transformer.swin_b(image_size=self.img_size,num_classes=self.class_size)

            else:
                model=models.resnet50(pretrained=False,num_classes=self.class_size)

            return model
        

    #train 函数
    def train_one_epoch(self,training_loader:DataLoader,model,epoch_index, tb_writer, loss_fn, optimizer,device="cpu",scaler=None):
        running_loss = 0.
        last_loss = 0.
        arc=0.
        
        correct = 0
        total = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        print_step=len(training_loader)/50
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs,labels=inputs.to(device),labels.to(device)
            

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                 # Make predictions for this batch
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

           

            # Compute the loss and its gradients
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # loss.backward()

            # # Adjust learning weights
            # optimizer.step()
            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                train_text='  train size {}/{} loss: {}'.format((i + 1)*training_loader.batch_size,len(training_loader.dataset), last_loss)
                self.printtext.emit(train_text)
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        if total>0:
            arc=float(correct / total)
        # print(f"traning accuracy:{arc:>3f}")
        # writer.add_scalars("Accuracy", {"Train": arc}, 1)

        return last_loss ,arc
    #结束训练
    def btn_exit_train(self):
        
        if self.train_exit==False:
            self.train_exit=True
            train_text='等待本批次训练完成.....'
            self.printtext.emit(train_text)
 #训练函数
    def btn_train_cleck(self):
            self.train_exit=False
            platform=sys.platform
            dataset_root="./dataset"
            
            train_name= os.path.join(dataset_root,'train.txt')
            test_name= os.path.join(dataset_root,'test.txt')
            classnames= os.path.join(dataset_root,'classnames.txt')
            # classes=create_classimage_dataset(root_dir=self.pic_dir,train_ratio=self.train_percent,train_name=train_name,
            # test_name=test_name,classnamestest=classnames)
            classes=dict()
            if os.path.exists(train_name) and os.path.exists(test_name) and os.path.exists(classnames):
                with open(classnames,'r',encoding='UTF-8') as f:
                    names=f.readlines()
                    for l in names:
                        val=l.split(":")
                        classes[int(val[0])]=val[1][:-1]         
            else:
                 classes=create_classimage_dataset(root_dir=self.pic_dir,train_ratio=self.train_percent,train_name=train_name,
            test_name=test_name,classnamestest=classnames)



            print(classes)
            train_weight=[]
            for key,val in classes.items():
                if '臭蛋' in val :
                    train_weight.append(2.0)
                else:
                    train_weight.append(1.0)
            train_weight_loss=torch.tensor(train_weight)
            print(train_weight_loss)

            # for x, y in thisdict.items():
            classtext='类别序号名称：'
            for x,y in classes.items():
                classtext+=str(x)
                classtext+=';'
                classtext+=str(y)


            # self.textBrowser.append(classtext)
            self.printtext.emit(classtext)
            #类别数量
            classes_size=len(classes)
            self.class_size=classes_size
            # 多GPU训练
            
            muli_gpu=self.mult_gpu
            batch_size =self.train_bench_size#训练批次
            batch_size_test =self.val_bench_size#验证批次
            ##给训练集和测试集分别创建一个数据集加载器 class_image/test.txt
            
            train_data = LoadData(train_name, True,self.img_size)
            valid_data = LoadData(test_name, False,self.img_size)
            num_work=3
            if platform=='linux':
                num_work=3
            
            train_dataloader = DataLoader(dataset=train_data, num_workers=num_work, pin_memory=True, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=valid_data, num_workers=num_work, pin_memory=True, batch_size=batch_size_test)


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
            
            # model=models.VisionTransformer(image_size=640,patch_size=32,num_layers=12,num_heads=12,hidden_dim=768,mlp_dim=3072,dropout=0.2,num_classes=classes_size+1)
            # model=models.efficientnet_b4(pretrained=False,num_classes=classes_size)
           
            # model=self.get_model()
            model=self.get_model_byname()
            # print(model)
                # 如果显卡可用，则用显卡进行训练
            device="cpu"
            device_count=0
            device,device_count=get_devices()
        
            if (device_count>1) & muli_gpu:
                    model=nn.DataParallel(model)
            
            # 保存训练好的模型
            # model_path='jidan_rest50_last.pth'
            # model_name='jidan_efficent4'
            model_name=self.model_name+"_"+str(self.img_size)
            model_path=model_name+'_last.pth' 
            #从最好的模型开始训练
            if os.path.exists((model_name+'_best.pth')):
                model.load_state_dict(torch.load(model_name+'_best.pth') )

            # device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            # input=torch.randn(1,3,664,664).to(device)
            # print(input) 
            loss_fn = nn.CrossEntropyLoss(weight=train_weight_loss)
            loss_fn.to(device)

            # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001)  # 初始学习率
            
            optimizer = torch.optim.RAdam(model.parameters(),lr=self.lr,weight_decay=0.0001)  # 初始学习率
            exp_lr_scheduler=None
            if self.lr_f==0:
                exp_lr_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0.000001, last_epoch=-1)
            elif self.lr_f==1:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)#按批次减小学习率
            # Initializing in a separate cell so we can easily add more epochs to the same run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            write_name='runs/egg_'+model_name+'_{}'.format(timestamp)
            # writer = SummaryWriter('runs/egg_rest50_trainer_{}'.format(timestamp))
            writer = SummaryWriter(write_name)
            epoch_number = 0
            EPOCHS = self.epochs
            best_vloss = 2.0
            
            scaler=torch.cuda.amp.GradScaler()
            
            torch.cuda.empty_cache()#清理cuda缓存
            train_text='modle:'+self.model_name+';'+ 'lr:'+str(self.lr)+';train_benchsize:'+str(self.train_bench_size)
            self.printtext.emit(train_text)
            

            for epoch in range(self.epochs):
                if self.train_exit:
                    break
                print('EPOCH {}:'.format(epoch_number + 1))
                print("___lr:" ,optimizer.state_dict()['param_groups'][0]['lr'] )
                train_text='EPOCH {}:'.format(epoch_number + 1)
                self.printtext.emit(train_text)
                train_text="___lr:" +str(optimizer.state_dict()['param_groups'][0]['lr'] )
                self.printtext.emit(train_text)

                # Make sure gradient tracking is on, and do a pass over the data
                model.train(True)
                avg_loss=0.0 
                arc=0.0
                time_start = time.time()

                if self.train_params["amp"]:
                    #混合精度训练
                    try:
                        avg_loss ,arc= self.train_one_epoch(train_dataloader,model,epoch_number, writer, loss_fn, optimizer,device,scaler=scaler)
                    except Exception as e:
                        self.train_exit=True
                        print(e)
                        train_text=str(e)
                        self.printtext.emit(train_text)
                        break
                else:
                    #正常精度训练
                    try:
                        avg_loss ,arc= self.train_one_epoch(train_dataloader,model,epoch_number, writer, loss_fn, optimizer,device,scaler=None)
                    except Exception as e:
                        self.train_exit=True
                        print(e)
                        train_text=str(e)
                        self.printtext.emit(train_text)
                        break
                time_end = time.time()
                print(f"train time: {(time_end-time_start)}")
                train_text=f"train time: {str(time_end-time_start)}"
                self.printtext.emit(train_text)
                if avg_loss<0.0005:
                    train_text=f"avg_loss: {str(avg_loss)} <0.005 "+"training exit"
                    self.printtext.emit(train_text)
                    break
                    # sys.exit(1)
                lr=optimizer.state_dict()['param_groups'][0]['lr']
                if lr>0.000001 or self.lr_f==0:
                    if exp_lr_scheduler!=None: 
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
                        c=c.reshape(-1)
                        for j in range(len(vinputs)):#4是每一个batch的个数
                            label = vlabels[j]
                            class_correct[label] += c[j].item()
                            class_total[label] += 1

                        vloss = loss_fn(voutputs, vlabels)
                        running_vloss += vloss
                        vloss_sum+=1
                        correct += (voutputs.argmax(1) == vlabels).type(torch.float).sum().item()
                        len_test=len(test_dataloader.dataset) /test_dataloader.batch_size 
                        print_epoch=int(len_test/5)
                        if (i%print_epoch)==(print_epoch-1):                   
                        # if (int(correct/size)*100)%10==0:
                            
                            print(f"correct:{correct},/{size}")
                            train_text=f"correct:{correct},/{size}"
                            self.printtext.emit(train_text)
                print(f"correct:{correct},/{size}")
                train_text=f"correct:{correct},/{size}"
                self.printtext.emit(train_text)
                for i in range(classes_size):
                    print('Accuracy of %5s : %3f%%: %d' % (
                    classes[i], 100.0 * class_correct[i] / class_total[i],class_total[i]))#每一个类别的准确率
                    train_text='Accuracy of %5s : %3f%%: %d' % (
                    classes[i], 100.0 * class_correct[i] / class_total[i],class_total[i])
                    # self.textBrowser.append(train_text)
                    self.printtext.emit(train_text)
                    arc1=100 * class_correct[i] / class_total[i]
                    writer.add_scalars('value arc classes',{str(classes[i]):arc1},epoch_number + 1)
                
                correct /= size

                avg_vloss = running_vloss / vloss_sum
                print('LOSS train {} LOSS valid {} train arc {} vaild arc {}'.format(avg_loss, avg_vloss,arc*100,correct*100))
                train_text=f'LOSS train {avg_loss} LOSS valid {avg_vloss} train arc {arc} vaild arc {correct}'
                # train_text='LOSS train: '+str(avg_loss)+'LOSS valid :'+str(avg_vloss)+ 'train arc:'+ str(arc*100)+'vaild arc :'+ str(correct*100)
                self.printtext.emit(train_text)
                time_endv = time.time()
                print(f"test time: {(time_endv-time_startv)}")
                train_text=(f"test time: {(time_endv-time_startv)}")
                # self.textBrowser.append(train_text)
                self.printtext.emit(train_text)

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
                    torch.save(modelbest.state_dict(),model_name+'_best.pth')


                epoch_number += 1
                torch.cuda.empty_cache()#清理cuda缓存
            print("Done!")
            self.printtext.emit("Done!")
            if self.train_exit==False:
                # 保存训练好的模型
                # model_path='jidan_rest50_last.pth'
                # model_path='jidan_efficent4_last.pth'
                # torch.save(model.state_dict(), "model.pth")
                model.eval()
                torch.save(model,model_path)
                # 多GPU
            
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
                    input=torch.randn(1,3,self.img_size,self.img_size).to(device)
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
                    input=torch.randn(1,3,self.img_size,self.img_size).to(device)
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
                # print("Saved PyTorch Model Success!")
                self.printtext.emit("Saved PyTorch Model Success!")
    #线程退出
    def stop(self):
        run=self.isRunning
        if run:
            self.terminate()
            print("stop thread:")
              
# create data text

def create_classimage_dataset(root_dir="./data",train_ratio=0.8,train_name='train_jidan.txt',
test_name='test_jidan.txt',classnamestest='./dataset/classnames.txt'):
        platform=sys.platform

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
                str_split="/"
                if platform=="win32":
                    str_split="\\"
                dir1=dir.split(str_split)
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
        with open(classnamestest,"w",encoding='UTF-8') as f:
            for i ,value1 in c.items():
                print(i)
                print(value1) 
                classs_name=str(i)+":"+str(value1)+'\n'
                f.write(classs_name)        
        return c 

        
# def test(dataloader, model,loss_fn,device="cpu"):
#     size = len(dataloader.dataset)
#     print("test:",size)
#     # 将模型转为验证模式
#     model.eval()
#     # 初始化test_loss 和 correct， 用来统计每次的误差
#     test_loss, correct = 0, 0
#     # 测试时模型参数不用更新，所以no_gard()
#     # 非训练， 推理期用到
#     with torch.no_grad():
#         # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
#         for X, y in dataloader:
#             # 将数据转到GPU
#             X, y = X.to(device), y.to(device)
#             # 将图片传入到模型当中就，得到预测的值pred
#             pred = model(X)
            
#             # print(pred.size())
#             # print(pred)
#             # 计算预测值pred和真实值y的差距
#             test_loss += loss_fn(pred, y).item()
#             # 统计预测正确的个数
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#             if (int(correct/size)*100)%10==0:
#                 print(f"correct:{correct},/{size}")
#     test_loss /= size
#     correct /= size
#     print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# def get_ResNet(classes,pretrained=True,loadfile = None):
#     # ResNet=resnet101(pretrained)# 这里自动下载官方的预训练模型 
#     # ResNet = resnet152(pretrained)
#     ResNet = models.efficientnet_b4(pretrained)
#     efficientnet=True

#     if loadfile!= None:
#         ResNet.load_state_dict(torch.load( loadfile))	#加载本地模型 
        
#     # 将所有的参数层进行冻结
#     for param in ResNet.parameters():
#         param.requires_grad = False
#     # 这里打印下全连接层的信息
#     if efficientnet==True:
#         # print(ResNet)
#         print(ResNet.classifier[1])
#         x = ResNet.classifier[1].in_features #获取到fc层的输入
#         ResNet.classifier[1]= nn.Linear(x, classes) # 定义一个新的FC层
#         print(ResNet.classifier[1]) # 最后再打印一下新的模型

#     else:
#         # print(ResNet)
#         print(ResNet.fc)
#         x = ResNet.fc.in_features #获取到fc层的输入
#         ResNet.fc = nn.Linear(x, classes) # 定义一个新的FC层
#         print(ResNet.fc) # 最后再打印一下新的模型
#     return ResNet

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
def get_devices():
    device='cpu'
    if torch.cuda.is_available():
            device="cuda:0"
    device_count=torch.cuda.device_count()
    return device,device_count
class mywindow(QtWidgets.QWidget,Ui_UI):
    def __init__(self,parent=None):
        super(mywindow,self).__init__(parent)
        self.picsdir='./class_imgs'
        self.t_text="train.txt"
        self.v_text="test.txt"
        self.class_size=80
        self.muli_gpu=False
        # self.model=models.resnet50(pretrained=False,num_classes=self.class_size)
        self.train_params={}
        self.setupUi(self)
        self.pre_models_11=["resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                            "resnext50_32x4d",
                            "resnext101_32x8d",
                            "resnext101_64x4d",
                            "wide_resnet50_2",
                            "wide_resnet101_2",
                            "densenet121",
                            "densenet161",
                            "densenet169",
                            "densenet201",
                            "efficientnet_b0",
                            "efficientnet_b1",
                            "efficientnet_b2",
                            "efficientnet_b3",
                            "efficientnet_b4",
                            "efficientnet_b5",
                            "efficientnet_b6",
                            "efficientnet_b7",
                            "vit_b_16",
                            "vit_b_32",
                            "vit_l_16",
                            "vit_l_32",
        ]
        self.pre_models_12=["resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
     "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
     "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "swin_t",
    "swin_s",
    "swin_b",]
        
        # self.add_comb_models()
        self.add_comb_models_12()
        self.worker=MyThread(self.train_params)
        self.init_ui()
    def add_comb_models_12(self):
            torch_ver=torch.__version__
            torch_ver_g=False#检查版本号是否大于1.11.0    
            ver=version.TorchVersion(torch_ver)
            self.comb_model.clear()
            if ver>(1.11,0):
                
                for modelname1 in self.pre_models_12:
                    self.comb_model.addItem(modelname1)
                    print(modelname1) 
                        
            else :
                for modelname2 in self.pre_models_11:
                    self.comb_model.addItem(modelname2)
                    print(modelname2) 

    def add_comb_models(self):
            torch_ver=torch.__version__
            torch_ver_g=False#检查版本号是否大于1.11.0    
            ver=version.TorchVersion(torch_ver)
            if ver>(1.11,0):
                torch_ver_g=True
            if torch_ver_g:
                self.comb_model.addItem("efficientnet_v2_m")
                self.comb_model.addItem("efficientnet_v2_l")
  
    def train_params_init(self):
        
        self.train_params["train_bench_size"]=self.train_benchsize.value()
        self.train_params["val_bench_size"]=self.val_benchsize.value()
        self.train_params["pic_dir"]=self.lineEdit_picdir.text()
        self.train_params["mult_gpu"]=self.checkBox_multGpu.isChecked()
        self.train_params["model_type"]=self.comb_model.currentIndex()
        self.train_params["model_type_name"]=self.comb_model.currentText()
        self.train_params["epochs"]=self.epochs.value()
        self.train_params["train_percent"]=self.doubleSpinBox_trainsize.value()#训练文件占总图片数的百分比
        self.train_params["img_size"]=self.img_size.value()#训练图片缩放大小
        self.train_params["lr"]=self.lr.value()#学习率
        self.train_params["model_name"]=self.comb_model.currentText()#模型名称
        self.train_params["amp"]=self.checkBox_amp.isChecked()#是否混合精度训练
        self.train_params["lr_f"]=self.comb_lr_f.currentIndex()
    def btn_picdir_cleck(self):
                
                self.picsdir= QFileDialog.getExistingDirectory()
                self.lineEdit_picdir.setText( self.picsdir)
    # def btn_trainT_cleck(self):
                
    #             self.t_text= QFileDialog.getOpenFileName()
    #             self.lineEdit_traintext.setText( self.t_text[0])
    # def btn_valT_cleck(self):
                
    #             self.v_text= QFileDialog.getOpenFileName()
    #             self.lineEdit_valtext.setText( self.v_text[0])
    def get_model(self):
        model_index=self.comb_model.currentIndex()
        if model_index==0:
            model=models.resnet50(pretrained=False,num_classes=self.class_size)
        elif model_index==1:
            model=models.resnet101(pretrained=False,num_classes=self.class_size)
        elif model_index==2:
                model=models.resnet152(pretrained=False,num_classes=self.class_size)
        elif model_index==3:
                model=models.efficientnet_b4(pretrained=False,num_classes=self.class_size)
        return model
            
    def woker(self):
        run=self.worker.isRunning()
        if run==False:
            self.train_params_init()
            self.textBrowser.clear()
            self.worker.start()

    def print_messge(self,text):
        self.textBrowser.append(text)
            

    def init_ui(self):        
                self.btn_picsdir.clicked.connect(self.btn_picdir_cleck)
                # self.btn_traintext.clicked.connect(self.btn_trainT_cleck)
                # self.btn_valtext.clicked.connect(self.btn_valT_cleck)
                self.btn_train.clicked.connect(self.woker)
                self.btn_exit_train.clicked.connect(self.worker.btn_exit_train)
                self.worker.printtext.connect(self.print_messge)
                
                
if __name__=='__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())

   