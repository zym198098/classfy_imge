
import time
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.functional as F

# 
training_data = datasets.FashionMNIST(
    root="dataset",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize(56),
                transforms.ToTensor()
])
)

test_data = datasets.FashionMNIST(
    root="dataset",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize(56),
               transforms.ToTensor()])
)

labels_map = {
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
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     print(img)
#     plt.imshow(img[0].squeeze(), cmap="gray")
# plt.show()

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

from torch.utils.data import DataLoader

# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer):
   
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()

        # 得到预测的结果pred
        pred = model(X)

        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练100次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
  
            
    

def test(dataloader, model):
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
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
if __name__ =='__main__':
    batch=800
    train_dataloader = DataLoader(training_data,num_workers=4,pin_memory=True, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_data,num_workers=2,pin_memory=True, batch_size=800, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img[0], cmap="gray")
    plt.show()
    print(f"Label: {label}")

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model=torchvision.models.resnet50(pretrained=False,num_classes=10).to(device)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

        # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 初始学习率

    epotchs=50
    for t in range(epotchs):
        print(f"Epoch {t+1}\n-------------------------------")
        time_start = time.time()
        train(train_dataloader, model, loss_fn, optimizer)
        time_end = time.time()
        print(f"train time: {(time_end-time_start)}") 
            # 5 epoch one test 
        if (t+1)%1==0:
            test(test_dataloader, model)
    print("Done!")
    # torch.save(model,"minst1.pth")
    torch.save(model.state_dict(),"minist_dict.pth")
   
    input=torch.randn(1,3,28,28,requires_grad=True).cuda()
    model=model.eval()

    torch.onnx.export(model,input,"last.onnx", input_names = ['input'],   # the model's input names
                  output_names = ['output'],opset_version=11)
    
    # Export the model
    torch.onnx.export(model,               # model being run
                    input,                         # model input (or a tuple for multiple inputs)
                    "last_resolution.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print("Saved PyTorch Model Success!")

