import time
import onnx
import torch
import torchvision.transforms as transforms
import onnxruntime
from PIL import Image
# 4.检查onnx计算图
def checknet():
    net = onnx.load("./swin_s_224_last.pth.onnx")
    onnx.checker.check_model(net)           # 检查文件模型是否正确


def runonnx():
    transform_BZ= transforms.Normalize(
    # mean=[0.485, 0.456, 0.406],#imagenet 标准化参数
    # std=[0.229, 0.224, 0.225]
#     [-0.0060067656, -0.006891677, -0.007734876]
# [0.009368785, 0.009223793, 0.009070563]
    # mean=[0.5, 0.5, 0.5],# 标准化参数
    # std=[0.5, 0.5, 0.5]
    mean=[0.03371112, 0.03064092, 0.027711032],# 标准化参数
    std=[0.028724361, 0.028832901, 0.028438283]
    )
    val_tf  = transforms.Compose([
                transforms.Resize((224,224)),#将图片压缩成224*224的大小
            
                # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                transform_BZ#图片标准化的步骤
            ])

    # image = torch.randn(2, 3, 224, 224).cuda()
    img_dir='pics/egg_fd.jpg'
    img=Image.open(img_dir)
    img = img.convert('RGB')#转换为RGB 格式
    img=val_tf(img)
    img=torch.unsqueeze(img,0)
    # img=torch.reshape(img,(1,3,224,224))

    session = onnxruntime.InferenceSession("./swin_s_224_last.pth.onnx")
    meta=session.get_modelmeta()
    print(meta)
    output2 = session.run(['output'], {"input": img.cpu().numpy()})
    output0=output2[0].reshape(-1)
    print(output2[0])
    t=torch.tensor(output2[0])
    print(t.shape)
    output3=torch.softmax(torch.tensor(output2[0]),dim=1 )
#     0:反蛋
# 1:无精蛋
# 2:有精蛋
# 3:空位蛋
# 4:臭蛋
    classnames=['反蛋','无精蛋','有精蛋','空位蛋','臭蛋']
    for ouput in output3:
        # print(ouput.shape)
        m= torch.max(ouput,0)
        print(m[0].data)
        print(classnames[m[1].data])
        # print(ouput)
    print(output2[0].shape)


if __name__ == '__main__':
    checknet()
    runonnx()