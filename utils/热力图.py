from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import *

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from torchcam.utils import overlay_mask
# # Grad-CAM++
    # from pytorch_grad_cam import GradCAMPlusPlus  


import torch
def get_relitu(model,target_layers,img_pil,test_transform,device='cpu'):
    '''
    model 完成了model.eval model.to(device)
    target_layers 生成热力图的层，不同模型层数不一样
    img_pil plt 读取的图片
    test_transform 图像预处理
    device 和模型to device 一致
    本函数热力图模式GradCAMPlusPlus

    '''
    # print(device)
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    # print(input_tensor.shape)
   
      # from pytorch_grad_cam import GradCAM
    # print(model)
    # target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam_map = cam(input_tensor=input_tensor)[0] # 不加平滑
    # print(cam_map.shape)
    # plt.imshow(cam_map)
    # plt.show()
    
    

    result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡   
    import os
    if os.path.exists('output')==False:
        os.mkdir('output')
    result.save('output/B1.jpg')
    return result
    
if(__name__=='__main__'):
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
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
    img_path = 'pics/cat_dog.jpg'
    img_pil = Image.open(img_path)
    # model=torch.load("runs/resnet50_224_20221127_200903/checkponts/resnet50_224_last.pth")
    model = resnet50(pretrained=True).eval().to(device)
    # model = resnet101(pretrained=True).eval().to(device)
    
    model.eval()
    model.to(device)
    target_layers = [model.layer4[-1]]
    result=get_relitu(model=model,target_layers=target_layers,img_pil=img_pil,test_transform=test_transform,device=device)
    plt.imshow(result)
    plt.show()
    print('done')

   
