from __future__ import print_function, division

import torch
from torchvision import datasets, models, transforms
import PIL.Image as Image


def Predict(path,vgg16):
    class_names = ['angry', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised'] #待识别的类别

    show_transforn = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         ]
    ) #对导入的PIL图像进行处理，转成三通道224*224的tensor

    iml = Image.open(path).convert('RGB') #以RGB的形式读入
    imlgray=iml.convert('L')              #转成灰度图
    imlrgb=imlgray.convert('RGB')         #转成三通道灰度图
    image = show_transforn(imlrgb).unsqueeze(0)  #转为三通道224*224的tensor
    outputs = vgg16(image)                       #将输入传入网络中
    _, preds = torch.max(outputs.data, 1)        #获得输出中最大值
    return class_names[preds.item()]             #返回标签


