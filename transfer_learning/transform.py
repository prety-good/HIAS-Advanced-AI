import torch
from torch import nn
import torch.utils
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

transfer = {
    'train': transforms.Compose([transforms.RandomRotation(45), # 随机旋转 -45度到45度之间
                                 transforms.CenterCrop(300), # 从中心处开始裁剪
                                 transforms.RandomHorizontalFlip(p = 0.5), # 随机水平翻转
                                 transforms.RandomVerticalFlip(p = 0.5), # 随机垂直翻转
                                 # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色调
                                 transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),
                                 transforms.RandomGrayscale(p = 0.025), # 概率转换为灰度图，三通道RGB
                                 # 灰度图转换以后也是三个通道，但是只是RGB是一样的
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值，标准差
                                ]),
    # resize成256 * 256 再选取 中心 224 * 224，然后转化为向量，最后正则化
    'val': transforms.Compose([  
                                 transforms.CenterCrop(300),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值和标准差和训练集相同
                                ]),
    }
img = Image.open("./data/flower_photos\sunflowers/6953297_8576bf4ea3.jpg").convert('RGB')

# 创建一个图形窗口，并分配两个子图
plt.figure(figsize=(10, 5))  # 设置整个图形窗口的大小

# 绘制第一张图片
plt.subplot(1, 2, 1)  # 参数意义：(行数, 列数, 子图编号)
plt.imshow(img)
plt.title('Image 1')  # 为第一张图片设置标题
plt.axis('off')  # 不显示坐标轴

y = transfer['train'](img) # y为tensor
img = np.transpose(y.numpy(), (1, 2, 0))

# 绘制第二张图片
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title('Image 2')  # 为第二张图片设置标题
plt.axis('off')

plt.show()  # 显示图形
