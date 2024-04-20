import torch
from torch import nn
import torchvision
import os
from torch.utils.data import Dataset
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class dataset(Dataset):
    def __init__(self, root_dir = "./data/flower_photos", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []  # 存储所有图像的路径以及它们的标签

        # 遍历所有文件夹，收集图像路径和它们的标签
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        # self.samples = random.sample(self.samples, len(self.samples) // 10) #测试用

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)
        return image, label
    

class dataset_v2(Dataset):
    def __init__(self, data = None, transform=None):
        self.transform = transform
        self.samples = data  # 存储所有图像的路径以及它们的标签
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)
        return image, label
    

def get_dataset(transform=None):
    root_dir = "./data/flower_photos"
    classes = os.listdir(root_dir)
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    train = []
    val = []
    for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            samples = []
            for img_name in os.listdir(class_dir):
                samples.append((os.path.join(class_dir, img_name), class_to_idx[class_name]))
            # samples = random.sample(samples, len(samples) // 10) #测试用
            print(len(samples))
            train.extend(samples[:int(len(samples) * 0.9)])
            val.extend(samples[int(len(samples) * 0.9):])
    return dataset_v2(train, transform["train"]), dataset_v2(val, transform["val"])

if __name__=="__main__":
    print(len(dataset()))
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
    'val': transforms.Compose([transforms.Resize(448),
                                 transforms.CenterCrop(300),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值和标准差和训练集相同
                                ]),
    }
    train_dataset, val_dataset = get_dataset(transfer)
    print(len(train_dataset), len(val_dataset))
    for x,y in val_dataset:
        print(x.shape, y)
        break