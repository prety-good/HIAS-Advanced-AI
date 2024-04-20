import os
import torch
from torch import nn
import argparse

import torch.utils
import torch.utils.data
from utils.model import feature_map_visualize
from utils import model
from data import VOC_Detection_Set
from torch.utils.data import DataLoader
from model import YOLOv1
from loss import YOLOv1_Loss
import torchvision
import cv2
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from torchvision import transforms
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_image_with_coords(image):
    # letterBox
    height = width = 448
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        scale_factor = scale_w
    else:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        scale_factor = scale_h
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
    return constant

if __name__=="__main__":
    transform_common = transforms.Compose([
        transforms.ToTensor(),  # height * width * channel -> channel * height * width
        transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
    ])

    model = YOLOv1().to(device)
    param_dict = torch.load("./weights/YOLOv1_40.pth", map_location=torch.device("cpu"))['model']
    model.load_state_dict(param_dict)

    imagefolder = "./data/VOC2012_test/JPEGImages"
    with torch.no_grad():
        for path in os.listdir(imagefolder):
            image = cv2.imread(os.path.join(imagefolder, path))
            image = resize_image_with_coords(image)
            image = transform_common(image)
            image = image.unsqueeze(dim=0).to(device)
            y_hat = model(image)
            print(y_hat)
            break


