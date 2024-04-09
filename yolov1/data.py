from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torchvision
from utils import image

torchvision.datasets.VOCDetection()
 
class VOC_Detection_Set(Dataset):
    def __init__(self, imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",
                 annotations_path="../DataSet/VOC2007+2012/Train/Annotations",
                 classes_file="../DataSet/VOC2007+2012/class.data", is_train = True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64, loss_mode="mse"):  # input_size:输入图像的尺度
        self.label_smooth_value = label_smooth_value
        self.class_num = class_num
        self.imgs_name = os.listdir(imgs_path)
        self.input_size = input_size
        self.grid_size = grid_size
        self.is_train = is_train
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.class_dict = {}
        self.loss_mode = loss_mode
 
        class_index = 0
        with open(classes_file, 'r') as file:
            for class_name in file:
                class_name = class_name.replace('\n', '')
                self.class_dict[class_name] = class_index  # 根据类别名制作索引
                class_index = class_index + 1
 
    def __getitem__(self, item):
 
        img_path = os.path.join(self.imgs_path, self.imgs_name[item])
        annotation_path = os.path.join(self.annotations_path, self.imgs_name[item].replace(".jpg", ".xml"))
        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()
 
        objects_xml = annotation_xml.findall("object")
        coords = []
 
        for object_xml in objects_xml:
            bnd_xml = object_xml.find("bndbox")
            class_name = object_xml.find("name").text
            if class_name not in self.class_dict:  # 不属于我们规定的类
                continue
            xmin = round((float)(bnd_xml.find("xmin").text))
            ymin = round((float)(bnd_xml.find("ymin").text))
            xmax = round((float)(bnd_xml.find("xmax").text))
            ymax = round((float)(bnd_xml.find("ymax").text))
            class_id = self.class_dict[class_name]
            coords.append([xmin, ymin, xmax, ymax, class_id])
 
        coords.sort(key=lambda coord : (coord[2] - coord[0]) * (coord[3] - coord[1]) )
 
        if self.is_train:
 
            transform_seed = random.randint(0, 4)
 
            if transform_seed == 0:  # 原图
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)
 
            elif transform_seed == 1:  # 缩放+中心裁剪
                img, coords = image.center_crop_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)
 
            elif transform_seed == 2:  # 平移
                img, coords = image.transplant_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)
 
            else:  # 曝光度调整
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = image.exposure(img, gamma=0.5)
                img = self.transform_common(img)
 
        else:
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)
 
        ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)
        return img, [ground_truth, ground_mask_positive, ground_mask_negative, img_path]
 
        #ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)
 
        # 通道变化方法: img = img[:, :, ::-1]
 
        #return img, ground_truth, ground_mask_positive, ground_mask_negative
 
    def __len__(self):
        return len(self.imgs_name)
 
    def getGroundTruth(self, coords):
 
        feature_size = self.input_size // self.grid_size
        #ground_mask_positive = np.zeros([feature_size, feature_size, 1], dtype=bool)
        #ground_mask_negative = np.ones([feature_size, feature_size, 1], dtype=bool)
        ground_mask_positive = np.full(shape=(feature_size, feature_size, 1), fill_value=False, dtype=bool)
        ground_mask_negative = np.full(shape=(feature_size, feature_size, 1), fill_value=True, dtype=bool)
 
        if self.loss_mode == "mse":
            ground_truth = np.zeros([feature_size, feature_size, 10 + self.class_num + 2])
        else:
            ground_truth = np.zeros([feature_size, feature_size, 10 + 1])
 
        for coord in coords:
 
            xmin, ymin, xmax, ymax, class_id = coord
 
            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)
 
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
 
            index_row = (int)(center_y * feature_size)
            index_col = (int)(center_x * feature_size)
 
            # 分类标签 label_smooth
            if self.loss_mode == "mse":
                # 转化为one_hot编码 对one_hot编码做平滑处理
                class_list = np.full(shape=self.class_num, fill_value=1.0, dtype=float)
                deta = 0.01
                class_list = class_list * deta / (self.class_num - 1)
                class_list[class_id] = 1.0 - deta
            elif self.loss_mode == "cross_entropy":
                class_list = [class_id]
            else:
                raise Exception("the loss mode can't be support now!")
 
            # 定位数据预设
            ground_box = [center_x * feature_size - index_col, center_y * feature_size - index_row,
                          ground_width, ground_height, 1,
                          round(xmin * self.input_size), round(ymin * self.input_size),
                          round(xmax * self.input_size), round(ymax * self.input_size),
                          round(ground_width * self.input_size * ground_height * self.input_size)
                          ]
            ground_box.extend(class_list)
            ground_box.extend([index_col, index_row])
 
            ground_truth[index_row][index_col] = np.array(ground_box)
            ground_mask_positive[index_row][index_col] = True
            ground_mask_negative[index_row][index_col] = False
 
        return ground_truth, torch.BoolTensor(ground_mask_positive), torch.BoolTensor(ground_mask_negative)