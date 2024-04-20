# coding: utf-8
import os
import torch
import cv2
import torchvision.transforms as transforms
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from model import YOLOv1
from tqdm import tqdm
from utils.image import resize_image_without_annotation

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def is_jpg(file_name):
    return file_name.find(".jpg") != -1

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

def iou_Para(max_conf_box, predict_boxes):

    LX = torch.max(max_conf_box[0], predict_boxes[:,0])
    LY = torch.max(max_conf_box[1], predict_boxes[:,1])
    RX = torch.min(max_conf_box[2], predict_boxes[:,2])
    RY = torch.min(max_conf_box[3], predict_boxes[:,3])

    max_conf_box_area = (max_conf_box[2] - max_conf_box[0]) * (max_conf_box[3] - max_conf_box[1])
    predict_boxes_area = (predict_boxes[:, 2] - predict_boxes[:, 0]) * (predict_boxes[:, 3] - predict_boxes[:, 1])
    zeros_tensor = torch.zeros_like(predict_boxes_area).to(device=predict_boxes_area.device)
    iou_tensor = torch.where((LX >= RX) | (LY >= RY), zeros_tensor, (RX - LX) * (RY - LY) / (max_conf_box_area + predict_boxes_area)).view(-1, 1)

    return iou_tensor

def NMS_Para(detection_result,grid_idx_martix,S=7,B=2,cls_num=20,img_size=448,conf_threshold=0.4,iou_threshold=0.8):
    device = detection_result.device
    grid_size = img_size // S # 64
    # get positive sample
    # 得到两个框的绝对的x1 y1
    detection_result[..., 0:2] = (detection_result[..., 0:2] + grid_idx_martix) * grid_size
    detection_result[..., 5:7] = (detection_result[..., 5:7] + grid_idx_martix) * grid_size
    # 得到绝对的 width 和 height
    detection_result[..., 2:4] = detection_result[..., 2:4] * img_size
    detection_result[..., 7:9] = detection_result[..., 7:9] * img_size
    # 找置信度大于0.5的框
    pos_girds_mark = (detection_result[..., 4] > conf_threshold) | (detection_result[..., 9] > conf_threshold)

    if pos_girds_mark.int().sum() == 0:# cannot detect box
        return []

    pos_girds_mark = pos_girds_mark.unsqueeze(3)
    # 过滤置信度大于0.5的所有框
    pos_grids = torch.masked_select(detection_result, pos_girds_mark).view(-1, B * 5 + cls_num)

    pos_girds_first_box = pos_grids[:, 0:5]
    pos_grids_second_box = pos_grids[:, 5:10]
    #同一个位置的2个框 找到置信度大的
    sample_boxes = torch.where((pos_girds_first_box[:, 4] > pos_grids_second_box[:, 4]).unsqueeze(1), pos_girds_first_box, pos_grids_second_box).view(-1, 5)

    zero_tensors = torch.zeros_like(sample_boxes[:,0]).to(device=device)
    imgsize_tensors = torch.ones_like(sample_boxes[:,0]).to(device=device)

    predict_boxes = torch.cat([
        torch.max(zero_tensors, sample_boxes[:,0] - sample_boxes[:,2] / 2).unsqueeze(1),
        torch.max(zero_tensors, sample_boxes[:,1] - sample_boxes[:,3] / 2).unsqueeze(1),
        torch.max(imgsize_tensors, sample_boxes[:,0] + sample_boxes[:,2] / 2).unsqueeze(1),
        torch.max(imgsize_tensors, sample_boxes[:,1] + sample_boxes[:,3] / 2).unsqueeze(1),
        sample_boxes[:, 4].unsqueeze(1),
        pos_grids[:, B * 5:]
    ],dim=1)

    assured_boxes = []
    # box overlap filter
    while len(predict_boxes) != 0:
        max_conf_box_idx = torch.argmax(predict_boxes[:, 4], dim=0)
        max_conf_box = predict_boxes[max_conf_box_idx].clone()
        boxes_iou = iou_Para(max_conf_box, predict_boxes)
        boxes_iou[max_conf_box_idx] = 1
        # 保留IOU低于iou_threshold的边界框
        rest_box_mark = (boxes_iou < iou_threshold).bool().to(device=device)
        predict_boxes = torch.masked_select(predict_boxes, rest_box_mark).view(-1, 5 + cls_num)

        assured_box_cls_idx = torch.argmax(max_conf_box[5:])
        assured_boxes.append([int(max_conf_box[0]), int(max_conf_box[1]), int(max_conf_box[2]), int(max_conf_box[3]), float(max_conf_box[4] * max_conf_box[5 + assured_box_cls_idx]), int(assured_box_cls_idx)])

    return assured_boxes

def NMS(detection_result,S=7,B=2,img_size=448,confidence_threshold=0.5,iou_threshold=0.8):
    bounding_boxes = detection_result.cpu().detach().numpy().tolist()
    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * j
                gridY = grid_size * i
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                bounding_box.extend(bounding_boxes[batch][i][j][10:])
                if bounding_box[4] >= confidence_threshold:
                    centerX = (int)(gridX + bounding_box[0] * grid_size)
                    centerY = (int)(gridY + bounding_box[1] * grid_size)
                    width = (int)(bounding_box[2] * img_size)
                    height = (int)(bounding_box[3] * img_size)
                    bounding_box[0] = max(0, (int)(centerX - width / 2))
                    bounding_box[1] = max(0, (int)(centerY - height / 2))
                    bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                    bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))
                    print("batch_idx:{} conf:{}".format(batch, bounding_box[4]))
                    predict_boxes.append(bounding_box)

        while len(predict_boxes) != 0:
            predict_boxes.sort(key=lambda box:box[4])
            assured_box = predict_boxes[0]
            temp = []
            classIndex = np.argmax(assured_box[5:])
            #assured_box[4] = assured_box[4] * assured_box[5 + classIndex] #修正置信度为 物体分类准确度 × 含有物体的置信度
            assured_box[5] = classIndex
            nms_boxes.append(assured_box)
            i = 1
            while i < len(predict_boxes):
                if iou(assured_box,predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp
    return nms_boxes

def inverse_transform_bbox(predicted_bbox, orig_shape, borders):
    orig_h, orig_w = orig_shape[:2]
    processed_h = processed_w = 448
    top, bottom, left, right = borders

    # 1. 获得原始图片的最长边 l
    longest_edge = max(orig_h, orig_w)
    # 2. 计算 resize 为 l*l 大小时的缩放比例
    scale = longest_edge / processed_h  # 因为 processed_h == processed_w
    
    # 3. 初始化逆变换后的边界框列表
    inverse_transformed_bboxes = []
    
    for box in predicted_bbox:
        # 解包预测的边界框坐标
        x1, y1, x2, y2, cls, conf = box
        
        # 3. 根据正变换中添加的边界调整坐标
        if top > 0:  # 添加了上下边界
            y1 = round(max(0, (y1 * scale) - top))
            y2 = round(min(orig_h, (y2 * scale) - top))
        else:  # 添加了左右边界
            x1 = round(max(0, (x1 * scale) - left))
            x2 = round(min(orig_w, (x2 * scale) - left))
        
        # 添加调整后的边界框到列表中
        inverse_transformed_bboxes.append([x1, y1, x2, y2, cls, conf])
    
    return inverse_transformed_bboxes


@torch.no_grad()
def detection_img(img_res, yolo, grid_idx_martix, img_size=448):
    img,  borders = resize_image_without_annotation(img_res, img_size, img_size)
    img_tensor = transform(img).unsqueeze(0).to(device=device)
    with torch.no_grad():
        detection_result = yolo(img_tensor)
        nms_boxes = NMS_Para(detection_result, grid_idx_martix)

        if debug:
            print("###########################")
            for box in nms_boxes:
                img = cv2.rectangle(img=img, pt1=(round(box[0]), round(box[1])), pt2=(round(box[2]), round(box[3])), color=(0, 255, 0), thickness=1)
                img = cv2.putText(img, "{} {}".format(classes_name[box[5]], round(box[4], 2)), (round(box[0]), round(box[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            print(box)
            cv2.imshow("detection", img)
            cv2.waitKey()

        nms_boxes = inverse_transform_bbox(nms_boxes, img_res.shape, borders)

        if debug:
            for box in nms_boxes:
                img_res = cv2.rectangle(img_res, pt1=(round(box[0]), round(box[1])), pt2=(round(box[2]), round(box[3])), color=(0, 255, 0), thickness=1)
                img_res = cv2.putText(img_res, "{} {}".format(classes_name[box[5]], round(box[4], 2)), (round(box[0]), round(box[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                print(box)
            cv2.imshow("detection", img_res)
            cv2.waitKey()
        
    return nms_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detection config")
    parser.add_argument("--path", type=str, default="./data/VOC2012_test/ImageSets/Main/test.txt")
    parser.add_argument("--wait_time", type=int, default=1000)
    parser.add_argument("--weight_path", type=str, default="./weights/YOLOv1_90.pth")
    # parser.add_argument("--weight_path", type=str, default="./weights/first/YOLOv1_40.pth")
    parser.add_argument("--classes_path", type=str, default="./data/class.data")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    debug = False
    output_dir = "./detect"
    path = args.path
    wait_time = args.wait_time
    weight_path = args.weight_path

    model_weight = torch.load(weight_path)["model"]

    classes_path = args.classes_path
    classes_name = {}
    class_index = 0
    with open(classes_path, "r") as f:
        for class_name in f:
            class_name = class_name.replace("\n", "")
            classes_name[class_index] = class_name
            class_index = class_index + 1

    transform = transforms.Compose([
        transforms.ToTensor(),  # height * width * channel -> channel * height * width
        transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
    ])
    img_size = args.img_size
    feature_map_side_len = img_size // 64
    grid_idx_martix = torch.ones(size=(feature_map_side_len, feature_map_side_len, 2))
    for row_idx in range(feature_map_side_len):#row => y
        for col_idx in range(feature_map_side_len):#col => x
            # 第i行j列的数据为 j,i
            grid_idx_martix[row_idx][col_idx] = torch.Tensor([col_idx, row_idx])
    grid_idx_martix = grid_idx_martix.to(device=device)

    yolo = YOLOv1().to(device).eval()
    yolo.load_state_dict(model_weight)

    imgs_path = []
    with open(path, 'r') as file:
        for line in file:
        # 使用strip()移除每行末尾的换行符'\n'
            imgs_path.append(line.strip())

    for img_path in tqdm(imgs_path, total=len(imgs_path)):
        nms_boxes = detection_img(cv2.imread(os.path.join("./data/VOC2012_test/JPEGImages" ,f"{img_path}.jpg")), yolo, grid_idx_martix)
        for box in nms_boxes:
            x1, y1, x2, y2, confidence, class_id = box
            # if class_id == 17 or class_id == 16:
            #     print("################################")
            cls = classes_name[class_id]
            line = f"{img_path} {round(confidence, 6)} {x1} {y1} {x2} {y2}\n"
            # print(cls, line)
            file_path = f"{output_dir}/Main/comp3_det_test_{cls}.txt"
            with open(file_path, 'a') as file:  # 使用 'a' 模式以追加的方式打开文件
                file.write(line)


        # img = cv2.imread(os.path.join(path ,img_path))
        # annotation_path = os.path.join(path, img_path).replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        # tree = ET.parse(annotation_path)
        # annotation_xml = tree.getroot()
        # objects_xml = annotation_xml.findall("object")
        # coords = []
        # for object_xml in objects_xml:
        #     bnd_xml = object_xml.find("bndbox")
        #     class_name = object_xml.find("name").text
        #     xmin = round((float)(bnd_xml.find("xmin").text))
        #     ymin = round((float)(bnd_xml.find("ymin").text))
        #     xmax = round((float)(bnd_xml.find("xmax").text))
        #     ymax = round((float)(bnd_xml.find("ymax").text))
        #     coords.append([xmin, ymin, xmax, ymax, class_name])
        # for box in coords:
        #     img = cv2.rectangle(img, pt1=(round(box[0]), round(box[1])), pt2=(round(box[2]), round(box[3])), color=(0, 255, 0), thickness=1)
        #     img = cv2.putText(img, f"{box[4]}", (round(box[0]), round(box[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #     print(box)
        # cv2.imshow("detection", img)
        # cv2.waitKey()
    if debug:
        cv2.destroyAllWindows()




