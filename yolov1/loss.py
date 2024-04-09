import math
import torch
import torch.nn as nn
 
class YOLOv1_Loss(nn.Module):
 
    def __init__(self, S=7, B=2, Classes=20, l_coord=5, pos_conf=1, pos_cls=1, l_noobj=0.5):
        # 有物体的box损失权重设为l_coord,没有物体的box损失权重设置为l_noobj
        super(YOLOv1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.pos_conf = pos_conf
        self.pos_cls = pos_cls
        self.l_noobj = l_noobj
 
    def iou_force(self, bounding_box, ground_box, gridX, gridY, img_size=448, grid_size=64):  # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
        # [xmin,ymin,xmax,ymax]
 
        predict_box = list([0, 0, 0, 0])
        predict_box[0] = (int)(gridX + bounding_box[0] * grid_size)
        predict_box[1] = (int)(gridY + bounding_box[1] * grid_size)
        predict_box[2] = (int)(bounding_box[2] * img_size)
        predict_box[3] = (int)(bounding_box[3] * img_size)
 
        predict_coord = list([max(0, predict_box[0] - predict_box[2] / 2),
                              max(0, predict_box[1] - predict_box[3] / 2),
                              min(img_size - 1, predict_box[0] + predict_box[2] / 2),
                              min(img_size - 1, predict_box[1] + predict_box[3] / 2)])
 
        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])
 
        ground_coord = list([ground_box[5].item(), ground_box[6].item(), ground_box[7].item(), ground_box[8].item()])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])
 
        # 存储格式 xmin ymin xmax ymax
 
        # 2.计算交集的面积 左边的大者 右边的小者 上边的大者 下边的小者
        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])
 
        if CrossRX < CrossLX or CrossDY < CrossUY:  # 没有交集
            return 0
 
        interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)
 
        return interSection / (predict_Area + ground_Area - interSection)
 
    def iou(self, bounding_boxes, ground_boxes, img_size=448, grid_size=64, device=torch.device("cuda:0")):  # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [xmin,ymin,xmax,ymax,gridX, gridY]
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
 
        gridX = ground_boxes[:,4]
        gridY = ground_boxes[:,5]
        # [center_x, center_y, width, height]
 
        center_x = ((gridX + bounding_boxes[:,0]) * grid_size).unsqueeze(1).int()
        center_y = ((gridY + bounding_boxes[:,1]) * grid_size).unsqueeze(1).int()
        width = (bounding_boxes[:,2] * img_size).unsqueeze(1).int()
        height = (bounding_boxes[:,3] * img_size).unsqueeze(1).int()
 
        predict_boxes = torch.cat([center_x, center_y, width, height], dim=1)
        # [xmin,ymin,xmax,ymax]
 
        predict_coords = torch.cat([torch.max(torch.Tensor([0]).to(device=device), predict_boxes[:,0] - predict_boxes[:,2] / 2).unsqueeze(1),
                                    torch.max(torch.Tensor([0]).to(device=device), predict_boxes[:,1] - predict_boxes[:,3] / 2).unsqueeze(1),
                                    torch.min(torch.Tensor([img_size - 1]).to(device=device), predict_boxes[:,0] + predict_boxes[:,2] / 2).unsqueeze(1),
                                    torch.min(torch.Tensor([img_size - 1]).to(device=device), predict_boxes[:,1] + predict_boxes[:,3] / 2).unsqueeze(1)], dim=1)
 
        predict_areas = (predict_coords[:,2] - predict_coords[:,0]) * (predict_coords[:,3] - predict_coords[:,1])
        ground_area = (ground_boxes[:,2] - ground_boxes[:,0]) * (ground_boxes[:,3] - ground_boxes[:,1])
 
        # 2.计算交集的面积 左边的大者 右边的小者 上边的大者 下边的小者
        cross_lx = torch.max(predict_coords[:,0], ground_boxes[:,0])
        cross_rx = torch.min(predict_coords[:,2], ground_boxes[:,2])
        cross_uy = torch.max(predict_coords[:,1], ground_boxes[:,1])
        cross_dy = torch.min(predict_coords[:,3], ground_boxes[:,3])
 
        inter_section = torch.where((cross_rx < cross_lx) | (cross_dy < cross_uy), 0, ((cross_rx - cross_lx) * (cross_dy - cross_uy)).long())
 
        return inter_section / (predict_areas + ground_area - inter_section)
 
    def forward(self, bounding_boxes, ground_labels, grid_size=64, img_size=448, device=torch.device("cuda:0")):  # 输入是 S * S * ( 2 * B + Classes)
 
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        loss = 0
        loss_coord = 0
        loss_positive_conf = 0
        loss_negative_conf = 0
        loss_classes = 0
        mseLoss = nn.MSELoss(size_average=False)
        batch_size = len(bounding_boxes)
        #print("bs:{}".format(batch_size))
        # optimize backward
        ground_truth, ground_mask_positive, ground_mask_negative, img_path = ground_labels
        predict_positive_boxes = torch.masked_select(bounding_boxes, ground_mask_positive).view(-1, 10 + self.Classes)
        predict_negative_boxes = torch.masked_select(bounding_boxes, ground_mask_negative).view(-1, 10 + self.Classes)
        ground_positive_boxes = torch.masked_select(ground_truth, ground_mask_positive).view(-1, 10 + self.Classes + 2)
        #print("pos mask{} neg mask{}".format(ground_mask_positive, ground_mask_negative))
 
        # positive samples
        predict_boxes_one = predict_positive_boxes[:, 0:5]
        predict_boxes_two = predict_positive_boxes[:, 5:10]
        #print("predict_boxes_one:{} predict_boxes_two:{}".format(predict_boxes_one, predict_boxes_two))
        ground_boxes = torch.cat([ground_positive_boxes[:,5:9], ground_positive_boxes[:,self.B * 5 + self.Classes:]], dim=1)
        boxes_one_iou = self.iou(predict_boxes_one, ground_boxes)
        boxes_two_iou = self.iou(predict_boxes_two, ground_boxes)
 
        #print("one:{} two:{}".format(boxes_one_iou, boxes_two_iou))
        positive_location = torch.where(boxes_one_iou > boxes_two_iou, 0, 1)
        positive_iou = torch.where(boxes_one_iou > boxes_two_iou, boxes_one_iou, boxes_two_iou)
        iou_sum = positive_iou.sum()
        #print("loc:{}".format(positive_location))
 
        object_num = len(positive_location)
        grid_positive_mask = torch.zeros(size=(object_num, 10)).to(device=device)
        grid_negative_mask = torch.ones(size=(object_num, 10)).to(device=device)
 
        # 分类
        ground_class = ground_positive_boxes[:, self.B * 5: self.B * 5 + self.Classes]
        predict_class = predict_positive_boxes[:, self.B * 5: self.B * 5 + self.Classes]
        # print("ground_class:{} predict_class:{}".format(ground_class, predict_class))
        # classes = self.pos_cls * torch.pow(ground_class - predict_class, 2).sum()
        # loss = loss + classes
        # loss_classes += classes.item()
        loss_class = self.pos_cls * mseLoss(ground_class, predict_class) / batch_size
        loss = loss + loss_class
 
        for location_idx in range(object_num):
            if positive_location[location_idx] == 0:
                grid_positive_mask[location_idx][0:5] = torch.ones(size=(5,))
                grid_negative_mask[location_idx][0:5] = torch.zeros(size=(5,))
            else:
                grid_positive_mask[location_idx][5:10] = torch.ones(size=(5,))
                grid_negative_mask[location_idx][5:10] = torch.zeros(size=(5,))
 
        predict_grid_positive_box = torch.masked_select(predict_positive_boxes[:,0:10], grid_positive_mask.bool()).view(-1, 5)
        predict_grid_negative_box = torch.masked_select(predict_positive_boxes[:,0:10], grid_negative_mask.bool()).view(-1, 5)
 
        # 正样本：
        # 定位
        loss_coord = self.l_coord * (mseLoss(predict_grid_positive_box[:,0:2], ground_positive_boxes[:,0:2]) + mseLoss(torch.sqrt(predict_grid_positive_box[:,2:4] + 1e-8), torch.sqrt(ground_positive_boxes[:,2:4] + 1e-8))) / batch_size
        loss = loss + loss_coord
        #coord = self.l_coord * torch.pow(predict_grid_positive_box[:,0:2] - ground_positive_boxes[:,0:2], 2).sum() / batch_size \
        #        + self.l_coord * torch.pow(torch.sqrt(predict_grid_positive_box[:,2:4]) - torch.sqrt(ground_positive_boxes[:, 2:4]), 2).sum() / batch_size
        #loss = loss + coord
        #loss_coord += coord.item()
        # positive 置信度
        loss_positive_conf = self.pos_conf * mseLoss(predict_grid_positive_box[:,4], torch.Tensor([1]).to(device=device)) / batch_size
        loss = loss + loss_positive_conf
        #positive_conf = self.pos_conf * torch.pow(predict_grid_positive_box[:, 4] - torch.Tensor([1]).to(device=device), 2).sum() / batch_size
        #loss = loss + positive_conf
        #loss_positive_conf += positive_conf.item()
        # negative 置信度
        predict_negative_boxes = torch.cat([predict_negative_boxes[:,0:5], predict_negative_boxes[:,5:10], predict_grid_negative_box], dim=0)
        loss_negative_conf = self.l_noobj * mseLoss(predict_negative_boxes[:,4], torch.Tensor([0]).to(device=device)) / batch_size
        loss = loss + loss_negative_conf
        #negative_conf = self.l_noobj * torch.pow(predict_negative_boxes[:,4] - torch.Tensor([0]).to(device=device), 2).sum() / batch_size
        #loss = loss + negative_conf
        #loss_negative_conf += negative_conf.item()
        #print("loss:{} loss_coord:{} loss_positive_conf:{} loss_negative_conf:{} loss_classes:{} iou_sum.item():{} object_num:{}".format(loss, loss_coord, loss_positive_conf, loss_negative_conf, loss_classes, iou_sum.item(), object_num))
 
        return loss, loss_coord.item(), loss_positive_conf.item(), loss_negative_conf.item(), loss_class.item(), iou_sum.item(), object_num