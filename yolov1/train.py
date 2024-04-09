#---------------step0:Common Definition-----------------
import os
import torch
import argparse
from utils.model import feature_map_visualize
from data import VOC_Detection_Set, voc_dataloader, voc_prefetcher
from torch.utils.data import DataLoader
from model import YOLOv1
from loss import YOLOv1_Loss
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
 
from prefetch_generator import BackgroundGenerator
 
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")
 
if __name__ == "__main__":
#def train():
#../../DataSet/VOC2007+2012/Train/JPEGImages/2008_002118.jpg
    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLOv1 train config")
    parser.add_argument('--num_workers', type=int, help="train num_workers num", default=4)
    parser.add_argument('--B', type=int, help="YOLOv1 predict box num every grid", default=2)
    parser.add_argument('--class_num', type=int, help="YOLOv1 predict class num", default=20)
    parser.add_argument('--lr', type=float, help="start lr", default=1e-3)
    parser.add_argument('--lr_mul_factor_epoch_1', type=float, help="lr mul factor when full YOLOv1 train in epoch1", default=1.04)
    parser.add_argument('--lr_epoch_2', type=int, help="lr when full YOLOv1 train in epoch2", default=0.001)
    parser.add_argument('--lr_epoch_77', type=int, help="lr when full YOLOv1 train in epoch77", default=0.0001)
    parser.add_argument('--lr_epoch_107', type=int, help="lr when full YOLOv1 train in epoch107", default=0.00001)
    parser.add_argument('--batch_size', type=int, help="YOLOv1 train batch size", default=32)
    parser.add_argument('--train_imgs', type=str, help="YOLOv1 train train_imgs", default="../../DataSet/VOC2007/Train/JPEGImages")
    parser.add_argument('--train_labels', type=str, help="YOLOv1 train train_labels", default="../../DataSet/VOC2007/Train/Annotations")
    parser.add_argument('--val_imgs', type=str, help="YOLOv1 train val_imgs", default="../../DataSet/VOC2007/Val/JPEGImages")
    parser.add_argument('--val_labels', type=str, help="YOLOv1 train val_labels", default="../../DataSet/VOC2007/Val/Annotations")
    parser.add_argument('--voc_classes_path', type=str, help="voc classes path", default="../../DataSet/VOC2007/class.data")
    parser.add_argument('--weight_decay', type=float, help="optim weight_decay", default=5e-4)
    parser.add_argument('--momentum', type=float, help="optim momentum", default=0.9)
    parser.add_argument('--pre_weight_file', type=str, help="YOLOv1 BackBone pre-train path", default="../PreTrain/weights/YOLO_Feature_150.pth")
    parser.add_argument('--epoch_interval', type=int, help="save YOLOv1 weight epoch interval", default=10)
    parser.add_argument('--epoch_unfreeze', type=int, help="YOLOv1 backbone unfreeze epoch", default=10)
    parser.add_argument('--epoch_num', type=int, help="YOLOv1 train epoch num", default=200)
    parser.add_argument('--grad_visualize', type=bool, help="YOLOv1 train grad visualize", default=True)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLOv1 train feature map visualize", default=False)
    parser.add_argument('--restart', type=bool, default=True)
    parser.add_argument('--weight_file', type=str, help="YOLOv1 weight path", default="./weights/YOLO_V1_110.pth")
    args = parser.parse_args()
 
    num_workers = args.num_workers
    class_num = args.class_num
    batch_size = args.batch_size
    lr_mul_factor_epoch_1 = args.lr_mul_factor_epoch_1
    lr_epoch_2 = args.lr_epoch_2
    lr_epoch_77 = args.lr_epoch_77
    lr_epoch_107 = args.lr_epoch_107
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    momentum = args.momentum
    epoch_interval = args.epoch_interval
    epoch_unfreeze = args.epoch_unfreeze
    loss_mode = "mse"
 
    if args.restart == True:
        pre_weight_file = args.pre_weight_file
        pre_param_dict = torch.load(pre_weight_file, map_location=torch.device("cpu"))
        lr = args.lr
        param_dict = {}
        epoch = 0
        epoch_val_loss_min = 999999999
 
    else:
        weight_file = args.weight_file
        param_dict = torch.load(weight_file, map_location=torch.device("cpu"))
        epoch = param_dict['epoch']
        epoch_val_loss_min = param_dict['epoch_val_loss_min']
 
    # 2.dataset
    train_dataSet = VOC_Detection_Set(imgs_path=args.train_imgs,
                                      annotations_path=args.train_labels,
                                      classes_file=args.voc_classes_path, class_num=class_num, is_train=True, loss_mode=loss_mode)
    val_dataSet = VOC_Detection_Set(imgs_path=args.val_imgs,
                                    annotations_path=args.val_labels,
                                    classes_file=args.voc_classes_path, class_num=class_num, is_train=False, loss_mode=loss_mode)
 
    # 3-4.network + optimizer
    YOLO = YOLOv1().to(device=device, non_blocking=True)
    if args.restart == True:
        YOLO.initialize_weights(pre_param_dict["min_loss_model"]) #load darknet pretrain weight
        optimizer_SGD = optim.SGD(YOLO.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        optimal_dict = {}
    else:
        YOLO.load_state_dict(param_dict['model']) #load yolov1 train weight
        optimizer_SGD = param_dict['optim']
        optimal_dict = param_dict['optimal']
    if epoch < epoch_unfreeze:
        model.set_freeze_by_idxs(YOLO, [0, 1])
 
    # 5.loss
    loss_function = YOLOv1_Loss().to(device=device, non_blocking=True)
 
    # 6.train and record
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
 
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
 
    while epoch < args.epoch_num:
 
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_iou = 0
        epoch_val_iou = 0
        epoch_train_object_num = 0
        epoch_val_object_num = 0
        epoch_train_loss_coord = 0
        epoch_val_loss_coord = 0
        epoch_train_loss_pos_conf = 0
        epoch_train_loss_neg_conf = 0
        epoch_val_loss_pos_conf = 0
        epoch_val_loss_neg_conf = 0
        epoch_train_loss_classes = 0
        epoch_val_loss_classes = 0
 
        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:
 
            #voc_train_loader = voc_dataloader(train_loader)
 
            #for batch_index, batch_train in BackgroundGenerator(train_loader):
            for batch_idx, [train_data, label_data] in enumerate(train_loader):
                optimizer_SGD.zero_grad()
                train_data = train_data.float().to(device=device, non_blocking=True)
                label_data[0] = label_data[0].float().to(device=device, non_blocking=True)
                label_data[1] = label_data[1].to(device=device, non_blocking=True)
                label_data[2] = label_data[2].to(device=device, non_blocking=True)
 
                loss = loss_function(bounding_boxes=YOLO(train_data), ground_labels=label_data)
                sample_avg_loss = loss[0]
                epoch_train_loss_coord = epoch_train_loss_coord + loss[1] * batch_size
                epoch_train_loss_pos_conf = epoch_train_loss_pos_conf + loss[2] * batch_size
                epoch_train_loss_neg_conf = epoch_train_loss_neg_conf + loss[3] * batch_size
                epoch_train_loss_classes = epoch_train_loss_classes + loss[4] * batch_size
                epoch_train_iou = epoch_train_iou + loss[5]
                epoch_train_object_num = epoch_train_object_num + loss[6]
 
                sample_avg_loss.backward()
                optimizer_SGD.step()
 
                batch_loss = sample_avg_loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss
 
                tbar.set_description(
                    "train: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} avg_iou:{}".format(
                        round(loss[1], 4),
                        round(loss[2], 4),
                        round(loss[3], 4),
                        round(loss[4], 4),
                        round(loss[5] / loss[6], 4)), refresh=True)
                tbar.update(1)
 
                if epoch == epoch_unfreeze + 1:
                    lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)
                    for param_group in optimizer_SGD.param_groups:
                        param_group["lr"] = lr
 
            if args.feature_map_visualize:
                feature_map_visualize(train_data[0][0], writer, YOLO)
            # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
 
            print("train-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                round(epoch_train_loss / train_len, 4),
                round(epoch_train_loss_coord / train_len, 4),
                round(epoch_train_loss_pos_conf / train_len, 4),
                round(epoch_train_loss_neg_conf / train_len, 4),
                round(epoch_train_loss_classes / train_len, 4),
                round(epoch_train_iou / epoch_train_object_num, 4)))
 
 
        val_len = val_loader.__len__()
        YOLO.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
 
                #voc_val_loader = voc_dataloader(val_loader)
                '''
                val_prefetcher = voc_prefetcher(val_loader, device)
                val_data, label_data = val_prefetcher.next()
                while val_data is not None:
                    loss = loss_function(bounding_boxes=YOLO(val_data), ground_labels=label_data)
                    sample_avg_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1] * batch_size
                    epoch_val_loss_positive_confidence = epoch_val_loss_positive_confidence + loss[2] * batch_size
                    epoch_val_loss_negative_confidence = epoch_val_loss_negative_confidence + loss[3] * batch_size
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[4] * batch_size
                    epoch_val_iou = epoch_val_iou + loss[5]
                    epoch_val_object_num = epoch_val_object_num + loss[6]
                    batch_loss = sample_avg_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss
                    tbar.set_description(
                        "val: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4),
                                                                                            round(loss[2], 4),
                                                                                            round(loss[3], 4),
                                                                                            round(loss[4], 4),
                                                                                            round(loss[5] / loss[6],
                                                                                                  4)), refresh=True)
                    tbar.update(1)
                    val_data, label_data = val_prefetcher.next()
                '''
                #for batch_index, batch_train in BackgroundGenerator(val_loader):
                for batch_idx, [val_data, label_data] in enumerate(train_loader):
                    val_data = val_data.float().to(device=device, non_blocking=True)
                    label_data[0] = label_data[0].float().to(device=device, non_blocking=True)
                    label_data[1] = label_data[1].to(device=device, non_blocking=True)
                    label_data[2] = label_data[2].to(device=device, non_blocking=True)
                    loss = loss_function(bounding_boxes=YOLO(val_data), ground_labels=label_data)
                    sample_avg_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1] * batch_size
                    epoch_val_loss_pos_conf = epoch_val_loss_pos_conf + loss[2] * batch_size
                    epoch_val_loss_neg_conf = epoch_val_loss_neg_conf + loss[3] * batch_size
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[4] * batch_size
                    epoch_val_iou = epoch_val_iou + loss[5]
                    epoch_val_object_num = epoch_val_object_num + loss[6]
                    batch_loss = sample_avg_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss
 
                    tbar.set_description("val: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                        round(loss[1], 4),
                        round(loss[2], 4),
                        round(loss[3], 4),
                        round(loss[4], 4),
                        round(loss[5]/ loss[6], 4)), refresh=True)
                    tbar.update(1)
 
                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, YOLO)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print("val-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                round(epoch_val_loss / val_len, 4),
                round(epoch_val_loss_coord / val_len, 4),
                round(epoch_val_loss_pos_conf / val_len, 4),
                round(epoch_val_loss_neg_conf / val_len, 4),
                round(epoch_val_loss_classes / val_len, 4),
                round(epoch_val_iou / epoch_val_object_num, 4)))
 
        epoch = epoch + 1
        print("epoch : {} ; loss : {}".format(epoch, epoch_train_loss))
 
        if epoch == epoch_unfreeze:
            model.unfreeze_by_idxs(YOLO, [0, 1])
 
        if epoch == 2 + epoch_unfreeze:
            lr = lr_epoch_2
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 77 + epoch_unfreeze:
            lr = lr_epoch_77
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 107 + epoch_unfreeze:
            lr = lr_epoch_107
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
 
        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOLO.state_dict()
 
        if epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['optimizer'] = optimizer_SGD
            param_dict['epoch'] = epoch
            param_dict['optimal'] = optimal_dict
            param_dict['epoch_val_loss_min'] = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLOv1_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
 
        if args.grad_visualize:
            for i, (name, layer) in enumerate(YOLO.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)
        '''
        for name, layer in YOLO.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
        '''
 
        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
        writer.add_scalar('Train/Loss_pos_conf', epoch_train_loss_pos_conf, epoch)
        writer.add_scalar('Train/Loss_neg_conf', epoch_train_loss_neg_conf, epoch)
        writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
        writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)
 
        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
        writer.add_scalar('Val/Loss_coord', epoch_val_loss_coord, epoch)
        writer.add_scalar('Val/Loss_pos_conf', epoch_val_loss_pos_conf, epoch)
        writer.add_scalar('Val/Loss_neg_conf', epoch_val_loss_neg_conf, epoch)
        writer.add_scalar('Val/Loss_classes', epoch_val_loss_classes, epoch)
        writer.add_scalar('Val/Epoch_iou', epoch_val_iou / epoch_val_object_num, epoch)
 
    writer.close()