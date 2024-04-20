import os
import torch
import argparse
from utils.model import feature_map_visualize
from utils import model
from data import VOC_Detection_Set
from torch.utils.data import DataLoader
from model import YOLOv1
from loss import YOLOv1_Loss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
 
if __name__ == "__main__":
    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLOv1 train config")
    parser.add_argument('--B', type=int, help="YOLOv1 predict box num every grid", default=2)
    parser.add_argument('--class_num', type=int, help="YOLOv1 predict class num", default=20)
    parser.add_argument('--lr', type=float, help="start lr", default=5e-3)
    parser.add_argument('--batch_size', type=int, help="YOLOv1 train batch size", default=12)
    parser.add_argument('--train_imgs', type=str, help="YOLOv1 train train_imgs", default="")
    parser.add_argument('--train_labels', type=str, help="YOLOv1 train train_labels", default="")
    parser.add_argument('--val_imgs', type=str, help="YOLOv1 train val_imgs", default="./data/VOC2007_test/JPEGImages")
    parser.add_argument('--val_labels', type=str, help="YOLOv1 train val_labels", default="./data/VOC2012_test/Annotations")
    parser.add_argument('--voc_classes_path', type=str, help="voc classes path", default="./data/class.data")
    parser.add_argument('--momentum', type=float, help="optim momentum", default=0.9)
    parser.add_argument('--epoch_interval', type=int, help="save YOLOv1 weight epoch interval", default=10)
    parser.add_argument('--epoch_unfreeze', type=int, help="YOLOv1 backbone unfreeze epoch", default=0)
    parser.add_argument('--epoch_num', type=int, help="YOLOv1 train epoch num", default=100)
    parser.add_argument('--grad_visualize', type=bool, help="YOLOv1 train grad visualize", default=False)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLOv1 train feature map visualize", default=False)
    parser.add_argument('--restart', type=bool, default=True)
    parser.add_argument('--weight_file', type=str, help="YOLOv1 weight path", default="./weights/efficientnet/YOLOv1_50.pth")
    args = parser.parse_args()

    num_epochs = args.epoch_num
    class_num = args.class_num
    batch_size = args.batch_size
    batch_size = args.batch_size
    momentum = args.momentum
    epoch_interval = args.epoch_interval
    epoch_unfreeze = args.epoch_unfreeze
    learning_rate = args.lr
    loss_mode = "mse"
 

    YOLO = YOLOv1().to(device=device)

    if args.restart == True:
        param_dict = {}
        epoch = 0
        epoch_val_loss_min = 999999999
 
    else:
        weight_file = args.weight_file
        param_dict = torch.load(weight_file)
        epoch = param_dict['epoch']
        epoch_val_loss_min = param_dict['epoch_val_loss_min']
        YOLO.load_state_dict(param_dict['model'])
 
    # 2.dataset
    # train_dataSet = VOC_Detection_Set(imgs_path=args.train_imgs,
    #                                   annotations_path=args.train_labels,
    #                                   classes_file=args.voc_classes_path, class_num=class_num, is_train=True, loss_mode=loss_mode)
    train_dataSet = VOC_Detection_Set(is_train = True)
    val_dataSet = VOC_Detection_Set(imgs_path=args.val_imgs,
                                    annotations_path=args.val_labels,
                                    classes_file=args.voc_classes_path, class_num=class_num, is_train=False, loss_mode=loss_mode)

    
    if epoch < epoch_unfreeze:
        model.set_freeze_by_idxs(YOLO, [0, 1])


    # optimizer = optim.SGD(YOLO.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(YOLO.parameters(), lr=learning_rate)
    loss_function = YOLOv1_Loss().to(device=device)
 
    # 6.train and record
    writer = SummaryWriter()
 
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=16)
 
    while epoch < num_epochs:
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
 
        train_len = len(train_loader)
        YOLO.train()
        loss = [0, 0, 0, 0, 0, 0, 1]
        for train_data, label_data in tqdm((train_loader), desc=f"epoch:{epoch}/{num_epochs}", total=len(train_loader)):
            optimizer.zero_grad()
            train_data = train_data.float().to(device=device)
            label_data[0] = label_data[0].float().to(device=device)
            label_data[1] = label_data[1].to(device=device)
            label_data[2] = label_data[2].to(device=device)

            loss = loss_function(bounding_boxes=YOLO(train_data), ground_labels=label_data)
            sample_avg_loss = loss[0]
            sample_avg_loss.backward()
            optimizer.step()
            epoch_train_loss_coord = epoch_train_loss_coord + loss[1] * batch_size
            epoch_train_loss_pos_conf = epoch_train_loss_pos_conf + loss[2] * batch_size
            epoch_train_loss_neg_conf = epoch_train_loss_neg_conf + loss[3] * batch_size
            epoch_train_loss_classes = epoch_train_loss_classes + loss[4] * batch_size
            epoch_train_iou = epoch_train_iou + loss[5]
            epoch_train_object_num = epoch_train_object_num + loss[6]

            batch_loss = sample_avg_loss.item() * batch_size
            epoch_train_loss = epoch_train_loss + batch_loss

            if epoch == epoch_unfreeze + 1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
            # print("train: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} avg_iou:{}".format(
            #         round(loss[1], 4),
            #         round(loss[2], 4),
            #         round(loss[3], 4),
            #         round(loss[4], 4),
            #         round(loss[5] / loss[6], 4)))

        if args.feature_map_visualize:
            feature_map_visualize(train_data[0][0], writer, YOLO)
        # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))

        print("\ntrain-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
            round(epoch_train_loss / train_len, 4),
            round(epoch_train_loss_coord / train_len, 4),
            round(epoch_train_loss_pos_conf / train_len, 4),
            round(epoch_train_loss_neg_conf / train_len, 4),
            round(epoch_train_loss_classes / train_len, 4),
            round(epoch_train_iou / epoch_train_object_num, 4)))
 
 
        val_len = len(val_loader)
        YOLO.eval()
        with torch.no_grad():
            for val_data, label_data in tqdm((val_loader), desc=f"epoch:{epoch}/{num_epochs}",total=len(val_loader)):
                    val_data = val_data.float().to(device=device)
                    label_data[0] = label_data[0].float().to(device=device)
                    label_data[1] = label_data[1].to(device=device)
                    label_data[2] = label_data[2].to(device=device)
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
 
            if args.feature_map_visualize:
                feature_map_visualize(train_data[0][0], writer, YOLO)

            print("\nval-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                round(epoch_val_loss / val_len, 4),
                round(epoch_val_loss_coord / val_len, 4),
                round(epoch_val_loss_pos_conf / val_len, 4),
                round(epoch_val_loss_neg_conf / val_len, 4),
                round(epoch_val_loss_classes / val_len, 4),
                round(epoch_val_iou / epoch_val_object_num, 4)))
 
        print("epoch : {} ; loss : {}".format(epoch, epoch_train_loss))
 
        # if epoch == epoch_unfreeze:
        #     model.unfreeze_by_idxs(YOLO, [0, 1])
 
        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
 
        if  epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['epoch'] = epoch
            param_dict['epoch_val_loss_min'] = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLOv1_' + str(epoch) + '.pth')

        if args.grad_visualize:
            for i, (name, layer) in enumerate(YOLO.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)

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

        epoch += 1

    writer.close()