import os
import torch
import argparse
from utils import model
from data import VOC_Detection_Set
from torch.utils.data import DataLoader
from model import YOLOv1
from loss import YOLOv1_Loss
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import warnings
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
 
if __name__ == "__main__":
    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLOv1 train config")
    parser.add_argument('--B', type=int, help="YOLOv1 predict box num every grid", default=2)
    parser.add_argument('--class_num', type=int, help="YOLOv1 predict class num", default=20)
    parser.add_argument('--lr', type=float, help="start lr", default=1e-4)
    parser.add_argument('--batch_size', type=int, help="YOLOv1 train batch size", default=12)
    parser.add_argument('--train_imgs', type=str, help="YOLOv1 train train_imgs", default="")
    parser.add_argument('--train_labels', type=str, help="YOLOv1 train train_labels", default="")
    parser.add_argument('--voc_classes_path', type=str, help="voc classes path", default="./data/class.data")
    parser.add_argument('--epoch_num', type=int, help="YOLOv1 train epoch num", default=100)
    parser.add_argument('--restart', type=bool, default=True)
    parser.add_argument('--weight_file', type=str, help="YOLOv1 weight path", default="./weights/efficientnet/YOLOv1_50.pth")
    args = parser.parse_args()

    num_epochs = args.epoch_num
    class_num = args.class_num
    batch_size = args.batch_size
    learning_rate = args.lr
    loss_mode = "mse"
    epoch_interval = 10
 
    wandb.init(
    # set the wandb project where this run will be logged
        project="yolo_v1",
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "yolo_v1 efficientnet",
        "dataset": "voc2012",
        "epochs": num_epochs,
        "batch_size": batch_size,
        }
    )

    YOLO = YOLOv1().to(device=device)



    if args.restart == True:
        param_dict = {}
        epoch = 0
    else:
        param_dict = torch.load(args.weight_file)
        epoch = param_dict['epoch']
        YOLO.load_state_dict(param_dict['model'])

    backbone_parameters = list(YOLO.children())[0].parameters()
    remaining_parameters = []
    for child in list(YOLO.children())[1:]:  # 从第二个开始到最后
        remaining_parameters.extend(list(child.parameters()))
    optimizer = optim.Adam([
            {'params': backbone_parameters, 'lr': learning_rate * 0.1},
            {'params': remaining_parameters, 'lr': learning_rate}])

    # optimizer = optim.Adam(YOLO.parameters(), lr= learning_rate)
    scheduler = CosineAnnealingLR(optimizer,T_max = num_epochs)

    loss_function = YOLOv1_Loss().to(device=device)

    train_dataSet = VOC_Detection_Set()
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True,num_workers=16)
 
    while epoch < num_epochs:
        epoch_train_loss = 0
        epoch_train_iou = 0
        epoch_train_object_num = 0
        epoch_train_loss_coord = 0
        epoch_train_loss_pos_conf = 0
        epoch_train_loss_neg_conf = 0
        epoch_train_loss_classes = 0

        train_len = len(train_loader)
        YOLO.train()
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

            # print("train: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} avg_iou:{}".format(
            #         round(loss[1], 4),
            #         round(loss[2], 4),
            #         round(loss[3], 4),
            #         round(loss[4], 4),
            #         round(loss[5] / loss[6], 4)))

        scheduler.step()
        print("\ntrain-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
            round(epoch_train_loss / train_len, 4),
            round(epoch_train_loss_coord / train_len, 4),
            round(epoch_train_loss_pos_conf / train_len, 4),
            round(epoch_train_loss_neg_conf / train_len, 4),
            round(epoch_train_loss_classes / train_len, 4),
            round(epoch_train_iou / epoch_train_object_num, 4)))
 
 
        if  epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['epoch'] = epoch
            torch.save(param_dict, './weights/YOLOv1_' + str(epoch) + '.pth')

        wandb.log({'Train/Loss_sum':epoch_train_loss,
                   'Train/Loss_coord':epoch_train_loss_coord,
                   'Train/Loss_pos_conf':epoch_train_loss_pos_conf,
                   'Train/Loss_neg_conf':epoch_train_loss_neg_conf,
                   'Train/Loss_classes':epoch_train_loss_classes, 
                   'Train/Epoch_iou':epoch_train_iou / epoch_train_object_num,
                   'learning_rate': optimizer.param_groups[1]['lr'],
                   })
        epoch += 1

    wandb.finish()
    os.system('shutdown')