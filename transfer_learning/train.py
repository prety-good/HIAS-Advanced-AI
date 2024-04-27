import torch
from torch import nn
import torch.utils
import torchvision
from torchvision import transforms
from model import inception
from dataset import dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from dataset import get_dataset

num_epochs = 50
batch_size = 32
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="transfer learning",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "inception v3",
        "dataset": "flowers",
        "epochs": num_epochs,
        "batch_size": batch_size,
        }
    )
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


    model = inception().to(device)

    # train_dataset = dataset(transform=transfer["train"])
    # train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    train_dataset, val_dataset = get_dataset(transfer)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()

    fc_parameters = list(model.backbone.children())[-1].parameters()
    remaining_parameters = []
    for child in list(model.backbone.children())[:-1]: 
        remaining_parameters.extend(list(child.parameters()))
        # child.requires_grad_ = False
    optimizer = torch.optim.Adam([
            {'params': fc_parameters, 'lr': learning_rate },
            {'params': remaining_parameters, 'lr': learning_rate* 0.1}])
    # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 2 * num_epochs)

    for epoch in range(num_epochs):
        train_loss = 0
        acc_num = 0
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"epoch:{epoch}/{num_epochs}", total=len(train_dataloader)):
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)[0]

            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            acc_num += (y == torch.argmax(y_hat, dim=1)).sum().item()
        scheduler.step()
        epoch_acc = acc_num / len(train_dataset)
        print(f"total loss:{train_loss},accuracy:{epoch_acc}")

        val_loss = 0
        val_acc_num = 0
        model.eval()
        with torch.no_grad():
            for X, y in tqdm(val_dataloader, desc=f"val epoch:{epoch}/{num_epochs}", total=len(val_dataloader)):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item()
                val_acc_num += (y == torch.argmax(y_hat, dim=1)).sum().item()
            val_epoch_acc = val_acc_num / len(val_dataset)
            print(f"total loss:{val_loss},accuracy:{val_epoch_acc}")



        wandb.log({
            "train total loss":train_loss,
            "train accuracy":epoch_acc,
            "val total loss":val_loss,
            "val accuracy":val_epoch_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"./data/weight/inception_{epoch}.pth")
    wandb.finish()