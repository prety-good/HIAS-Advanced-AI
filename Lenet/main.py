import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from model import lenet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter()
batch_size = 128
num_epochs = 20
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    transform = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root="data", train=True,download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="data", train=False,download=True, transform=transform)

    train_dataloder = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloder = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = lenet().to(device)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        for X, y in tqdm((train_dataloder), desc=f"Train Epoch: [{epoch}/{num_epochs}]", total=len(train_dataloder)):
            X = X.to(device, dtype = torch.float32)
            y = y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            pred = torch.argmax(y_hat, dim = 1)
            correct += (pred == y).sum().item()

        train_loss /= len(train_dataloder)
        train_accuracy  = correct / len(train_data)
        print(f"Train loss:{train_loss}，Train acc:{train_accuracy}")

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X, y in tqdm((test_dataloder), desc=f"Val Epoch: [{epoch}/{num_epochs}]", total=len(test_dataloder)):
                X = X.to(device, dtype = torch.float32)
                y = y.to(device)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                test_loss += loss.item()
                pred = torch.argmax(y_hat, dim = 1)
                correct += (pred == y).sum().item()

        test_loss /= len(test_dataloder)
        test_accuracy  = correct / len(test_data)
        print(f"Test loss: {test_loss},Test acc:{test_accuracy}")
        
        tb_writer.add_scalars('acc', {'train': train_accuracy, 'test': test_accuracy}, epoch) 
        tb_writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch) 
        
    # tb_writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},{'hparam/accuracy': accuracy, 'hparam/loss': test_loss})