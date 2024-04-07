import torch
import torchvision
from torch import nn



class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.poll = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, X):
        X = torch.sigmoid(self.conv1(X))
        X = self.poll(X)
        X = self.poll(torch.sigmoid(self.conv2(X)))
        X = torch.flatten(X)
        X = self.fc3(torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(X)))))
        return X 

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),nn.Sigmoid(),nn.AvgPool2d(2),
    nn.Conv2d(6, 16, kernel_size=5),nn.Sigmoid(),nn.AvgPool2d(2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10))


if __name__=="__main__":
    model = lenet()
    
    print(model)

