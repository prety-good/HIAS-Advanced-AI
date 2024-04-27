import torch
import torchvision
from torch import nn
from torch.nn import init

class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()
        # self.activate = nn.Sigmoid()
        self.activate = nn.ReLU()
        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
    def forward(self, X):
        X = self.activate(self.conv1(X))
        X = self.pool(X)
        X = self.pool(self.activate(self.conv2(X)))
        X = self.flatten(X)
        X = self.fc3(self.activate(self.fc2(self.activate(self.fc1(X)))))
        return X 

if __name__=="__main__":
    model = lenet()
    x = torch.randn((32,1,28,28))
    y = model(x)
    print(model)
    print(y.shape)
    print(y)
    predict = torch.argmax(y, dim = 1)
    print(predict)