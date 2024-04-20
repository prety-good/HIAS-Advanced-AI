import torch
from torch import nn
import torchvision


class inception(nn.Module):
    def __init__(self, num_class = 5):
        super().__init__()
        self.backbone = torchvision.models.inception_v3(weights="DEFAULT")
        self.backbone.fc.out_features = 5
        # self.fc = nn.Sequential(nn.Linear(self.backbone.fc.out_features, 500),
        #                         nn.ReLU(),
        #                         nn.Linear(500, num_class))
    def forward(self, x):
        x = self.backbone(x)
        # print(x)
        # x = self.fc(x)
        return x
    
if __name__=="__main__":
    model = inception()
    x = torch.rand((4, 3, 448,448))
    y = model(x)
    print(y)
