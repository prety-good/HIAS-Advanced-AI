import torch
import torch.nn as nn
import torchvision

class Convention(nn.Module):
    def __init__(self,in_channels,out_channels, conv_size,conv_stride, padding,need_bn = True):
        super(Convention,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_size, conv_stride, padding, bias=False if need_bn else True)
        self.leaky_relu = nn.LeakyReLU()
        self.need_bn = need_bn
        if need_bn:
            self.bn = nn.BatchNorm2d(out_channels)
 
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x))) if self.need_bn else self.leaky_relu(self.conv(x))
 
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # backbone = torchvision.models.resnet34(pretrained=True)
        backbone = torchvision.models.resnet18(weights='DEFAULT')

        self.feature_num = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
    def forward(self, x):
        return self.backbone(x)


class YOLOv1(nn.Module):

    def __init__(self,B=2,classes_num=20):
        super(YOLOv1,self).__init__()
        self.B = B
        self.classes_num = classes_num
 
        # self.Conv_Feature = nn.Sequential(
        #     Convention(3, 64, 7, 2, 3),
        #     nn.MaxPool2d(2, 2),
 
        #     Convention(64, 192, 3, 1, 1),
        #     nn.MaxPool2d(2, 2),
 
        #     Convention(192, 128, 1, 1, 0),
        #     Convention(128, 256, 3, 1, 1),
        #     Convention(256, 256, 1, 1, 0),
        #     Convention(256, 512, 3, 1, 1),
        #     nn.MaxPool2d(2, 2),
 
        #     Convention(512, 256, 1, 1, 0),
        #     Convention(256, 512, 3, 1, 1),
        #     Convention(512, 256, 1, 1, 0),
        #     Convention(256, 512, 3, 1, 1),
        #     Convention(512, 256, 1, 1, 0),
        #     Convention(256, 512, 3, 1, 1),
        #     Convention(512, 256, 1, 1, 0),
        #     Convention(256, 512, 3, 1, 1),
        #     Convention(512, 512, 1, 1, 0),
        #     Convention(512, 1024, 3, 1, 1),
        #     nn.MaxPool2d(2, 2),
        # )

        # self.Conv_Semanteme = nn.Sequential(
        #     Convention(1024, 512, 1, 1, 0),
        #     Convention(512, 1024, 3, 1, 1),
        #     Convention(1024, 512, 1, 1, 0),
        #     Convention(512, 1024, 3, 1, 1),
        # )

        self.backbone = backbone()
        
        self.Conv_Back = nn.Sequential(
            Convention(self.backbone.feature_num, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 2, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
        )
 
        self.Fc = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7 * 7 * (B*5 + classes_num)),
        )
 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=3)
 
    def forward(self, x):
        # x = self.Conv_Feature(x)
        # x = self.Conv_Semanteme(x)

        x = self.backbone(x)
        
        x = self.Conv_Back(x)
        # batch_size * channel * height * weight -> batch_size * height * weight * channel
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1)
        x = self.Fc(x)
        x = x.view(-1, 7, 7, (self.B * 5 + self.classes_num))
        #print("x seg:{}".format(x[:,:,:,0 : self.B * 5]))
        bnd_coord = self.sigmoid(x[:,:,:,0 : self.B * 5])
        #print("bnd_coord:{}".format(bnd_coord))
        bnd_cls = self.softmax(x[:,:,:, self.B * 5 : ])
        bnd = torch.cat([bnd_coord, bnd_cls], dim=3)
        #x = self.sigmoid(x.view(-1,7,7,(self.B * 5 + self.classes_num)))
        #x[:,:,:, 0 : self.B * 5] = self.sigmoid(x[:,:,:, 0 : self.B * 5])
        #x[:,:,:, self.B * 5 : ] = self.softmax(x[:,:,:, self.B * 5 : ])
        return bnd
 
    # 定义权值初始化
    def initialize_weights(self, net_param_dict):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, Convention):
                m.weight_init()
 
        self_param_dict = self.state_dict()
        for name, layer in self.named_parameters():
            if name in net_param_dict:
                self_param_dict[name] = net_param_dict[name]
        self.load_state_dict(self_param_dict)
if __name__=="__main__":
    model = YOLOv1().cuda()
    x = torch.randn((32,3,448,448)).cuda()
    y = model(x)
    # print(model)
    print(y.shape)