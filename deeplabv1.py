# -*- coding: utf-8 -*-
# @Time: 2020/5/27
# @Author: ZHL
# @FileName: deeplabv1.py
# @Description:
# 微调VGG16，使用了空洞卷积(dialation)扩大感受野
# 具体操作：1）FC改为CONV；2）最后2个pool改为stride=1；
# 3）最后3个卷积层的dialation=2，第一个FC层output=1024, k=3, dialation=12
# FCN是对预测结果进行上采样，deeplab使用CRF对预测的结果进行后处理，细化语义轮廓

import torch.nn as nn
import torch
from torchvision.models import VGG

def make_layers(input_channels, output_channels, edit_pool=False, len=2):
    layers = []
    for i in range(len):
        conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv, nn.ReLU(inplace=True)]
        input_channels = output_channels
    if edit_pool:
        maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    else:
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    layers += [maxpool]
    return nn.Sequential(*layers)

class DeepLabv1(nn.Module):
    def __init__(self, input_channels=3, n_classes=21):
        super(DeepLabv1, self).__init__()
        self.block1 = make_layers(input_channels, 64, len=2)
        self.block2 = make_layers(64, 128, len=2)
        self.block3 = make_layers(128, 256, len=3)
        self.block4 = make_layers(256, 512, edit_pool=True, len=3)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc8 = nn.Conv2d(1024, n_classes, kernel_size=1)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x


if __name__ == '__main__':
    batch_size, n_classes, h, w = 2, 21, 160, 160

    # deeplabv1 = DeepLabv1(n_classes=n_classes)
    # input = torch.randn(batch_size, 3, h, w)
    # output = deeplabv1(input)
    # assert output.size() == torch.Size([batch_size, n_classes, h//8, w//8])
    # print('deeplabv1 checked!')

    import torch.optim as optim
    from tqdm import tqdm
    deeplabv1 = DeepLabv1(n_classes=n_classes)
    input = torch.randn(batch_size, 3, h, w)
    label = torch.randn(batch_size, n_classes, h//8, w//8)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(deeplabv1.parameters(), lr=0.1, momentum=0.9)

    for iter in tqdm(range(10)):
        optimizer.zero_grad()
        output = deeplabv1(input)
        loss = criterion(output, label)
        loss.backward()
        print("iter:{}, loss:{}".format(iter, loss.item()))
        optimizer.step()

