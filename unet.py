# -*- coding: utf-8 -*-
# @Time: 2020/5/27
# @Author: ZHL
# @FileName: unet.py
# @Description: unet分为两部分
# 收缩路径（contraction path）：每个block的最后一层copy一份给扩展路径，最大池化进行下采样
# 扩展路径（expansive path）：进行上采样（Upsample/ConvTranspose2d），获得收缩路径的feature map，连接起来再进行卷积

import torch.nn as nn
import torch

def make_layers(input_channels, output_channels):
    layers = []
    for i in range(2):
        conv = nn.Conv2d(input_channels, output_channels, kernel_size=3)
        input_channels = output_channels
        layers += [conv, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class UNet_C(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet_C, self).__init__()
        self.block1 = make_layers(input_channels, 64)
        self.block2 = make_layers(64, 128)
        self.block3 = make_layers(128, 256)
        self.block4 = make_layers(256, 512)
        self.block5 = make_layers(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.block1(x)
        x11 = self.maxpool(x1)
        x2 = self.block2(x11)
        x22 = self.maxpool(x2)
        x3 = self.block3(x22)
        x33 = self.maxpool(x3)
        x4 = self.block4(x33)
        x44 = self.maxpool(x4)
        x5 = self.block5(x44)
        output_feature_map = [x1, x2, x3, x4]
        return x5, output_feature_map

class UNet(nn.Module):
    def __init__(self, input_channels=3, n_classes=32):
        super(UNet, self).__init__()
        self.contraction = UNet_C(input_channels)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.eblock1 = make_layers(1024, 512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.eblock2 = make_layers(512, 256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.eblock3 = make_layers(256, 128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.eblock4 = make_layers(128, 64)

        self.classifire = nn.Conv2d(64, n_classes, kernel_size=1)

    # 疑惑：论文中所说的是crop map1，而参考他人的代码则是pad map2，且最终输出的图像也与原图不一致。。。
    def connect(self, map1, map2):
        diffY = map1.size()[2] - map2.size()[2]
        diffX = map1.size()[3] - map2.size()[3]
        map2 = torch.nn.functional.pad(map2, [diffX//2, diffX-diffX//2,
                                              diffY//2, diffY-diffY//2])
        x = torch.cat([map1, map2], dim=1)
        return x

    def forward(self, x):
        x, feature_maps = self.contraction(x)
        x = self.deconv1(x)
        x = self.connect(feature_maps[3], x)
        x = self.eblock1(x)
        x = self.deconv2(x)
        x = self.connect(feature_maps[2], x)
        x = self.eblock2(x)
        x = self.deconv3(x)
        x = self.connect(feature_maps[1], x)
        x = self.eblock3(x)
        x = self.deconv4(x)
        x = self.connect(feature_maps[0], x)
        x = self.eblock4(x)
        x = self.classifire(x)
        return x

if __name__ == '__main__':
    batch_size, n_classes, h, w = 2, 32, 572, 572

    # unet = UNet_C()
    # input = torch.randn(batch_size, 3, h, w)
    # output, feature_map = unet(input)
    # assert output.size() == torch.Size([batch_size, 1024, 28, 28])
    # print("contraction path checked!")

    import torch.optim as optim
    import torch.utils

    unet = UNet()
    input = torch.randn(batch_size, 3, h, w)
    label = torch.randn(batch_size, n_classes, 564, 564)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(unet.parameters(), lr=1e-3, momentum=0.9)

    for iter in range(10):
        optimizer.zero_grad()
        output = unet(input)
        output = torch.sigmoid(output)
        loss = criterion(output, label)
        loss.backward()
        print("iter:{}, loss:{}".format(iter, loss.item()))
        optimizer.step()