# -*- coding: utf-8 -*-
# @Time: 2020/5/27
# @Author: ZHL
# @FileName: segnet.py
# @Description:
# encoder：5 blocks，vggnet16的前13层构成每个blocks，每个blocks最后一层是一个最大池化层
# decoder：5 blocks，上采样，每个block连接最大池化层索引
# 最后一层：softmax分类层

import torch
import torch.nn as nn
from torchvision.models import VGG

# vggnet 每层输出通道
vgg16_cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

def make_layers(input_channels, output_channels):
    layer = []
    conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    layer += [conv, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)

class encoder1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(encoder1, self).__init__()
        self.layer1 = make_layers(input_channels, output_channels)
        self.layer2 = make_layers(output_channels, output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        unpooled_size = x.size()
        x, index = self.pool(x)
        return x, index, unpooled_size

class encoder2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(encoder2, self).__init__()
        self.layer1 = make_layers(input_channels, output_channels)
        self.layer2 = make_layers(output_channels, output_channels)
        self.layer3 = make_layers(output_channels, output_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        unpooled_size = x.size()
        x, index = self.pool2(x)
        return x, index, unpooled_size


class decoder2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(decoder2, self).__init__()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer1 = make_layers(input_channels, output_channels)
        self.layer2 = make_layers(output_channels, output_channels)
        self.layer3 = make_layers(output_channels, output_channels)


    def forward(self, x, index, output_size):
        x = self.unpool1(input=x, indices=index, output_size=output_size)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class decoder1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(decoder1, self).__init__()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer1 = make_layers(input_channels, output_channels)
        self.layer2 = make_layers(output_channels, output_channels)


    def forward(self, x, index, output_size):
        x = self.unpool2(input=x, indices=index, output_size=output_size)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SegNet(nn.Module):
    def __init__(self, input_channels, n_classes=20):
        super(SegNet, self).__init__()
        self.block1 = encoder1(input_channels, 64)
        self.block2 = encoder1(64, 128)
        self.block3 = encoder2(128, 256)
        self.block4 = encoder2(256, 512)
        self.block5 = encoder2(512, 512)

        self.dblock5 = decoder2(512, 512)
        self.dblock4 = decoder2(512, 256)
        self.dblock3 = decoder1(256, 128)
        self.dblock2 = decoder1(128, 64)
        self.dblock1 = decoder1(64, n_classes)

    def forward(self, x):
        enc1, index1, unpooled_size1 = self.block1(x)
        enc2, index2, unpooled_size2 = self.block2(enc1)
        enc3, index3, unpooled_size3 = self.block3(enc2)
        enc4, index4, unpooled_size4 = self.block4(enc3)
        enc5, index5, unpooled_size5 = self.block5(enc4)

        dec5 = self.dblock5(enc5, index5, unpooled_size5)
        dec4 = self.dblock4(dec5, index4, unpooled_size4)
        dec3 = self.dblock3(dec4, index3, unpooled_size3)
        dec2 = self.dblock2(dec3, index2, unpooled_size2)
        dec1 = self.dblock1(dec2, index1, unpooled_size1)

        return dec1

    # 导入预训练模型，只加载指定卷积层的参数
    # 其中encoder的权重与vgg16的权重保持一致
    def vgg16_init(self, vgg16):
        convNets = [self.block1, self.block2, self.block3, self.block4, self.block5]
        vgg16Features = list(vgg16.features.children())
        vgg16Conv = []
        for layer in vgg16Features:
            if isinstance(layer, nn.Conv2d):
                vgg16Conv.append(layer)

        outLayers = []
        for id, conv in enumerate(convNets):
            if id < 2:
                layers = list(conv.layer1.children()) + list(conv.layer2.children())
            else:
                layers = list(conv.layer1.children()) + list(conv.layer2.children()) + list(conv.layer3.children())
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    outLayers.append(layer)
        assert len(vgg16Conv) == len(outLayers)

        for xx, yy in zip(vgg16Conv, outLayers):
            if isinstance(xx, nn.Conv2d) and isinstance(yy, nn.Conv2d):
                assert xx.weight.size() == yy.weight.size()
                assert xx.bias.size() == yy.bias.size()
                yy.weight.data = xx.weight.data
                yy.bias.data = xx.bias.data

if __name__ == '__main__':


    batch_size, n_classes, h, w = 2, 20, 224, 224

    # block1 = encoder1(3, 64)
    # input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    # output, index1, unpooled_size1 = block1(input)
    # assert output.size() == torch.Size([batch_size, 64, 112, 112])
    # print("block1 checked!")
    #
    # block5 = encoder2(512, 512)
    # input = torch.randn(batch_size, 512, 14, 14, requires_grad=True)
    # output, index5, unpooled_size5 = block5(input)
    # assert output.size() == torch.Size([batch_size, 512, 7, 7])
    # print("block5 checked!")
    #
    #
    # d_block1 = decoder2(512, 512)
    # input = torch.randn(batch_size, 512, 7, 7, requires_grad=True)
    # output = d_block1(input, index5, unpooled_size5)
    # assert output.size() == torch.Size([batch_size, 512, 14, 14])
    # print("d_block1 checked!")
    #
    # d_block5 = decoder1(64, n_classes)
    # input = torch.randn(batch_size, 64, 112, 112, requires_grad=True)
    # output = d_block5(input, index1, unpooled_size1)
    # assert output.size() == torch.Size([batch_size, n_classes, 224, 224])
    # print("d_block5 checked!")

    from torchvision import models
    import torch.optim as optim

    segnet = SegNet(3, n_classes)
    vgg16 = models.vgg16(pretrained=True)
    segnet.vgg16_init(vgg16)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(segnet.parameters(), lr=0.001, momentum=0.9)

    input = torch.randn(batch_size, 3, h, w)
    label = torch.randn(batch_size, n_classes, h, w)

    for iter in range(10):
        optimizer.zero_grad()
        output = segnet(input)
        output = torch.sigmoid(output) # 用sigmoid将数据压缩到[0,1]之间求概率
        loss = criterion(output, label)
        loss.backward()
        print("iter:{}, loss:{}".format(iter, loss.item()))
        optimizer.step()
