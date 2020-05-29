# -*- coding: utf-8 -*-
# @Time: 2020/5/27
# @Author: ZHL
# @FileName: fcn_vggnet.py
# @Description: 编写fcn_vggnet模型

import torch
import torch.nn as nn
from torchvision.models import VGG
from torchvision import models
import torch.optim as optim

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# VGG网络中每个block（每个块以MaxPool2d结束）的范围
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# 使用循环创建VGG的卷积层
def make_layers(cfg, batch_norm=False):
    layers = [] # 用列表保存网络层
    in_channels = 3 # 初始的输入通道，即图片的输入通道=3
    for v in cfg: # 输出通道
        if v == 'M': # 如果当前是最大池化层（M）
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: # 添加BN层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 修改VGG网络：移除fc层，添加“skip”连接
# “skip”连接：获取5个池化层的输出
class VGGNet(VGG):
    def __init__(self, model='vgg16', pretrained=True, remove_fc=True):
        super().__init__(make_layers(cfgs[model])) # 初始化VGG的卷积层，self.features=make_layers(cfg[models])
        self.ranges = ranges[model]

        if pretrained:
            # 执行加载预训练模型参数的语法：models.net_name(pretrained=True)
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if remove_fc: # 移除fc层，fc层在VGG网络中命名为 classifier
            del self.classifier

    def forward(self, x):
        output = {} # 存放每个MaxPool2d的输出结果，用于"skip"连接
        # 所有blocks进行前向传播
        for idx in range(len(self.ranges)):
            # 每个block里面的网络层进行前向传播
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            # 每个块前向传播结束后，获得MaxPool2d的输出
            output['pool%d' % (idx+1)] = x

        return output

# 上采样层的顺序是：deconv → relu → bn
def make_upsampling_layers():
    layers = []
    in_channel = 512
    upsampling_cfg = [512, 256, 128, 64, 32]
    for v in upsampling_cfg:
        deconv = nn.ConvTranspose2d(in_channel, v, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        bn = nn.BatchNorm2d(v)
        layers += [deconv, nn.ReLU(inplace=True), bn]
        in_channel = v
    return nn.Sequential(*layers)

class FCN32s(nn.Module):
    """
    FCN32s：VGG16-pool5 → 上采样 → 像素分类器
    """
    def __init__(self, pretrain_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrain_net = pretrain_net
        self.relu = nn.ReLU(inplace=True)
        self.upsampling = make_upsampling_layers()
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.pretrain_net(x)
        pool5 = x['pool5']
        for idx, layer in enumerate(self.upsampling):
            if idx == 0: # 第一个deconv层，输入 pool5，图像 x2
                x = layer(pool5)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x

class FCN16s(nn.Module):
    """
    FCN16s：VGG16-pool5 → 上采样 + VGG16-pool4  → 上采样 → 像素分类器
    """
    def __init__(self, pretrain_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrain_net = pretrain_net
        self.relu = nn.ReLU(inplace=True)
        self.upsampling = make_upsampling_layers()
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.pretrain_net(x)
        pool5 = x['pool5']
        pool4 = x['pool4']
        for idx, layer in enumerate(self.upsampling):
            if idx == 0: # 第一个deconv层，输入 pool5，图像 x2
                x = layer(pool5)
            elif idx == 2: # 第一个bn层，进行"skip"连接，输入relu(x)+pool4
                x = layer(x + pool4)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x

class FCN8s(nn.Module):
    """
    FCN8s：VGG16-pool5→ 上采样 + VGG16-pool4  → 上采样 + VGG16-pool3 → 上采样 → 像素分类器
    """
    def __init__(self, pretrain_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrain_net = pretrain_net
        self.relu = nn.ReLU(inplace=True)
        self.upsampling = make_upsampling_layers()
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.pretrain_net(x)
        pool5 = x['pool5']
        pool4 = x['pool4']
        pool3 = x['pool3']
        for idx, layer in enumerate(self.upsampling):
            if idx == 0: # 第一个deconv层，输入 pool5，图像 x2
                x = layer(pool5)
            elif idx == 2: # 第一个bn层，进行"skip"连接，输入relu(x)+pool4
                x = layer(x + pool4)
            elif idx == 5: # 第二个bn层，进行"skip"连接，输入relu(x)+pool3
                x = layer(x + pool3)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x

class FCNs(nn.Module):
    """
    FCNs：将所有池化层输出进行"skip"连接
    """
    def __init__(self, pretrain_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrain_net = pretrain_net
        self.relu = nn.ReLU(inplace=True)
        self.upsampling = make_upsampling_layers()
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.pretrain_net(x)
        pool5 = x['pool5']
        pool4 = x['pool4']
        pool3 = x['pool3']
        pool2 = x['pool2']
        pool1 = x['pool1']
        for idx, layer in enumerate(self.upsampling):
            if idx == 0: # 第一个deconv层，输入 pool5，图像 x2
                x = layer(pool5)
            elif idx == 2: # 第一个bn层，进行"skip"连接，输入relu(x)+pool4
                x = layer(x + pool4)
            elif idx == 5: # 第二个bn层，进行"skip"连接，输入relu(x)+pool3
                x = layer(x + pool3)
            elif idx == 8:  # 第三个bn层，进行"skip"连接，输入relu(x)+pool2
                x = layer(x + pool2)
            elif idx == 11:  # 第四个bn层，进行"skip"连接，输入relu(x)+pool1
                x = layer(x + pool1)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    batch_size, n_class, h, w = 2, 20, 224, 224
    # 测试网络的输出大小
    vgg = VGGNet()
    input = torch.randn(batch_size, 3, h, w, requires_grad=True) # requires_grad=True，input需要求梯度
    output = vgg(input)
    assert output['pool5'].size() == torch.Size([batch_size, 512, 7, 7])
    print("VGGNet checked!")

    # fcn32s = FCN32s(vgg, n_class)
    # input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    # output = fcn32s(input)
    # assert output.size() == torch.Size([batch_size, n_class, 224, 224])
    # print("FCN32s checked!")
    #
    # fcn16s = FCN16s(vgg, n_class)
    # input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    # output = fcn16s(input)
    # assert output.size() == torch.Size([batch_size, n_class, 224, 224])
    # print("FCN16s checked!")
    #
    # fcn8s = FCN8s(vgg, n_class)
    # input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    # output = fcn8s(input)
    # assert output.size() == torch.Size([batch_size, n_class, 224, 224])
    # print("FCN8s checked!")
    #
    # fcns = FCNs(vgg, n_class)
    # input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    # output = fcns(input)
    # assert output.size() == torch.Size([batch_size, n_class, 224, 224])
    # print("FCNs checked!")
    #
    # # 查看网络信息（参数量、计算量等）
    # from ptflops import get_model_complexity_info
    # fcn = FCNs(vgg, n_class)
    # flops, params = get_model_complexity_info(model=fcn, input_res=(3,h,w), as_strings=True,print_per_layer_stat=True)
    # print("%s |%s |%s" % ('fcn model', flops, params))

    # 测试一个batch，损失函数如果减小，则成功
    # 实例化模型
    fcn_vggnet = FCNs(vgg, n_class)
    # 损失函数
    criterion = nn.BCELoss()
    # 梯度下降优化算法
    optimizer = optim.SGD(fcn_vggnet.parameters(), lr=1e-3, momentum=0.9)
    # 定义一个训练集batch
    input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    label = torch.randn(batch_size, n_class, h, w)
    # 训练
    iters = 10
    for iter in range(iters):
        optimizer.zero_grad()
        output = fcn_vggnet(input)
        output = torch.sigmoid(output) # 用sigmoid将数据压缩到[0,1]之间求概率
        loss = criterion(output, label)
        loss.backward()
        print("iter:{}, loss:{}".format(iter, loss.item()))
        optimizer.step()