# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/16 20:24
    @filename: deeplab.py
    @software: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .utils import *

backbones = {
    "resnet50":resnet50,
    "resnet101":resnet101,
    "resnest50" : resnest50,
    "resnest101" : resnest101,
    "resnest200" : resnest200,
    "resnest269" : resnest269,
    "seresnet50" : seresnet50,
    "resnest26" : resnest26,
    "resnest14" : resnest14
}

class LayerAttentionModule(nn.Module):
    def __init__(self, in_ch1, in_ch2, expansion=2.0,norm_layer=nn.BatchNorm2d):
        """
        Implementation of our simple layer attention module
        Args:
            in_ch (int): number of channels for inputs
            expansion (float): expansion rate for hidden+layer
            norm_layer (nn.Module): normalization part
        """
        super(LayerAttentionModule, self).__init__()
        hidden_size = int(in_ch1 * expansion)
        self.fc_q = nn.Conv2d(in_ch1, hidden_size, kernel_size=1, stride=1, bias=False)
        self.fc_k = nn.Conv2d(in_ch2, hidden_size, kernel_size=1, stride=1, bias=False)
        self.fc_value = nn.Conv2d(in_ch2, hidden_size, kernel_size=1, stride=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_size, in_ch2, kernel_size=1, stride=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, value, query):
        bs, _, h, w = value.size()
        key = self.fc_k(value).reshape(bs, -1, h*w)
        query = self.fc_q(query).reshape(bs, -1, h*w)
        proj_value = self.fc_value(value).reshape(bs, -1, h*w)
        atten = torch.bmm(query, key.transpose(1, 2))
        atten = torch.softmax(atten, dim=-1)
        net = torch.bmm(proj_value.transpose(1, 2), atten).reshape(bs, -1, h, w)
        net = self.fc2(net)
        net = net * self.gamma + value
        return net


class ImagePooling(nn.Module):
    def __init__(self, in_planes, out_ch=256, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        '''
            implementation of ImagePooling in deeplabv3,section 3.3
            paper:https://arxiv.org/abs/1706.05587
            args:
                in_planes (int):input planes for the pooling
                out_ch (int): output channels, in paper is 256
                norm_layer (nn.Module): the batch normalization module
                activation(nn.Module): the activation function module
        '''
        super(ImagePooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = Conv2d(in_planes, out_ch, ksize=1, stride=1, padding=0, norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        net = self.avgpool(x)
        net = self.conv(net)
        return net


class ASPP(nn.Module):
    def __init__(self, in_planes, out_ch, rates, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        '''
            implementation of ASPP(Atrous Spatial Pyramid Pooling) in deeplabv3,section 3.3
            References:
                https://arxiv.org/abs/1706.05587
            args:
                in_planes (int):input planes for the pooling
                out_ch (int): output channels, in paper is 256
                norm_layer (nn.Module): the batch normalization module
                activation(nn.Module): the activation function module
        '''
        super(ASPP, self).__init__()
        self.branch1 = Conv2d(in_planes, 256, 1, stride=1, padding=0, dilation=rates[0], norm_layer=norm_layer, activation=activation)
        self.branch2 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[1], dilation=rates[1], norm_layer=norm_layer,
                              activation=activation)
        self.branch3 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[2], dilation=rates[2], norm_layer=norm_layer,
                              activation=activation)
        self.branch4 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[3], dilation=rates[3], norm_layer=norm_layer,
                              activation=activation)
        self.branch5 = ImagePooling(in_planes, 256)

        self.conv = Conv2d(1280, out_ch, 1, stride=1, padding=0)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)
        branch5 = F.interpolate(branch5, size=branch4.size()[2:], mode="bilinear", align_corners=True)
        concat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        conv = self.conv(concat)
        return conv

class DeeplabV3(nn.Module):
    def __init__(self, in_ch, num_classes, backbone="resnet50",  output_stride=16, **kwargs):
        super(DeeplabV3, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))

        multi_grids = [1, 2, 4]
        self.backbone = backbones[backbone](in_ch=in_ch, strides=strides,
                                            dilations=dilations, multi_grids=multi_grids, **kwargs)
        del self.backbone.avg_pool
        del self.backbone.fc
        self.aspp = ASPP(in_planes=2048, out_ch=256, rates=rates)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        net = self.backbone.conv1(x)
        net = self.backbone.max_pool(net)
        net = self.backbone.layer1(net)
        net = self.backbone.layer2(net)
        net = self.backbone.layer3(net)
        net = self.backbone.layer4(net)
        net = self.aspp(net)
        net = self.conv5(net)
        net = F.interpolate(net, size=size, mode="bilinear", align_corners=True)
        return net

class RCAB(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4,
                 norm=nn.BatchNorm2d,activation=nn.ReLU(inplace=True)):
        super(RCAB, self).__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            norm(out_ch),
            activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
            norm(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_ch),
        )
        self.se = SEModule(out_ch, reduction=reduction, norm_layer=norm,
                           sigmoid=nn.Sigmoid(), activation=activation)
        self.act = activation

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.se(net)*net + net
        net = self.act(net)
        return net

class DeeplabV3Plus(nn.Module):
    def __init__(self, in_ch, num_classes, backbone="resnet50",  output_stride=16,
                 fgam=False, middle_layer=False, CAB=False,**kwargs):
        super(DeeplabV3Plus,self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))

        multi_grids = [1, 2, 4]
        self.backbone = backbones[backbone](in_ch=in_ch, strides=strides,
                                            dilations=dilations, multi_grids=multi_grids, **kwargs)
        try:
            norm_layer = kwargs["norm_layer"]
        except KeyError as e:
            norm_layer = nn.BatchNorm2d
        try:
            activation = kwargs["activation"]
        except KeyError as e:
            activation = nn.ReLU(inplace=True)
        try:
            reduction = kwargs["reduction"]
        except KeyError as e:
            reduction = 16

        del self.backbone.avg_pool
        del self.backbone.fc
        self.fgam = fgam
        self.aspp = ASPP(in_planes=2048, out_ch=256, rates=rates)
        if middle_layer:
            low_ch = 512
        else:
            low_ch = 256
        self.conv5 = nn.Conv2d(low_ch, 48, kernel_size=1, stride=1, bias=False)
        # if self.fgam:
        #     self.attention = LayerAttentionModule(256, 304,expansion=expansion)
        fusion_ch = 304
        self.conv6 = nn.Sequential(
            Conv2d(fusion_ch, out_ch=256, ksize=3, stride=1, padding=1),
            Conv2d(256, 256, ksize=3, stride=1, padding=1),
        )
        if CAB:
            self.conv6 = RCAB(fusion_ch, 256, reduction=reduction, norm=norm_layer, activation=activation)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.middle_layer = middle_layer

    def forward(self, x):
        size = x.size()[2:]
        net = self.backbone.conv1(x)
        net = self.backbone.max_pool(net)
        net = self.backbone.layer1(net)
        h = net
        net = self.backbone.layer2(net)
        if self.middle_layer:
            h = net
        net = self.backbone.layer3(net)
        net = self.backbone.layer4(net)
        net = self.aspp(net)
        # net = self.conv5(net)
        h = self.conv5(h)
        net = F.interpolate(net, size=h.size()[2:], mode="bilinear", align_corners=True)
        net = torch.cat([h, net], dim=1)
        net = self.conv6(net)
        if self.fgam:
            net = net * torch.softmax(h, dim=1) + net
        net = self.conv7(net)
        net = F.interpolate(net, size=size, mode="bilinear", align_corners=True)
        return net

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = DeeplabV3Plus(3, 20, fgam=True)
    model.eval()
    out = model(x)
    print(out.shape)