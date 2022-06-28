# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 10:58
    @filename: utils.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

from .splat import SplAtConv2d

__all__ = ["Conv2d", "Downsample", "DoubleConv", "Upsample", "ResBlock", "SplAtBlock", "SEModule", "DepthWiseSeparableConv2d", "RRBlock"]



class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=16, norm_layer=None, sigmoid=None, activation=None):
        """
        SEModule of SENet and MobileNetV3
        Args:
            in_ch (int): the number of input channels
            reduction (int): the reduction rate
            norm_layer (nn.Module): the normalization module
            sigmoid （nn.Module): the sigmoid activation function for the last of fc
            activation (nn.Module): the middle activation function
        """
        super(SEModule, self).__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if sigmoid is None:
            sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        inter_channel = in_ch // reduction
        self.fc = nn.Sequential(
            Conv2d(in_ch, inter_channel, ksize=1, stride=1, norm_layer=norm_layer, activation=activation),
            Conv2d(inter_channel, in_ch, ksize=1, stride=1, norm_layer=norm_layer, activation=sigmoid)
        )
    def forward(self, x):
        net = self.avg_pool(x)
        net = self.fc(net) * x
        return net

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), dropout_rate=0.0, gn_groups=32,
                 **kwargs):
        """
        The conv2d with normalization layer and activation layer.
        Args:
            in_ch (int): the number of channels for input
            out_ch (int): the number of channels for output
            ksize (Union[int,tuple]): the size of conv kernel, default is 1
            stride (Union[int,tuple]): the stride of the slide window, default is 1
            padding (Union[int, tuple]): the padding size, default is 0
            dilation (Union[int,tuple]): the dilation rate, default is 1
            groups (int): the number of groups, default is 1
            bias (bool): whether use bias, default is False
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the nonlinear module
            dropout_rate (float): dropout rate
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, **kwargs)
        self.norm_layer = norm_layer
        if not norm_layer is None:
            if isinstance(norm_layer, nn.GroupNorm):
                self.norm_layer = norm_layer(gn_groups, out_ch)
            else:
                self.norm_layer = norm_layer(out_ch)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        net = self.conv(x)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout_rate > 0:
            self.dropout(net)
        return net

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False):
        """
        Implmentation of Unet's double 3x3 conv.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
        """
        super(DoubleConv, self).__init__()
        expansion_out = int(round(out_ch*expansion))
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv2d(in_ch, expansion_out, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)
        self.conv2 = Conv2d(expansion_out, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        return net

class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1, bias=False, **kwargs):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depth_wise = nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation,
                                    bias=bias,**kwargs)
        self.point_wise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, bias=bias, **kwargs)

    def forward(self, x):
        net = self.depth_wise(x)
        net = self.point_wise(net)
        return net


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,radix=2, drop_prob=0.0,
                 dilation=1, padding=1, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True),avd=False, avd_first=False, num_current=2):
        """
        Downsample part with different block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for inputs
            out_ch (int): number of channels for outputs
            convblock (nn.Module): block to extract features
            expansion (float):  expansion rate for channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (float): dilation rate for conv
            padding (Union[int, tuple]): the padding size
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """

        super(Downsample, self).__init__()
        self.conv = convblock(in_ch, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction, expansion=expansion, num_current=num_current)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feature = self.conv(x)
        downsample = self.down(feature)
        return feature, downsample


class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2,out_ch, convblock=DoubleConv,
                 radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2):
        """
        Upsample part with different conv block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch1 (int): number of channels for inputs1 (upsampled from last level)
            in_ch2 (int): number of channels for inputs2 (same level features)
            out_ch (int): number of channels for outputs
            convblock (nn.Module): block to extract features
            expansion (float):  expansion rate for channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (float): dilation rate for conv
            padding (Union[int, tuple]): the padding size
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        self.upsample_conv = Conv2d(in_ch1, out_ch, norm_layer=norm_layer, activation=activation)
        self.conv = convblock(out_ch+in_ch2, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction, expansion=expansion, num_current=num_current)

    def forward(self, x, x1):
        net = self.upsample(x)
        net = self.upsample_conv(net)
        net = torch.cat([net, x1], dim=1)
        net = self.conv(net)
        return net

class  ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2):
        """
        Implementation of ResNet's block.
        References:
            "Deep Residual Learning for Image Recognition"<https://arxiv.org/pdf/1512.03385.pdf>
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(ResBlock, self).__init__()
        expansion_out = int(round(out_ch * expansion))
        self.conv1 = Conv2d(in_ch, expansion_out, ksize=3, stride=1,
                            padding=dilation, dilation=dilation,
                            activation=activation, norm_layer=norm_layer)
        self.conv2 = Conv2d(expansion_out,  out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            activation=None, norm_layer=norm_layer)
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0,
                            activation=None, norm_layer=norm_layer)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcut(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net

class RConv(nn.Module):
    def __init__(self, out_ch, num_recurrent=2, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True)):
        """
        Recurrent conv.
        References:
            "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>
        Args:
            out_ch (int): number of output channels
            num_recurrent (int): times or recurrent
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
        """
        super(RConv, self).__init__()
        self.conv = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation)
        self.num_recurrent = num_recurrent

    def forward(self, x):
        x1 = x
        for i in range(self.num_recurrent):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class RRBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1,
                 padding=1, expansion=3, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False):
        """
        Recurrent block in R2Unet.
        References:
             "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>

        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            num_current(int): times or recurrent
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
        """
        super(RRBlock, self).__init__()
        self.conv1x1 = Conv2d(in_ch, out_ch, norm_layer=None, activation=None)
        self.conv = nn.Sequential(
            RConv(out_ch, num_current, norm_layer=norm_layer),
            RConv(out_ch, num_current, norm_layer=norm_layer)
        )

    def forward(self, x):
        net = self.conv1x1(x)
        identify = net
        net = self.conv(net)
        net = net + identify
        return net


class SplAtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=3,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2):
        """
        Implementation of block with Split Attention Module.
        References:
            "ResNeSt: Split-Attention Networks",https://hangzhang.org/files/resnest.pdf
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(SplAtBlock, self).__init__()
        expansion_out = int(round(out_ch * expansion))
        self.conv1 = Conv2d(in_ch, expansion_out, norm_layer=norm_layer, activation=activation)
        self.conv2 = SplAtConv2d(expansion_out, expansion_out, ksize=3, stride=1, padding=padding, dilation=dilation,
                                 radix=radix, drop_prob=drop_prob, norm_layer=norm_layer,
                                 nolinear=activation, reduction=reduction)
        self.conv3 = Conv2d(expansion_out, out_ch, norm_layer=norm_layer, activation=None)
        self.avd_layer = nn.Identity()
        if avd:
            self.avd_layer = nn.AvgPool2d(3, stride=1, padding=1)
        self.avd_first = avd_first
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0,
                                   activation=None, norm_layer=norm_layer)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcut(x)
        net = self.conv1(x)
        if self.avd_first:
            net = self.avd_layer(net)
        net = self.conv2(net)
        if not self.avd_first:
            net = self.avd_layer(net)
        net = self.conv3(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net