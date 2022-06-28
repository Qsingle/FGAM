# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 16:46
    @filename: unet.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import *
from .superpixel import SuperResolutionModule

__all__ = ["Unet", "NestedUNet", "AttUnet"]

class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def forward(self, x):
        """
        H-Swish activation
        Math:
            x*\frac{relu6(x+3)}{6}
        Args:
            x (tensor): input value

        Returns:
            tensor
        """
        return x*F.relu6(x+3, inplace=True) / 6

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x+3) / 6

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)




class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 fgam=False, super_reso=False):
        """
        Unet with different block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            convblock (nn.Module):
            radix (int): number of groups, default 2
            drop_prob (float): dropout rate, default 0.0
            expansion (float): expansion rate for channels, default 1.0
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer before SplAtConv
            fgam(bool): whether use FGAM module
            multi_head (bool): whether use multi-head attention
            num_head (int): number of head for multi-head attention
        """
        super(Unet, self).__init__()
        self.down1 = Downsample(in_ch, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down2 = Downsample(64, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down3 = Downsample(128, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down4 = Downsample(256, 512, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down5 = convblock(512, 1024, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.fgam = fgam
        self.super_reso = super_reso
        if super_reso:
            self.sup = SuperResolutionModule(64)
            self.sup_conv = convblock(3, 64, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
            self.sup_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.up6 = Upsample(1024, 512, 512, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.up7 = Upsample(512, 256, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.up8 = Upsample(256, 128, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)

        self.up9 = Upsample(128, 64, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        up6 = self.up6(down5, down4_f)
        up7 = self.up7(up6, down3_f)
        up8 = self.up8(up7, down2_f)
        up9 = self.up9(up8, down1_f)
        if self.fgam:
            up9 = up9 * torch.softmax(down1_f, dim=1) + up9

        out = self.out_conv(up9)
        out = F.softmax(out, dim=1)
        return out

class EffDoubleConv(nn.Module):
    def __init__(self,  in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=SiLU(),
                 avd=False, avd_first=False):
        """

        Args:
            in_ch:
            out_ch:
            num_current:
            radix:
            drop_prob:
            dilation:
            padding:
            expansion:
            reduction:
            norm_layer:
            activation:
            avd:
            avd_first:
        """
        super(EffDoubleConv, self).__init__()
        hidden_size = int(in_ch*expansion)
        self.conv1 = nn.Sequential(
            DepthWiseSeparableConv2d(in_ch, hidden_size, ksize=3, stride=1, padding=1),
            norm_layer(hidden_size),
            activation
        )

        self.conv2 = nn.Sequential(
            DepthWiseSeparableConv2d(hidden_size, out_ch, ksize=3, stride=1, padding=1),
            norm_layer(out_ch),
            activation
        )

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        return net

class EffUNet(nn.Module):
    def __init__(self, in_ch, num_classes=3, expansion=4, layer_attention=False,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=SiLU(), avd=False, avd_first=False,):
        """

        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            convblock (nn.Module):
            radix (int): number of groups, default 2
            drop_prob (float): dropout rate, default 0.0
            expansion (float): expansion rate for channels, default 1.0
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer before SplAtConv
            layer_attention (bool): whether use layer attention
        """
        super(EffUNet, self).__init__()
        convblock = DoubleConv
        eff_convblock = DoubleConv
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.down1 = Downsample(in_ch, 24, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down2 = Downsample(24, 48, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down3 = Downsample(48, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)

        self.down4 = Downsample(64, 128, convblock=eff_convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down5 = eff_convblock(128, 256, radix=radix, drop_prob=drop_prob,
                               dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                               norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.layer_attention = layer_attention
        #
        self.up6 = Upsample(256, 128, 128, convblock=eff_convblock, radix=radix, drop_prob=drop_prob,
                            dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                            norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.up7 = Upsample(128, 64, 64, convblock=eff_convblock, radix=radix, drop_prob=drop_prob,
                            dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                            norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.up8 = Upsample(64, 48, 48, convblock=convblock, radix=radix, drop_prob=drop_prob,
                            dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                            norm_layer=norm_layer, activation=activation, expansion=expansion)


        self.up9 = Upsample(48, 24, 24, convblock=convblock, radix=radix, drop_prob=drop_prob,
                            dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                            norm_layer=norm_layer, activation=activation, expansion=expansion)

        self.out_conv = nn.Conv2d(24, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        up6 = self.up6(down5, down4_f)
        up7 = self.up7(up6, down3_f)
        up8 = self.up8(up7, down2_f)
        up9 = self.up9(up8, down1_f)
        if self.layer_attention:
            up9 = up9 * torch.softmax(down1_f, dim=1) + up9
            # up9 = up9 * torch.sigmoid(down1_f) + up9
            #up9 = up9 * torch.softmax(F.adaptive_avg_pool2d(down1_f, 1), dim=1) + up9* torch.softmax(down1_f.mean(1),dim=1) + up9

        out = self.out_conv(up9)
        if out.size() != x.size():
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, fgam=False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.fgam = fgam

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        if self.fgam:
            x0_4 = torch.sigmoid(x0_0) * x0_4 + x0_4
            if self.deep_supervision:
                x0_3 = torch.softmax(x0_0, dim=1)*x0_3 + x0_3
                x0_2 = torch.softmax(x0_0, dim=1)*x0_2 + x0_2
                x0_1 = torch.softmax(x0_0, dim=1)*x0_1 + x0_1

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttentionBlock(nn.Module):
    def __init__(self, in_ch_g, in_ch_l, out_ch):
        super(AttentionBlock, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch_l, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_g = Conv2d(in_ch_g, out_ch, ksize=1, stride=1, padding=0)
        self.conv_l = Conv2d(out_ch, out_ch, ksize=1, stride=1, padding=0)
        self.psi = Conv2d(out_ch, 1, activation=nn.Sigmoid())

    def forward(self, x1, x2):
        x2 = self.up_conv(x2)
        x2 = self.conv_l(x2)
        x1 = self.conv_g(x1)
        net = x1+x2
        net = F.relu(net, inplace=True)
        psi = self.psi(net)
        return net * psi

class AttUnet(nn.Module):
    def __init__(self, in_ch, num_classes, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 fgam=False):
        super(AttUnet, self).__init__()
        self.down1 = Downsample(in_ch, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down2 = Downsample(64, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down3 = Downsample(128, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down4 = Downsample(256, 512, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down5 = convblock(512, 1024, radix=radix, drop_prob=drop_prob,
                               dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                               norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.fgam = fgam

        self.att6 = AttentionBlock(512, 1024, 512)
        self.up6 = convblock(512, 512)
        self.att7 = AttentionBlock(256, 512, 256)
        self.up7 = convblock(256, 256)
        self.att8 = AttentionBlock(128, 256, 128)
        self.up8 = convblock(128, 128)
        self.att9 = AttentionBlock(64, 128, 64)
        self.up9 = convblock(64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        att6 = self.att6(down4_f, down5)
        up6 = self.up6(att6)
        att7 = self.att7(down3_f, up6)
        up7 = self.up7(att7)
        att8 = self.att8(down2_f, up7)
        up8 = self.up8(att8)
        att9 = self.att9(down1_f, up8)
        up9 = self.up9(att9)
        if self.fgam:
            up9 = F.softmax(down1_f, dim=1) * up9 + up9
        out = self.out_conv(up9)
        return out

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample3D, self).__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.down = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        net = self.conv(x)
        down = self.down(net)
        return net, down

class Upsample3D(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(Upsample3D, self).__init__()
        self.conv = DoubleConv3D(in_ch1+in_ch2, out_ch)


    def forward(self, x1, x2):
        up = F.interpolate(x2, x1.size()[2:], mode="trilinear", align_corners=True)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net


class UNet3DMultiModal(nn.Module):
    def __init__(self, in_ch=1, num_classes=7):
        super(UNet3DMultiModal, self).__init__()
        self.modal1_down1 = Downsample3D(in_ch, 64)
        # self.modal2_down1 = Downsample3D(in_ch, 64)
        # self.modal3_down1 = Downsample3D(in_ch, 64)

        self.modal1_down2 = Downsample3D(64, 128)
        # self.modal2_down2 = Downsample3D(64, 128)
        # self.modal3_down2 = Downsample3D(64, 128)
        #
        self.modal1_down3 = Downsample3D(128, 256)
        # self.modal2_down3 = Downsample3D(128, 256)
        # self.modal3_down3 = Downsample3D(128, 256)
        #
        self.modal1_down4 = DoubleConv3D(256, 512)
        # self.modal2_down4 = DoubleConv3D(256, 512)
        # self.modal3_down4 = DoubleConv3D(256, 512)

        self.up5 = Upsample3D(256, 512, 256)
        self.up6 = Upsample3D(128, 256, 128)
        self.up7 = Upsample3D(64, 128, 64)

        self.out_conv = nn.Conv3d(
            64, num_classes, 1, 1
        )

    def forward(self, x):
        modal1_1, modal1_down = self.modal1_down1(x)
        modal1_2, modal1_down = self.modal1_down2(modal1_down)
        modal1_3, modal1_down = self.modal1_down3(modal1_down)
        modal1_4 = self.modal1_down4(modal1_down)

        # modal2_1, modal2_down = self.modal2_down1(x2)
        # modal2_2, modal2_down = self.modal2_down2(modal2_down)
        # modal2_3, modal2_down = self.modal2_down3(modal2_down)
        # modal2_4 = self.modal2_down4(modal2_down)
        #
        # modal3_1, modal3_down = self.modal3_down1(x3)
        # modal3_2, modal3_down = self.modal3_down2(modal3_down)
        # modal3_3, modal3_down = self.modal3_down3(modal3_down)
        # modal3_4 = self.modal3_down4(modal3_down)

        # down_1 = modal1_1 + modal2_1 + modal3_1
        # down_2 = modal1_2 + modal2_2 + modal3_2
        # down_3 = modal1_3 + modal2_3 + modal3_3
        # down_4 = modal1_4 + modal2_4 + modal3_4
        up5 = self.up5(modal1_3, modal1_4)
        up6 = self.up6(modal1_2, up5)
        up7 = self.up7(modal1_1, up6)
        out = self.out_conv(up7)
        return out





if __name__ == "__main__":
    #model = Unet(3, 3, convblock=SplAtBlock, expansion=4.0, avd=True, layer_attention=True)
    #model = MultiHeadLayerAttention(3, 64)
    # model = LayerAttentionModule(3, 3, expansion=1.0)
    model = UNet3DMultiModal()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    x = torch.randn((2, 1, 20, 32, 32))
    x1 = torch.randn((2, 1, 20, 32, 32))
    with torch.no_grad():
        out = model(x, x1, x1)
        print("Out Shape:", out.shape)