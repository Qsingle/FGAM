# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  effsegnet.py
@Time    :  2021/7/19 11:33
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_division(chs, multiplier=1.0, min_depth=8, divisor=8):
    """Round number of filters based on depth multiplier."""

    chs *= multiplier
    min_depth = min_depth or multiplier
    new_filters = max(min_depth, int(chs + divisor / 2) // divisor * divisor)
    return int(new_filters)

class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=4.0, activation=nn.Sigmoid()):
        super(SEModule, self).__init__()
        if activation is None:
            activation = nn.Sigmoid()
        hidden_ch = int(round(in_ch*(1/reduction)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_ch, in_ch, kernel_size=1, stride=1)
        self.activation = activation

    def forward(self, x):
        identity = x
        net = self.avg_pool(x)
        net = self.fc1(net)
        net = self.act1(net)
        net = self.fc2(net)
        net = self.activation(net) * identity
        return net

if hasattr(nn, "SiLU"):
    SiLU = nn.SiLU
else:
    class SiLU(nn.Module):
        def __init__(self):
            super(SiLU, self).__init__()

        def forward(self, x):
            return x*torch.sigmoid(x)

class MBConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4.0, se=True, se_reduction=4, activation=nn.ReLU(),
                 norm_layer=None, se_activation=nn.Sigmoid()):
        super(MBConv2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU()

        hidden_size = int(round(in_ch * expansion))
        #pw
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_size, kernel_size=1, stride=1, bias=False),
            norm_layer(hidden_size),
            activation
        )
        #dw
        self.conv2  = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, stride=stride,padding=1,
                      groups=hidden_size, bias=False),
            norm_layer(hidden_size),
            activation
        )
        self.se = None
        if se:
            self.se = SEModule(hidden_size, se_reduction, se_activation)

        #pw linear
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, out_ch, 1, 1, bias=False),
            norm_layer(out_ch)
        )
        self.short_cut = True
        if stride > 1 or in_ch != out_ch:
            self.short_cut = False
        self.activation = activation

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        if self.se is not None:
            net = self.se(net)
        net = self.conv3(net)
        if self.short_cut:
            net = net + x
        net = self.activation(net)
        return net

class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4.0, stride=1, se=True, se_reduction=4,
                 activation=nn.ReLU(), se_activation=nn.Sigmoid(), norm_layer=None):
        super(FusedMBConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU()
        hidden_size = int(round(in_ch*expansion))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_size, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(hidden_size),
            activation
        )
        self.se = None
        if se:
            self.se = SEModule(hidden_size, reduction=se_reduction, activation=se_activation)
        self.identity = True
        if in_ch != out_ch or stride > 1:
            self.identity = False
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, out_ch, 1,  bias=False),
            norm_layer(out_ch)
        )
        self.activation = activation

    def forward(self,x):
        net = self.conv1(x)
        if self.se is not None:
            net = self.se(net)
        net = self.conv2(net)
        if self.identity:
            net = net + x
        net = self.activation(net)
        return net

class CSPBlock(nn.Module):
    def __init__(self, block, num_layers, stride, in_ch, out_ch, expansion=4.0,
                 norm_layer=None, activation=SiLU(),
                 se=True, se_activation=nn.Sigmoid(), se_reduction=4.0):
        super(CSPBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.transition_a = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 1), stride=(1, 1)),
            norm_layer(out_ch),
            activation
        )
        self.down = None
        if stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                norm_layer(out_ch),
                activation
            )
            pre_ch = out_ch
        else:
            pre_ch = in_ch
        self.pre_conv = nn.Sequential(
            nn.Conv2d(pre_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_ch),
            activation
        )
        pre_ch = out_ch // 2
        blocks = []
        for i in range(num_layers):
            blocks.append(block(pre_ch, out_ch, stride=1, expansion=expansion, norm_layer=norm_layer,
                             activation=activation, se=se,se_activation=se_activation,
                             se_reduction=se_reduction))
            pre_ch = out_ch
        self.block = nn.Sequential(*blocks)
        self.transition_b = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, kernel_size=(1,1), stride=(1, 1)),
            norm_layer(out_ch // 2),
            activation
        )

    def forward(self,x):
        if self.down is not None:
            x = self.down(x)
        x = self.pre_conv(x)
        xa, xb = torch.split(x, x.shape[1]//2, dim=1)[:]
        xb = self.block(xb)
        xb = self.transition_b(xb)
        out = torch.cat([xa,xb], dim=1)
        out = self.transition_a(out)
        return out

class DecodeBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, expansion=1/4,
                 norm_layer=nn.BatchNorm2d, activation=SiLU(),
                 dilations=[1, 2, 4]):
        super(DecodeBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.resample = nn.Sequential(
            nn.Conv2d(in_ch1, out_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch2+out_ch, out_ch,kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )
        hidden_ch = make_division(out_ch, expansion)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_ch, hidden_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=dilations[0],
                      dilation=dilations[0], bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(out_ch, hidden_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=dilations[1],
                      dilation=dilations[1], bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(out_ch, hidden_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=dilations[2],
                      dilation=dilations[2], bias=False),
            norm_layer(hidden_ch),
            activation,
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(out_ch),
            activation
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode="bilinear", align_corners=True)
        x1 = self.resample(x1)
        net = torch.cat([x1, x2], dim=1)
        net = self.conv1(net)
        b1 = self.branch1(net)
        b2 = self.branch2(net)
        b3 = self.branch3(net)
        net = torch.cat([b1, b2, b3], dim=1)
        net = self.fusion_conv(net)
        return net



class EffSegNet(nn.Module):
    def __init__(self, in_ch, cfgs, num_classes=3, multiplier=1.0, min_depth=8, divisor=8,
                 norm_layer=nn.BatchNorm2d, activation=nn.PReLU(), se_reduction=4.0,
                 se_activation=nn.Sigmoid(), super_reso=False, upscale_rate=4):
        super(EffSegNet, self).__init__()
        self.blocks = nn.ModuleList()
        in_chs = in_ch
        out_ch = make_division(cfgs[0][2], multiplier, min_depth, divisor)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_ch),
            activation
        )
        block_type = {0:MBConv2d, 1:FusedMBConv}
        in_planes = out_ch
        for e, n, c, se, s, b in cfgs:
            block = block_type[b]
            in_ch = in_planes
            out_ch = make_division(c, multiplier, min_depth=min_depth, divisor=divisor)
            layers = []
            for i in range(n):
                layers.append(block(in_ch, out_ch, stride=s if i==0 else 1, expansion=e, se=se, norm_layer=norm_layer,
                                    se_reduction=se_reduction, se_activation=se_activation))
                in_ch = out_ch
            self.blocks.append(nn.Sequential(*layers))
            # self.blocks.append(CSPBlock(block, n, s, in_ch, out_ch, expansion=e, norm_layer=norm_layer,
            #                             activation=activation, se=se, se_activation=se_activation,
            #                             se_reduction=se_reduction))
            in_planes = out_ch
        up1_in_ch = make_division(cfgs[len(cfgs) // 2][2], multiplier, min_depth=min_depth, divisor=divisor)
        self.up1_conv = nn.Conv2d(in_planes, up1_in_ch, 1, bias=False)
        self.up1_resample = nn.Conv2d(up1_in_ch * 2, up1_in_ch, 1, bias=False)
        self.up1 = block_type[cfgs[0][-1]](up1_in_ch, up1_in_ch, se=cfgs[len(cfgs)//2][3],expansion=cfgs[len(cfgs)//2][0],
                             activation=activation, se_activation=se_activation, norm_layer=norm_layer,
                             se_reduction=se_reduction)

        # self.up1 = DecodeBlock(in_planes, up1_in_ch, up1_in_ch, norm_layer=norm_layer, activation=activation)
        up2_in_ch = make_division(cfgs[0][2], multiplier, min_depth=min_depth, divisor=divisor)
        self.up2 = DecodeBlock(up1_in_ch, up2_in_ch, up2_in_ch, norm_layer=norm_layer, activation=activation)
        # self.up2_conv = nn.Conv2d(up1_in_ch, up2_in_ch, 1, bias=False)
        # self.up2_resample = nn.Conv2d(up2_in_ch*2, up2_in_ch, 1, bias=False)
        # self.up2 = block_type[cfgs[0][-1]](up2_in_ch, up2_in_ch, se=cfgs[0][3],expansion=cfgs[0][0],
        #                      activation=activation, se_activation=se_activation, norm_layer=norm_layer,
        #                      se_reduction=se_reduction)
        self.out_conv = nn.Conv2d(up2_in_ch, num_classes, 1)

        self.up2_de = nn.Sequential(
            nn.Conv2d(up2_in_ch, up2_in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(up2_in_ch),
            activation,
            nn.Conv2d(up2_in_ch, up2_in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(up2_in_ch),
            activation
        )
        self.out_conv = nn.Conv2d(up2_in_ch, num_classes, 1)
        self.super_reso = super_reso
        if self.super_reso:
            self.sr = nn.Sequential(
                nn.Conv2d(up2_in_ch, 64, 3, stride=1, padding=1),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.Conv2d(32, in_chs*(upscale_rate**2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            )
            self.upscale_rate = upscale_rate

    def forward(self, x):
        net = self.conv1(x)
        outputs = []
        for block in self.blocks:
            net = block(net)
            outputs.append(net)

        # fusion1 = torch.cat([self.pool(self.pool(outputs[0])), self.pool(outputs[1]), outputs[2]], dim=1)
        # fusion1 = self.fusion1_conv1(fusion1)
        # fusion1 = self.fusion1_de(fusion1)
        # fusion2 = torch.cat([self.pool(outputs[3]), outputs[4], outputs[5]], dim=1)
        # fusion2 = self.fusion2_conv1(fusion2)
        # fusion2 = self.fusion2_de(fusion2)
        # fusion = torch.cat([fusion1, F.interpolate(fusion2, size=fusion1.size()[2:], mode="bilinear", align_corners=True)], dim=1)
        # up = self.up_conv(fusion)
        # up = F.interpolate(up, size=x.size()[2:], mode="bilinear", align_corners=True)
        # up = self.up(up)
        # out = self.out_conv(up)

        up1 = self.up1_conv(outputs[-1])
        up1 = F.interpolate(up1, size=outputs[len(self.blocks)//2].size()[2:], mode="bilinear", align_corners=True)
        up1 = torch.cat([outputs[len(self.blocks)//2],up1], dim=1)
        up1 = self.up1_resample(up1)
        up1 = self.up1(up1)
        # up2 = self.up2_conv(up1)
        # up2 = F.interpolate(up2, size=outputs[0].size()[2:], mode="bilinear", align_corners=True)
        # up2 = torch.cat([outputs[0], up2], dim=1)
        # up2 = self.up2_resample(up2)
        # up2 = self.up2(up2)

        # up1 = self.up1(outputs[-1], outputs[len(outputs)//2])
        # up2 = self.up2_conv(up1)
        # up2 = F.interpolate(up2, size=outputs[0].size()[2:], mode="bilinear", align_corners=True)
        # up2 = torch.cat([outputs[0], up2], dim=1)
        # up2 = self.up2_resample(up2)
        # up2 = self.up2(up2)
        up2 = self.up2(up1, outputs[0])
        if self.super_reso:
            up2_sr = None
            if self.training:
                up2_sr = torch.softmax(outputs[0], dim=1)*up2 + \
                      torch.softmax(torch.mean(outputs[0], dim=1, keepdim=True),dim=1)*up2 + \
                      torch.softmax(F.adaptive_avg_pool2d(outputs[0], 1), dim=1)*up2 + up2
            out = self.out_conv(up2)
            out = F.interpolate(out, mode="bilinear", size=[out.size(2)*self.upscale_rate, out.size(3)*self.upscale_rate],
                                align_corners=True)
            if self.training:
                sr = self.sr(up2_sr)
                return out, sr
            return out
        out = self.out_conv(up2)
        return out

base_cfgs = [
        #e, n, c, se, s, block
        [4, 2, 24, False, 1, 1],
        [4, 2, 48, False, 2, 1],
        [4, 2, 96, False, 2, 1],
        [4, 4, 128, True, 2, 0],
        [4, 4, 160, True, 2, 0],
        [4, 3, 256, True, 1, 0]
    ]

def _build_effsegnet(in_ch=3,cfgs=base_cfgs,  **kwargs):
    return EffSegNet(in_ch, cfgs,  **kwargs)

def effsegnets1(in_ch=3, **kwargs):
    return _build_effsegnet(in_ch, base_cfgs, **kwargs)

if __name__ == "__main__":
    import torch
    model = effsegnets1(super_reso=True, activation=nn.SiLU(inplace=True))
    x = torch.randn((4, 3, 256, 512))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device, dtype=torch.float32)
    # model.eval()
    model.to(device)
    # with torch.no_grad():
    print(model(x)[1].shape)
