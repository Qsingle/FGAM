# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:fpn
    author: 12718
    time: 2021/9/22 11:23
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_chs, num_classes):
        super(FPN, self).__init__()
        blocks = []
        for i in range(len(in_chs)):
            blocks.append(
                nn.Conv2d(in_chs[i], 256, 1, 1)
            )
        self.linears = nn.ModuleList()
        for _ in range(len(in_chs)):
            self.linears.append(nn.Linear(256, num_classes))
        self.blocks = nn.ModuleList(blocks[-1:])

    def forward(self, inputs:list):
        inputs.reverse()
        preds = []
        net = self.blocks[0](inputs[0])
        flatten = F.adaptive_avg_pool2d(net, output_size=1)
        flatten = torch.flatten(flatten, start_dim=1)
        preds.append(self.linears[0](flatten))
        for i in range(1,len(self.blocks)):
            net = F.interpolate(net, inputs[i].size()[2:], mode="bilinear", align_corners=True) \
                  + self.blocks[i](inputs[i])
            flatten = F.adaptive_avg_pool2d(net, output_size=1)
            flatten = torch.flatten(flatten, dims=1)
            preds.append(self.linears[i](flatten))
        return preds