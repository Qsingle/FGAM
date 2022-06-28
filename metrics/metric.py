# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/13 11:09
    @filename: metric.py
    @software: PyCharm
"""

import torch
import numpy as np

from .confusion_matrix import confusion_matrix

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = torch.zeros((self.num_classes, self.num_classes))

    def update(self, output, target):
        if output.dim() == 4:
            output = torch.max(output, dim=1)[1]
        matrix = confusion_matrix(output, target, self.num_classes)
        self.matrix += matrix.detach().cpu()

    def evalutate(self):
        FP = self.matrix.sum(0) - torch.diag(self.matrix)
        FN = self.matrix.sum(1) - torch.diag(self.matrix)
        TP = torch.diag(self.matrix)
        TN = self.matrix.sum() - (FP + FN + TP)
        precision = TP / (TP + FP)
        acc = (TP + TN) / (TP+FP+FN+TN)
        recall = TP / (TP + FN)
        f1 =  2 * (precision * recall) / (precision + recall)
        specficity = TN / (TN + FP)
        iou = TP / (TP + FN +FP)
        dice = (2*TP) / (2*TP + FN + FP)
        if self.num_classes > 2:
            mean_precision = np.nanmean(precision.detach().cpu().numpy())
            mean_acc = np.nanmean(acc.detach().cpu().numpy())
            mean_recall = np.nanmean(recall.detach().cpu().numpy())
            mean_f1 = np.nanmean(f1.detach().cpu().numpy()) #f1-score
            mean_specficity = np.nanmean(specficity.detach().cpu().numpy())
            mean_iou = np.nanmean(iou.cpu().numpy())
            mean_dice = np.nanmean(dice.cpu().numpy())
        else:
            mean_precision = precision.detach().cpu().numpy()[1]
            mean_acc = acc.detach().cpu().numpy()[1]
            mean_recall = recall.detach().cpu().numpy()[1]
            mean_f1 = f1.detach().cpu().numpy()[1] #f1-score
            mean_specficity = specficity.detach().cpu().numpy()[1]
            mean_iou = iou.cpu().numpy()[1]
            mean_dice = dice.cpu().numpy()[1]
        return mean_precision, mean_acc, mean_recall,  mean_f1, mean_specficity, mean_iou, mean_dice