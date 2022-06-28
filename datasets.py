# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/12 20:29
    @filename: datasets.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from albumentations import ChannelDropout, ColorJitter, ShiftScaleRotate, GaussianBlur
from albumentations import Compose, Normalize, Resize, HorizontalFlip, VerticalFlip

import os
from PIL import Image

class SegPathDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augmentation=False, image_size=256, divide=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.image_size = image_size
        assert len(img_paths) == len(mask_paths)
        self.length = len(self.img_paths)
        self.divide = divide

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        if self.augmentation:
            aug_task = [
                ChannelDropout(),
                ColorJitter(),
                ShiftScaleRotate(),
                HorizontalFlip(),
                VerticalFlip(),
                GaussianBlur()
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=img, mask=mask)
            img = aug_data["image"]
            mask = aug_data["mask"]
        nor_resize = Compose(
           [
               Resize(self.image_size, self.image_size, always_apply=True),
               Normalize([0.5]*3, std=[0.5]*3, always_apply=True)
           ]
        )
        if self.divide:
            mask = mask // 255
        nor_resize_data = nor_resize(image=img, mask=mask)
        img = nor_resize_data["image"]
        mask = nor_resize_data["mask"]
        if img.ndim == 3:
            img = np.transpose(img, axes=[2, 0, 1])
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

class DRIVEDataset(Dataset):
    def __init__(self, paths, mask_dir, image_size=256, augmentation=False):
        self.paths = paths
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augmentation = augmentation
        self.length = len(paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.paths[index]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, filename.split("_")[0] + "_manual1"+".gif")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_path))
        if self.augmentation:
            aug_task = [
                ShiftScaleRotate(),
                HorizontalFlip(),
                VerticalFlip(),
                GaussianBlur()
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=img, mask=mask)
            img = aug_data["image"]
            mask = aug_data["mask"]
        nor_resize = Compose(
            [
                Resize(self.image_size, self.image_size, always_apply=True),
                Normalize([0.5] , std=[0.5] , always_apply=True)
            ]
        )
        mask = mask // 255
        nor_resize_data = nor_resize(image=img, mask=mask)
        img = nor_resize_data["image"]
        mask = nor_resize_data["mask"]
        if img.ndim == 3:
            img = np.transpose(img, axes=[2, 0, 1])
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

class CityScapeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, output_size=(256, 512), augmentaion=False, super_reso=False,
                 shift_limit=0.0625, scale_limit=0.1, rotate_limit=45):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_size = output_size
        self.augmentation = augmentaion
        self.super_reso = super_reso
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        # self._regular_init()


    # def _regular_init(self):
    #     self.init_shift_limit = np.random.uniform(0.0, self.shift_limit*0.6)
    #     self.init_scale_limit = np.random.uniform(0, self.scale_limit*0.6)
    #     self.init_rotate_limit = np.random.randint(0, int(self.rotate_limit*0.6))
    #     self.init_h = np.random.randint(0, int(self.output_size[0]*0.85))
    #     self.init_w = np.random.randint(0, int(self.output_size[1]*0.85))

    def __len__(self):
        return len(self.image_paths)

    # def regular_update(self, stage, total_stage):
    #     factor = stage/total_stage
    #     self._shift_limit = self.init_shift_limit + (self.shift_limit - self.init_shift_limit) * factor
    #     self._scale_limit = self.init_scale_limit + (self.scale_limit - self.init_scale_limit) * factor
    #     self._rotate_limit = self.init_rotate_limit + (self.rotate_limit - self.init_rotate_limit) * factor
    #     output_h = self.init_h + (self.output_size[0] - self.init_h) * factor
    #     output_w = self.init_w + (self.output_size[1] - self.init_w) * factor
    #     self._output_size = (int(output_h), int(output_w))

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None
        assert os.path.exists(self.mask_paths[index]), self.mask_paths[index]
        mask = cv2.imread(self.mask_paths[index], 0)
        assert mask is not None, "Cannot open {}".format(self.mask_paths[index])
        if self.augmentation:
            task_list = [
                ChannelDropout(),
                ColorJitter(),
                ShiftScaleRotate(shift_limit=self.shift_limit, scale_limit=self.scale_limit, rotate_limit=self.rotate_limit),
                HorizontalFlip(),
                VerticalFlip(),
                GaussianBlur(),
            ]
            aug = Compose(task_list)
            aug_data = aug(image=img, mask=mask)
            img = aug_data["image"]
            mask = aug_data["mask"]
        nor_resize = Compose([
            Resize(height=self.output_size[0], width=self.output_size[1]),
            Normalize()
        ])
        if self.super_reso:
            img_lr = nor_resize(image=img)["image"]
            img = Normalize()(image=img)["image"]
            if img.ndim > 2:
                img = np.transpose(img, axes=[2, 0, 1])
                img_lr = np.transpose(img_lr, axes=[2, 0, 1])
            return torch.from_numpy(img_lr), torch.from_numpy(img), torch.from_numpy(mask)
        nor_re_data = nor_resize(image=img, mask=mask)
        img = nor_re_data["image"]
        mask = nor_re_data["mask"]
        if img.ndim > 2:
            img = np.transpose(img, axes=[2, 0, 1])
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

class OCTADataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=256, augmentation=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.length = len(image_paths)
        self.augmentation = augmentation

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        mask = np.where(mask == 100, 1, 0) + np.where(mask == 255, 2, 0)
        if self.augmentation:
            aug_task = [
                ChannelDropout(),
                ColorJitter(),
                ShiftScaleRotate(),
                HorizontalFlip(),
                VerticalFlip(),
                GaussianBlur()
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        normalize= Compose([
            Normalize([0.5]*3, [0.5]*3, always_apply=True)
        ])
        normalize_data = normalize(image=image, mask=mask)
        image = normalize_data["image"]
        mask = normalize_data["mask"]
        if image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)

class Decathlon(Dataset):
    def __init__(self, image_paths, mask_paths, augmentation=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        img = img.astype(np.float32)
        # img = (img - img.min()) / (img.max() - img.min() + 1e-9)

        # mask = mask.astype(np.uint8)
        # if img.shape[0] != 320 or img.shape[1] != 320:
        #     img = cv2.resize(img, (320, 320))
        #     mask = cv2.resize(mask, (320, 320))
        if self.augmentation:
            aug_task = [
                # ColorJitter(),
                ShiftScaleRotate(),
                HorizontalFlip(),
                VerticalFlip(),
                # GaussianBlur()
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=img, mask=mask)
            img = aug_data["image"]
            mask = aug_data["mask"]

        img = np.expand_dims(img, axis=0)
        return img, mask


class Kits19Dataset(Dataset):
    def __init__(self, img_paths, mask_paths, hu_scale=[-79, 304], augmentation=False):
        """
        Dataset to load the kits 2019 data pair.
        Args:
            img_paths:
            mask_paths:
            hu_scale:
            augmentation:
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.hu_scale = [min(hu_scale), max(hu_scale)]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_paths)

    def _normal(self, img, hu_scale):
        img = np.clip(img, hu_scale[0], hu_scale[1])
        img = (img - 101) / 76.9
        return img

    def __getitem__(self, index):
        img = np.load(self.img_paths[index])
        mask = cv2.imread(self.mask_paths[index], 0)
        img = self._normal(img, self.hu_scale)
        img = img.astype(np.float32)
        mask = mask.astype(np.uint8)
        if img.shape[0] != 512 or img.shape[1] != 512:
            img = cv2.resize(img, (512, 512))
            mask = cv2.resize(mask, (512, 512))
        if self.augmentation:
            aug_task = [
                ColorJitter(),
                ShiftScaleRotate(),
                HorizontalFlip(),
                VerticalFlip(),
                GaussianBlur()
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=img, mask=mask)
            img = aug_data["image"]
            mask = aug_data["mask"]

        img = np.expand_dims(img, axis=0)
        return img, mask

def get_paths(image_dir, mask_dir):
    files = os.listdir(image_dir)
    IMAGE_FORMAT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif")
    image_paths = []
    mask_paths = []
    for file in files:
        if os.path.isfile(os.path.join(image_dir, file)):
            base, format = os.path.splitext(file)
            if format.lower() in IMAGE_FORMAT:
                img_path = os.path.join(image_dir, file)
                mask_path = os.path.join(mask_dir, file)
                image_paths.append(img_path)
                mask_paths.append(mask_path)
    return image_paths, mask_paths