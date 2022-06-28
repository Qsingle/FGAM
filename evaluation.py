# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/14 9:22
    @filename: test.py.py
    @software: PyCharm
"""
import torch
from albumentations import Compose, Resize, Normalize
import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
import glob

from datasets import get_paths
from metrics.metric import Metric
from modules import Unet,SplAtBlock,ResBlock,DoubleConv, RRBlock, NestedUNet, AttUnet
from modules.deeplab import DeeplabV3Plus,DeeplabV3
from modules.encnet import EncNet
from modules.danet import DANet

def main():
    args = parser.parse_args()
    gpu = args.gpu
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    num_classes = args.num_classes
    image_size = args.image_size
    channels = args.channels
    block_name = args.block_name
    weights = args.weights
    model_name = args.model_name
    backbone = args.backbone
    if model_name == "unet":
        if block_name == "splat":
            block = SplAtBlock
        elif block_name == "res":
            block = ResBlock
        elif block_name == "novel":
            block = DoubleConv
        elif block_name == "rrblock":
            block = RRBlock
        else:
            raise ValueError("unknown type of block, except in [splat, res, novel] but got {}".format(block_name))
        model = Unet(in_ch=channels, out_ch=num_classes, convblock=block, expansion=args.expansion,
                     drop_prob=args.drop_prob, reduction=args.reduction, avd=args.avd, avd_first=args.avd_first,
                     fgam=args.fgam)
    elif model_name == "deeplabv3":
        model = DeeplabV3(in_ch=channels, num_classes=num_classes, backbone=backbone)
    elif model_name == "deeplabv3plus":
        model = DeeplabV3Plus(in_ch=channels, num_classes=num_classes, backbone=backbone,
                              fgam=args.fgam)
    elif model_name == "encnet":
        model = EncNet(in_ch=channels, num_classes=num_classes, backbone=backbone)
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes, input_channels=channels, deep_supervision=True, fgam=args.fgam)
    elif model_name == "attunet":
        model = AttUnet(channels, num_classes=num_classes)
    elif model_name == "danet":
        model = DANet(num_classes, backbone)
    else:
        raise ValueError("unknown model name")
    att = "no"
    if args.fgam:
        att = "attention"
    output_dir = f"./output/{model_name}_{block_name}_{args.dataset}_{att}_pre"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert os.path.exists(weights), "Weight file {} not exits".format(weights)

    state_dict = torch.load(weights, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except Exception:
        new_state = {k[7:]:v for k,v in state_dict.items()}
        model.load_state_dict(new_state)
    paths, masks = get_paths(image_dir, mask_dir)
    dataset_name = args.dataset
    if dataset_name == "drive":
        for i in range(len(masks)):
            masks[i] = os.path.join(os.path.dirname(masks[i]), os.path.basename(masks[i]).split("_")[0] + "_manual1.gif")
    elif dataset_name == "octa":
        csv_path = args.csv_path
        assert csv_path is not None, "Octa dataset must give csv path"
        data = pd.read_csv(csv_path)
        filenames = data["filenames"].values
        paths = [os.path.join(image_dir, f) for f in filenames]
        masks = [os.path.join(mask_dir, f) for f in filenames]
    elif dataset_name == "oct":
        csv_path = args.csv_path
        assert csv_path is not None, "Oct dataset must give csv path"
        data = pd.read_csv(csv_path)
        image_paths = data["image_path"]
        mask_paths = data["mask_path"]
        paths = [os.path.join(image_dir, p) for p in image_paths]
        masks = [os.path.join(mask_dir, p) for p in mask_paths]
    elif dataset_name == "chest":
        csv_path = args.csv_path
        assert csv_path is not None, "Oct dataset must give csv path"
        data = pd.read_csv(csv_path)
        mask_paths = data["mask"].values
        image_paths = data["image"].values
        paths = [os.path.join(image_dir, f) for f in image_paths]
        masks = [os.path.join(mask_dir, f) for f in mask_paths]
        args.divide = True
    elif dataset_name == "gland":
        paths = glob.glob(os.path.join(image_dir, "test[AB]_*[0-9].bmp"))
        masks = glob.glob(os.path.join(image_dir, "test[AB]_*[0-9]_anno.bmp"))
    elif dataset_name == "skin":
        paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        masks = [os.path.splitext(path)[0] + "_Segmentation.png" for path in paths]
        args.divide = True
    elif dataset_name == "origa":
        masks = [os.path.splitext(path)[0] + ".png" for path in masks]
    metric = Metric(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mean = [0.5]*3
    std = [0.5]*3
    if dataset_name == "drive":
        mean = [0.5]*3
        std = [0.5]*3
    nor_resize = Compose(
        [
            Resize(image_size, image_size, always_apply=True),
            Normalize(mean=mean, std=std, always_apply=True)
        ]
    )
    with torch.no_grad():
        for i in range(len(masks)):
            img = cv2.imread(paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            base, ext = os.path.splitext(masks[i])
            mask = cv2.imread(masks[i], 0)
            if ext == ".gif":
                mask = Image.open(masks[i])
                mask = np.asarray(mask)
            if dataset_name == "drive" or dataset_name == "skin" or dataset_name == "chest" or args.divide:
                mask = mask // 255
            elif dataset_name == "octa":
                mask = np.where(mask == 100, 1, 0) + np.where(mask == 255, 2, 0)
            # if dataset_name == "drive":
            #     img = img[..., 1]

            img = nor_resize(image=img)["image"]
            if dataset_name == "drive":
                # x = np.expand_dims(img, axis=0)
                x = np.transpose(img, axes=[2, 0, 1])
            else:
                x = np.transpose(img, axes=[2, 0, 1])
            x = torch.from_numpy(x).unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device, dtype=torch.long)
            filename = os.path.basename(paths[i])
            h, w = mask.shape[:2]
            pred = model(x)
            if isinstance(pred, list) or isinstance(pred, tuple):
                if model_name == "nestedunet":
                    pred = pred[-1]
                else:
                    pred = pred[0]
            pred = torch.softmax(pred, dim=1)
            pred = torch.max(pred, dim=1)[1]
            pred = pred.detach().cpu().numpy().squeeze()
            pred = cv2.resize(pred.astype("uint8"), (w, h))
            pred = torch.from_numpy(pred).to(device, dtype=torch.long)
            metric.update(pred.squeeze(), mask.squeeze())
            pred = pred.detach().cpu().numpy()
            pred = (pred-pred.min()) / (pred.max() - pred.min() + 1e-9) * 255
            if dataset_name == "drive":
                cv2.imwrite(os.path.join(output_dir, filename.split("_")[0]+".png"), pred.astype("uint8"))
            else:
                cv2.imwrite(os.path.join(output_dir, filename), pred.astype("uint8"))
            #pred = torch.max(pred, dim=1)[1].detach().cpu().numpy()
            # for pre in pred:
            #     io.imsave(os.path.join(output_dir, os.path.basename(paths[cnt])),pre)
            #     cnt+=1
        precision, acc, recall, f1, specificity, mean_iou, mean_dice = metric.evalutate()
        print(f"precision:{precision} acc:{acc} recall:{recall} f1:{f1}  specificity:{specificity} "
              f" mean_iou:{mean_iou} {mean_dice}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=20,
                        help="number of classes")
    parser.add_argument("--image_size", type=int, default=256,
                        help="image size for resize")
    parser.add_argument("-j", "--num_workers", type=int, default=4,
                        help="number of workers to load data")
    parser.add_argument("-b","--batch_size", type=int, default=4,
                        help="number of images for each train step")
    parser.add_argument("--image_dir", type=str, default="./train",
                        help="path of directory that stored image")
    parser.add_argument("--mask_dir", type=str, default="./train_mask",
                        help="path of directory that stored mask")
    parser.add_argument("--divide", action="store_true")
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument("--channels", type=int, default=3,
                        help="number of channels for input")
    parser.add_argument("--block_name", type=str, choices=["res", "splat", "novel", "rrblock"],
                        default="splat", help="type of block used to ")
    parser.add_argument("--avd", action="store_true",
                        help="whether use avd layer")
    parser.add_argument("--avd_first", action="store_true",
                        help="whether use avd layer before splat conv")
    parser.add_argument("--reduction", type=int, default=4,
                        help="reduction rate")
    parser.add_argument("--drop_prob", type=float, default=0.0,
                        help="dropout rate")
    parser.add_argument("--fgam", action="store_true",
                        help="whether use feature guided attention")
    parser.add_argument("--expansion", type=float, default=2.0,
                        help="expansion rate for hidden channel")
    parser.add_argument("--weights", type=str, default="./ckpt",
                        help="path of directory to store weights")
    parser.add_argument("--dataset", type=str, default=None,
                        help="name of dataset")
    parser.add_argument("--model_name", type=str,
                        choices=["deeplabv3", "deeplabv3plus", "encnet", "unet", "nestedunet", "attunet", "danet"],
                        help="name of model")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "resnest50", "resnest101", "seresnet50"],
                        default="resnet50",
                        help="name of backbone for deeplab and encnet")
    parser.add_argument("--csv_path", type=str, default=None)
    main()