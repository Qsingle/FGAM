# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/12 20:39
    @filename: train.py
    @software: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import glob



from modules.unet import Unet, NestedUNet, AttUnet
from modules.utils import SplAtBlock,ResBlock, DoubleConv, RRBlock
from modules.deeplab import DeeplabV3Plus, DeeplabV3
from modules.danet import DANet
from modules.encnet import EncNet
from datasets import get_paths, DRIVEDataset, SegPathDataset, OCTADataset
from metrics.metric import Metric
from modules.saunet import SAUnet

class EncLoss(nn.Module):
    def __init__(self, num_classes):
        super(EncLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        loss1 = self.cross_entropy(outputs[0], target)
        seg_target = self._get_batch_label_vector(target, self.num_classes)
        loss2 = self.bce(torch.sigmoid(outputs[1]), seg_target.to(outputs[1].device))
        loss = loss1 + loss2
        return loss

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.autograd.Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect

class IterScheduler:
    def __init__(self, optimizer, num_images, epochs,  batch_size, base_lr=0.01, power=0.9):
        self.total_iterations = int(num_images * epochs / batch_size)
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.power = power
        self.reset()

    def reset(self):
        self.iteration = 0

    def step(self):
        for param in self.optimizer.param_groups:
            param["lr"] = self.base_lr * ((1 - self.iteration / self.total_iterations) ** self.power)

        self.iteration += 1


best_iou = 0


def main():
    args = parser.parse_args()
    print("prepare")
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        args.ngpus_per_node = ngpus_per_node if args.ngpus_per_node is None else args.ngpus_per_node
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_iou
    args.gpu = gpu

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    channels = args.channels
    block_name = args.block_name
    num_classes = args.num_classes
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
        model = Unet(in_ch=channels,out_ch=num_classes,convblock=block, expansion=args.expansion,
                     drop_prob=args.drop_prob, reduction=args.reduction, avd=args.avd, avd_first=args.avd_first,
                     fgam=args.fgam)
        model_name = model_name + "_" + block_name
    elif model_name == "deeplabv3":
        model = DeeplabV3(in_ch=channels, num_classes=num_classes, backbone=backbone)
    elif model_name == "deeplabv3plus":
        model = DeeplabV3Plus(in_ch=channels, num_classes=num_classes, backbone=backbone, fgam=args.fgam)
    elif model_name == "encnet":
        model = EncNet(in_ch=channels, num_classes=num_classes, backbone=backbone)
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes, input_channels=channels, deep_supervision=True, fgam=args.fgam)
    elif model_name == "attunet":
        model = AttUnet(channels, num_classes=num_classes, fgam=args.fgam)
    elif model_name == "danet":
        model = DANet(num_classes, backbone=backbone)
    elif model_name == "saunet":
        model = SAUnet(channels, num_classes=num_classes)
    else:
        raise ValueError("unknown model name")
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # batch_size = int(args.batch_size / ngpus_per_node)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if torch.cuda.device_count() > 1:
            model = DP(model).cuda()
        else:
            model = model.cuda()

    cudnn.benchmark = True
    image_dir = args.image_dir
    mask_dir = args.mask_dir

    dataset = args.dataset
    csv_path = args.csv_path
    if csv_path is not None:
        data = pd.read_csv(csv_path)
        if "octa" == dataset:
            filenames = data["filenames"].values
            image_paths = [os.path.join(image_dir, filename) for filename in filenames]
            mask_paths = [os.path.join(mask_dir, filename) for filename in filenames]
        elif "oct" == dataset:
            mask_dir = image_dir
            image_paths = data["image_path"].values
            mask_paths = data["mask_path"].values
            image_paths = [os.path.join(image_dir, p) for p in image_paths]
            mask_paths = [os.path.join(mask_dir, p) for p in mask_paths]
        elif "chest" == dataset:
            mask_paths = data["mask"].values
            image_paths = data["image"].values
            image_paths = [os.path.join(image_dir, f) for f in image_paths]
            mask_paths = [os.path.join(mask_dir, f) for f in mask_paths]
            args.divide = True
        else:
            raise ValueError("Unknown dataset name")
    else:
        image_paths, mask_paths = get_paths(image_dir, mask_dir)
    if dataset == "skin":
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        mask_paths = [os.path.splitext(path)[0] + "_Segmentation.png" for path in image_paths]
        args.divide = True
    elif dataset == "origa":
        mask_paths = [os.path.splitext(path)[0] + ".png" for path in mask_paths]
    elif dataset == "gland":
        image_paths = glob.glob(os.path.join(image_dir, "train_*[0-9].bmp"))
        mask_paths = glob.glob(os.path.join(image_dir, "train_*[0-9]_anno.bmp"))

    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths,
                                                                                            mask_paths, test_size=0.3,
                                                                                            random_state=0)
    train_dataset = SegPathDataset(train_image_paths, train_mask_paths, augmentation=True,
                                   image_size=args.image_size, divide=args.divide)
    val_dataset = SegPathDataset(val_image_paths, val_mask_paths, augmentation=False,
                                 image_size=args.image_size, divide=args.divide)
    if dataset is not None and dataset == "drive":
        train_dataset = DRIVEDataset(train_image_paths, mask_dir, image_size=args.image_size, augmentation=True)
        val_dataset = DRIVEDataset(val_image_paths, mask_dir, image_size=args.image_size, augmentation=False)
    elif dataset == "octa":
        train_dataset = OCTADataset(train_image_paths, train_mask_paths, image_size=args.image_size, augmentation=True)
        val_dataset = OCTADataset(val_image_paths, val_mask_paths, image_size=args.image_size, augmentation=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    batch_size = args.batch_size
    # lr = args.lr * batch_size / 256
    if args.distributed:
        batch_size = batch_size // ngpus_per_node
    lr = args.lr
    num_workers = args.num_workers
    if args.distributed:
       num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              num_workers=num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    momentum = args.momentum
    weight_decay = args.weight_decay
    optimizer = opt.SGD([{"params":model.parameters(), "initia_lr":lr}], lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=True)
    epochs = args.epochs
    lr_sche_type = args.lr_sche
    if lr_sche_type == "cosine":
        lr_sc = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        lr_sc = IterScheduler(optimizer, len(train_dataset), batch_size, lr)
    train_metric = Metric(num_classes)
    val_metric = Metric(num_classes)
    criterion = nn.CrossEntropyLoss()
    if model_name == "encnet":
        criterion = EncLoss(num_classes=num_classes)
    ckpt_dir = args.ckpt_dir
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if args.gpu is not None:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fgam:
        att = "fgam"
    else:
        att = "nos"
    writer = SummaryWriter(comment=f"{block_name}_{args.expansion}_{att}")
    global_step = 0
    local_rank = args.rank

    for epoch in range(epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        total = 0
        losses = 0.0
        index = 0
        model.train()
        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            bs = image.size(0)
            total += bs
            if bs == 1 and block_name == "splat":
                continue
            optimizer.zero_grad()
            pred = model(image)
            if model_name == "nestedunet":
                loss = criterion(pred[0], mask) + criterion(pred[1], mask) + criterion(pred[2], mask) + criterion(pred[3], mask)
            elif model_name == "danet":
                loss = criterion(pred[0], mask)
            else:
                loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            losses += loss.item() * bs
            if isinstance(pred, list):
                pred = pred[0]
            pred = torch.softmax(pred, dim=1)
            train_metric.update(pred, mask)
            precision, acc, recall, f1, specificity, miou, mdice = train_metric.evalutate()
            if (args.distributed and local_rank==0) or (not args.distributed):
                print(f"training:[{epoch}:{index}] loss:{losses / total} precision:{precision} "
                      f"acc:{acc} recall:{recall} f1:{f1} specificity:{specificity} miou:{miou} mdice:{mdice}")
                writer.add_scalar("train/losses", losses / total, global_step=global_step)
                writer.add_scalar("train/acc", acc, global_step=global_step)
                writer.add_scalar("train/precision", precision, global_step=global_step)
                writer.add_scalar("train/recall", recall, global_step=global_step)
                writer.add_scalar("train/f1", f1, global_step=global_step)
                writer.add_scalar("train/specificity", specificity, global_step=global_step)
                global_step+=1
                index +=1
            if lr_sche_type != "cosine":
                lr_sc.step()
        train_metric.reset()
        if lr_sche_type == "cosine":
            lr_sc.step()
        model.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image = image.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.long)
                bs = image.size(0)
                total += bs
                pred = model(image)
                if model_name == "nestedunet":
                    loss = criterion(pred[0], mask) + criterion(pred[1], mask) + criterion(pred[2], mask) + criterion(
                        pred[3], mask)
                elif model_name == "danet":
                    loss = criterion(pred[0], mask)
                else:
                    loss = criterion(pred, mask)
                losses += loss.item() * bs
                if isinstance(pred, list):
                    pred = pred[0]
                pred = torch.softmax(pred, dim=1)
                val_metric.update(pred, mask)
            precision, acc, recall, f1, specificity, miou, mdice = val_metric.evalutate()
            if (args.distributed and local_rank == 0) or (not args.distributed):
                print(f"validation:{epoch}-loss:{losses / total} precision:{precision} "
                      f"acc:{acc} recall:{recall} f1:{f1}  specificity:{specificity} miou:{miou} mdice:{mdice}")
                writer.add_scalar("val/losses", losses / total, global_step=epoch)
                writer.add_scalar("val/acc", acc, global_step=epoch)
                writer.add_scalar("val/precision", precision, global_step=epoch)
                writer.add_scalar("val/recall", recall, global_step=epoch)
                writer.add_scalar("val/f1", f1, global_step=epoch)
                writer.add_scalar("val/specificity", specificity, global_step=epoch)
            val_metric.reset()
            if best_iou < miou:
                if (args.distributed and local_rank == 0) or (not args.distributed):
                    torch.save(model.state_dict(),os.path.join(ckpt_dir, f"{model_name}_{att}.pt"))
                    best_iou = miou
            if (args.distributed and local_rank == 0) or (not args.distributed):
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_{att}_last.pt"))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=4e-5,
                        help="weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="number of images for each train step")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="number of classes")
    parser.add_argument("--image_size", type=int, default=256,
                        help="image size for resize")
    parser.add_argument("-j", "--num_workers", type=int, default=4,
                        help="number of workers to load data")
    parser.add_argument("--epochs", type=int, default=64,
                        help="number of circle")
    parser.add_argument("--image_dir", type=str, default="./train",
                        help="path of directory that stored image")
    parser.add_argument("--mask_dir", type=str, default="./train_mask",
                        help="path of directory that stored mask")
    parser.add_argument("--ngpus_per_node", type=int, default=None,
                        help="number of gpus for per node")
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:8090",
                        help="url used to set distribution training")
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument("--gpu_index", type=str, default=None,
                        help="indexes of gpu to use, default is None")
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
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt",
                        help="path of directory to store weights")
    parser.add_argument("--model_name", type=str,
                        choices=["deeplabv3", "deeplabv3plus", "encnet", "unet", "nestedunet", "attunet", "danet", "saunet"],
                        help="name of model")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "resnest50", "resnest101","seresnet50", "resnest200"],
                        default="resnet50",
                        help="name of backbone for deeplab and encnet")
    parser.add_argument("--divide", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset name")
    parser.add_argument("--lr_sche", type=str, default="cosine")
    parser.add_argument("--csv_path", type=str, default=None)
    main()