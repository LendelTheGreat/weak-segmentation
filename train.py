import argparse
from datetime import datetime
import logging
import os
from shutil import copyfile
import time

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from data.data_iterators import SimpleISPRSVaihingenDataset, StrongWeakISPRSVaihingenDataset
from losses import SegmentationLoss, ClassLoss
import models.dummynet
import models.unet
from utils.visuals import make_grid

seed = 8671
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

dir_output = os.path.join(os.getenv('HOME'), 'data/crops/')
dir_train_strong = os.path.join(dir_output, 'train_strong')
dir_train_weak = os.path.join(dir_output, 'train_weak')
dir_train_strong_img = os.path.join(dir_output, 'train_strong', 'img')
dir_train_strong_seg = os.path.join(dir_output, 'train_strong', 'seg')
dir_train_weak_img = os.path.join(dir_output, 'train_weak', 'img')
dir_train_weak_seg = os.path.join(dir_output, 'train_weak', 'seg')
dir_val_img = os.path.join(dir_output, 'val', 'img')
dir_val_seg = os.path.join(dir_output, 'val', 'seg')

def train(opt):
    dir_logs = os.path.join(os.getenv('HOME'), 'logs', opt.model+'_'+str(datetime.now().strftime("%Y%m%d_%H-%M-%S")))
    os.makedirs(dir_logs)
    copyfile('train.py', os.path.join(dir_logs, 'train.py'))
    copyfile('models/{}.py'.format(opt.model), os.path.join(dir_logs, '{}.py'.format(opt.model)))
    
    logging.basicConfig(level=logging.INFO if not opt.debug else logging.DEBUG,
                        handlers=[logging.FileHandler(os.path.join(dir_logs, 'console_output.log')),
                                  logging.StreamHandler()])
    logger = logging.getLogger('train')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    tb_writer = SummaryWriter(log_dir=dir_logs)
    
    logger.info('Starting training with opt {}'.format(opt))

    device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    logger.info('Found available device {}'.format(device))

    if opt.no_weak_supervision:
        train_dataset = SimpleISPRSVaihingenDataset(dir_train_strong_img, dir_train_strong_seg, transforms.ToTensor())
    elif opt.no_strong_supervision:
        train_dataset = StrongWeakISPRSVaihingenDataset(None, dir_train_weak, transforms.ToTensor())
    else:
        train_dataset = StrongWeakISPRSVaihingenDataset(dir_train_strong, dir_train_weak, transforms.ToTensor())
    logger.info('Found {} images and segmaps in train_dataset'.format(len(train_dataset)))
    train_data_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True)

    train_for_eval_dataset = SimpleISPRSVaihingenDataset(dir_train_strong_img, dir_train_strong_seg,
                                                         transforms.ToTensor(), preshuffle=True)
    logger.info('Found {} images and segmaps in train_for_eval_dataset'.format(len(train_for_eval_dataset)))
    train_for_eval_data_loader = DataLoader(train_for_eval_dataset, opt.batch_size, shuffle=False)
    val_dataset = SimpleISPRSVaihingenDataset(dir_val_img, dir_val_seg,
                                              transforms.ToTensor(), preshuffle=True)
    logger.info('Found {} images and segmaps in val_dataset'.format(len(val_dataset)))
    val_data_loader = DataLoader(val_dataset, opt.batch_size, shuffle=False)

    if opt.model == 'dummynet':
        model = models.dummynet.DummyNet()
    elif opt.model == 'unet':
        model = models.unet.UNet(logger)
    else:
        logger.error('Unknown model name! {}'.format(opt.model))
    model.to(device)

    segmap_loss_func = SegmentationLoss()
    class_loss_func = ClassLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        start_time = time.time()

        # Training
        model.train()
        for i, data in enumerate(train_data_loader):
            image = data['image'].to(device)
            segmap_gt = data['segmap'].to(device)
            class_vec_gt = data['class_vec'].to(device)
            supervision = data['strong_supervision'].to(device)

            optimizer.zero_grad()
            segmap_pred = model(image)
            loss_strong = loss_weak = 0
            if not opt.no_strong_supervision:
                loss_strong = segmap_loss_func(segmap_pred, segmap_gt, supervision)
            if not opt.no_weak_supervision:
                loss_weak = class_loss_func(segmap_pred, class_vec_gt)
                loss_weak *= opt.lambda_weak
            loss = loss_strong + loss_weak
            loss.backward()
            optimizer.step()
            if opt.debug:
                if i == 0:
                    grid = make_grid(image, segmap_pred, segmap_gt)
                    tb_writer.add_image('train_debug/images', grid, epoch, dataformats='HWC')
                    tb_writer.close()
                logger.debug('train iter {: >4d}  |  loss {: >2.4f}'.format(i, loss.item()))
                break

        # Evaluate on train
        model.eval()
        train_loss_strong = 0
        train_loss_weak = 0
        for i, data in enumerate(train_for_eval_data_loader):
            with torch.no_grad():
                image = data['image'].to(device)
                segmap_gt = data['segmap'].to(device)
                class_vec_gt = data['class_vec'].to(device)
                segmap_pred = model(image)
                loss_strong = loss_weak = 0
                if not opt.no_strong_supervision:
                    loss_strong = segmap_loss_func(segmap_pred, segmap_gt).item()
                if not opt.no_weak_supervision:
                    loss_weak = class_loss_func(segmap_pred, class_vec_gt).item()

            train_loss_strong += loss_strong
            train_loss_weak += loss_weak
            if i == 0:
                grid = make_grid(image, segmap_pred, segmap_gt)
                tb_writer.add_image('train/images', grid, epoch, dataformats='HWC')
                tb_writer.close()

            if opt.debug:
                if i >= 0:
                    break
        train_loss_strong /= i+1
        train_loss_weak /= i+1
        train_loss = train_loss_strong + (train_loss_weak * opt.lambda_weak)


        # Evaluate on val
        model.eval()
        val_loss_strong = 0
        val_loss_weak = 0
        for i, data in enumerate(val_data_loader):
            with torch.no_grad():
                image = data['image'].to(device)
                segmap_gt = data['segmap'].to(device)
                class_vec_gt = data['class_vec'].to(device)
                segmap_pred = model(image)
                loss_strong = loss_weak = 0
                if not opt.no_strong_supervision:
                    loss_strong = segmap_loss_func(segmap_pred, segmap_gt)
                if not opt.no_weak_supervision:
                    loss_weak = class_loss_func(segmap_pred, class_vec_gt)

            val_loss_strong += loss_strong
            val_loss_weak += loss_weak
            if i == 0:
                grid = make_grid(image, segmap_pred, segmap_gt)
                tb_writer.add_image('val/images', grid, epoch, dataformats='HWC')
                tb_writer.close()

            if opt.debug:
                if i >= 0:
                    break

        logger.info('Epoch {: >3d}  |  Train loss {: >2.6f}  |  Val loss {: >2.6f}  |  Time: {}'.format(
            epoch, train_loss, val_loss, time.time() - start_time))
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/loss_strong', train_loss_strong, epoch)
        tb_writer.add_scalar('train/loss_weak', train_loss_weak, epoch)
        tb_writer.add_scalar('val/loss', val_loss, epoch)
        tb_writer.add_scalar('val/loss_strong', val_loss_strong, epoch)
        tb_writer.add_scalar('val/loss_weak', val_loss_weak, epoch)
        tb_writer.close()
        
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='From models folder (e.g. dummynet)')
    parser.add_argument('--gpu', default='0', help='GPU to run on (0 or 1)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lambda_weak', default=0.1, type=float, help='Scaling factor of the weakly supervised loss')
    parser.add_argument('--debug', '-d', action='store_true', help='Run in debug mode')
    parser.add_argument('--no_strong_supervision', action='store_true', help='Skip strong supervision loss')
    parser.add_argument('--no_weak_supervision', action='store_true', help='Skip weak supervision loss')
    opt = parser.parse_args()
    train(opt)
    