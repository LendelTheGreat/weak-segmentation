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
    start_time = time.time()
    if opt.debug:
        dir_logs = '{}_{}_DEBUG_{}'.format(opt.model, opt.tag, str(datetime.now().strftime("%Y%m%d_%H-%M-%S")))
    else:
        dir_logs = '{}_{}_{}'.format(opt.model, opt.tag, str(datetime.now().strftime("%Y%m%d_%H-%M-%S")))
    dir_logs = os.path.join(os.getenv('HOME'), 'logs', dir_logs)
    os.makedirs(dir_logs)
    copyfile('train.py', os.path.join(dir_logs, 'train.py'))
    copyfile('models/{}.py'.format(opt.model), os.path.join(dir_logs, '{}.py'.format(opt.model)))
    
    logging.basicConfig(level=logging.INFO if not opt.debug else logging.DEBUG,
                        handlers=[logging.FileHandler(os.path.join(dir_logs, 'console_output.log')),
                                  logging.StreamHandler()])
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    logger.info('Saving model logs into {}'.format(dir_logs))
    tb_writer = SummaryWriter(log_dir=dir_logs)
    
    logger.info('Starting training with opt {}'.format(opt))

    device = torch.device("cuda:{}".format(opt.gpu) if int(opt.gpu) >= 0 else "cpu")
    logger.info('Found available device {}'.format(device))

    if opt.no_weak_supervision:
        train_dataset = StrongWeakISPRSVaihingenDataset(dir_train_strong, None, transforms.ToTensor())
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
    
    best_val_loss_strong = np.inf

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        for i, data in enumerate(train_data_loader):
            image = data['image'].to(device)
            segmap_gt = data['segmap'].to(device)
            class_vec_gt = data['class_vec'].to(device)
            supervision = data['strong_supervision'].to(device)

            optimizer.zero_grad()
            segmap_pred = model(image)
            loss_strong = loss_weak = loss_flip = 0
            if not opt.no_strong_supervision:
                loss_strong = segmap_loss_func(segmap_pred, segmap_gt, supervision)
            if not opt.no_weak_supervision:
                loss_weak = class_loss_func(segmap_pred, class_vec_gt)
                loss_weak *= opt.lambda_weak
            if opt.lambda_flip > 0:
                with torch.no_grad():
                    flip_dims = random.choice([[2], [3], (2, 3)])
                    image_flipped = image.flip(dims=flip_dims)
                    segmap_flipped_pred = model(image_flipped)
                    segmap_unflipped_pred = segmap_flipped_pred.flip(dims=flip_dims)
                loss_flip = segmap_loss_func(segmap_pred, segmap_unflipped_pred)
                loss_flip *= opt.lambda_flip
            loss = loss_strong + loss_weak + loss_flip
            loss.backward()
            optimizer.step()
            if opt.debug:
                if i == 0:
                    grid = make_grid(image, segmap_pred, segmap_gt)
                    tb_writer.add_image('train_debug/images', grid, epoch, dataformats='HWC')
                    tb_writer.close()
                logger.debug('train iter {: >4d}  |  loss {: >2.4f}'.format(i, loss.item()))
                if i >= 0:
                    break

        # Evaluate on train
        model.eval()
        train_loss_strong = 0
        train_loss_weak = 0
        train_loss_flip = 0
        for i, data in enumerate(train_for_eval_data_loader):
            with torch.no_grad():
                image = data['image'].to(device)
                segmap_gt = data['segmap'].to(device)
                class_vec_gt = data['class_vec'].to(device)
                segmap_pred = model(image)
                loss_strong = loss_weak = loss_flip = 0
                loss_strong = segmap_loss_func(segmap_pred, segmap_gt).item()
                loss_weak = class_loss_func(segmap_pred, class_vec_gt).item()
                if opt.lambda_flip > 0:
                    flip_dims = random.choice([[2], [3], (2, 3)])
                    image_flipped = image.flip(dims=flip_dims)
                    segmap_flipped_pred = model(image_flipped)
                    segmap_unflipped_pred = segmap_flipped_pred.flip(dims=flip_dims)
                    loss_flip = segmap_loss_func(segmap_pred, segmap_unflipped_pred)

            train_loss_strong += loss_strong
            train_loss_weak += loss_weak
            train_loss_flip += loss_flip
            if i == 0:
                grid = make_grid(image, segmap_pred, segmap_gt)
                tb_writer.add_image('train/images', grid, epoch, dataformats='HWC')
                tb_writer.close()

            if opt.debug:
                if i >= 0:
                    break
        train_loss_strong /= i+1
        train_loss_weak /= i+1
        train_loss_flip /= i+1
        train_loss = 0
        if not opt.no_strong_supervision:
            train_loss += train_loss_strong
        if not opt.no_weak_supervision:
            train_loss += train_loss_weak * opt.lambda_weak
        if opt.lambda_flip > 0:
            train_loss += train_loss_flip * opt.lambda_flip

        # Evaluate on val
        model.eval()
        val_loss_strong = 0
        val_loss_weak = 0
        val_loss_flip = 0
        for i, data in enumerate(val_data_loader):
            with torch.no_grad():
                image = data['image'].to(device)
                segmap_gt = data['segmap'].to(device)
                class_vec_gt = data['class_vec'].to(device)
                segmap_pred = model(image)
                loss_strong = loss_weak = loss_flip = 0
                loss_strong = segmap_loss_func(segmap_pred, segmap_gt)
                loss_weak = class_loss_func(segmap_pred, class_vec_gt)
                if opt.lambda_flip > 0:
                    flip_dims = random.choice([[2], [3], (2, 3)])
                    image_flipped = image.flip(dims=flip_dims)
                    segmap_flipped_pred = model(image_flipped)
                    segmap_unflipped_pred = segmap_flipped_pred.flip(dims=flip_dims)
                    loss_flip = segmap_loss_func(segmap_pred, segmap_unflipped_pred)

            val_loss_strong += loss_strong
            val_loss_weak += loss_weak
            val_loss_flip += loss_flip
            if i == 0:
                grid = make_grid(image, segmap_pred, segmap_gt)
                tb_writer.add_image('val/images', grid, epoch, dataformats='HWC')
                tb_writer.close()

            if opt.debug:
                if i >= 0:
                    break

        val_loss_strong /= i+1
        val_loss_weak /= i+1
        val_loss_flip /= i+1
        val_loss = 0
        if not opt.no_strong_supervision:
            val_loss += val_loss_strong
        if not opt.no_weak_supervision:
            val_loss += val_loss_weak * opt.lambda_weak
        if opt.lambda_flip > 0:
            val_loss += val_loss_flip * opt.lambda_flip
        
        if val_loss_strong < best_val_loss_strong:
            best_val_loss_strong = val_loss_strong
            logger.info('Saving best model with val loss {} at epoch {}'.format(best_val_loss_strong, epoch))
            torch.save(model.state_dict(), os.path.join(dir_logs, 'model_weights_best.pth'))

        logger.info('Epoch {: >3d}  |  Train loss {: >2.6f} = strong {: >2.6f} + weak {: >2.6f} + flip {: >2.6f} |  Val loss {: >2.6f} = strong {: >2.6f} + weak {: >2.6f} + flip {: >2.6f} |  Time: {}'.format(
            epoch, train_loss, train_loss_strong, train_loss_weak, train_loss_flip,
            val_loss, val_loss_strong, val_loss_weak, val_loss_flip, time.time() - epoch_start_time))
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/loss_strong', train_loss_strong, epoch)
        tb_writer.add_scalar('train/loss_weak', train_loss_weak, epoch)
        tb_writer.add_scalar('train/loss_flip', train_loss_flip, epoch)
        tb_writer.add_scalar('val/loss', val_loss, epoch)
        tb_writer.add_scalar('val/loss_strong', val_loss_strong, epoch)
        tb_writer.add_scalar('val/loss_weak', val_loss_weak, epoch)
        tb_writer.add_scalar('val/loss_flip', val_loss_flip, epoch)
        tb_writer.close()
    logger.info('Finished training in {:.0f}'.format(time.time() - start_time))
        
        
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='From models folder (e.g. dummynet)')
    parser.add_argument('--gpu', default='0', help='GPU to run on (0 or 1) (-1 for cpu)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lambda_weak', default=0.1, type=float, help='Scaling factor of the weakly supervised loss')
    parser.add_argument('--lambda_flip', default=0.0, type=float, help='Scaling factor of the flip loss')
    parser.add_argument('--no_strong_supervision', action='store_true', help='Skip strong supervision loss')
    parser.add_argument('--no_weak_supervision', action='store_true', help='Skip weak supervision loss')
    parser.add_argument('--tag', default='', help='Tag added to model logs folder name')
    parser.add_argument('--debug', '-d', action='store_true', help='Run in debug mode')
    opt = parser.parse_args()
    train(opt)
    