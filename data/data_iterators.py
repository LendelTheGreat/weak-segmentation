"""Data iterators to serve data for pytorch training

Here we have 2 data iterators:
SimpleISPRSVaihingenDataset for evalutation
StrongWeakISPRSVaihingenDataset for training on strongly and weakly supervised data.
"""

import os
import random

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

from utils.utils import segmap_colors_to_segmap_classes, segmap_classes_to_class_vec


class StrongWeakISPRSVaihingenDataset(Dataset):
    """ISPRS Vaihingen dataset of crops and semantic segmentation segmaps for strong and weak supervision
    """

    def __init__(self, dir_strong, dir_weak, transform=None, preshuffle=False):
        """Initialize StrongWeakISPRSVaihingenDataset
        
        Parameters
        ----------
        dir_strong : str
            Path to a directory with input images and ground truth segmaps for strong supervision
        dir_weak : str
            Path to a directory with input images and ground truth segmaps for weak supervision
        transform : tourchvision.Transform, optional
            Transform the images before returning them e.g. transforms.ToTensor()
        preshuffle : bool, optional
            Randomly shuffle the data during initialization
        """
        self.dir_strong = dir_strong
        if dir_strong is not None:
            self.imagenames_strong = os.listdir(os.path.join(dir_strong, 'img'))
            self.imagenames_strong = [file for file in self.imagenames_strong if file.split('.')[-1] == 'png']
            self.imagenames_strong.sort()
            segmapnames_strong = os.listdir(os.path.join(dir_strong, 'seg'))
            segmapnames_strong = [file for file in segmapnames_strong if file.split('.')[-1] == 'png']
            assert len(self.imagenames_strong) == len(segmapnames_strong), 'Images and segmaps strong folders should contain same number of files!\nimages: {}\nsegmaps: {}'.format(len(self.imagenames_strong), len(segmapnames_strong))
        else:
            self.imagenames_strong = []
        
        self.dir_weak = dir_weak
        if dir_weak is not None:
            self.imagenames_weak = os.listdir(os.path.join(dir_weak, 'img'))
            self.imagenames_weak = [file for file in self.imagenames_weak if file.split('.')[-1] == 'png']
            self.imagenames_weak.sort()
            segmapnames_weak = os.listdir(os.path.join(dir_weak, 'seg'))
            segmapnames_weak = [file for file in segmapnames_weak if file.split('.')[-1] == 'png']
            assert len(self.imagenames_weak) == len(segmapnames_weak), 'Images and segmaps weak folders should contain same number of files!\nimages: {}\nsegmaps: {}'.format(len(self.imagenames_weak), len(segmapnames_weak))
            self.imagenames_weak = self.imagenames_weak[:1000]
        else:
            self.imagenames_weak = []
            
        self.transform = transform
        
        if preshuffle:
            random.shuffle(self.imagenames_strong, lambda: 0.1)
            random.shuffle(self.imagenames_weak, lambda: 0.1)

    def __len__(self):
        return len(self.imagenames_strong) + len(self.imagenames_weak)

    def __getitem__(self, idx):
        if idx < len(self.imagenames_strong):
            strong_supervision = True
            image_path = os.path.join(self.dir_strong, 'img', self.imagenames_strong[idx])
            segmap_path = os.path.join(self.dir_strong, 'seg', self.imagenames_strong[idx].replace('img_', 'seg_'))
        else:
            strong_supervision = False
            idx = idx - len(self.imagenames_strong)
            image_path = os.path.join(self.dir_weak, 'img', self.imagenames_weak[idx])
            segmap_path = os.path.join(self.dir_weak, 'seg', self.imagenames_weak[idx].replace('img_', 'seg_'))
        
        image = io.imread(image_path).astype(np.float32) / 255.
        segmap_colors = io.imread(segmap_path).astype(np.float32) / 255.
        segmap_classes = segmap_colors_to_segmap_classes(segmap_colors)
        class_vec = segmap_classes_to_class_vec(segmap_classes)
        
        sample = {'image': image.astype(np.float32),
                  'segmap': segmap_classes.astype(np.float32),
                  'class_vec': class_vec.astype(np.float32),
                  'strong_supervision': np.array(strong_supervision)}

        # This will only work correctly with deterministic transforms!
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['segmap'] = self.transform(sample['segmap'])
        sample['class_vec'] = torch.tensor(sample['class_vec'])
        sample['strong_supervision'] = torch.tensor(sample['strong_supervision'])

        return sample


class SimpleISPRSVaihingenDataset(Dataset):
    """ISPRS Vaihingen dataset of crops and semantic segmentation maps"""

    def __init__(self, dir_images, dir_segmaps, transform=None, preshuffle=False):
        """Initialize SimpleISPRSVaihingenDataset
        
        Parameters
        ----------
        dir_images : str
            Path to a directory with input images
        dir_segmaps : str
            Path to a directory with ground truth segmaps
        transform : tourchvision.Transform, optional
            Transform the images before returning them e.g. transforms.ToTensor()
        preshuffle : bool, optional
            Randomly shuffle the data during initialization
        """
        self.dir_images = dir_images
        self.images = [file for file in os.listdir(dir_images) if file.split('.')[-1] == 'png']
        self.images.sort()
        
        self.dir_segmaps = dir_segmaps
        self.segmaps = [file for file in os.listdir(dir_segmaps) if file.split('.')[-1] == 'png']
        self.segmaps.sort()
        
        assert len(self.images) == len(self.segmaps), 'Images and segmaps folders should contain same number of files!\nimages: {}\nsegmaps: {}'.format(len(self.images), len(self.segmaps))
        
        self.transform = transform
        
        if preshuffle:
            random.shuffle(self.images, lambda: 0.5)
            random.shuffle(self.segmaps, lambda: 0.5)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.dir_images, self.images[idx])).astype(np.float32) / 255.
        segmap_colors = io.imread(os.path.join(self.dir_segmaps, self.segmaps[idx])).astype(np.float32) / 255.
        segmap_classes = segmap_colors_to_segmap_classes(segmap_colors)
        class_vec = segmap_classes_to_class_vec(segmap_classes)
        
        sample = {'image': image.astype(np.float32),
                  'segmap': segmap_classes.astype(np.float32),
                  'class_vec': class_vec.astype(np.float32)}

        # This will only work correctly with deterministic transforms!
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['segmap'] = self.transform(sample['segmap'])

        return sample