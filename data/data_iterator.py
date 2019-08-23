import os
import random

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

from utils.utils import segmap_colors_to_segmap_classes, segmap_classes_to_class_vec


class SimpleISPRSVaihingenDataset(Dataset):
    """ISPRS Vaihingen dataset of crops and semantic segmentation segmaps"""

    def __init__(self, dir_images, dir_segmaps, transform=None, preshuffle=False):
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