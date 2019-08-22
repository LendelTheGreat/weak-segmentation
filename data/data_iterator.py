import os

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

from utisl.utils import segmap_to_class_vec, image_to_segmap


class SimpleISPRSVaihingenDataset(Dataset):
    """ISPRS Vaihingen dataset of crops and semantic segmentation segmaps"""

    def __init__(self, dir_images, dir_segmaps, transform=None):
        self.dir_images = dir_images
        self.images = [file for file in os.listdir(dir_images) if file.split('.')[-1] == 'tif']
        self.images.sort()
        
        self.dir_segmaps = dir_segmaps
        self.segmaps = [file for file in os.listdir(dir_segmaps) if file.split('.')[-1] == 'tif']
        self.segmaps.sort()
        
        assert [file.split('.')[0] for file in self.images] == [file.split('.')[0] for file in self.segmaps], 'Images and segmaps folders should contain same files!\nimages: {}\nsegmaps: {}'.format(self.images, self.segmaps)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.dir_images, self.images[idx])
        image = io.imread(filename)
        filename = os.path.join(self.dir_segmaps, self.segmaps[idx])
        segmap = image_to_segmap(io.imread(filename))
        class_vec = segmap_to_class_vec(segmap)
        
        sample = {'image': image, 'segmap': segmap, 'class_vec': class_vec}

        # This will only work correctly with deterministic transforms!
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['segmap'] = self.transform(sample['segmap'])

        return sample