import os

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

from utils.utils import segmap_to_class_vec, image_to_segmap, image_to_segmap_full


class SimpleISPRSVaihingenDataset(Dataset):
    """ISPRS Vaihingen dataset of crops and semantic segmentation segmaps"""

    def __init__(self, dir_images, dir_segmaps, transform=None):
        self.dir_images = dir_images
        self.images = [file for file in os.listdir(dir_images) if file.split('.')[-1] == 'png']
        self.images.sort()
        
        self.dir_segmaps = dir_segmaps
        self.segmaps = [file for file in os.listdir(dir_segmaps) if file.split('.')[-1] == 'png']
        self.segmaps.sort()
        
        assert len(self.images) == len(self.segmaps), 'Images and segmaps folders should contain same number of files!\nimages: {}\nsegmaps: {}'.format(len(self.images), len(self.segmaps))
        
        print('Found {} images and segmaps'.format(len(self.images)))
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.dir_images, self.images[idx]))
        segmap = io.imread(os.path.join(self.dir_segmaps, self.segmaps[idx]))
        class_vec = segmap_to_class_vec(image_to_segmap(segmap))
        segmap = image_to_segmap_full(segmap)
        
        sample = {'image': image.astype(np.float32),
                  'segmap': segmap.astype(np.float32),
                  'class_vec': class_vec.astype(np.float32)}

        # This will only work correctly with deterministic transforms!
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['segmap'] = self.transform(sample['segmap'])

        return sample