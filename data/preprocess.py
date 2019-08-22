import os

import numpy as np
from skimage import io

from data.dataset_split import train_segmap_ids, train_class_ids, val_ids, test_ids
from utils.utils import segmap_to_class_vec, image_to_segmap

dir_images = '~data/raw/top'
dir_segmaps = '~data/raw/gts'
dir_output = '~data/crops'

crop_size = 192
crop_halfsize = int(crop_size / 2)

def run():

    images = [file for file in os.listdir(dir_images) if file.split('.')[-1] == 'tif']
    images.sort()
    
    segmaps = [file for file in os.listdir(dir_segmaps) if file.split('.')[-1] == 'tif']
    segmaps.sort()
    
    assert [file.split('.')[0] for file in images] == [file.split('.')[0] for file in segmaps], 'Images and segmaps folders should contain same files!\nimages: {}\nsegmaps: {}'.format(images, segmaps)
    
    total_class_sums = np.zeros(5)
    total_n_crops = 0
    for i in range(len(images)):
        class_sums = np.zeros(5)
        n_crops = 0
        
        filename = os.path.join(dir_images, images[i])
        image = io.imread(filename)
        filename = os.path.join(dir_segmaps, segmaps[i])
        segmap = io.imread(filename)
        
        assert image.shape == segmap.shape, 'Shape of image and segmap must be equal'
        
        # y, x are all possible cropped center points
        for y in range(crop_halfsize, segmap.shape[0]-crop_halfsize, crop_halfsize):
            for x in range(crop_halfsize, segmap.shape[1]-crop_halfsize, crop_halfsize):
                image_crop = image[y-crop_halfsize:y+crop_halfsize, x-crop_halfsize:x+crop_halfsize, :]
                segmap_crop = segmap[y-crop_halfsize:y+crop_halfsize, x-crop_halfsize:x+crop_halfsize, :]
                
                #Save image_crop and segmap_crop
                if i in train_class_ids:
                    io.imsave(os.path.join(dir_output, 'train_class', 'img_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x), image_crop)
                    io.imsave(os.path.join(dir_output, 'train_class', 'seg_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x)), segmap_crop)
                elif i in train_segmap_ids:
                    io.imsave(os.path.join(dir_output, 'train_segmap', 'img_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x), image_crop)
                    io.imsave(os.path.join(dir_output, 'train_segmap', 'seg_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x)), segmap_crop)
                elif i in val_ids:
                    io.imsave(os.path.join(dir_output, 'val', 'img_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x), image_crop)
                    io.imsave(os.path.join(dir_output, 'val', 'seg_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x)), segmap_crop)
                elif i in test_ids:
                    io.imsave(os.path.join(dir_output, 'test', 'img_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x), image_crop)
                    io.imsave(os.path.join(dir_output, 'test', 'seg_{:0>2d}_{:0>4d}_{:0>4d}.tif'.format(i, y, x)), segmap_crop)
                
                class_vec = segmap_to_class_vec(image_to_segmap(segmap_crop), class_presence_threshold=160)
                class_sums += class_vec
                n_crops += 1
    
        total_class_sums += class_sums
        total_n_crops += n_crops
        
    print('Total class frequencies: {}'.format(total_class_sums / total_n_crops))
    print('Saved {} crops into {}'.format(n_crops, dir_output))
    
if __name__== "__main__":
  run()