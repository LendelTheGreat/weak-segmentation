import os

import numpy as np
from skimage import io

print(os.getcwd())
from data.dataset_split import train_strong_ids, train_weak_ids, val_ids, test_ids
from utils.utils import segmap_to_class_vec, image_to_segmap

dir_images = os.path.join(os.getenv('HOME'), 'data/raw/top/')
dir_segmaps = os.path.join(os.getenv('HOME'), 'data/raw/gts/')
dir_output = os.path.join(os.getenv('HOME'), 'data/crops/')

dir_train_strong_img = os.path.join(dir_output, 'train_strong', 'img')
dir_train_strong_seg = os.path.join(dir_output, 'train_strong', 'seg')
dir_train_weak_img = os.path.join(dir_output, 'train_weak', 'img')
dir_train_weak_seg = os.path.join(dir_output, 'train_weak', 'seg')
dir_val_img = os.path.join(dir_output, 'val', 'img')
dir_val_seg = os.path.join(dir_output, 'val', 'seg')
dir_test_img = os.path.join(dir_output, 'test', 'img')
dir_test_seg = os.path.join(dir_output, 'test', 'seg')
os.makedirs(dir_train_strong_img, exist_ok=True)
os.makedirs(dir_train_strong_seg, exist_ok=True)
os.makedirs(dir_train_weak_img, exist_ok=True)
os.makedirs(dir_train_weak_seg, exist_ok=True)
os.makedirs(dir_val_img, exist_ok=True)
os.makedirs(dir_val_seg, exist_ok=True)
os.makedirs(dir_test_img, exist_ok=True)
os.makedirs(dir_test_seg, exist_ok=True)

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
        print('Cropping image {} / {}'.format(i, len(images)))
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
                if i in train_strong_ids:
                    io.imsave(os.path.join(dir_train_strong_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              image_crop, check_contrast=False)
                    io.imsave(os.path.join(dir_train_strong_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              segmap_crop, check_contrast=False)
                elif i in train_weak_ids:
                    io.imsave(os.path.join(dir_train_weak_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              image_crop, check_contrast=False)
                    io.imsave(os.path.join(dir_train_weak_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              segmap_crop, check_contrast=False)
                elif i in val_ids:
                    io.imsave(os.path.join(dir_val_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              image_crop, check_contrast=False)
                    io.imsave(os.path.join(dir_val_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              segmap_crop, check_contrast=False)
                elif i in test_ids:
                    io.imsave(os.path.join(dir_test_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              image_crop, check_contrast=False)
                    io.imsave(os.path.join(dir_test_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(i, y, x)),
                              segmap_crop, check_contrast=False)
                
                class_vec = segmap_to_class_vec(image_to_segmap(segmap_crop), class_presence_threshold=160)
                class_sums += class_vec
                n_crops += 1
    
        total_class_sums += class_sums
        total_n_crops += n_crops
        
    print('Total class frequencies: {}'.format(total_class_sums / total_n_crops))
    print('Saved {} crops into {}'.format(total_n_crops, dir_output))
    
if __name__== "__main__":
  run()

# Last output of this script:
# Total class frequencies: [0.81808962 0.77574951 0.81352602 0.81644166 0.29758509]
# Saved 15777 crops into /home/ubuntu/data/crops/