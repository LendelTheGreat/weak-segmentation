import os

import numpy as np
from skimage import io

from data.dataset_split import all_ids, train_strong_ids, train_weak_ids, val_ids, test_ids
from utils.utils import segmap_to_class_vec, image_to_segmap, contains_clutter_class

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
class_presence_threshold = (crop_size * crop_size) / 100

def run():

    total_class_sums = np.zeros(5)
    total_n_crops = 0
    total_filtered_crops = 0
    for i, img_id in enumerate(all_ids):
        filename = 'top_mosaic_09cm_area{}.tif'.format(img_id)
        print('Cropping image {} / {} - {}'.format(i, len(all_ids), filename))
        class_sums = np.zeros(5)
        n_crops = 0
        
        image = io.imread(os.path.join(dir_images, filename))
        segmap = io.imread(os.path.join(dir_segmaps, filename))
        
        assert image.shape == segmap.shape, 'Shape of image and segmap must be equal'
        
        # y, x are all possible cropped center points
        for y in range(crop_halfsize, segmap.shape[0]-crop_halfsize, crop_halfsize):
            for x in range(crop_halfsize, segmap.shape[1]-crop_halfsize, crop_halfsize):
                segmap_crop = segmap[y-crop_halfsize:y+crop_halfsize, x-crop_halfsize:x+crop_halfsize, :]
                
                if contains_clutter_class(segmap_crop, class_presence_threshold):
                    total_filtered_crops += 1 # TODO: this does not work
                else:
                    image_crop = image[y-crop_halfsize:y+crop_halfsize, x-crop_halfsize:x+crop_halfsize, :]
                    
                    #Save image_crop and segmap_crop
                    if img_id in train_strong_ids:
                        io.imsave(os.path.join(dir_train_strong_img,
                                               'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  image_crop, check_contrast=False)
                        io.imsave(os.path.join(dir_train_strong_seg,
                                               'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  segmap_crop, check_contrast=False)
                    elif img_id in train_weak_ids:
                        io.imsave(os.path.join(dir_train_weak_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  image_crop, check_contrast=False)
                        io.imsave(os.path.join(dir_train_weak_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  segmap_crop, check_contrast=False)
                    elif img_id in val_ids:
                        io.imsave(os.path.join(dir_val_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  image_crop, check_contrast=False)
                        io.imsave(os.path.join(dir_val_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  segmap_crop, check_contrast=False)
                    elif img_id in test_ids:
                        io.imsave(os.path.join(dir_test_img, 'img_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  image_crop, check_contrast=False)
                        io.imsave(os.path.join(dir_test_seg, 'seg_{:0>2d}_{:0>4d}_{:0>4d}.png'.format(img_id, y, x)),
                                  segmap_crop, check_contrast=False)

                    class_vec = segmap_to_class_vec(image_to_segmap(segmap_crop), class_presence_threshold)
                    class_sums += class_vec
                    n_crops += 1
    
        total_class_sums += class_sums
        total_n_crops += n_crops
        
    print('Total class frequencies: {}'.format(total_class_sums / total_n_crops))
    print('Saved {} crops into {}'.format(total_n_crops, dir_output))
    print('Total filtered crops {}'.format(total_filtered_crops))
    
if __name__== "__main__":
  run()

# TODO: this is out of date
# Last output of this script:
# Total class frequencies: [0.81808962 0.77574951 0.81352602 0.81644166 0.29758509]
# Saved 15777 crops into /home/ubuntu/data/crops/