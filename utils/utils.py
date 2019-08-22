import numpy as np
import torch

n_classes = 5

# TODO: rename segmaps to more informative segmap_color for raw, segmap_5c, segmap_1c, segmap_tensor???

def segmap_to_class_vec(segmap, class_presence_threshold=1):
    class_vec = np.zeros(n_classes)
    for i in range(n_classes):
        if np.count_nonzero(segmap == i) >= class_presence_threshold:
            class_vec[i] = 1
            
    return class_vec
    
# TODO: Refactor this out and use image_to_segmap_full instead
def image_to_segmap(image):
    segmap = np.zeros_like(image[:, :, 0])
    
    indexes_blue = np.where(np.all(image == np.array([0,0,255]), axis=-1))
    segmap[indexes_blue] = 1
    indexes_cyan = np.where(np.all(image == np.array([0,255,255]), axis=-1))
    segmap[indexes_cyan] = 2
    indexes_green = np.where(np.all(image == np.array([0,255,0]), axis=-1))
    segmap[indexes_green] = 3
    indexes_yellow = np.where(np.all(image == np.array([255,255,0]), axis=-1))
    segmap[indexes_yellow] = 4
    
    # unique, counts = np.unique(segmap, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    return segmap

def image_to_segmap_full(image):
    segmap = np.zeros((image.shape[0], image.shape[1], n_classes))
    
    indexes_white = np.where(np.all(image == np.array([255,255,255]), axis=-1))
    segmap[indexes_white, 0] = 1
    indexes_blue = np.where(np.all(image == np.array([0,0,255]), axis=-1))
    segmap[indexes_blue, 1] = 1
    indexes_cyan = np.where(np.all(image == np.array([0,255,255]), axis=-1))
    segmap[indexes_cyan, 2] = 1
    indexes_green = np.where(np.all(image == np.array([0,255,0]), axis=-1))
    segmap[indexes_green, 3] = 1
    indexes_yellow = np.where(np.all(image == np.array([255,255,0]), axis=-1))
    segmap[indexes_yellow, 4] = 1
    
    return segmap

def contains_clutter_class(image, class_presence_threshold=1):
    indexes_red = np.where(np.all(image == np.array([255,0,0]), axis=-1))
 
    if len(indexes_red[0]) >= class_presence_threshold:
        return True
    return False
