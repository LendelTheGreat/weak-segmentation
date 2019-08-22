import numpy as np
import torch

n_classes = 5

def segmap_to_class_vec(segmap, class_presence_threshold=1):
    if torch.is_tensor(segmap):
        segmap = segmap.numpy()
        
    class_vec = np.zeros(n_classes)
    for i in range(n_classes):
        if np.count_nonzero(segmap == i) >= class_presence_threshold:
            class_vec[i] = 1
            
    return class_vec
    
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