import numpy as np
import torch

n_classes = 5
class_colors = np.array([[1., 1., 1.],
                         [0., 0., 1.],
                         [0., 1., 1.],
                         [0., 1., 0.],
                         [1., 1., 0.]])


def segmap_classes_to_class_vec(segmap_classes, class_presence_threshold=1):
    """Converts segmentation map per class to a class vector by thresholding on the number of class pixels.

    Parameters
    ------
    segmap_classes : np array (shape: (H, W, n_classes))
        Segmentation map with 1 channel per class
    class_presence_threshold : int, optional
        Threshold of how many pixels of a class need to be present in the segmap

    Returns
    -------
    class_vec : np array (shape: (n_classes))
        Binary vector with 1 for each class present in the segmap
    """
    class_vec = np.zeros(n_classes)
    for i in range(n_classes):
        if np.count_nonzero(segmap_classes[:, :, i]) >= class_presence_threshold:
            class_vec[i] = 1
            
    return class_vec
    
    
def segmap_colors_to_segmap_stacked(segmap_colors):
    """Converts segmentation map image to a stacked segmentation map

    Parameters
    ------
    segmap_colors : np array (shape: (H, W, 3))
        Segmentation map as image where each class is a specific color

    Returns
    -------
    segmap_stacked : np array (shape: (H, W, 1))
        Segmap where each pixel contains the class number
    """
    segmap_stacked = np.zeros_like(segmap_colors[:, :, 0])
    
    indexes_blue = np.where(np.all(segmap_colors == class_colors[1], axis=-1))
    segmap_stacked[indexes_blue] = 1
    indexes_cyan = np.where(np.all(segmap_colors == class_colors[2], axis=-1))
    segmap_stacked[indexes_cyan] = 2
    indexes_green = np.where(np.all(segmap_colors == class_colors[3], axis=-1))
    segmap_stacked[indexes_green] = 3
    indexes_yellow = np.where(np.all(segmap_colors == class_colors[4], axis=-1))
    segmap_stacked[indexes_yellow] = 4
    
    # unique, counts = np.unique(segmap_stacked, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    return segmap_stacked


def segmap_stacked_to_segmap_colors(segmap_stacked):
    """Converts stacked segmentation map to a segmentation map color image

    Parameters
    ------
    segmap_stacked : np array (shape: (H, W, 1))
        Segmap where each pixel contains the class number

    Returns
    -------
    segmap_colors : np array (shape: (H, W, 3))
        Segmentation map as image where each class is a specific color
    """
    segmap_colors = np.zeros((segmap_stacked.shape[0], segmap_stacked.shape[1], 3))

    indexes_white = np.where(segmap_stacked == 0)
    segmap_colors[indexes_white[0], indexes_white[1], :] = class_colors[0]
    indexes_blue = np.where(segmap_stacked == 1)
    segmap_colors[indexes_blue[0], indexes_blue[1], :] = class_colors[1]
    indexes_cyan = np.where(segmap_stacked == 2)
    segmap_colors[indexes_cyan[0], indexes_cyan[1], :] = class_colors[2]
    indexes_green = np.where(segmap_stacked == 3)
    segmap_colors[indexes_green[0], indexes_green[1], :] = class_colors[3]
    indexes_yellow = np.where(segmap_stacked == 4)
    segmap_colors[indexes_yellow[0], indexes_yellow[1], :] = class_colors[4]
    
    return segmap_colors


def segmap_colors_to_segmap_classes(segmap_colors):
    """Converts segmentation map image to a stacked segmentation map

    Parameters
    ------
    segmap_colors : np array (shape: (H, W, 3))
        Segmentation map as image where each class is a specific color

    Returns
    -------
    segmap_classes : np array (shape: (H, W, n_classes))
        Segmentation map with 1 channel per class
    """
    segmap_classes = np.zeros((segmap_colors.shape[0], segmap_colors.shape[1], n_classes))
    
    indexes_white = np.where(np.all(segmap_colors == class_colors[0], axis=-1))
    segmap_classes[indexes_white[0], indexes_white[1], 0] = 1
    indexes_blue = np.where(np.all(segmap_colors == class_colors[1], axis=-1))
    segmap_classes[indexes_blue[0], indexes_blue[1], 1] = 1
    indexes_cyan = np.where(np.all(segmap_colors == class_colors[2], axis=-1))
    segmap_classes[indexes_cyan[0], indexes_cyan[1], 2] = 1
    indexes_green = np.where(np.all(segmap_colors == class_colors[3], axis=-1))
    segmap_classes[indexes_green[0], indexes_green[1], 3] = 1
    indexes_yellow = np.where(np.all(segmap_colors == class_colors[4], axis=-1))
    segmap_classes[indexes_yellow[0], indexes_yellow[1], 4] = 1
    
    return segmap_classes


def segmap_classes_to_segmap_stacked(segmap_classes):
    """Converts segmentation map per class to a stacked segmentation map

    Parameters
    ------
    segmap_classes : np array (shape: (H, W, n_classes))
        Segmentation map with 1 channel per class

    Returns
    -------
    segmap_stacked : np array (shape: (H, W, 1))
        Segmap where each pixel contains the class number
    """
    segmap_stacked = segmap_classes.argmax(axis=2)
    return segmap_stacked


def contains_clutter_class(segmap_colors, class_presence_threshold=1):
    """Checks if the segmentation map contains a clutter class (class 6)

    Parameters
    ------
    segmap_colors : np array (shape: (H, W, 3))
        Segmentation map as image where each class is a specific color
    class_presence_threshold : int, optional
        Threshold of how many pixels of a class need to be present in the segmap

    Returns
    -------
    bool
        True if clutter class is present
    """
    indexes_red = np.where(np.all(segmap_colors == np.array([255,0,0]), axis=-1))
 
    if len(indexes_red[0]) >= class_presence_threshold:
        return True
    return False


def assert_segmap_conversions():
    """Assertion check that converting segmentation map from colors to stacked
    and back results in the same segmentation map.
    """
    segmap_colors = np.zeros((192, 192, 3), dtype=np.float32)
    for y in range(192):
        for x in range(192):
            segmap_colors[y, x, :] = class_colors[np.random.choice(n_classes)]
    assert segmap_colors.shape == (192, 192, 3)
    
    segmap_classes = segmap_colors_to_segmap_classes(segmap_colors)
    assert segmap_classes.shape == (192, 192, n_classes)
    
    segmap_stacked = segmap_classes_to_segmap_stacked(segmap_classes)
    assert segmap_stacked.shape == (192, 192)
        
    segmap_colors_2 = segmap_stacked_to_segmap_colors(segmap_stacked)
    assert segmap_colors_2.shape == (192, 192, 3)    
        
    assert np.all(segmap_colors == segmap_colors_2), 'Converting segmap colors into classes and back should result in the same values!'
    
    print('Segmap conversions work correctly!')


if __name__== "__main__":
    assert_segmap_conversions()
