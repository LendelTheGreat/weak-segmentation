import numpy as np

from utils.utils import segmap_classes_to_segmap_stacked, segmap_stacked_to_segmap_colors

def make_grid(image_batch, segmap_classes_pred_batch, segmap_classes_gt_batch, max_grid_length=10):
    image_batch = image_batch.cpu().detach().numpy()
    image_batch = np.moveaxis(image_batch, 1, 3)
    segmap_classes_pred_batch = segmap_classes_pred_batch.cpu().detach().numpy()
    segmap_classes_pred_batch = np.moveaxis(segmap_classes_pred_batch, 1, 3)
    segmap_classes_gt_batch = segmap_classes_gt_batch.cpu().detach().numpy()
    segmap_classes_gt_batch = np.moveaxis(segmap_classes_gt_batch, 1, 3)
    n_images_in_grid = min(image_batch.shape[0], max_grid_length)
    image_h = image_batch.shape[1]
    image_w = image_batch.shape[2]
    grid_h = image_h * 3
    grid_w = image_w * n_images_in_grid
    grid = np.zeros((grid_h, grid_w, 3))
    
    for i in range(n_images_in_grid):
        grid[0:image_h, i*image_w:(i+1)*image_w, :] = image_batch[i, :, :, :]
        segmap_stacked = segmap_classes_to_segmap_stacked(segmap_classes_pred_batch[i, :, :, :])
        segmap_colors = segmap_stacked_to_segmap_colors(segmap_stacked)
        grid[image_h:image_h*2, i*image_w:(i+1)*image_w, :] = segmap_colors
        segmap_stacked = segmap_classes_to_segmap_stacked(segmap_classes_gt_batch[i, :, :, :])
        segmap_colors = segmap_stacked_to_segmap_colors(segmap_stacked)
        grid[image_h*2:image_h*3, i*image_w:(i+1)*image_w, :] = segmap_colors
    
    return grid