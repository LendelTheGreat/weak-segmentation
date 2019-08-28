import numpy as np
import torch
import torch.nn as nn


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, segmap_pred, segmap_gt, strong_supervision=None):
        if strong_supervision is None:
            loss = self.mse(segmap_pred, segmap_gt)
        else:
            loss = self.mse(segmap_pred[strong_supervision, :, :, :], segmap_gt[strong_supervision, :, :, :])
        return loss

    
class ClassLoss(torch.nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def global_weighted_rank_pooling(self, probs, class_vec_gt):
        sh = probs.size() # probs size: B C H W
        probs = probs.view(sh[0], sh[1], sh[2]*sh[3]).contiguous()
        probs_sorted, _ = probs.sort(dim=2) # probs_sorted size: B C H*W
        
        weights = np.zeros((sh[0], sh[1], sh[2]*sh[3])) # weights size: B C H*W
        for b, class_gt_batch in enumerate(class_vec_gt):
            for c, class_gt in enumerate(class_gt_batch):
                q_fg = 0 if class_gt.item() == 0 else 0.99
                weights[b, c, :] = np.array([ q_fg ** i for i in range(sh[2]*sh[3] - 1, -1, -1)])
        weights = torch.Tensor(weights)
        weights = weights.to(probs_sorted.get_device())
        Z_fg = weights.sum(dim=2, keepdim=True)
        probs_normalized = probs_sorted * weights
        probs_normalized = probs_normalized / Z_fg # probs_normalized size: B C H*W
        probs_mean = torch.sum(probs_normalized, dim=2) # probs_mean size: B C
        
        return probs_mean
        
    def forward(self, segmap_pred, class_vec_gt):
        class_vec_pred = self.global_weighted_rank_pooling(segmap_pred, class_vec_gt)
        
        loss = self.loss(class_vec_pred, class_vec_gt)
        return loss