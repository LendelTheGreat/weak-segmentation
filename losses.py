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
        self.avgpool = nn.AvgPool2d(192)
        self.loss = nn.BCELoss()
        
        
    def global_weighted_rank_pooling(x):
        # TODO: finish this
        
        # initial shape: B C H W
        sh = x.size()
        x = x.view(sh[0], sh[1], sh[2]*sh[3]).contiguous()
        x = x.permute(0, 2, 1) # resulting shape (B, H*W, C)
        probs_sort = x.sort(axis=1) # TODO
        
        q_fg = 0.99
        weights = np.array([ q_fg ** i for i in range(sh[2]*sh[3]-1, -1, -1)])
        weights = np.reshape(weights,(1,-1,1))
        weights = torch.Tensor(weights)
        Z_fg = weights.sum()
        probs_mean = tf.reduce_sum((probs_sort*weights)/Z_fg, axis=1)
        
        
        return class_vec
        
    def forward(self, segmap_pred, class_vec_gt):
        #class_vec_gt = class_vec_gt * 0.5 # TODO: maybe adjust for each class frequency - only if we use MSE loss
        
        class_vec_pred = self.avgpool(segmap_pred).squeeze()
        
        loss = self.loss(class_vec_pred, class_vec_gt)
        return loss