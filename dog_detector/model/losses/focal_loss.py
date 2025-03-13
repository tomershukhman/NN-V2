import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA

class FocalLoss(nn.Module):
    """
    Focal Loss as described in https://arxiv.org/abs/1708.02002
    Helps address class imbalance by down-weighting easy examples.
    """
    def __init__(self, alpha=None, gamma=None, reduction='none'):
        super().__init__()
        self.alpha = alpha if alpha is not None else FOCAL_LOSS_ALPHA
        self.gamma = gamma if gamma is not None else FOCAL_LOSS_GAMMA
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss