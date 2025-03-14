import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA

class FocalLoss(nn.Module):
    """
    Focal Loss as described in https://arxiv.org/abs/1708.02002
    Helps address class imbalance by down-weighting easy examples.
    
    Improved implementation with better stability and performance.
    """
    def __init__(self, alpha=None, gamma=None, reduction='none'):
        super().__init__()
        self.alpha = alpha if alpha is not None else FOCAL_LOSS_ALPHA
        self.gamma = gamma if gamma is not None else FOCAL_LOSS_GAMMA
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Compute focal loss with proper gradient flow.
        
        Args:
            inputs: Predicted logits or raw scores
            targets: Ground truth labels (0 or 1)
            
        Returns:
            Loss tensor with gradients
        """
        # Calculate probabilities while maintaining gradients
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
            
        # Use sigmoid directly on inputs for better numerical stability
        p = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy with better stability
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Calculate p_t (probability of the target class)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Add a small epsilon to avoid numerical instability
        p_t = torch.clamp(p_t, min=1e-7, max=1.0)
        
        # Calculate focal weight
        focal_weight = torch.pow((1 - p_t), self.gamma)
        
        # Apply the focal weight to the cross entropy loss
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        # Apply reduction strategy
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss