import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, IOU_THRESHOLD, 
    NEG_POS_RATIO, LOC_LOSS_WEIGHT
)
from .utils.box_utils import box_iou, diou_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, reduction='none'):
        super().__init__()
        self.alpha = FOCAL_LOSS_ALPHA if alpha is None else alpha
        self.gamma = FOCAL_LOSS_GAMMA if gamma is None else gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Add small epsilon to prevent log(0)
        eps = 1e-7
        
        # Use logits directly instead of sigmoid for better numerical stability
        if inputs.requires_grad:
            p = torch.sigmoid(inputs)
        else:
            p = inputs
            
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        p_t = torch.clamp(p_t, min=eps, max=1.0 - eps)
        
        # Calculate focal weight with gradient
        focal_weight = ((1 - p_t) ** self.gamma).detach()
        
        # Calculate final loss
        loss = focal_weight * ce_loss
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=None, neg_pos_ratio=None, use_focal_loss=True):
        super().__init__()
        self.iou_threshold = IOU_THRESHOLD if iou_threshold is None else iou_threshold
        self.neg_pos_ratio = NEG_POS_RATIO if neg_pos_ratio is None else neg_pos_ratio
        self.use_focal_loss = use_focal_loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        
    def forward(self, predictions, targets):
        # Extract predictions and ensure gradients
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        
        if not bbox_pred.requires_grad:
            bbox_pred.requires_grad_(True)
        if not conf_pred.requires_grad:
            conf_pred.requires_grad_(True)
            
        batch_size = len(targets)
        device = bbox_pred.device
        
        # Initialize loss components with gradients
        total_loc_loss = torch.tensor(0., device=device, requires_grad=True)
        total_conf_loss = torch.tensor(0., device=device, requires_grad=True)
        num_pos = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']
            num_gt = len(gt_boxes)
            
            if num_gt == 0:
                # Handle empty ground truth case
                conf_loss = self.focal_loss(conf_pred[i], torch.zeros_like(conf_pred[i]))
                total_conf_loss = total_conf_loss + conf_loss.mean()
                continue
            
            # Calculate IoU matrix between predicted and ground truth boxes
            ious = box_iou(bbox_pred[i].unsqueeze(0), gt_boxes.unsqueeze(0))
            
            # Get best matches
            best_gt_iou, best_gt_idx = ious.max(dim=1)
            best_anchor_iou, best_anchor_idx = ious.max(dim=0)
            
            # Create positive mask with gradient
            positive_mask = (best_gt_iou > self.iou_threshold).float()
            
            # Ensure each gt box has at least one positive anchor
            for gt_idx in range(num_gt):
                best_anchor_for_gt = best_anchor_idx[gt_idx]
                positive_mask[best_anchor_for_gt] = 1.0
            
            positive_indices = torch.nonzero(positive_mask).squeeze(1)
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            if num_positive > 0:
                # Localization loss with gradient
                matched_gt_boxes = gt_boxes[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                loc_loss = diou_loss(pred_boxes, matched_gt_boxes)
                loc_loss = loc_loss.sum() * LOC_LOSS_WEIGHT
                
                # Confidence loss with hard negative mining
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = 1
                
                # Calculate loss for all anchors
                if self.use_focal_loss:
                    all_conf_loss = self.focal_loss(conf_pred[i], conf_target)
                else:
                    all_conf_loss = self.bce_loss(conf_pred[i], conf_target)
                
                # Hard negative mining
                conf_loss_pos = all_conf_loss[positive_indices]
                conf_loss_neg = all_conf_loss[~positive_mask.bool()]
                
                # Sort negative losses
                conf_loss_neg, _ = conf_loss_neg.sort(descending=True)
                num_neg = min(conf_loss_neg.size(0), num_positive * self.neg_pos_ratio)
                conf_loss_neg = conf_loss_neg[:num_neg]
                
                conf_loss = (conf_loss_pos.sum() + conf_loss_neg.sum()) / (num_positive + num_neg)
                
                # Add losses with gradient
                total_loc_loss = total_loc_loss + loc_loss
                total_conf_loss = total_conf_loss + conf_loss
        
        # Normalize losses
        num_pos = max(num_pos, 1)  # Avoid division by zero
        avg_loc_loss = total_loc_loss / num_pos
        avg_conf_loss = total_conf_loss / num_pos
        
        # Total loss with gradient
        total_loss = avg_loc_loss + avg_conf_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': avg_conf_loss.item(),
            'bbox_loss': avg_loc_loss.item()
        }