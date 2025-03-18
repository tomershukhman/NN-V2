import torch
import torch.nn as nn
import torch.nn.functional as F
from dog_detector.utils import compute_iou
from torchvision.ops import generalized_box_iou
from config import POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, REG_LOSS_WEIGHT

def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='sum'):
    """
    Compute focal loss for better handling of class imbalance.
    Args:
        pred: predictions tensor of shape [N, C] where C is the number of classes
        target: ground truth labels tensor of shape [N]
        alpha: weighting factor for rare class (positive samples)
        gamma: focusing parameter to down-weight easy examples
        reduction: 'sum' or 'mean' or 'none'
    """
    # Compute cross entropy loss first
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    
    # Apply label-dependent alpha weighting
    alpha_t = torch.ones_like(target, dtype=torch.float) * (1 - alpha)  # Weight for background
    alpha_t[target > 0] = alpha  # Weight for foreground
    
    # Calculate focal term
    focal_term = (1 - p_t) ** gamma
    
    # Combine all terms
    loss = alpha_t * focal_term * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class DetectionLoss(nn.Module):
    def __init__(self, model):
        super(DetectionLoss, self).__init__()
        self.model = model.module if hasattr(model, 'module') else model
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, predictions, targets):
        """Calculate detection loss using smooth L1 loss for regression and focal loss for classification"""
        cls_output, reg_output, anchors = predictions
        batch_size = cls_output.size(0)
        device = cls_output.device
        
        cls_losses = torch.zeros(batch_size, device=device)
        reg_losses = torch.zeros(batch_size, device=device)
        total_positive_samples = 0
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            # Get outputs for this image
            reg_output_i = reg_output[i].permute(1, 0, 2, 3).reshape(4, -1).t()
            cls_output_i = cls_output[i].permute(1, 0, 2, 3).reshape(cls_output.size(1), -1).t()
            
            # For negative samples (no objects), use focal loss with stronger negative weight
            if len(target_boxes) == 0:
                neg_target_cls = torch.zeros(cls_output_i.size(0), dtype=torch.long, device=device)
                cls_loss = focal_loss(cls_output_i, neg_target_cls, alpha=0.1, gamma=3.0, reduction='sum')
                # Normalize by number of anchors to prevent dominating the loss
                cls_loss = cls_loss / (cls_output_i.size(0) * batch_size)
                cls_losses[i] = cls_loss
                continue

            # Compute IoU between decoded predictions and targets
            decoded_boxes = self.model._decode_boxes(reg_output_i, anchors)
            ious = compute_iou(decoded_boxes, target_boxes)
            max_ious, max_idx = ious.max(dim=1)

            # Assign positive and negative samples with dynamic thresholding
            pos_mask = max_ious >= POS_IOU_THRESHOLD
            neg_mask = max_ious < NEG_IOU_THRESHOLD
            
            # Ensure minimum positive samples by taking highest IoUs
            if pos_mask.sum() < 16 and len(target_boxes) > 0:
                top_k = min(16, len(max_ious))
                _, top_anchor_idx = max_ious.topk(top_k)
                new_pos_mask = torch.zeros_like(pos_mask)
                new_pos_mask[top_anchor_idx] = True
                pos_mask = new_pos_mask

            # More balanced negative sampling - use IoU-based hard negative mining
            num_pos = pos_mask.sum()
            if num_pos > 0:
                # Take hardest negative samples (highest IoU below threshold)
                neg_ious = max_ious[~pos_mask]
                k = min(len(neg_ious), int(4 * num_pos))  # 4:1 neg:pos ratio
                if k > 0:
                    _, hard_neg_idx = neg_ious.topk(k)
                    neg_mask = torch.zeros_like(neg_mask)
                    neg_mask[torch.where(~pos_mask)[0][hard_neg_idx]] = True

            if pos_mask.sum() > 0:
                # Classification loss for positive samples
                pos_pred_cls = cls_output_i[pos_mask]
                pos_target_cls = target_labels[max_idx[pos_mask]]
                cls_loss_pos = focal_loss(pos_pred_cls, pos_target_cls, alpha=0.25, gamma=2.0, reduction='sum')
                
                # Compute regression targets with better scale handling
                pos_anchors = anchors[pos_mask]
                pos_gt_boxes = target_boxes[max_idx[pos_mask]]
                
                # Convert to center format
                pos_anchor_w = pos_anchors[:, 2] - pos_anchors[:, 0]
                pos_anchor_h = pos_anchors[:, 3] - pos_anchors[:, 1]
                pos_anchor_cx = pos_anchors[:, 0] + 0.5 * pos_anchor_w
                pos_anchor_cy = pos_anchors[:, 1] + 0.5 * pos_anchor_h
                
                gt_w = pos_gt_boxes[:, 2] - pos_gt_boxes[:, 0]
                gt_h = pos_gt_boxes[:, 3] - pos_gt_boxes[:, 1]
                gt_cx = pos_gt_boxes[:, 0] + 0.5 * gt_w
                gt_cy = pos_gt_boxes[:, 1] + 0.5 * gt_h
                
                # Compute regression targets with better normalization
                reg_targets = torch.zeros_like(reg_output_i[pos_mask])
                reg_targets[:, 0] = (gt_cx - pos_anchor_cx) / (pos_anchor_w + 1e-6)
                reg_targets[:, 1] = (gt_cy - pos_anchor_cy) / (pos_anchor_h + 1e-6)
                reg_targets[:, 2] = torch.log(gt_w / (pos_anchor_w + 1e-6))
                reg_targets[:, 3] = torch.log(gt_h / (pos_anchor_h + 1e-6))
                
                # Compute regression loss with better normalization
                reg_pred = reg_output_i[pos_mask]
                reg_loss = self.smooth_l1(reg_pred, reg_targets)
                reg_loss = reg_loss.sum() / (pos_mask.sum() + 1e-6)  # Normalize by positive samples
                reg_losses[i] = reg_loss
                total_positive_samples += pos_mask.sum()
            else:
                cls_loss_pos = torch.tensor(0.0, device=device)

            # Classification loss for negative samples with better weighting
            if neg_mask.sum() > 0:
                neg_pred_cls = cls_output_i[neg_mask]
                neg_target_cls = torch.zeros(neg_mask.sum(), dtype=torch.long, device=device)
                cls_loss_neg = focal_loss(neg_pred_cls, neg_target_cls, alpha=0.1, gamma=3.0, reduction='sum')
            else:
                cls_loss_neg = torch.tensor(0.0, device=device)

            # Normalize classification loss by total samples with minimum denominator
            total_samples = max(pos_mask.sum() + neg_mask.sum(), 1)
            cls_losses[i] = (cls_loss_pos + cls_loss_neg) / total_samples

        # Calculate final losses with proper batch normalization
        cls_loss_final = cls_losses.mean()
        reg_loss_final = reg_losses.mean() if total_positive_samples > 0 else torch.tensor(0.0, device=device)
        
        # Apply dynamic loss weighting based on positive samples
        reg_weight = min(1.0, total_positive_samples / (100 * batch_size))  # Scale up gradually
        total_loss = cls_loss_final + reg_weight * REG_LOSS_WEIGHT * reg_loss_final

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss_final,
            'reg_loss': reg_loss_final,
            'num_positive_samples': total_positive_samples
        }
