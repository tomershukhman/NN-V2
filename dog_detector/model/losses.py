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
        pred: predictions
        target: ground truth labels
        alpha: weighting factor for rare class
        gamma: focusing parameter
        reduction: 'sum' or 'mean' or 'none'
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    loss = alpha * (1 - p_t) ** gamma * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class DetectionLoss(nn.Module):
    def __init__(self, model):
        super(DetectionLoss, self).__init__()
        self.model = model.module if hasattr(model, 'module') else model
        
    def forward(self, predictions, targets):
        """Calculate detection loss using GIoU loss for regression and focal loss for classification"""
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
            
            # For negative samples (no objects)
            if len(target_boxes) == 0:
                neg_target_cls = torch.zeros(cls_output_i.size(0), dtype=torch.long, device=device)
                cls_loss = focal_loss(cls_output_i, neg_target_cls, reduction='sum') / cls_output_i.size(0)
                cls_losses[i] = cls_loss
                continue

            # Decode regression outputs to get actual box coordinates
            decoded_boxes = self.model._decode_boxes(reg_output_i, anchors)
            
            # Compute IoU between decoded predictions and targets
            ious = compute_iou(decoded_boxes, target_boxes)
            max_ious, max_idx = ious.max(dim=1)

            # Assign positive and negative samples
            pos_mask = max_ious >= POS_IOU_THRESHOLD
            neg_mask = max_ious < NEG_IOU_THRESHOLD
            
            # Ensure minimum positive samples with highest IoU
            if pos_mask.sum() < 10 and len(target_boxes) > 0:
                top_k = min(10, len(max_ious))
                _, top_anchor_idx = max_ious.topk(top_k)
                new_pos_mask = torch.zeros_like(pos_mask)
                new_pos_mask[top_anchor_idx] = True
                pos_mask = new_pos_mask

            # Balance negative samples to be 3x positive samples
            num_pos = pos_mask.sum()
            if num_pos > 0 and neg_mask.sum() > 3 * num_pos:
                # Randomly sample negative examples
                neg_indices = torch.where(neg_mask)[0]
                perm = torch.randperm(len(neg_indices), device=device)
                sampled_neg = neg_indices[perm[:3 * num_pos]]
                neg_mask = torch.zeros_like(neg_mask)
                neg_mask[sampled_neg] = True

            total_samples = pos_mask.sum() + neg_mask.sum()
            
            if pos_mask.sum() > 0:
                # Classification loss for positive samples
                pos_pred_cls = cls_output_i[pos_mask]
                pos_target_cls = target_labels[max_idx[pos_mask]]
                cls_loss_pos = focal_loss(pos_pred_cls, pos_target_cls, reduction='sum')
                
                # Regression loss using GIoU
                pos_pred_boxes = decoded_boxes[pos_mask]
                pos_gt_boxes = target_boxes[max_idx[pos_mask]]
                giou = generalized_box_iou(pos_pred_boxes, pos_gt_boxes)
                giou_diag = torch.diag(giou)
                reg_loss = (1 - giou_diag).sum()  # Sum instead of mean for proper normalization
                
                reg_losses[i] = reg_loss / (pos_mask.sum() or 1)
                total_positive_samples += pos_mask.sum()
            else:
                cls_loss_pos = torch.tensor(0.0, device=device)

            # Classification loss for negative samples
            if neg_mask.sum() > 0:
                neg_pred_cls = cls_output_i[neg_mask]
                neg_target_cls = torch.zeros(neg_mask.sum(), dtype=torch.long, device=device)
                cls_loss_neg = focal_loss(neg_pred_cls, neg_target_cls, reduction='sum')
            else:
                cls_loss_neg = torch.tensor(0.0, device=device)

            # Normalize classification loss by total samples
            cls_losses[i] = (cls_loss_pos + cls_loss_neg) / (total_samples or 1)

        # Calculate final losses - already normalized per image
        cls_loss_final = cls_losses.mean()
        reg_loss_final = reg_losses.mean() if total_positive_samples > 0 else torch.tensor(0.0, device=device)
        
        # Total loss with proper weighting
        total_loss = cls_loss_final + REG_LOSS_WEIGHT * reg_loss_final

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss_final,
            'reg_loss': reg_loss_final,
            'num_positive_samples': total_positive_samples
        }
