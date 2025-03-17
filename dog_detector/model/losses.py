import torch
import torch.nn as nn
import torch.nn.functional as F
from dog_detector.utils import compute_iou
from config import POS_IOU_THRESHOLD, NEG_IOU_THRESHOLD, BOX_REG_SCALE, REG_LOSS_WEIGHT

class DetectionLoss(nn.Module):
    def __init__(self, model):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # Use L1Loss instead of SmoothL1Loss for stronger gradients
        self.reg_loss_fn = nn.L1Loss(reduction='none')
        self.reg_loss_scale = BOX_REG_SCALE  # Use constant from config
        self.model = model.module if hasattr(model, 'module') else model

    def forward(self, predictions, targets):
        """Calculate detection loss with properly scaled regression targets"""
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
            
            # For negative samples
            if len(target_boxes) == 0:
                neg_target_cls = torch.zeros(cls_output_i.size(0), dtype=torch.long, device=device)
                cls_loss = F.cross_entropy(cls_output_i, neg_target_cls, reduction='mean')
                cls_losses[i] = cls_loss
                continue

            # Decode regression outputs to get actual box coordinates
            decoded_boxes = self.model._decode_boxes(reg_output_i, anchors)
            
            # Compute IoU between decoded predictions and targets
            ious = compute_iou(decoded_boxes, target_boxes)
            max_ious, max_idx = ious.max(dim=1)  # For each anchor, get best GT box

            # Assign positive and negative samples using constants from config
            pos_mask = max_ious >= POS_IOU_THRESHOLD
            neg_mask = max_ious < NEG_IOU_THRESHOLD
            
            # Forcefully ensure we have at least some positive anchors
            # If no anchors exceed the threshold, take the top-K highest IoU anchors
            if pos_mask.sum() < 10 and len(target_boxes) > 0:
                top_k = min(10, len(max_ious))
                _, top_anchor_idx = max_ious.topk(top_k)
                new_pos_mask = torch.zeros_like(pos_mask)
                new_pos_mask[top_anchor_idx] = True
                pos_mask = new_pos_mask
            
            # Ensure at least one positive match per GT box if available
            if pos_mask.sum() > 0:
                # Classification loss for positive samples
                pos_pred_cls = cls_output_i[pos_mask]
                pos_target_cls = target_labels[max_idx[pos_mask]]
                cls_loss_pos = F.cross_entropy(pos_pred_cls, pos_target_cls, reduction='mean')
                cls_losses[i] += cls_loss_pos

                # For regression, directly compare the decoded boxes with GT boxes
                pos_pred_boxes = decoded_boxes[pos_mask]
                pos_gt_boxes = target_boxes[max_idx[pos_mask]]
                
                # Normalize boxes by image dimensions for better scaling
                h, w = 512, 512  # IMAGE_SIZE from config
                pos_pred_boxes_norm = pos_pred_boxes.clone()
                pos_gt_boxes_norm = pos_gt_boxes.clone()
                
                # Normalize coordinates to [0, 1] range
                pos_pred_boxes_norm[:, 0::2] /= w  # x coordinates
                pos_pred_boxes_norm[:, 1::2] /= h  # y coordinates
                pos_gt_boxes_norm[:, 0::2] /= w
                pos_gt_boxes_norm[:, 1::2] /= h
                
                # Direct L1 loss on normalized box coordinates (stronger signal)
                reg_loss = self.reg_loss_fn(pos_pred_boxes_norm, pos_gt_boxes_norm).mean(dim=1).mean()
                
                reg_losses[i] = reg_loss
                total_positive_samples += pos_mask.sum()

            # Classification loss for negative samples
            if neg_mask.sum() > 0:
                neg_pred_cls = cls_output_i[neg_mask]
                neg_target_cls = torch.zeros(neg_mask.sum(), dtype=torch.long, device=device)
                cls_loss_neg = F.cross_entropy(neg_pred_cls, neg_target_cls, reduction='mean')
                cls_losses[i] += cls_loss_neg
                
        # Calculate final losses
        cls_loss_final = cls_losses.mean()
        
        # Ensure regression loss isn't zero even with few positive samples
        if total_positive_samples > 0:
            reg_loss_final = reg_losses.mean()  # Use mean instead of sum/count
        else:
            reg_loss_final = torch.tensor(0.0, device=device)
            
        # Apply REG_LOSS_WEIGHT from config
        total_loss = cls_loss_final + REG_LOSS_WEIGHT * reg_loss_final

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss_final,
            'reg_loss': reg_loss_final,
            'num_positive_samples': total_positive_samples
        }
