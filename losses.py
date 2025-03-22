import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, CLASS_NAMES

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0):
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        anchors = predictions['anchors']
        
        batch_size = bbox_pred.shape[0]
        total_loss = 0
        total_conf_loss = 0
        total_bbox_loss = 0
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            # Skip empty targets
            if len(target_boxes) == 0:
                # Handle background-only case
                conf_loss = -torch.log(1 - torch.softmax(conf_pred[i], dim=1)[:, 1:].max(dim=1)[0] + 1e-6).mean()
                total_conf_loss += conf_loss
                continue
            
            # Calculate IoU between anchors and target boxes
            ious = self._calculate_iou(anchors, target_boxes)
            
            # Assign targets to anchors
            max_ious, matched_targets = ious.max(dim=1)
            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.4
            
            # Ensure at least one anchor per ground truth box
            for t_idx in range(len(target_boxes)):
                if not pos_mask.any():
                    # If no positive matches, take the best match
                    best_anchor = ious[:, t_idx].argmax()
                    pos_mask[best_anchor] = True
                    neg_mask[best_anchor] = False
                    matched_targets[best_anchor] = t_idx
            
            num_pos = pos_mask.sum()
            
            # Calculate classification loss
            target_conf = torch.zeros_like(conf_pred[i])
            if num_pos > 0:
                target_conf[pos_mask, target_labels[matched_targets[pos_mask]]] = 1
            
            conf_loss = F.cross_entropy(
                conf_pred[i],
                target_conf.argmax(dim=1),
                reduction='none'
            )
            
            # Hard negative mining
            if num_pos > 0:
                num_neg = min(3 * num_pos, neg_mask.sum())
                conf_loss_neg = conf_loss[neg_mask]
                _, neg_idx = conf_loss_neg.sort(descending=True)
                neg_idx = neg_idx[:num_neg]
                
                conf_loss = (conf_loss[pos_mask].sum() + conf_loss_neg[neg_idx].sum()) / (num_pos + num_neg)
            else:
                # If no positive samples, use all negative samples
                conf_loss = conf_loss[neg_mask].mean()
            
            # Calculate box regression loss only for positive anchors
            if num_pos > 0:
                bbox_loss = F.smooth_l1_loss(
                    bbox_pred[i, pos_mask],
                    target_boxes[matched_targets[pos_mask]],
                    reduction='sum'
                ) / num_pos
            else:
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            
            # Combine losses with weights
            total_conf_loss += conf_loss
            total_bbox_loss += bbox_loss
        
        # Average over batch
        conf_loss = total_conf_loss / batch_size
        bbox_loss = total_bbox_loss / batch_size
        
        # Apply weights and combine
        total_loss = conf_weight * conf_loss + bbox_weight * bbox_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': conf_loss.item(),
            'bbox_loss': bbox_loss.item()
        }
    
    def _calculate_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Convert to x1y1x2y2 format if normalized
        boxes1 = boxes1.clone()
        boxes2 = boxes2.clone()
        
        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _giou_loss(self, boxes1, boxes2):
        """Calculate GIoU loss between boxes"""
        # Calculate IoU
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        
        # Calculate the smallest enclosing box
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        # Calculate GIoU
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        
        return 1 - giou