import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, CLASS_NAMES

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Class weights based on inverse frequency: background=1.0, person=1.0, dog=4.93 (~94.1%/19.1%)
        self.class_weights = torch.tensor([1.0, 1.0, 4.93])
        # Size-based weights to handle the significant size difference between classes
        self.person_size_mean = 0.093  # 9.30% average area
        self.dog_size_mean = 0.1767    # 17.67% average area
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0):
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        anchors = predictions['anchors']
        
        batch_size = bbox_pred.shape[0]
        total_loss = 0
        total_conf_loss = 0
        total_bbox_loss = 0
        
        # Move class weights to prediction device
        class_weights = self.class_weights.to(conf_pred.device)
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            if len(target_boxes) == 0:
                # Handle background-only case with class weighting
                conf_loss = -class_weights[0] * torch.log(1 - torch.softmax(conf_pred[i], dim=1)[:, 1:].max(dim=1)[0] + 1e-6).mean()
                total_conf_loss += conf_loss
                continue
            
            # Calculate IoU between anchors and target boxes
            ious = self._calculate_iou(anchors, target_boxes)
            
            # Dynamic thresholds based on object size
            target_sizes = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
            is_large = target_sizes > 0.15  # Threshold between person and dog mean sizes
            
            # Assign targets to anchors with dynamic IoU thresholds
            max_ious, matched_targets = ious.max(dim=1)
            pos_mask = torch.zeros_like(max_ious, dtype=torch.bool)
            
            # Apply different IoU thresholds based on object size
            for idx, (size, label) in enumerate(zip(target_sizes, target_labels)):
                if label == 1:  # person
                    threshold = 0.4
                else:  # dog
                    threshold = 0.35  # More permissive for rare class
                    
                if size > 0.15:  # Large object
                    threshold += 0.05  # Stricter for larger objects
                
                mask = ious[:, idx] >= threshold
                pos_mask |= mask
            
            neg_mask = max_ious < 0.3  # More aggressive negative mining
            
            # Ensure at least one anchor per ground truth box
            for t_idx in range(len(target_boxes)):
                if not pos_mask.any():
                    best_anchor = ious[:, t_idx].argmax()
                    pos_mask[best_anchor] = True
                    neg_mask[best_anchor] = False
            
            num_pos = pos_mask.sum()
            
            # Calculate classification loss with class weights
            target_conf = torch.zeros_like(conf_pred[i])
            if num_pos > 0:
                target_conf[pos_mask, target_labels[matched_targets[pos_mask]]] = 1
            
            conf_loss = F.cross_entropy(
                conf_pred[i],
                target_conf.argmax(dim=1),
                weight=class_weights,
                reduction='none'
            )
            
            # Hard negative mining with dynamic ratio
            if num_pos > 0:
                # Use more negative samples for the rare class (dogs)
                has_dog = (target_labels == 2).any()
                neg_ratio = 6 if has_dog else 4
                num_neg = min(neg_ratio * num_pos, neg_mask.sum())
                
                conf_loss_neg = conf_loss[neg_mask]
                _, neg_idx = conf_loss_neg.sort(descending=True)
                neg_idx = neg_idx[:num_neg]
                
                conf_loss = (conf_loss[pos_mask].sum() + conf_loss_neg[neg_idx].sum()) / (num_pos + num_neg)
            else:
                conf_loss = conf_loss[neg_mask].mean()
            
            # Calculate box regression loss with size-aware weighting
            if num_pos > 0:
                # Calculate size-based weights
                target_sizes = target_sizes[matched_targets[pos_mask]]
                size_weights = torch.ones_like(target_sizes)
                
                # Apply different weights based on class and size
                target_labels_pos = target_labels[matched_targets[pos_mask]]
                for idx, label in enumerate(target_labels_pos):
                    if label == 1:  # person
                        # Increase weight for smaller people
                        size_weights[idx] = torch.sqrt(self.person_size_mean / (target_sizes[idx] + 1e-6))
                    else:  # dog
                        # Increase weight for smaller dogs and apply class rarity factor
                        size_weights[idx] = 4.93 * torch.sqrt(self.dog_size_mean / (target_sizes[idx] + 1e-6))
                
                bbox_loss = F.smooth_l1_loss(
                    bbox_pred[i, pos_mask],
                    target_boxes[matched_targets[pos_mask]],
                    reduction='none'
                )
                bbox_loss = (bbox_loss.mean(dim=1) * size_weights).sum() / num_pos
            else:
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            
            total_conf_loss += conf_loss
            total_bbox_loss += bbox_loss
        
        # Average over batch
        conf_loss = total_conf_loss / batch_size
        bbox_loss = total_bbox_loss / batch_size
        
        # Combine losses with adaptive weighting
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