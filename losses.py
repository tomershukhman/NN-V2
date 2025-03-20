import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive_threshold = 0.35  # Lowered from 0.4
        self.negative_threshold = 0.3
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0, underdetection_weight=1.0):
        batch_size = len(targets)
        total_conf_loss = 0
        total_bbox_loss = 0
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            
            # Handle completely empty predictions
            if len(target_boxes) == 0:
                conf_loss = -torch.log(1 - predictions['conf_pred'][i] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=predictions['bbox_pred'].device)
            else:
                # Calculate IoU matrix between predicted boxes and target boxes
                ious = self._calculate_box_iou(predictions['bbox_pred'][i], target_boxes)
                max_ious, best_target_idx = ious.max(dim=1)
                
                # For each target, find best matching prediction
                target_best_ious, _ = ious.max(dim=0)
                
                # Positive/negative sample assignment
                positive_mask = max_ious >= self.positive_threshold
                negative_mask = max_ious < self.negative_threshold
                
                # Force best prediction for each target to be positive
                _, force_positive_idx = ious.max(dim=0)
                positive_mask[force_positive_idx] = True
                negative_mask[force_positive_idx] = False
                
                # Set target confidence scores
                conf_target = torch.zeros_like(predictions['conf_pred'][i])
                conf_target[positive_mask] = 1
                
                # Calculate confidence loss with underdetection penalty
                base_conf_loss = -(conf_target * torch.log(predictions['conf_pred'][i] + 1e-6) + 
                                 (1 - conf_target) * torch.log(1 - predictions['conf_pred'][i] + 1e-6))
                
                # Apply underdetection penalty when we have fewer detections than targets
                if len(target_boxes) > positive_mask.sum():
                    missed_ratio = (len(target_boxes) - positive_mask.sum()) / len(target_boxes)
                    # Scale up loss for positive samples based on underdetection
                    base_conf_loss[positive_mask] *= (1 + missed_ratio * underdetection_weight)
                
                conf_loss = base_conf_loss.mean()
                
                # Calculate bbox loss only for positive samples
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = predictions['bbox_pred'][i][positive_mask]
                    
                    # IoU loss for positive samples
                    pos_ious = self._calculate_box_iou(pred_boxes, matched_target_boxes)
                    iou_loss = -torch.log(pos_ious.diag() + 1e-6)
                    
                    # L1 loss weighted by IoU quality
                    l1_loss = torch.abs(pred_boxes - matched_target_boxes).mean(dim=1)
                    
                    # Combined box loss with underdetection penalty
                    box_losses = iou_loss + l1_loss
                    if len(target_boxes) > positive_mask.sum():
                        box_losses *= (1 + missed_ratio * (underdetection_weight * 0.5))  # Less aggressive for bbox loss
                    
                    bbox_loss = box_losses.mean()
                else:
                    bbox_loss = torch.tensor(0.0, device=predictions['bbox_pred'].device)
            
            total_conf_loss += conf_loss
            total_bbox_loss += bbox_loss
        
        # Average losses over batch
        total_conf_loss = total_conf_loss / batch_size
        total_bbox_loss = total_bbox_loss / batch_size
        
        # Apply weights
        weighted_conf_loss = conf_weight * total_conf_loss
        weighted_bbox_loss = bbox_weight * total_bbox_loss
        total_loss = weighted_conf_loss + weighted_bbox_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_bbox_loss.item()
        }
        
    def _calculate_box_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Calculate intersection
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
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