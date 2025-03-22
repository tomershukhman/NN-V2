import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, CLASS_NAMES

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive_threshold = 0.5
        self.negative_threshold = 0.4
        self.alpha = 0.25
        self.gamma = 2.0
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0):
        # Handle both training and validation outputs
        if isinstance(predictions, list):
            # Convert validation format to training format
            batch_size = len(predictions)
            device = predictions[0]['boxes'].device
            
            # Get number of anchors from the first prediction that includes anchors
            default_anchors = None
            for pred in predictions:
                if 'anchors' in pred and pred['anchors'] is not None:
                    default_anchors = pred['anchors']
                    break
            
            if default_anchors is None:
                raise ValueError("No default anchors found in predictions")
                
            num_anchors = len(default_anchors)
            
            # Create tensors in training format
            bbox_pred = torch.zeros((batch_size, num_anchors, 4), device=device)
            conf_pred = torch.zeros((batch_size, num_anchors, NUM_CLASSES), device=device)
            
            for i, pred in enumerate(predictions):
                valid_preds = pred['boxes'].shape[0]
                if valid_preds > 0:
                    bbox_pred[i, :valid_preds] = pred['boxes']
                    # One-hot encode the labels
                    for j, label in enumerate(pred['labels']):
                        conf_pred[i, j, label] = pred['scores'][j]
            
            predictions = {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors
            }
        
        # Extract predictions
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        default_anchors = predictions['anchors']
        
        batch_size = bbox_pred.shape[0]
        num_anchors = bbox_pred.shape[1]
        
        # Initialize losses
        conf_losses = []
        bbox_losses = []
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            if len(target_boxes) == 0:
                # Handle empty targets with background-only loss
                conf_loss = -torch.log(1 - conf_pred[i, :, 1:].max(dim=1)[0] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            else:
                # Calculate IoU between anchors and target boxes
                ious = self._calculate_box_iou(default_anchors, target_boxes)
                
                # For each anchor, get IoU with best matching target
                max_ious, best_target_idx = ious.max(dim=1)
                
                # Create masks for positive and negative anchors
                positive_mask = max_ious >= self.positive_threshold
                negative_mask = max_ious < self.negative_threshold
                
                # Ensure at least one anchor per ground truth box
                for t_idx in range(len(target_boxes)):
                    box_ious = ious[:, t_idx]
                    best_anchor_idx = box_ious.argmax()
                    positive_mask[best_anchor_idx] = True
                    negative_mask[best_anchor_idx] = False
                
                # Create confidence targets
                conf_target = torch.zeros_like(conf_pred[i])
                if positive_mask.any():
                    matched_target_labels = target_labels[best_target_idx[positive_mask]]
                    conf_target[positive_mask, matched_target_labels] = 1.0
                
                # Calculate focal loss
                probs = torch.softmax(conf_pred[i], dim=1)
                pt = torch.where(conf_target == 1.0, probs, 1 - probs)
                alpha_factor = torch.where(conf_target == 1.0, self.alpha, 1 - self.alpha)
                focal_weight = (1 - pt).pow(self.gamma)
                
                focal_loss = -alpha_factor * focal_weight * torch.log(torch.clamp(pt, min=1e-6))
                
                # Balance positive and negative samples
                num_positive = positive_mask.sum().item()
                if num_positive > 0:
                    pos_loss = focal_loss[positive_mask].sum()
                    
                    # Sample negative examples
                    neg_loss = focal_loss[negative_mask]
                    
                    if len(neg_loss) > 0:
                        # Calculate number of negative samples to use
                        num_neg = min(num_positive * 3, negative_mask.sum().item())
                        
                        if num_neg > 0:
                            # Sort negative losses and take top k
                            neg_loss_values, neg_loss_indices = torch.topk(neg_loss.view(-1), k=num_neg)
                            neg_loss = neg_loss_values.sum()
                        else:
                            neg_loss = torch.tensor(0.0, device=bbox_pred.device)
                            
                        conf_loss = (pos_loss + neg_loss) / (num_positive + num_neg)
                    else:
                        conf_loss = pos_loss / num_positive
                else:
                    # If no positives, just use the hardest negatives
                    neg_loss = focal_loss[negative_mask]
                    if len(neg_loss) > 0:
                        num_neg = min(100, len(neg_loss))  # Cap at reasonable number
                        neg_loss_values, _ = torch.topk(neg_loss.view(-1), k=num_neg)
                        conf_loss = neg_loss_values.mean()
                    else:
                        conf_loss = focal_loss.mean()  # Fallback if no negatives
                
                # Calculate bbox loss only for positive samples
                if positive_mask.sum() > 0:
                    # Get matched target boxes
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Combine IoU loss and L1 loss
                    iou_loss = self._giou_loss(pred_boxes, matched_target_boxes)
                    l1_loss = F.smooth_l1_loss(pred_boxes, matched_target_boxes, reduction='none').mean(dim=1)
                    
                    bbox_loss = (iou_loss + 0.5 * l1_loss).mean()
                else:
                    bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            
            conf_losses.append(conf_loss)
            bbox_losses.append(bbox_loss)
        
        # Average losses across batch
        conf_loss = torch.stack(conf_losses).mean()
        bbox_loss = torch.stack(bbox_losses).mean()
        
        # Apply weights and combine
        total_loss = conf_weight * conf_loss + bbox_weight * bbox_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': conf_loss.item(),
            'bbox_loss': bbox_loss.item()
        }
    
    def _calculate_box_iou(self, boxes1, boxes2):
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