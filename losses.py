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
            conf_pred = torch.zeros((batch_size, num_anchors), device=device)
            
            for i, pred in enumerate(predictions):
                valid_preds = pred['boxes'].shape[0]
                if valid_preds > 0:
                    bbox_pred[i, :valid_preds] = pred['boxes']
                    conf_pred[i, :valid_preds] = pred['scores']
            
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
            
            if len(target_boxes) == 0:
                # Handle empty targets
                conf_loss = -torch.log(1 - conf_pred[i] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            else:
                # Calculate IoU between anchors and target boxes
                ious = self._calculate_box_iou(default_anchors, target_boxes)
                
                # For each anchor, get IoU with best matching target
                max_ious, best_target_idx = ious.max(dim=1)
                
                # For each target, ensure at least one anchor is assigned
                best_anchor_per_target = []
                for t_idx in range(len(target_boxes)):
                    # Get candidate anchors for this target (IoU > 0.3)
                    target_ious = ious[:, t_idx]
                    candidate_mask = target_ious > 0.3
                    
                    if candidate_mask.sum() > 0:
                        # Take top-k anchors for each target to ensure multiple good matches
                        # This helps with multi-object scenarios by providing more positive anchors
                        k = min(3, candidate_mask.sum().item())  # Get up to 3 anchors per target
                        topk_ious, topk_indices = torch.topk(target_ious, k=k)
                        best_anchor_per_target.extend(topk_indices.tolist())
                    else:
                        # If no good candidates, force assign the best one
                        best_idx = target_ious.argmax().item()
                        best_anchor_per_target.append(best_idx)
                
                # Create masks for positive and negative anchors
                positive_mask = max_ious >= self.positive_threshold
                for idx in best_anchor_per_target:
                    positive_mask[idx] = True  # Force-assigned anchors are positive
                
                negative_mask = max_ious < self.negative_threshold
                negative_mask[positive_mask] = False  # Ensure no overlap
                
                # Calculate confidence target for focal loss
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_mask] = 1.0
                
                # Apply focal loss with improved weighting for multi-object detection
                pt = torch.where(conf_target == 1.0, conf_pred[i], 1 - conf_pred[i])
                alpha_factor = torch.where(conf_target == 1.0, self.alpha, 1 - self.alpha)
                focal_weight = (1 - pt).pow(self.gamma)
                
                focal_loss = -alpha_factor * focal_weight * torch.log(torch.clamp(
                    torch.where(conf_target == 1.0, conf_pred[i], 1 - conf_pred[i]),
                    min=1e-6, max=1-1e-6
                ))
                
                # Balance positive and negative samples
                num_positive = positive_mask.sum().item()
                if num_positive > 0:
                    # Sample negative examples to maintain a reasonable pos:neg ratio
                    # For multi-object cases, we want more negative samples
                    neg_ratio = min(3, int(num_anchors / num_positive))
                    max_neg = neg_ratio * num_positive
                    
                    if negative_mask.sum() > max_neg:
                        # Sort negative anchors by loss value
                        neg_losses = focal_loss[negative_mask]
                        neg_values, neg_indices = torch.topk(neg_losses, k=max_neg)
                        
                        # Create a new mask with only the selected negatives
                        selected_neg_mask = torch.zeros_like(negative_mask)
                        selected_neg_indices = torch.where(negative_mask)[0][neg_indices]
                        selected_neg_mask[selected_neg_indices] = True
                        
                        # Only compute loss on positives and selected negatives
                        loss_mask = positive_mask | selected_neg_mask
                        conf_loss = focal_loss[loss_mask].mean()
                    else:
                        # If we don't have too many negatives, use all of them
                        loss_mask = positive_mask | negative_mask
                        conf_loss = focal_loss[loss_mask].mean()
                else:
                    # If no positives, just use the hardest negatives
                    num_neg = min(num_anchors // 4, 100)  # Cap at a reasonable number
                    neg_values, neg_indices = torch.topk(focal_loss[negative_mask], k=min(num_neg, negative_mask.sum().item()))
                    conf_loss = neg_values.mean()
                
                # Calculate bbox loss only for positive samples with better scaling
                if positive_mask.sum() > 0:
                    # Multi-target assignment: each anchor might be associated with different GT boxes
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Use combined IoU and L1 loss for better localization
                    iou_loss = self._giou_loss(pred_boxes, matched_target_boxes)
                    l1_loss = F.l1_loss(pred_boxes, matched_target_boxes, reduction='none').mean(dim=1)
                    
                    # Combine losses with adaptive weighting
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