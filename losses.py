import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive_threshold = 0.5
        self.negative_threshold = 0.25  # Lowered to catch more potential dogs
        
        # Adjusted focal loss parameters for multi-dog scenarios
        self.alpha = 0.6  # Increased to focus more on positive examples
        self.gamma = 1.0  # Reduced to prevent over-suppression of hard negatives
        
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
                # Handle empty targets with modified background confidence loss
                conf_loss = -torch.log(1 - conf_pred[i] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            else:
                # Calculate IoU between anchors and target boxes
                ious = self._calculate_box_iou(default_anchors, target_boxes)
                
                # For each anchor, get IoU with best matching target
                max_ious, best_target_idx = ious.max(dim=1)
                
                # For each target, find best matching anchor
                best_target_ious, _ = ious.max(dim=0)
                
                # For each target, ensure multiple anchors can be assigned if they have good overlap
                good_ious = ious > self.positive_threshold
                
                # Create positive mask considering multiple conditions
                positive_mask = (max_ious >= self.positive_threshold) | good_ious.any(dim=1)
                
                # Create negative mask avoiding any decent overlaps
                negative_mask = max_ious < self.negative_threshold
                negative_mask[positive_mask] = False
                
                # Calculate IoU-aware confidence targets
                conf_target = torch.zeros_like(max_ious)
                conf_target[positive_mask] = torch.clamp(max_ious[positive_mask], min=0.5)
                
                # Apply focal weighting with IoU awareness
                alpha_factor = torch.ones_like(conf_target) * self.alpha
                alpha_factor[positive_mask] = 1 - self.alpha
                
                focal_weight = (1 - conf_pred[i]).pow(self.gamma)
                focal_weight[positive_mask] = (1 - conf_pred[i][positive_mask] * max_ious[positive_mask]).pow(self.gamma)
                
                # Calculate confidence loss with IoU-aware weighting
                conf_loss = -(conf_target * torch.log(conf_pred[i] + 1e-6) + 
                            (1 - conf_target) * torch.log(1 - conf_pred[i] + 1e-6))
                conf_loss = (conf_loss * focal_weight * alpha_factor).mean()
                
                # Calculate bbox loss only for positive samples
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Calculate IoU loss for better handling of overlapping boxes
                    iou_loss = self._giou_loss(pred_boxes, matched_target_boxes)
                    
                    # L1 loss weighted by IoU to focus on accurate localization
                    l1_loss = F.l1_loss(pred_boxes, matched_target_boxes, reduction='none')
                    l1_loss = (l1_loss * max_ious[positive_mask].unsqueeze(1)).mean(dim=1)
                    
                    # Combine losses with adaptive weighting based on number of targets
                    num_targets = len(target_boxes)
                    target_weight = torch.sqrt(torch.tensor(num_targets, device=bbox_pred.device))
                    bbox_loss = (iou_loss * 0.7 + l1_loss * 0.3).mean() * target_weight
                else:
                    bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
            
            conf_losses.append(conf_loss)
            bbox_losses.append(bbox_loss)
        
        # Average losses across batch with count-based scaling
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