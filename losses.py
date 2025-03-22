import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduce threshold gap for more consistent predictions
        self.positive_threshold = 0.5  # Back to original
        self.negative_threshold = 0.3  # Lower for better recall
        
        # Adjusted focal loss parameters
        self.alpha = 0.5  # Increased from 0.25 for better balance
        self.gamma = 1.5  # Reduced from 2.0 to prevent over-suppression
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0, cls_weight=1.0):
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
            cls_pred = torch.zeros((batch_size, num_anchors, NUM_CLASSES-1), device=device)
            
            for i, pred in enumerate(predictions):
                valid_preds = pred['boxes'].shape[0]
                if valid_preds > 0:
                    bbox_pred[i, :valid_preds] = pred['boxes']
                    conf_pred[i, :valid_preds] = pred['scores']
                    cls_pred[i, :valid_preds] = pred['classes']
            
            predictions = {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'cls_pred': cls_pred,
                'anchors': default_anchors
            }

        # Extract predictions
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        cls_pred = predictions['cls_pred']
        default_anchors = predictions['anchors']
        
        batch_size = bbox_pred.shape[0]
        num_anchors = bbox_pred.shape[1]
        
        # Initialize losses
        conf_losses = []
        bbox_losses = []
        cls_losses = []
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            if len(target_boxes) == 0:
                # Handle empty targets
                conf_loss = -torch.log(1 - conf_pred[i] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
                cls_loss = torch.tensor(0.0, device=bbox_pred.device)
            else:
                # Calculate IoU between anchors and target boxes
                ious = self._calculate_box_iou(default_anchors, target_boxes)
                
                # For each anchor, get IoU with best matching target
                max_ious, best_target_idx = ious.max(dim=1)
                
                # For each target, ensure at least one anchor is assigned
                _, best_anchor_idx = ious.max(dim=0)
                
                # Assign positive and negative samples
                positive_mask = max_ious >= self.positive_threshold
                positive_mask[best_anchor_idx] = True  # Force best anchors to be positive
                
                negative_mask = max_ious < self.negative_threshold
                negative_mask[positive_mask] = False
                
                # Calculate confidence loss with modified focal loss
                conf_target = positive_mask.float()
                
                # Adjust focal weights for multiple dogs
                alpha_factor = torch.ones_like(conf_target) * self.alpha
                alpha_factor[positive_mask] = 1 - self.alpha  # Inverse alpha for positive samples
                
                # Calculate focal weight with temperature scaling
                focal_weight = (1 - conf_pred[i]) * conf_target + conf_pred[i] * (1 - conf_target)
                focal_weight = focal_weight.pow(self.gamma)
                
                conf_loss = -(conf_target * torch.log(conf_pred[i] + 1e-6) + 
                            (1 - conf_target) * torch.log(1 - conf_pred[i] + 1e-6))
                conf_loss = (conf_loss * focal_weight * alpha_factor).mean()
                
                # Calculate bbox loss only for positive samples with better scaling
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Use combined IoU and L1 loss for better localization
                    iou_loss = self._giou_loss(pred_boxes, matched_target_boxes)
                    l1_loss = F.l1_loss(pred_boxes, matched_target_boxes, reduction='none').mean(dim=1)
                    
                    # Combine losses with adaptive weighting
                    bbox_loss = (iou_loss + 0.5 * l1_loss).mean()
                else:
                    bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
                
                # Classification loss only for positive samples
                if positive_mask.sum() > 0:
                    matched_target_labels = target_labels[best_target_idx[positive_mask]]
                    pred_classes = cls_pred[i][positive_mask]
                    
                    # Convert target labels to one-hot (subtract 1 to match prediction indices)
                    target_one_hot = F.one_hot(matched_target_labels - 1, num_classes=NUM_CLASSES-1)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        pred_classes, target_one_hot.float()
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=bbox_pred.device)
            
            conf_losses.append(conf_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        
        # Average losses across batch
        conf_loss = torch.stack(conf_losses).mean()
        bbox_loss = torch.stack(bbox_losses).mean()
        cls_loss = torch.stack(cls_losses).mean()
        
        # Apply weights and combine
        total_loss = conf_weight * conf_loss + bbox_weight * bbox_loss + cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': conf_loss.item(),
            'bbox_loss': bbox_loss.item(),
            'cls_loss': cls_loss.item()
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