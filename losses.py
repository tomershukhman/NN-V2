import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Lower positive threshold to catch more potential matches
        self.positive_threshold = 0.35  # Lowered from 0.4 to catch more positives
        self.negative_threshold = 0.2   # Lowered from 0.3 to be more strict with negatives
        
        # Adjusted focal loss parameters
        self.alpha = 0.7    # Increased from 0.6 to give even more weight to positive samples
        self.gamma = 2.0    # Increased from 1.5 for harder negative mining
        
        # IOU thresholds for loss calculation
        self.iou_threshold = 0.4
        self.iou_good_match = 0.7
        
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
                max_ious, best_target_idx = ious.max(dim=1)
                
                # For each target, get best matching anchor
                target_best_ious, _ = ious.max(dim=0)
                
                # Assign positive and negative samples
                positive_mask = max_ious >= self.positive_threshold
                
                # Force best anchor for each target to be positive
                _, force_positive_idx = ious.max(dim=0)
                positive_mask[force_positive_idx] = True
                
                negative_mask = max_ious < self.negative_threshold
                negative_mask[positive_mask] = False
                
                # Initialize conf_loss for this batch item
                conf_loss = -torch.log(1 - conf_pred[i] + 1e-6)  # Start with background loss
                
                # Calculate confidence target with adaptive weighting
                conf_target = positive_mask.float()
                
                # Calculate focal weights with stronger emphasis on hard examples
                alpha_factor = torch.ones_like(conf_target) * self.alpha
                alpha_factor[positive_mask] = 1 - self.alpha
                
                # Enhanced focal loss weighting
                focal_weight = (1 - conf_pred[i]) * conf_target + conf_pred[i] * (1 - conf_target)
                focal_weight = focal_weight.pow(self.gamma)
                
                # Additional weight for underdetection cases
                if len(target_boxes) > positive_mask.sum():
                    underdetection_factor = 1.4  # Increased from 1.2
                    focal_weight[positive_mask] *= underdetection_factor
                    conf_loss = conf_loss * 1.2  # Additional global boost for underdetection cases
                
                # Enhanced penalty for high confidence predictions with low IoU
                if len(target_boxes) > 0:
                    # Find predictions with high confidence but low IoU
                    high_conf_mask = conf_pred[i] > 0.7
                    if high_conf_mask.any():
                        ious = self._calculate_box_iou(default_anchors[high_conf_mask], target_boxes)
                        max_ious, _ = ious.max(dim=1)
                        # Add extra penalty for high confidence but low IoU predictions
                        low_iou_mask = max_ious < 0.3
                        if low_iou_mask.any():
                            conf_loss = conf_loss + 0.3 * (-torch.log(1 - conf_pred[i][high_conf_mask][low_iou_mask] + 1e-6)).mean()

                conf_loss = (conf_loss * focal_weight * alpha_factor).mean()
                
                # Calculate bbox loss only for positive samples with enhanced underdetection handling
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Calculate IoU-based loss
                    ious = self._calculate_box_iou(pred_boxes, matched_target_boxes)
                    iou_loss = -torch.log(ious + 1e-6)  # Convert to loss
                    
                    # L1 loss weighted by IoU quality
                    l1_loss = F.l1_loss(pred_boxes, matched_target_boxes, reduction='none')
                    l1_loss = l1_loss.mean(dim=1)
                    
                    # Combine losses with adaptive weighting
                    iou_quality = torch.clamp((ious - self.iou_threshold) / (self.iou_good_match - self.iou_threshold), 0, 1)
                    bbox_loss = (iou_loss + (1 - iou_quality) * l1_loss).mean()
                    
                    # Stronger penalty for missed detections in bbox loss
                    if len(target_boxes) > positive_mask.sum():
                        bbox_loss *= 1.4  # Increased from 1.2

                    bbox_losses.append(bbox_loss)
                else:
                    bbox_losses.append(torch.tensor(0.0, device=bbox_pred.device))
            
            conf_losses.append(conf_loss)
        
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