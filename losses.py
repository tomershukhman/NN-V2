import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Refined threshold values for better precision
        self.positive_threshold = 0.6  # Increased from 0.5
        self.negative_threshold = 0.4  # Decreased from 0.5
        
        # Focal loss parameters for better handling of class imbalance
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
                max_ious, best_target_idx = ious.max(dim=1)
                
                # Assign positive and negative samples with refined thresholds
                positive_mask = max_ious >= self.positive_threshold
                negative_mask = max_ious < self.negative_threshold
                
                # Ensure minimum positive samples
                if positive_mask.sum() < 3:
                    k = min(3, len(max_ious))
                    top_ious, top_idx = max_ious.topk(k)
                    positive_mask[top_idx] = True
                    negative_mask[top_idx] = False
                
                # Calculate confidence loss with focal loss
                conf_target = positive_mask.float()
                alpha_factor = self.alpha * conf_target + (1 - self.alpha) * (1 - conf_target)
                focal_weight = (1 - conf_pred[i]) * conf_target + conf_pred[i] * (1 - conf_target)
                focal_weight = alpha_factor * focal_weight.pow(self.gamma)
                
                conf_loss = -(conf_target * torch.log(conf_pred[i] + 1e-6) + 
                            (1 - conf_target) * torch.log(1 - conf_pred[i] + 1e-6))
                conf_loss = (conf_loss * focal_weight).mean()
                
                # Calculate bbox loss only for positive samples
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # GIoU Loss for better localization
                    bbox_loss = self._giou_loss(pred_boxes, matched_target_boxes)
                    bbox_loss = bbox_loss.mean()
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