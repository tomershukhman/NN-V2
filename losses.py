import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Optimize thresholds for better detection
        self.positive_threshold = 0.5  # Maintain original positive threshold
        self.negative_threshold = 0.35  # Raised slightly to reduce false negatives
        
        # Enhanced focal loss parameters for better confidence learning
        self.alpha = 0.6  # Increased to give more weight to positive samples
        self.gamma = 1.5  # Moderate gamma to avoid over-suppression
        
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
                # Handle empty targets with smooth focal loss
                conf_loss = -torch.log(1 - conf_pred[i] + 1e-6).mean()
                bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
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
                
                # For targets with lower IoU, ensure some anchors are assigned
                for target_idx in range(len(target_boxes)):
                    # Find top-k anchors per target
                    if target_idx not in best_target_idx[positive_mask]:
                        target_ious = ious[:, target_idx]
                        # Get top 2 matching anchors that aren't already positive
                        candidate_anchors = torch.where((target_ious >= 0.3) & ~positive_mask)[0]
                        if len(candidate_anchors) > 0:
                            # Take top 2 at most
                            k = min(2, len(candidate_anchors))
                            _, top_k_indices = torch.topk(target_ious[candidate_anchors], k=k)
                            top_anchors = candidate_anchors[top_k_indices]
                            positive_mask[top_anchors] = True
                
                negative_mask = max_ious < self.negative_threshold
                negative_mask[positive_mask] = False
                
                # Calculate confidence loss with enhanced focal loss
                conf_target = positive_mask.float()
                
                # Adaptive focal loss weighting based on IoU quality
                iou_weights = torch.ones_like(conf_target)
                iou_weights[positive_mask] = torch.pow(max_ious[positive_mask], 2)  # Higher weight for better IoU
                
                # Alpha weighting for class imbalance
                alpha_factor = torch.ones_like(conf_target) * self.alpha
                alpha_factor[positive_mask] = 1 - self.alpha  # Inverse alpha for positive samples
                
                # Calculate focal weight
                pt = torch.where(conf_target == 1, conf_pred[i], 1 - conf_pred[i])
                focal_weight = torch.pow(1 - pt, self.gamma)
                
                # Combine weights
                combined_weight = focal_weight * alpha_factor * iou_weights
                
                # Binary cross entropy with weights
                bce_loss = F.binary_cross_entropy(conf_pred[i], conf_target, reduction='none')
                conf_loss = (bce_loss * combined_weight).sum() / (combined_weight.sum() + 1e-6)
                
                # Calculate bbox loss only for positive samples with better scaling
                if positive_mask.sum() > 0:
                    matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                    pred_boxes = bbox_pred[i][positive_mask]
                    
                    # Use Distance-IoU loss for better localization
                    diou_loss = self._diou_loss(pred_boxes, matched_target_boxes)
                    
                    # Scale loss based on prediction confidence to prioritize confident predictions
                    conf_values = conf_pred[i][positive_mask]
                    # Weighted loss gives more importance to high-confidence predictions
                    weighted_diou_loss = diou_loss * conf_values
                    
                    bbox_loss = weighted_diou_loss.mean()
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
        """Calculate IoU between two sets of boxes with safe broadcasting"""
        # Handle different dimensions properly
        if boxes1.dim() == 2 and boxes2.dim() == 2:
            # Get the coordinates
            boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            
            # Calculate intersection
            left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
            top = torch.max(boxes1[:, None, 1], boxes2[:, 1])   # [N,M]
            right = torch.min(boxes1[:, None, 2], boxes2[:, 2]) # [N,M]
            bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3]) # [N,M]
            
            width = (right - left).clamp(min=0)  # [N,M]
            height = (bottom - top).clamp(min=0)  # [N,M]
            intersection = width * height  # [N,M]
            
            # Calculate union without using transpose
            union = boxes1_area[:, None] + boxes2_area - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-6)  # [N,M]
            
            return iou
        else:
            # Fall back to element-wise calculation for different dimensions
            iou_matrix = torch.zeros(boxes1.size(0), boxes2.size(0), device=boxes1.device)
            for i in range(boxes1.size(0)):
                for j in range(boxes2.size(0)):
                    iou_matrix[i, j] = self._calculate_single_iou(boxes1[i], boxes2[j])
            return iou_matrix
    
    def _calculate_single_iou(self, box1, box2):
        """Calculate IoU between two individual boxes"""
        # Calculate intersection
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        # Calculate intersection area
        width = (x2 - x1).clamp(min=0)
        height = (y2 - y1).clamp(min=0)
        intersection = width * height
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        return iou
    
    def _diou_loss(self, boxes1, boxes2):
        """
        Distance-IoU Loss for better bounding box regression
        Directly optimizes the object detection evaluation metric
        """
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
        
        # Calculate centers
        c1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
        c1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
        c2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
        c2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        # Calculate distance between centers
        center_dist = torch.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        
        # Calculate diagonal distance of the smallest enclosing box
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_diag = torch.sqrt((enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2)
        
        # Calculate DIoU
        diou = iou - (center_dist**2) / (enc_diag**2 + 1e-6)
        
        # DIoU loss
        return 1 - diou