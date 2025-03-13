import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3, use_focal_loss=True):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.use_focal_loss = use_focal_loss
        self.bce_loss = nn.BCELoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, predictions, targets):
        # Handle both training and validation outputs
        is_val_mode = isinstance(predictions, list)
        if is_val_mode:
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
            # Track valid predictions to avoid computing loss on padded zeros
            valid_pred_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
            
            for i, pred in enumerate(predictions):
                valid_preds = pred['boxes'].shape[0]
                if valid_preds > 0:
                    bbox_pred[i, :valid_preds] = pred['boxes']
                    conf_pred[i, :valid_preds] = pred['scores']
                    valid_pred_mask[i, :valid_preds] = True
            
            predictions = {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors,
                'valid_pred_mask': valid_pred_mask
            }

        # Extract predictions
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4]
        valid_pred_mask = predictions.get('valid_pred_mask', None)  # For validation mode
        
        batch_size = len(targets)
        num_anchors = default_anchors.size(0)
        device = bbox_pred.device
        
        # Initialize loss components
        total_loc_loss = torch.tensor(0., device=device)
        total_conf_loss = torch.tensor(0., device=device)
        num_pos = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']  # [num_gt, 4]
            num_gt = len(gt_boxes)
            
            if num_gt == 0:
                # If no ground truth boxes, all predictions should have low confidence
                if is_val_mode and valid_pred_mask is not None:
                    # For validation, only compute loss on valid predictions
                    if valid_pred_mask[i].sum() > 0:
                        # Use either BCE or Focal Loss based on configuration
                        if self.use_focal_loss:
                            conf_logits = torch.log(conf_pred[i][valid_pred_mask[i]] / (1 - conf_pred[i][valid_pred_mask[i]] + 1e-10))
                            conf_loss = self.focal_loss(conf_logits, torch.zeros_like(conf_pred[i][valid_pred_mask[i]]))
                        else:
                            conf_loss = self.bce_loss(conf_pred[i][valid_pred_mask[i]], 
                                                torch.zeros_like(conf_pred[i][valid_pred_mask[i]]))
                        total_conf_loss += conf_loss.mean()
                else:
                    # Use either BCE or Focal Loss based on configuration
                    if self.use_focal_loss:
                        conf_logits = torch.log(conf_pred[i] / (1 - conf_pred[i] + 1e-10))
                        conf_loss = self.focal_loss(conf_logits, torch.zeros_like(conf_pred[i]))
                    else:
                        conf_loss = self.bce_loss(conf_pred[i], torch.zeros_like(conf_pred[i]))
                    total_conf_loss += conf_loss.mean()
                continue
            
            # Calculate IoU between all anchor boxes and gt boxes
            # Expand dimensions for broadcasting
            gt_boxes = gt_boxes.unsqueeze(0)  # [1, num_gt, 4]
            default_anchors_exp = default_anchors.unsqueeze(1)  # [num_anchors, 1, 4]
            
            # Calculate IoU matrix: [num_anchors, num_gt]
            ious = self._box_iou(default_anchors_exp, gt_boxes)
            
            # Find best gt for each anchor and best anchor for each gt
            best_gt_iou, best_gt_idx = ious.max(dim=1)  # [num_anchors]
            best_anchor_iou, best_anchor_idx = ious.max(dim=0)  # [num_gt]
            
            # Create targets for positive anchors
            positive_mask = best_gt_iou > self.iou_threshold
            
            # Ensure each gt box has at least one positive anchor
            for gt_idx in range(num_gt):
                best_anchor_for_gt = best_anchor_idx[gt_idx]
                positive_mask[best_anchor_for_gt] = True
            
            # For validation mode, restrict calculations to valid predictions
            if is_val_mode and valid_pred_mask is not None:
                # Combine valid_pred_mask with positive_mask to only include valid predictions
                positive_mask = positive_mask & valid_pred_mask[i]
            
            # Get positive anchors
            positive_indices = torch.where(positive_mask)[0]
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            if num_positive > 0:
                # Localization loss for positive anchors - using the matched GT boxes
                matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                
                # Calculate GIoU or DIoU loss instead of simple L1 for better localization
                loc_loss = self._diou_loss(pred_boxes, matched_gt_boxes)
                total_loc_loss += loc_loss.sum()
                
                # Create confidence targets
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = 1
                
                # Hard Negative Mining with improved logic for validation mode
                if is_val_mode and valid_pred_mask is not None:
                    neg_conf_loss = self.bce_loss(conf_pred[i], conf_target) 
                    # Zero out loss for non-valid predictions
                    neg_conf_loss[~valid_pred_mask[i]] = 0
                else:
                    neg_conf_loss = self.bce_loss(conf_pred[i], conf_target)
                
                # Remove positive examples from negative mining
                neg_conf_loss[positive_indices] = 0
                
                # Sort and select hard negatives
                _, neg_indices = neg_conf_loss.sort(descending=True)
                num_neg = min(num_positive * self.neg_pos_ratio, 
                             (valid_pred_mask[i].sum() if is_val_mode and valid_pred_mask is not None 
                              else num_anchors) - num_positive)
                neg_indices = neg_indices[:num_neg]
                
                # Final confidence loss - using Focal Loss for better handling of class imbalance
                if self.use_focal_loss:
                    pos_logits = torch.log(conf_pred[i][positive_indices] / (1 - conf_pred[i][positive_indices] + 1e-10))
                    neg_logits = torch.log(conf_pred[i][neg_indices] / (1 - conf_pred[i][neg_indices] + 1e-10))
                    
                    pos_loss = self.focal_loss(pos_logits, conf_target[positive_indices])
                    neg_loss = self.focal_loss(neg_logits, conf_target[neg_indices])
                    
                    conf_loss = pos_loss.sum() + neg_loss.sum()
                else:
                    pos_loss = self.bce_loss(conf_pred[i][positive_indices], conf_target[positive_indices])
                    neg_loss = self.bce_loss(conf_pred[i][neg_indices], conf_target[neg_indices])
                    
                    conf_loss = pos_loss.sum() + neg_loss.sum()
                
                total_conf_loss += conf_loss
            
            elif is_val_mode and valid_pred_mask is not None and valid_pred_mask[i].sum() > 0:
                # If no positive matches but we have valid predictions, 
                # they should all have low confidence
                if self.use_focal_loss:
                    conf_logits = torch.log(conf_pred[i][valid_pred_mask[i]] / (1 - conf_pred[i][valid_pred_mask[i]] + 1e-10))
                    conf_loss = self.focal_loss(conf_logits, torch.zeros_like(conf_pred[i][valid_pred_mask[i]]))
                else:
                    conf_loss = self.bce_loss(conf_pred[i][valid_pred_mask[i]], 
                                         torch.zeros_like(conf_pred[i][valid_pred_mask[i]]))
                total_conf_loss += conf_loss.sum()
        
        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / num_pos
        
        # Weighted sum - higher weight on localization to improve box accuracy
        total_loss = 1.5 * total_loc_loss + total_conf_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item()
        }

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """
        Calculate IoU between all pairs of boxes between boxes1 and boxes2
        boxes1: [N, M, 4] boxes
        boxes2: [N, M, 4] boxes
        Returns: [N, M] IoU matrix
        """
        # Calculate intersection areas
        left = torch.max(boxes1[..., 0], boxes2[..., 0])
        top = torch.max(boxes1[..., 1], boxes2[..., 1])
        right = torch.min(boxes1[..., 2], boxes2[..., 2])
        bottom = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        width = (right - left).clamp(min=0)
        height = (bottom - top).clamp(min=0)
        intersection = width * height
        
        # Calculate union areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    @staticmethod
    def _diou_loss(boxes1, boxes2):
        """
        Calculate DIoU loss between boxes1 and boxes2
        DIoU = 1 - IoU + ρ²(b,b^gt)/c² where ρ is the Euclidean distance between centers
        and c is the diagonal length of the smallest enclosing box
        """
        # Calculate IoU
        left = torch.max(boxes1[..., 0], boxes2[..., 0])
        top = torch.max(boxes1[..., 1], boxes2[..., 1])
        right = torch.min(boxes1[..., 2], boxes2[..., 2])
        bottom = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        width = (right - left).clamp(min=0)
        height = (bottom - top).clamp(min=0)
        intersection = width * height
        
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        
        # Calculate center distance
        center1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
        center2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
        center_dist = torch.sum((center1 - center2) ** 2, dim=-1)
        
        # Calculate diagonal distance of smallest enclosing box
        enclose_left = torch.min(boxes1[..., 0], boxes2[..., 0])
        enclose_top = torch.min(boxes1[..., 1], boxes2[..., 1])
        enclose_right = torch.max(boxes1[..., 2], boxes2[..., 2])
        enclose_bottom = torch.max(boxes1[..., 3], boxes2[..., 3])
        
        enclose_width = (enclose_right - enclose_left)
        enclose_height = (enclose_bottom - enclose_top)
        enclose_diag = enclose_width**2 + enclose_height**2 + 1e-6
        
        # Calculate DIoU
        diou = 1 - iou + center_dist / enclose_diag
        
        return diou