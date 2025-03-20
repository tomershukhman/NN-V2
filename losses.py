import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IOU_THRESHOLD, NEG_POS_RATIO

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=IOU_THRESHOLD, neg_pos_ratio=NEG_POS_RATIO):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.giou_loss = GIoULoss(reduction='none')
        
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
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4]
        
        batch_size = len(targets)
        num_anchors = default_anchors.size(0)
        device = bbox_pred.device
        
        # Initialize loss components
        total_loc_loss = torch.tensor(0., device=device)
        total_conf_loss = torch.tensor(0., device=device)
        total_count_loss = torch.tensor(0., device=device)
        num_pos = 0
        
        # Count statistics to track mismatches
        pred_counts = []
        gt_counts = []
        total_iou = 0.0
        num_matched_boxes = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']  # [num_gt, 4]
            num_gt = len(gt_boxes)
            gt_counts.append(num_gt)
            
            if num_gt == 0:
                # If no ground truth boxes, all predictions should have low confidence
                # Use focal loss to focus on hard negative examples
                conf_loss = self._focal_loss(conf_pred[i], torch.zeros_like(conf_pred[i]))
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
            
            # Get positive anchors
            positive_indices = torch.where(positive_mask)[0]
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            # Count the number of predicted boxes (where confidence > 0.5)
            pred_count = torch.sum(conf_pred[i] > 0.5).item()
            pred_counts.append(pred_count)
            
            if num_positive > 0:
                # Localization loss for positive anchors
                matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                
                # Convert boxes to center form for regression
                matched_gt_centers = (matched_gt_boxes[:, :2] + matched_gt_boxes[:, 2:]) / 2
                matched_gt_sizes = matched_gt_boxes[:, 2:] - matched_gt_boxes[:, :2]
                
                pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
                pred_sizes = pred_boxes[:, 2:] - pred_boxes[:, :2]
                
                # Use combination of smooth L1 and GIoU loss for better localization
                # Smooth L1 for center points, GIoU for overall box
                center_loss = self.smooth_l1(pred_centers, matched_gt_centers).sum(dim=1)
                size_loss = self.smooth_l1(pred_sizes, matched_gt_sizes).sum(dim=1)
                giou_loss = self.giou_loss(pred_boxes, matched_gt_boxes)
                
                # Combine losses with weights
                box_loss = 0.5 * (center_loss + size_loss) + 0.5 * giou_loss
                total_loc_loss += box_loss.sum()
                
                # Calculate IoUs for metrics tracking
                batch_ious = self._box_iou(pred_boxes.unsqueeze(1), matched_gt_boxes.unsqueeze(0)).diagonal()
                total_iou += batch_ious.sum().item()
                num_matched_boxes += num_positive
                
                # Create confidence targets
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = batch_ious.detach()  # Use IoU as target confidence
                
                # Hard Negative Mining with focal loss for negatives
                num_neg = min(num_positive * self.neg_pos_ratio, num_anchors - num_positive)
                
                # Use focal loss for better handling of class imbalance
                neg_conf_loss = self._focal_loss(conf_pred[i], conf_target)
                
                # Remove positive examples from negative mining
                neg_conf_loss[positive_indices] = 0
                
                # Sort and select hard negatives
                _, neg_indices = neg_conf_loss.sort(descending=True)
                neg_indices = neg_indices[:num_neg]
                
                # Final confidence loss - use focal loss for positives too
                conf_loss = neg_conf_loss[neg_indices].sum() + self._focal_loss(
                    conf_pred[i][positive_indices],
                    conf_target[positive_indices]
                ).sum()
                
                total_conf_loss += conf_loss
        
        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / num_pos
        
        # Apply weights to loss components
        weighted_loc_loss = bbox_weight * total_loc_loss
        weighted_conf_loss = conf_weight * total_conf_loss
        
        # Total loss
        total_loss = weighted_loc_loss + weighted_conf_loss
        
        # Calculate mean IoU for metrics
        mean_iou = total_iou / max(1, num_matched_boxes)
        
        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item(),
            'pred_counts': pred_counts,
            'gt_counts': gt_counts,
            'mean_iou': mean_iou,
            'bbox_weight': bbox_weight
        }
    
    def _focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal loss for better handling of class imbalance
        """
        bce_loss = self.bce_loss(pred, target)
        
        # Calculate focal weights
        if gamma > 0:
            pt = torch.where(target > 0, pred, 1-pred)
            focal_weight = (1 - pt) ** gamma
            
            # Apply alpha for positive/negative balance
            if alpha > 0:
                alpha_weight = torch.ones_like(target) * alpha
                alpha_weight = torch.where(target > 0, alpha_weight, 1-alpha_weight)
                focal_weight = focal_weight * alpha_weight
                
            bce_loss = focal_weight * bce_loss
            
        return bce_loss
    
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


class GIoULoss(nn.Module):
    """
    Generalized IoU loss for better box regression
    Provides better gradients when boxes don't overlap
    """
    def __init__(self, reduction='mean'):
        super(GIoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        # Calculate IoU
        iou = self._box_iou(pred_boxes, target_boxes)
        
        # Find enclosing box
        left = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        top = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        right = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        bottom = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Calculate area of enclosing box
        width = (right - left).clamp(min=0)
        height = (bottom - top).clamp(min=0)
        enclosing_area = width * height + 1e-6
        
        # Calculate areas of original boxes
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Find union area of original boxes
        union = pred_area + target_area - iou * pred_area * target_area
        
        # Calculate the extra penalty term
        giou = iou - (enclosing_area - union) / enclosing_area
        
        # Convert to loss (1 - GIoU)
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    @staticmethod
    def _box_iou(boxes1, boxes2):
        # Calculate intersection areas
        left = torch.max(boxes1[:, 0], boxes2[:, 0])
        top = torch.max(boxes1[:, 1], boxes2[:, 1])
        right = torch.min(boxes1[:, 2], boxes2[:, 2])
        bottom = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        width = (right - left).clamp(min=0)
        height = (bottom - top).clamp(min=0)
        intersection = width * height
        
        # Calculate box areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)