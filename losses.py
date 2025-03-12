import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4]
        
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
            
            # Get positive anchors
            positive_indices = torch.where(positive_mask)[0]
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            if num_positive > 0:
                # Localization loss for positive anchors
                matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                
                # Convert boxes to center form for regression
                matched_gt_centers = (matched_gt_boxes[:, :2] + matched_gt_boxes[:, 2:]) / 2
                matched_gt_sizes = matched_gt_boxes[:, 2:] - matched_gt_boxes[:, :2]
                
                pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
                pred_sizes = pred_boxes[:, 2:] - pred_boxes[:, :2]
                
                # Calculate regression targets
                loc_loss = self.smooth_l1(
                    torch.cat([pred_centers, pred_sizes], dim=1),
                    torch.cat([matched_gt_centers, matched_gt_sizes], dim=1)
                )
                total_loc_loss += loc_loss.sum()
                
                # Create confidence targets
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = 1
                
                # Hard Negative Mining
                num_neg = min(num_positive * self.neg_pos_ratio, num_anchors - num_positive)
                neg_conf_loss = self.bce_loss(conf_pred[i], conf_target)
                
                # Remove positive examples from negative mining
                neg_conf_loss[positive_indices] = 0
                
                # Sort and select hard negatives
                _, neg_indices = neg_conf_loss.sort(descending=True)
                neg_indices = neg_indices[:num_neg]
                
                # Final confidence loss
                conf_loss = neg_conf_loss[neg_indices].sum() + self.bce_loss(
                    conf_pred[i][positive_indices],
                    conf_target[positive_indices]
                ).sum()
                
                total_conf_loss += conf_loss
        
        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / num_pos
        
        total_loss = total_loc_loss + total_conf_loss
        
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