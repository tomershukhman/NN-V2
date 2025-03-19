import torch
import torch.nn as nn
from utils import box_iou

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=2, conf_weight=1.0, loc_weight=5.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio  # Reduced from 3 to 2
        self.conf_weight = conf_weight
        self.loc_weight = loc_weight  # Increased from 1.0 to 5.0
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=0.11)  # Reduced beta for stronger gradients

    def forward(self, predictions, targets):
        # Handle both training and validation outputs
        if isinstance(predictions, list):
            # Convert validation format to training format
            batch_size = len(predictions)
            device = predictions[0]['boxes'].device if predictions[0]['boxes'].numel() > 0 else predictions[0]['anchors'].device
            
            # Get default anchors from the first prediction that has them
            default_anchors = None
            for pred in predictions:
                if 'anchors' in pred and pred['anchors'] is not None:
                    default_anchors = pred['anchors']
                    break
            
            if default_anchors is None:
                raise ValueError("No default anchors found in predictions")

            # Initialize empty lists for batch processing
            batch_boxes = []
            batch_scores = []
            batch_anchors = []
            
            # Process each prediction in the batch
            for pred in predictions:
                boxes = pred['boxes'] if pred['boxes'].numel() > 0 else torch.zeros((0, 4), device=device)
                scores = pred['scores'] if pred['scores'].numel() > 0 else torch.zeros(0, device=device)
                anchors = default_anchors  # Use the same anchors for all predictions
                
                batch_boxes.append(boxes)
                batch_scores.append(scores)
                batch_anchors.append(anchors)
            
            # Stack into batched tensors
            max_preds = max(boxes.size(0) for boxes in batch_boxes) if any(boxes.size(0) > 0 for boxes in batch_boxes) else 1
            padded_boxes = []
            padded_scores = []
            
            for boxes, scores in zip(batch_boxes, batch_scores):
                if boxes.size(0) < max_preds:
                    # Pad boxes with zeros
                    pad_boxes = torch.zeros((max_preds - boxes.size(0), 4), device=device)
                    pad_scores = torch.zeros(max_preds - scores.size(0), device=device)
                    boxes = torch.cat([boxes, pad_boxes], dim=0)
                    scores = torch.cat([scores, pad_scores], dim=0)
                padded_boxes.append(boxes)
                padded_scores.append(scores)
            
            predictions = {
                'bbox_pred': torch.stack(padded_boxes),  # Shape: [batch_size, max_preds, 4]
                'conf_pred': torch.stack(padded_scores),  # Shape: [batch_size, max_preds]
                'anchors': torch.stack(batch_anchors)  # Shape: [batch_size, num_anchors, 4]
            }

        # Extract predictions
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4] or [num_valid_preds, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors] or [num_valid_preds]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4] or [num_valid_preds, 4]
        
        # Initialize loss components
        device = bbox_pred.device
        total_loc_loss = torch.tensor(0., device=device)
        total_conf_loss = torch.tensor(0., device=device)
        num_pos = 0
        
        # No need to handle validation format differently anymore since we standardized the format above
        batch_size = bbox_pred.size(0)

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']  # [num_gt, 4]
            num_gt = len(gt_boxes)
            
            if num_gt == 0:
                # If no ground truth boxes, all predictions should have low confidence
                conf_loss = self.bce_loss(conf_pred[i], torch.zeros_like(conf_pred[i]))
                total_conf_loss += conf_loss.mean()
                continue
            
            # Calculate IoU between all anchor boxes and gt boxes
            gt_boxes = gt_boxes.unsqueeze(0)  # [1, num_gt, 4]
            default_anchors_exp = default_anchors[i].unsqueeze(1) if len(default_anchors.shape) > 2 else default_anchors.unsqueeze(1)
            
            # Calculate IoU matrix: [num_anchors, num_gt]
            ious = box_iou(default_anchors_exp, gt_boxes, batch_dim=True)
            
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
            num_neg = min(num_positive * self.neg_pos_ratio, len(conf_pred[i]) - num_positive)
            neg_conf_loss = self.bce_loss(conf_pred[i], conf_target)
            
            # Remove positive examples from negative mining
            neg_conf_loss[positive_indices] = 0
            
            # Sort and select hard negatives
            _, neg_indices = neg_conf_loss.sort(descending=True)
            neg_indices = neg_indices[:num_neg]
            
            # Calculate positive and negative confidence losses separately
            pos_conf_loss = self.bce_loss(conf_pred[i][positive_indices], conf_target[positive_indices]).sum()
            neg_conf_loss = neg_conf_loss[neg_indices].sum()
            
            # Normalize confidence losses by their respective counts
            pos_conf_loss = pos_conf_loss / max(num_positive, 1)
            neg_conf_loss = neg_conf_loss / max(num_neg, 1)
            
            total_conf_loss += pos_conf_loss + neg_conf_loss
        
        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = (total_loc_loss / num_pos) * self.loc_weight
        total_conf_loss = (total_conf_loss / batch_size) * self.conf_weight
        
        total_loss = total_loc_loss + total_conf_loss
        
        return {
            'total_loss': total_loss,  # Keep tensor for backward()
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item(),
            'loss_values': {  # Add float values for logging
                'total_loss': total_loss.item(),
                'conf_loss': total_conf_loss.item(),
                'bbox_loss': total_loc_loss.item()
            }
        }