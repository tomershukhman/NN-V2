import torch
import torch.nn as nn
from .focal_loss import FocalLoss
from ..utils.box_utils import box_iou, diou_loss
from config import IOU_THRESHOLD, NEG_POS_RATIO, LOC_LOSS_WEIGHT


class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=None, neg_pos_ratio=None, use_focal_loss=True):
        super().__init__()
        self.iou_threshold = iou_threshold if iou_threshold is not None else IOU_THRESHOLD
        self.neg_pos_ratio = neg_pos_ratio if neg_pos_ratio is not None else NEG_POS_RATIO
        self.use_focal_loss = use_focal_loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        
    def forward(self, predictions, targets):
        # Handle different prediction formats
        is_val_mode = isinstance(predictions, list)
        if is_val_mode:
            predictions = self._convert_val_predictions(predictions)
            
        # Extract predictions
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        default_anchors = predictions['anchors']
        
        batch_size = len(targets)
        device = bbox_pred.device
        
        # Initialize loss components
        total_loc_loss = torch.tensor(0., device=device, requires_grad=True)
        total_conf_loss = torch.tensor(0., device=device, requires_grad=True)
        num_pos = 0
        
        # Process each image in the batch
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']
            num_gt = len(gt_boxes)
            
            if num_gt == 0:
                # Handle empty ground truth case - only confidence loss for all predictions as negatives
                conf_target = torch.zeros_like(conf_pred[i])
                conf_loss = self.focal_loss(conf_pred[i], conf_target).mean()
                total_conf_loss = total_conf_loss + conf_loss
                continue
                
            # Calculate IoU between all anchors and ground truth boxes
            # Ensure both tensors have proper dimensions for broadcasting
            # default_anchors shape: [num_anchors, 4]
            # gt_boxes shape: [num_gt, 4]
            if gt_boxes.dim() == 1 and gt_boxes.size(0) == 4:
                # If we got a single box as a 1D tensor, reshape it to [1, 4]
                gt_boxes = gt_boxes.unsqueeze(0)
                
            iou_matrix = box_iou(default_anchors, gt_boxes)
            
            # For each anchor, get the best matching ground truth box
            best_gt_iou, best_gt_idx = iou_matrix.max(dim=1)
            
            # For each ground truth box, get the best matching anchor
            best_anchor_iou, best_anchor_idx = iou_matrix.max(dim=0)
            
            # Create positive mask
            positive_mask = best_gt_iou > self.iou_threshold
            
            # Ensure each ground truth box has at least one positive anchor
            for gt_idx in range(num_gt):
                best_anchor_for_gt = best_anchor_idx[gt_idx]
                positive_mask[best_anchor_for_gt] = True
            
            positive_indices = torch.where(positive_mask)[0]
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            if num_positive > 0:
                # Calculate localization loss for positive anchors
                matched_gt_boxes = gt_boxes[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                
                # Use DIoU loss for better convergence
                loc_loss = diou_loss(pred_boxes, matched_gt_boxes)
                loc_loss = loc_loss.sum() * LOC_LOSS_WEIGHT
                
                # Calculate confidence loss with hard negative mining
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = 1
                
                # Compute confidence loss for all anchors
                if self.use_focal_loss:
                    all_conf_loss = self.focal_loss(conf_pred[i], conf_target)
                else:
                    all_conf_loss = self.bce_loss(conf_pred[i], conf_target)
                
                # Split confidence loss for positive and negative samples
                conf_loss_pos = all_conf_loss[positive_indices].sum()
                
                # Hard negative mining
                negative_mask = ~positive_mask
                conf_loss_neg = all_conf_loss[negative_mask]
                
                if len(conf_loss_neg) > 0:
                    # Sort negative losses for hard negative mining
                    conf_loss_neg, _ = conf_loss_neg.sort(descending=True)
                    num_neg = min(int(num_positive * self.neg_pos_ratio), len(conf_loss_neg))
                    
                    if num_neg > 0:
                        hard_neg_loss = conf_loss_neg[:num_neg].sum()
                        conf_loss = (conf_loss_pos + hard_neg_loss) / (num_positive + num_neg)
                    else:
                        conf_loss = conf_loss_pos / num_positive
                else:
                    conf_loss = conf_loss_pos / num_positive
                
                # Update total losses
                total_loc_loss = total_loc_loss + loc_loss
                total_conf_loss = total_conf_loss + conf_loss
        
        # Normalize losses by batch size and number of positives
        num_pos = max(1, num_pos)  # Prevent division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / batch_size
        
        # Ensure losses maintain gradients
        total_loss = total_loc_loss + total_conf_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item()
        }


    def _convert_val_predictions(self, predictions):
        """Convert validation predictions to training format with improved error handling"""
        batch_size = len(predictions)
        device = predictions[0]['boxes'].device if len(
            predictions) > 0 and 'boxes' in predictions[0] else torch.device('cpu')

        # Get default anchors from the first prediction
        default_anchors = None
        for pred in predictions:
            if pred is not None and 'anchors' in pred and pred['anchors'] is not None:
                default_anchors = pred['anchors']
                break

        if default_anchors is None:
            raise ValueError("No default anchors found in predictions")

        num_anchors = len(default_anchors)

        # Create tensors in training format with improved initialization
        bbox_pred = torch.zeros((batch_size, num_anchors, 4), device=device)
        conf_pred = torch.zeros((batch_size, num_anchors), device=device)
        valid_pred_mask = torch.zeros(
            (batch_size, num_anchors), dtype=torch.bool, device=device)

        for i, pred in enumerate(predictions):
            if pred is None:
                continue

            valid_preds = pred['boxes'].shape[0] if 'boxes' in pred and pred['boxes'].numel() > 0 else 0
            if valid_preds > 0:
                # Ensure we don't try to copy more elements than the tensor has
                valid_preds = min(valid_preds, num_anchors)
                bbox_pred[i, :valid_preds] = pred['boxes'][:valid_preds]
                conf_pred[i, :valid_preds] = torch.clamp(
                    pred['scores'][:valid_preds], min=1e-7, max=1-1e-7)
                valid_pred_mask[i, :valid_preds] = True

        return {
            'bbox_pred': bbox_pred,
            'conf_pred': conf_pred,
            'anchors': default_anchors,
            'valid_pred_mask': valid_pred_mask
        }
