import torch
import torch.nn as nn
from .focal_loss import FocalLoss
from ..utils.box_utils import box_iou, diou_loss
from config import IOU_THRESHOLD, NEG_POS_RATIO

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
            
        # Extract predictions and ensure they require gradients
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4]
        valid_pred_mask = predictions.get('valid_pred_mask', None)  # For validation mode
        
        batch_size = len(targets)
        device = bbox_pred.device
        
        # Initialize loss components with zeros
        total_loc_loss = torch.tensor(0., device=device, requires_grad=True)
        total_conf_loss = torch.tensor(0., device=device, requires_grad=True)
        num_pos = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']  # [num_gt, 4]
            num_gt = len(gt_boxes)
            
            if num_gt == 0:
                # For empty ground truth, all predictions should have low confidence
                conf_scores = torch.sigmoid(conf_pred[i])  # Convert to probabilities
                conf_loss = torch.mean(-torch.log(1 - conf_scores + 1e-6))
                total_conf_loss = total_conf_loss + conf_loss
                continue
            
            # Calculate IoU between anchors and ground truth boxes
            gt_boxes = gt_boxes.unsqueeze(0)  # [1, num_gt, 4]
            default_anchors_exp = default_anchors.unsqueeze(1)  # [num_anchors, 1, 4]
            
            # Calculate IoUs with gradient flow
            ious = box_iou(default_anchors_exp, gt_boxes)  # [num_anchors, num_gt]
            
            # Find best matches
            best_gt_iou, best_gt_idx = ious.max(dim=1)  # [num_anchors]
            best_anchor_iou, best_anchor_idx = ious.max(dim=0)  # [num_gt]
            
            # Create positive mask
            positive_mask = (best_gt_iou > self.iou_threshold)
            
            # Ensure each gt box has at least one positive anchor
            for gt_idx in range(num_gt):
                best_anchor_for_gt = best_anchor_idx[gt_idx]
                positive_mask[best_anchor_for_gt] = True
            
            # Apply validation mask if needed
            if is_val_mode and valid_pred_mask is not None:
                positive_mask = positive_mask & valid_pred_mask[i]
            
            positive_indices = torch.where(positive_mask)[0]
            num_positive = len(positive_indices)
            num_pos += num_positive
            
            if num_positive > 0:
                # Localization loss for positive anchors
                matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]
                
                # Compute localization loss with gradient clipping
                loc_loss = diou_loss(pred_boxes, matched_gt_boxes)
                loc_loss = torch.clamp(loc_loss, max=100.0)  # Prevent exploding gradients
                total_loc_loss = total_loc_loss + loc_loss.sum()
                
                # Prepare confidence targets
                conf_target = torch.zeros_like(conf_pred[i])
                conf_target[positive_indices] = 1
                
                # Calculate confidence loss
                if self.use_focal_loss:
                    all_conf_loss = self.focal_loss(conf_pred[i], conf_target)
                else:
                    # Apply sigmoid here for BCE loss
                    conf_scores = torch.sigmoid(conf_pred[i])
                    all_conf_loss = self.bce_loss(conf_scores, conf_target)
                
                # Hard negative mining
                conf_loss_pos = all_conf_loss[positive_indices]
                negative_mask = ~positive_mask
                if valid_pred_mask is not None:
                    negative_mask = negative_mask & valid_pred_mask[i]
                
                conf_loss_neg = all_conf_loss[negative_mask]
                conf_loss_neg, _ = conf_loss_neg.sort(descending=True)
                num_neg = min(conf_loss_neg.size(0), num_positive * self.neg_pos_ratio)
                conf_loss_neg = conf_loss_neg[:num_neg]
                
                conf_loss = torch.cat([conf_loss_pos, conf_loss_neg])
                conf_loss = torch.clamp(conf_loss, max=100.0)
                total_conf_loss = total_conf_loss + conf_loss.sum()
        
        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / num_pos
        
        # Total loss
        total_loss = total_loc_loss + total_conf_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item()
        }
    
    def _compute_negative_loss(self, conf_pred, valid_pred_mask, is_val_mode):
        """Compute loss for negative examples with improved stability"""
        # Add small epsilon to prevent log(0)
        eps = 1e-7
        
        if is_val_mode and valid_pred_mask is not None and valid_pred_mask.sum() > 0:
            if self.use_focal_loss:
                conf_logits = torch.clamp(conf_pred[valid_pred_mask], min=eps, max=1-eps)
                conf_loss = self.focal_loss(torch.log(conf_logits / (1 - conf_logits)), 
                                         torch.zeros_like(conf_pred[valid_pred_mask]))
            else:
                conf_loss = self.bce_loss(conf_pred[valid_pred_mask],
                                     torch.zeros_like(conf_pred[valid_pred_mask]))
        else:
            if self.use_focal_loss:
                conf_logits = torch.clamp(conf_pred, min=eps, max=1-eps)
                conf_loss = self.focal_loss(torch.log(conf_logits / (1 - conf_logits)),
                                         torch.zeros_like(conf_pred))
            else:
                conf_loss = self.bce_loss(conf_pred, torch.zeros_like(conf_pred))
        
        return torch.clamp(conf_loss.mean(), max=100.0)  # Prevent exploding gradients
    
    def _convert_val_predictions(self, predictions):
        """Convert validation predictions to training format with improved error handling"""
        batch_size = len(predictions)
        device = predictions[0]['boxes'].device if len(predictions) > 0 else torch.device('cpu')
        
        # Get default anchors from the first prediction
        default_anchors = None
        for pred in predictions:
            if 'anchors' in pred and pred['anchors'] is not None:
                default_anchors = pred['anchors']
                break
        
        if default_anchors is None:
            raise ValueError("No default anchors found in predictions")
        
        num_anchors = len(default_anchors)
        
        # Create tensors in training format with improved initialization
        bbox_pred = torch.zeros((batch_size, num_anchors, 4), device=device)
        conf_pred = torch.zeros((batch_size, num_anchors), device=device)
        valid_pred_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
        
        for i, pred in enumerate(predictions):
            if pred is None:
                continue
                
            valid_preds = pred['boxes'].shape[0] if 'boxes' in pred else 0
            if valid_preds > 0:
                bbox_pred[i, :valid_preds] = pred['boxes']
                conf_pred[i, :valid_preds] = torch.clamp(pred['scores'], min=1e-7, max=1-1e-7)
                valid_pred_mask[i, :valid_preds] = True
        
        return {
            'bbox_pred': bbox_pred,
            'conf_pred': conf_pred,
            'anchors': default_anchors,
            'valid_pred_mask': valid_pred_mask
        }