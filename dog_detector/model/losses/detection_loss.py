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
        self.bce_loss = nn.BCELoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')

    def forward(self, predictions, targets):
        # Handle both training and validation outputs
        is_val_mode = isinstance(predictions, list)
        if is_val_mode:
            predictions = self._convert_val_predictions(predictions)

        # Extract predictions
        bbox_pred = predictions['bbox_pred']  # Shape: [batch_size, num_anchors, 4]
        conf_pred = predictions['conf_pred']  # Shape: [batch_size, num_anchors]
        default_anchors = predictions['anchors']  # Shape: [num_anchors, 4]
        valid_pred_mask = predictions.get('valid_pred_mask', None)  # For validation mode

        batch_size = len(targets)
        device = bbox_pred.device

        # Initialize loss components
        total_loc_loss = torch.tensor(0., device=device)
        total_conf_loss = torch.tensor(0., device=device)
        num_pos = 0

        for i in range(batch_size):
            batch_loc_loss, batch_conf_loss, batch_num_pos = self._compute_batch_loss(
                bbox_pred[i], conf_pred[i], default_anchors,
                targets[i], valid_pred_mask[i] if valid_pred_mask is not None else None,
                is_val_mode
            )

            total_loc_loss += batch_loc_loss
            total_conf_loss += batch_conf_loss
            num_pos += batch_num_pos

        # Normalize losses
        num_pos = max(1, num_pos)  # Avoid division by zero
        total_loc_loss = total_loc_loss / num_pos
        total_conf_loss = total_conf_loss / num_pos

        # Weighted sum - higher weight on localization early in training
        loc_weight = 2.0  # Increased from 1.5 to emphasize localization
        total_loss = loc_weight * total_loc_loss + total_conf_loss

        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item()
        }

    def _compute_batch_loss(self, bbox_pred, conf_pred, default_anchors, target, valid_pred_mask, is_val_mode):
        gt_boxes = target['boxes']  # [num_gt, 4]
        num_gt = len(gt_boxes)
        device = bbox_pred.device

        if num_gt == 0:
            return self._compute_negative_loss(conf_pred, valid_pred_mask, is_val_mode)

        # Calculate IoU between all anchor boxes and gt boxes
        gt_boxes = gt_boxes.unsqueeze(0)  # [1, num_gt, 4]
        default_anchors_exp = default_anchors.unsqueeze(1)  # [num_anchors, 1, 4]

        ious = box_iou(default_anchors_exp, gt_boxes)
        best_gt_iou, best_gt_idx = ious.max(dim=1)  # [num_anchors]
        best_anchor_iou, best_anchor_idx = ious.max(dim=0)  # [num_gt]

        # Create targets for positive anchors
        positive_mask = best_gt_iou > self.iou_threshold

        # Ensure each gt box has at least one positive anchor
        for gt_idx in range(num_gt):
            best_anchor_for_gt = best_anchor_idx[gt_idx]
            positive_mask[best_anchor_for_gt] = True

        if is_val_mode and valid_pred_mask is not None:
            positive_mask = positive_mask & valid_pred_mask

        positive_indices = torch.where(positive_mask)[0]
        num_positive = len(positive_indices)

        loc_loss = torch.tensor(0., device=device)
        conf_loss = torch.tensor(0., device=device)

        if num_positive > 0:
            # Localization loss
            matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
            pred_boxes = bbox_pred[positive_indices]
            loc_loss = diou_loss(pred_boxes, matched_gt_boxes).sum()

            # Confidence loss with hard negative mining
            conf_target = torch.zeros_like(conf_pred)
            conf_target[positive_indices] = 1

            conf_loss = self._compute_conf_loss(
                conf_pred, conf_target, positive_indices,
                valid_pred_mask if is_val_mode else None,
                num_positive
            )

        return loc_loss, conf_loss, num_positive

    def _compute_negative_loss(self, conf_pred, valid_pred_mask, is_val_mode):
        device = conf_pred.device
        if is_val_mode and valid_pred_mask is not None and valid_pred_mask.sum() > 0:
            if self.use_focal_loss:
                conf_logits = torch.log(conf_pred[valid_pred_mask] / (1 - conf_pred[valid_pred_mask] + 1e-10))
                conf_loss = self.focal_loss(conf_logits, torch.zeros_like(conf_pred[valid_pred_mask]))
            else:
                conf_loss = self.bce_loss(conf_pred[valid_pred_mask],
                                     torch.zeros_like(conf_pred[valid_pred_mask]))
            conf_loss = conf_loss.mean()
        else:
            if self.use_focal_loss:
                conf_logits = torch.log(conf_pred / (1 - conf_pred + 1e-10))
                conf_loss = self.focal_loss(conf_logits, torch.zeros_like(conf_pred))
            else:
                conf_loss = self.bce_loss(conf_pred, torch.zeros_like(conf_pred))
            conf_loss = conf_loss.mean()

        return torch.tensor(0., device=device), conf_loss, 0

    def _compute_conf_loss(self, conf_pred, conf_target, positive_indices, valid_pred_mask, num_positive):
        neg_conf_loss = self.bce_loss(conf_pred, conf_target)

        if valid_pred_mask is not None:
            neg_conf_loss[~valid_pred_mask] = 0

        neg_conf_loss[positive_indices] = 0

        _, neg_indices = neg_conf_loss.sort(descending=True)
        num_neg = min(num_positive * self.neg_pos_ratio,
                     (valid_pred_mask.sum() if valid_pred_mask is not None else len(conf_pred)) - num_positive)
        neg_indices = neg_indices[:num_neg]

        if self.use_focal_loss:
            pos_logits = torch.log(conf_pred[positive_indices] / (1 - conf_pred[positive_indices] + 1e-10))
            neg_logits = torch.log(conf_pred[neg_indices] / (1 - conf_pred[neg_indices] + 1e-10))

            pos_loss = self.focal_loss(pos_logits, conf_target[positive_indices])
            neg_loss = self.focal_loss(neg_logits, conf_target[neg_indices])

            conf_loss = pos_loss.sum() + neg_loss.sum()
        else:
            pos_loss = self.bce_loss(conf_pred[positive_indices], conf_target[positive_indices])
            neg_loss = self.bce_loss(conf_pred[neg_indices], conf_target[neg_indices])

            conf_loss = pos_loss.sum() + neg_loss.sum()

        return conf_loss

    def _convert_val_predictions(self, predictions):
        batch_size = len(predictions)
        device = predictions[0]['boxes'].device

        # Get default anchors from the first prediction
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
        valid_pred_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)

        for i, pred in enumerate(predictions):
            valid_preds = pred['boxes'].shape[0]
            if valid_preds > 0:
                bbox_pred[i, :valid_preds] = pred['boxes']
                conf_pred[i, :valid_preds] = pred['scores']
                valid_pred_mask[i, :valid_preds] = True

        return {
            'bbox_pred': bbox_pred,
            'conf_pred': conf_pred,
            'anchors': default_anchors,
            'valid_pred_mask': valid_pred_mask
        }