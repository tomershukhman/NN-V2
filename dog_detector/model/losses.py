import torch
import torch.nn as nn
import torch.nn.functional as F
from dog_detector.utils import compute_iou


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        """Calculate the total detection loss

        Args:
            predictions (tuple): (classification_output, regression_output)
            targets (list): List of target dictionaries containing 'boxes' and 'labels'
        """
        cls_output, reg_output = predictions
        batch_size = cls_output.size(0)

        total_loss = 0
        cls_losses = []
        reg_losses = []

        for i in range(batch_size):
            # Get target boxes and labels for this image
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']

            if len(target_boxes) == 0:
                continue

            # Reshape outputs for loss computation
            reg_output_reshaped = reg_output[i].permute(
                1, 0, 2, 3).reshape(4, -1).t()
            cls_output_reshaped = cls_output[i].permute(
                1, 0, 2, 3).reshape(cls_output.size(1), -1).t()
            # Calculate IoU between predicted and target boxes
            ious = compute_iou(reg_output_reshaped, target_boxes)

            # Assign targets to predictions based on IoU
            max_ious, max_idx = ious.max(dim=1)
            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.3

            if pos_mask.sum() > 0:
                # Classification loss for positive samples
                pos_pred_cls = cls_output_reshaped[pos_mask]
                pos_target_cls = target_labels[max_idx[pos_mask]]
                cls_loss_pos = F.cross_entropy(pos_pred_cls, pos_target_cls)
                cls_losses.append(cls_loss_pos)

                # Regression loss for positive samples
                pos_pred_reg = reg_output[i][pos_mask]
                pos_target_reg = target_boxes[max_idx[pos_mask]]
                reg_loss = self.smooth_l1_loss(
                    pos_pred_reg, pos_target_reg).mean()
                reg_losses.append(reg_loss)

            if neg_mask.sum() > 0:
                # Classification loss for negative samples (background)
                neg_pred_cls = cls_output_reshaped[neg_mask]
                neg_target_cls = torch.zeros(neg_mask.sum(), dtype=torch.long,
                                             device=cls_output.device)
                cls_loss_neg = F.cross_entropy(neg_pred_cls, neg_target_cls)
                cls_losses.append(cls_loss_neg)

        # Combine all losses
        if cls_losses:
            total_loss += torch.stack(cls_losses).mean()
        if reg_losses:
            total_loss += torch.stack(reg_losses).mean()

        return total_loss
