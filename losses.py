import torch
import torch.nn as nn
from utils import box_iou

class DetectionLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3, conf_weight=2.0, loc_weight=1.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.conf_weight = conf_weight
        self.loc_weight = loc_weight
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=0.05)

    def forward(self, predictions, targets):
        if isinstance(predictions, list):
            batch_size = len(predictions)
            device = predictions[0]['boxes'].device if predictions[0]['boxes'].numel() > 0 else predictions[0]['anchors'].device

            default_anchors = None
            for pred in predictions:
                if 'anchors' in pred and pred['anchors'] is not None:
                    default_anchors = pred['anchors']
                    break

            if default_anchors is None:
                raise ValueError("No default anchors found in predictions")

            anchor_count = default_anchors.shape[0]

            batch_boxes = []
            batch_scores = []

            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']

                # Truncate or pad to anchor_count
                if boxes.size(0) < anchor_count:
                    pad_boxes = torch.zeros((anchor_count - boxes.size(0), 4), device=device)
                    boxes = torch.cat([boxes, pad_boxes], dim=0)
                else:
                    boxes = boxes[:anchor_count]

                if scores.size(0) < anchor_count:
                    pad_scores = torch.zeros(anchor_count - scores.size(0), device=device)
                    scores = torch.cat([scores, pad_scores], dim=0)
                else:
                    scores = scores[:anchor_count]

                batch_boxes.append(boxes)
                batch_scores.append(scores)

            predictions = {
                'bbox_pred': torch.stack(batch_boxes),
                'conf_pred': torch.stack(batch_scores),
                'anchors': default_anchors
            }

        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        default_anchors = predictions['anchors']

        if bbox_pred.shape[1] != default_anchors.shape[0] or conf_pred.shape[1] != default_anchors.shape[0]:
            raise ValueError(f"Prediction and anchor count mismatch: bbox_pred={bbox_pred.shape[1]}, conf_pred={conf_pred.shape[1]}, anchors={default_anchors.shape[0]}")

        device = bbox_pred.device
        total_loc_loss = torch.tensor(0., device=device)
        total_conf_loss = torch.tensor(0., device=device)
        num_pos = 0

        batch_size = bbox_pred.size(0)

        for i in range(batch_size):
            # Apply heavy penalty if no predictions when we know there should be at least one
            if bbox_pred[i].sum() == 0 or conf_pred[i].max() < 0.01:
                total_conf_loss += 5.0  # Significant penalty for missing guaranteed detection
                continue

            gt_boxes = targets[i]['boxes']
            num_gt = len(gt_boxes)

            if num_gt == 0:
                conf_loss = self.bce_loss(conf_pred[i], torch.zeros_like(conf_pred[i]))
                total_conf_loss += conf_loss.mean()
                continue

            gt_boxes = gt_boxes.unsqueeze(0)
            default_anchors_exp = default_anchors.unsqueeze(1)
            ious = box_iou(default_anchors_exp, gt_boxes, batch_dim=True)

            best_gt_iou, best_gt_idx = ious.max(dim=1)
            best_anchor_iou, best_anchor_idx = ious.max(dim=0)

            positive_mask = best_gt_iou > self.iou_threshold
            for gt_idx in range(num_gt):
                positive_mask[best_anchor_idx[gt_idx]] = True

            positive_indices = torch.where(positive_mask)[0]

            if positive_indices.numel() > 0 and positive_indices.max() >= conf_pred[i].size(0):
                raise IndexError(f"Positive indices out of range. Max={positive_indices.max().item()}, conf_pred size={conf_pred[i].size(0)}")

            num_positive = len(positive_indices)
            num_pos += num_positive

            if num_positive > 0:
                matched_gt_boxes = gt_boxes.squeeze(0)[best_gt_idx[positive_indices]]
                pred_boxes = bbox_pred[i][positive_indices]

                matched_gt_centers = (matched_gt_boxes[:, :2] + matched_gt_boxes[:, 2:]) / 2
                matched_gt_sizes = matched_gt_boxes[:, 2:] - matched_gt_boxes[:, :2]

                pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
                pred_sizes = pred_boxes[:, 2:] - pred_boxes[:, :2]

                loc_loss = self.smooth_l1(
                    torch.cat([pred_centers, pred_sizes], dim=1),
                    torch.cat([matched_gt_centers, matched_gt_sizes], dim=1)
                )
                total_loc_loss += loc_loss.sum()

            conf_target = torch.zeros_like(conf_pred[i])
            if positive_indices.numel() > 0:
                conf_target[positive_indices] = 1

            num_neg = min(num_positive * self.neg_pos_ratio, len(conf_pred[i]) - num_positive)
            neg_conf_loss = self.bce_loss(conf_pred[i], conf_target)
            neg_conf_loss[positive_indices] = 0

            _, neg_indices = neg_conf_loss.sort(descending=True)
            neg_indices = neg_indices[:num_neg]

            pos_conf_loss = self.bce_loss(conf_pred[i][positive_indices], conf_target[positive_indices]).sum()
            neg_conf_loss = neg_conf_loss[neg_indices].sum()

            pos_conf_loss = pos_conf_loss / max(num_positive, 1)
            neg_conf_loss = neg_conf_loss / max(num_neg, 1)

            # Add stronger weighting to positive examples since we know they must exist
            total_conf_loss += (1.5 * pos_conf_loss + neg_conf_loss)

        num_pos = max(1, num_pos)
        total_loc_loss = (total_loc_loss / num_pos) * self.loc_weight
        total_conf_loss = (total_conf_loss / batch_size) * self.conf_weight

        total_loss = total_loc_loss + total_conf_loss

        return {
            'total_loss': total_loss,
            'conf_loss': total_conf_loss.item(),
            'bbox_loss': total_loc_loss.item(),
            'loss_values': {
                'total_loss': total_loss.item(),
                'conf_loss': total_conf_loss.item(),
                'bbox_loss': total_loc_loss.item()
            }
        }
