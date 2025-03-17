import torch
import torch.nn as nn
import torch.nn.functional as F
from dog_detector.utils import compute_iou


class DetectionLoss(nn.Module):
    def __init__(self, model, reg_loss_scale=4.0):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.reg_loss_scale = reg_loss_scale  # Should match scale in _decode_boxes
        self.model = model.module if hasattr(model, 'module') else model

    def forward(self, predictions, targets):
        """Calculate detection loss with properly scaled regression targets"""
        cls_output, reg_output, anchors = predictions  # Now receiving anchors from forward pass
        batch_size = cls_output.size(0)
        device = cls_output.device
        
        cls_losses = torch.zeros(batch_size, device=device)
        reg_losses = torch.zeros(batch_size, device=device)
        total_positive_samples = 0

        # Debug info
        total_images_with_dogs = 0
        total_images_with_pos_matches = 0
        pos_matches_per_image = []
        gt_box_sizes = []
        max_gt_iou_values = []

        print("\n--- Starting batch processing ---")
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']
            
            # Get outputs for this image
            reg_output_i = reg_output[i].permute(1, 0, 2, 3).reshape(4, -1).t()
            cls_output_i = cls_output[i].permute(1, 0, 2, 3).reshape(cls_output.size(1), -1).t()
            
            # For negative samples
            if len(target_boxes) == 0:
                neg_target_cls = torch.zeros(cls_output_i.size(0), dtype=torch.long, device=device)
                cls_loss = F.cross_entropy(cls_output_i, neg_target_cls, reduction='mean')
                cls_losses[i] = cls_loss
                continue

            # Count images with dogs
            total_images_with_dogs += 1
            
            # Log GT box sizes for debugging
            for box in target_boxes:
                width = box[2] - box[0]
                height = box[3] - box[1]
                gt_box_sizes.append((width.item(), height.item()))

            # Decode regression outputs to get actual box coordinates
            decoded_boxes = self.model._decode_boxes(reg_output_i, anchors)
            
            # Compute IoU between decoded predictions and targets
            ious = compute_iou(decoded_boxes, target_boxes)  # Shape: [num_anchors, num_gt]
            max_ious, max_idx = ious.max(dim=1)  # For each anchor, get best GT box
            
            # Store max IoU for debugging
            if len(max_ious) > 0:
                max_gt_iou = max_ious.max().item()
                max_gt_iou_values.append(max_gt_iou)
                print(f"Image {i}: Best anchor IoU: {max_gt_iou:.4f}")

            # Assign positive and negative samples
            pos_mask = max_ious >= 0.25  # Use same threshold as in model's post-processing
            neg_mask = max_ious < 0.1

            # Ensure at least one positive match per GT box if available
            if pos_mask.sum() > 0:
                total_images_with_pos_matches += 1
                pos_matches_per_image.append(pos_mask.sum().item())

                # Classification loss for positive samples
                pos_pred_cls = cls_output_i[pos_mask]
                pos_target_cls = target_labels[max_idx[pos_mask]]
                cls_loss_pos = F.cross_entropy(pos_pred_cls, pos_target_cls, reduction='mean')
                cls_losses[i] += cls_loss_pos

                # For regression, convert GT boxes to offsets relative to anchors
                pos_pred_reg = reg_output_i[pos_mask]
                pos_gt_boxes = target_boxes[max_idx[pos_mask]]
                
                # Calculate regression targets in the same format as predictions
                pos_anchors = anchors[pos_mask]
                anchor_widths = pos_anchors[:, 2] - pos_anchors[:, 0]
                anchor_heights = pos_anchors[:, 3] - pos_anchors[:, 1]
                anchor_ctr_x = pos_anchors[:, 0] + 0.5 * anchor_widths
                anchor_ctr_y = pos_anchors[:, 1] + 0.5 * anchor_heights
                
                gt_widths = pos_gt_boxes[:, 2] - pos_gt_boxes[:, 0]
                gt_heights = pos_gt_boxes[:, 3] - pos_gt_boxes[:, 1]
                gt_ctr_x = pos_gt_boxes[:, 0] + 0.5 * gt_widths
                gt_ctr_y = pos_gt_boxes[:, 1] + 0.5 * gt_heights

                # Compute regression targets using inverse of _decode_boxes transformation
                tx = (gt_ctr_x - anchor_ctr_x) * (4.0 / anchor_widths)  # Scale matches _decode_boxes
                ty = (gt_ctr_y - anchor_ctr_y) * (4.0 / anchor_heights)
                tw = torch.log(gt_widths / anchor_widths)
                th = torch.log(gt_heights / anchor_heights)

                # Convert tx, ty back to logits for sigmoid
                tx = (tx + 1) / 2  # Map [-1, 1] to [0, 1] for sigmoid
                ty = (ty + 1) / 2

                reg_targets = torch.stack([tx, ty, tw, th], dim=1)
                reg_loss = self.smooth_l1_loss(pos_pred_reg, reg_targets).mean()
                reg_losses[i] = reg_loss
                total_positive_samples += pos_mask.sum()

            # Classification loss for negative samples
            if neg_mask.sum() > 0:
                neg_pred_cls = cls_output_i[neg_mask]
                neg_target_cls = torch.zeros(neg_mask.sum(), dtype=torch.long, device=device)
                cls_loss_neg = F.cross_entropy(neg_pred_cls, neg_target_cls, reduction='mean')
                cls_losses[i] += cls_loss_neg
                
        # Calculate final losses
        cls_loss_final = cls_losses.mean()
        reg_loss_final = reg_losses.sum() / (total_positive_samples if total_positive_samples > 0 else 1)
        total_loss = cls_loss_final + reg_loss_final

        # Print debug information
        print(f"Batch size: {batch_size}, Images with dogs: {total_images_with_dogs}, Images with pos matches: {total_images_with_pos_matches}")
        if pos_matches_per_image:
            print(f"Positive matches per image: {pos_matches_per_image}")
        if gt_box_sizes:
            print(f"GT box sizes (w,h): {gt_box_sizes}")
        if max_gt_iou_values:
            print(f"Max IoU values: {max_gt_iou_values}")
        print(f"Total positive samples: {total_positive_samples}, Reg loss: {reg_loss_final.item():.6f}")
        print("--- End batch processing ---\n")

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss_final,
            'reg_loss': reg_loss_final,
            'num_positive_samples': total_positive_samples
        }
