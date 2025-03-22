import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

@torch.jit.script
def calculate_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Vectorized IoU calculation with JIT compilation"""
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - intersection
    
    return intersection / (union + 1e-6)

@torch.jit.script
def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """JIT-compiled focal loss calculation"""
    pt = torch.where(target == 1.0, pred, 1 - pred)
    alpha_factor = torch.where(target == 1.0, alpha, 1 - alpha)
    focal_weight = (1 - pt).pow(gamma)
    
    return -alpha_factor * focal_weight * torch.log(torch.clamp(
        torch.where(target == 1.0, pred, 1 - pred),
        min=1e-6, max=1-1e-6
    ))

@torch.jit.script
def giou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """JIT-compiled GIoU loss calculation"""
    # Calculate IoU components
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    # Calculate enclosing box
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
    
    # Calculate GIoU
    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return 1 - giou

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('positive_threshold', torch.tensor(0.5))
        self.register_buffer('negative_threshold', torch.tensor(0.4))
        self.register_buffer('alpha', torch.tensor(0.25))
        self.register_buffer('gamma', torch.tensor(2.0))
        
    def forward(self, predictions, targets, conf_weight=1.0, bbox_weight=1.0):
        # Handle both training and validation outputs
        if isinstance(predictions, list):
            batch_size = len(predictions)
            device = predictions[0]['boxes'].device
            
            # Get number of anchors from the first prediction
            default_anchors = None
            for pred in predictions:
                if 'anchors' in pred and pred['anchors'] is not None:
                    default_anchors = pred['anchors']
                    break
            
            if default_anchors is None:
                raise ValueError("No default anchors found in predictions")
                
            num_anchors = len(default_anchors)
            
            # Create tensors in training format efficiently
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
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        default_anchors = predictions['anchors']
        
        batch_size = bbox_pred.shape[0]
        
        # Initialize losses with proper device and dtype
        conf_losses = torch.zeros(batch_size, device=bbox_pred.device)
        bbox_losses = torch.zeros(batch_size, device=bbox_pred.device)
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            
            if len(target_boxes) == 0:
                # Handle empty targets efficiently
                conf_losses[i] = -torch.log(1 - conf_pred[i] + 1e-6).mean()
                continue
                
            # Calculate IoU between anchors and target boxes
            ious = calculate_box_iou(default_anchors, target_boxes)
            
            # Get best matching target for each anchor
            max_ious, best_target_idx = ious.max(dim=1)
            
            # Find best anchors for each target
            best_anchor_per_target = []
            target_ious = ious.permute(1, 0)  # [num_targets, num_anchors]
            
            # Use vectorized operations for matching
            candidate_mask = target_ious > 0.3
            num_candidates = candidate_mask.sum(dim=1)
            
            for t_idx in range(len(target_boxes)):
                if num_candidates[t_idx] > 0:
                    # Get top-k anchors efficiently
                    k = min(3, num_candidates[t_idx].item())
                    topk_ious, topk_indices = torch.topk(target_ious[t_idx], k=k)
                    best_anchor_per_target.extend(topk_indices.tolist())
                else:
                    best_idx = target_ious[t_idx].argmax().item()
                    best_anchor_per_target.append(best_idx)
            
            # Create masks for positive and negative anchors
            positive_mask = max_ious >= self.positive_threshold
            positive_mask[best_anchor_per_target] = True
            
            negative_mask = max_ious < self.negative_threshold
            negative_mask[positive_mask] = False
            
            # Calculate confidence target for focal loss
            conf_target = torch.zeros_like(conf_pred[i])
            conf_target[positive_mask] = 1.0
            
            # Apply focal loss with vectorized operations
            focal_loss_result = focal_loss(conf_pred[i], conf_target, self.alpha, self.gamma)
            
            # Balance positive and negative samples efficiently
            num_positive = positive_mask.sum().item()
            if num_positive > 0:
                neg_ratio = min(3, int(len(default_anchors) / num_positive))
                max_neg = neg_ratio * num_positive
                
                if negative_mask.sum() > max_neg:
                    neg_losses = focal_loss_result[negative_mask]
                    neg_values, neg_indices = torch.topk(neg_losses, k=max_neg)
                    
                    selected_neg_mask = torch.zeros_like(negative_mask)
                    selected_neg_indices = torch.where(negative_mask)[0][neg_indices]
                    selected_neg_mask[selected_neg_indices] = True
                    
                    loss_mask = positive_mask | selected_neg_mask
                    conf_losses[i] = focal_loss_result[loss_mask].mean()
                else:
                    loss_mask = positive_mask | negative_mask
                    conf_losses[i] = focal_loss_result[loss_mask].mean()
            else:
                # Handle no positives case efficiently
                num_neg = min(len(default_anchors) // 4, 100)
                neg_values, _ = torch.topk(focal_loss_result[negative_mask], k=min(num_neg, negative_mask.sum().item()))
                conf_losses[i] = neg_values.mean()
            
            # Calculate bbox loss only for positive samples
            if positive_mask.sum() > 0:
                matched_target_boxes = target_boxes[best_target_idx[positive_mask]]
                pred_boxes = bbox_pred[i][positive_mask]
                
                # Combined IoU and L1 loss with vectorized operations
                box_giou_loss = giou_loss(pred_boxes, matched_target_boxes)
                box_l1_loss = F.l1_loss(pred_boxes, matched_target_boxes, reduction='none').mean(dim=1)
                
                bbox_losses[i] = (box_giou_loss + 0.5 * box_l1_loss).mean()
        
        # Average losses across batch
        conf_loss = conf_losses.mean()
        bbox_loss = bbox_losses.mean()
        
        # Apply weights and combine
        total_loss = conf_weight * conf_loss + bbox_weight * bbox_loss
        
        return {
            'total_loss': total_loss,
            'conf_loss': conf_loss.item(),
            'bbox_loss': bbox_loss.item()
        }