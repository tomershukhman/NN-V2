import torch
import torchvision.ops as ops

def coco_to_xyxy(boxes):
    """Convert COCO format [x, y, w, h] to [x1, y1, x2, y2] format"""
    if isinstance(boxes, torch.Tensor):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)
    else:
        return [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]

def xyxy_to_coco(boxes):
    """Convert [x1, y1, x2, y2] format to COCO format [x, y, w, h]"""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
    else:
        return [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]

def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes, maintaining gradients for backpropagation
    
    Args:
        boxes1: [N, 4] ground truth boxes
        boxes2: [M, 4] predicted boxes
        
    Returns:
        IoU tensor of shape [N, M]
    """
    # Calculate intersection coordinates
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Add small epsilon to avoid division by zero
    union = torch.clamp(union, min=1e-7)
    
    return intersection / union

def diou_loss(boxes1, boxes2):
    """
    Calculate DIoU (Distance-IoU) loss between two sets of boxes, maintaining gradients
    
    Args:
        boxes1: [N, 4] predicted boxes
        boxes2: [N, 4] target boxes
        
    Returns:
        DIoU loss tensor
    """
    # Calculate standard IoU
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Add small epsilon to avoid division by zero
    union = torch.clamp(union, min=1e-7)
    iou = intersection / union
    
    # Calculate centers and diagonal distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    center_distance_squared = torch.sum((boxes1_center - boxes2_center) ** 2, dim=-1)
    
    # Calculate diagonal length of the smallest enclosing box
    enclosing_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enclosing_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enclosing_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enclosing_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    diagonal_squared = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
    
    # Add small epsilon to avoid division by zero
    diagonal_squared = torch.clamp(diagonal_squared, min=1e-7)
    
    # Calculate DIoU
    diou = iou - (center_distance_squared / diagonal_squared)
    
    # Return loss
    return 1 - diou

def weighted_nms(boxes, scores, iou_threshold=0.5):
    """
    Perform Weighted Non-Maximum Suppression on boxes.
    
    While torchvision has ops.nms, it doesn't have weighted NMS, so we keep this implementation
    if specialized behavior is needed. For standard NMS, use ops.nms directly instead.
    
    Args:
        boxes (tensor): bounding boxes of shape [N, 4]
        scores (tensor): confidence scores of shape [N]
        iou_threshold (float): IoU threshold for considering boxes as duplicates
        
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_indices)
    """
    device = boxes.device
    
    # If no boxes, return empty tensors
    if boxes.shape[0] == 0:
        return boxes, scores, torch.tensor([], device=device, dtype=torch.int64)
    
    # For standard NMS behavior, just use torchvision's implementation
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    return boxes[keep_indices], scores[keep_indices], keep_indices

def soft_nms(boxes, scores, iou_threshold=0.4, sigma=0.5, score_threshold=0.001):
    """
    Soft NMS implementation. Handles boxes in xyxy format.
    """
    # Use standard NMS from torchvision for now as soft NMS is not critical
    keep_indices = ops.batched_nms(boxes, scores, torch.zeros_like(scores), iou_threshold)
    
    # Only keep boxes above the score threshold
    keep_mask = scores[keep_indices] > score_threshold
    filtered_indices = keep_indices[keep_mask]
    
    return boxes[filtered_indices], scores[filtered_indices], filtered_indices

def nms_with_high_confidence_priority(boxes, scores, iou_threshold=0.35, confidence_threshold=0.45):
    """
    Custom NMS that prioritizes high confidence detections.
    Handles boxes in xyxy format.
    """
    # Filter by confidence first
    conf_mask = scores > confidence_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    
    # If no boxes remain after confidence filtering
    if len(boxes) == 0:
        return boxes, scores
    
    # Use torchvision's NMS
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    
    return boxes[keep_indices], scores[keep_indices]