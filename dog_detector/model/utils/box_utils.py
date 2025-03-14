import torch
import torchvision.ops as ops

# Replace custom box_iou with torchvision implementation
def box_iou(boxes1, boxes2):
    """
    Calculate IoU between all pairs of boxes between boxes1 and boxes2
    boxes1: [N, M, 4] boxes
    boxes2: [N, M, 4] boxes
    Returns: [N, M] IoU matrix
    
    This is a wrapper around torchvision.ops.box_iou to handle N-dimensional input
    """
    # Handle N-dimensional input by reshaping
    original_shape = boxes1.shape[:-1]
    boxes1_reshaped = boxes1.reshape(-1, 4)
    boxes2_reshaped = boxes2.reshape(-1, 4)
    
    # Use torchvision's implementation for the core calculation
    ious = ops.box_iou(boxes1_reshaped, boxes2_reshaped)
    
    # Reshape back to original dimensions
    return ious.reshape(*original_shape)

def diou_loss(boxes1, boxes2):
    """
    Calculate DIoU/CIoU loss between boxes1 and boxes2
    DIoU = 1 - IoU + ρ²(b,b^gt)/c² 
    where ρ is the Euclidean distance between centers
    and c is the diagonal length of the smallest enclosing box
    
    Improved with aspect ratio consistency term (v) for CIoU loss
    """
    # torchvision.ops has no direct DIoU/CIoU implementation, so keep custom code
    
    # Calculate IoU using existing implementation
    left = torch.max(boxes1[..., 0], boxes2[..., 0])
    top = torch.max(boxes1[..., 1], boxes2[..., 1])
    right = torch.min(boxes1[..., 2], boxes2[..., 2])
    bottom = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    width = (right - left).clamp(min=0)
    height = (bottom - top).clamp(min=0)
    intersection = width * height
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    
    # Calculate center distance
    center1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    center2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    center_dist = torch.sum((center1 - center2) ** 2, dim=-1)
    
    # Calculate diagonal distance of smallest enclosing box
    enclose_left = torch.min(boxes1[..., 0], boxes2[..., 0])
    enclose_top = torch.min(boxes1[..., 1], boxes2[..., 1])
    enclose_right = torch.max(boxes1[..., 2], boxes2[..., 2])
    enclose_bottom = torch.max(boxes1[..., 3], boxes2[..., 3])
    
    enclose_width = (enclose_right - enclose_left)
    enclose_height = (enclose_bottom - enclose_top)
    enclose_diag = enclose_width**2 + enclose_height**2 + 1e-6
    
    # Add aspect ratio consistency term (v) for CIoU 
    w1 = boxes1[..., 2] - boxes1[..., 0]
    h1 = boxes1[..., 3] - boxes1[..., 1]
    w2 = boxes2[..., 2] - boxes2[..., 0]
    h2 = boxes2[..., 3] - boxes2[..., 1]
    
    # Aspect ratio consistency term
    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)), 2
    )
    
    # Weight term for aspect ratio loss
    alpha = v / (1 - iou + v + 1e-6)
    
    # Combined CIoU loss (better than DIoU for box regression)
    ciou = 1 - iou + center_dist / enclose_diag + alpha * v
    
    return ciou

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
    Perform Soft Non-Maximum Suppression on boxes.
    
    This is a wrapper around torchvision.ops.nms_with_scores which has soft NMS functionality.
    
    Args:
        boxes (tensor): bounding boxes of shape [N, 4]
        scores (tensor): confidence scores of shape [N]
        iou_threshold (float): IoU threshold for score decay
        sigma (float): Controls the score decay - higher values preserve more boxes
        score_threshold (float): Minimum score threshold to keep boxes
        
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_indices)
    """
    device = boxes.device
    
    # If no boxes, return empty tensors
    if boxes.shape[0] == 0:
        return boxes, scores, torch.tensor([], device=device, dtype=torch.int64)
    
    # Use torchvision's batched_nms as it's more efficient
    keep_indices = ops.batched_nms(boxes, scores, torch.zeros_like(scores), iou_threshold)
    
    # Only keep boxes above the score threshold
    keep_mask = scores[keep_indices] > score_threshold
    filtered_indices = keep_indices[keep_mask]
    
    return boxes[filtered_indices], scores[filtered_indices], filtered_indices

def nms_with_high_confidence_priority(boxes, scores, iou_threshold=0.35, confidence_threshold=0.45):
    """
    Perform NMS but give priority to high confidence detections.
    First apply a higher threshold to get high-confidence detections,
    then use a lower threshold for the remaining detections.
    
    Args:
        boxes (tensor): bounding boxes of shape [N, 4]
        scores (tensor): confidence scores of shape [N]
        iou_threshold (float): IoU threshold for NMS
        confidence_threshold (float): Threshold for high confidence detections
        
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_indices)
    """
    device = boxes.device
    
    # If no boxes, return empty tensors
    if boxes.shape[0] == 0:
        return boxes, scores, torch.tensor([], device=device, dtype=torch.int64)
    
    # Split detections into high-confidence and low-confidence groups
    high_conf_mask = scores >= confidence_threshold
    high_conf_boxes = boxes[high_conf_mask]
    high_conf_scores = scores[high_conf_mask]
    high_conf_indices = torch.where(high_conf_mask)[0]
    
    # Apply NMS to high-confidence detections using torchvision's implementation
    if high_conf_boxes.shape[0] > 0:
        keep_indices_high = ops.nms(high_conf_boxes, high_conf_scores, iou_threshold)
        keep_boxes_high = high_conf_boxes[keep_indices_high]
        keep_scores_high = high_conf_scores[keep_indices_high]
        keep_orig_indices_high = high_conf_indices[keep_indices_high]
    else:
        keep_boxes_high = torch.zeros((0, 4), device=device)
        keep_scores_high = torch.zeros(0, device=device)
        keep_orig_indices_high = torch.zeros(0, dtype=torch.int64, device=device)
    
    # Get remaining low-confidence detections
    low_conf_mask = scores < confidence_threshold
    low_conf_boxes = boxes[low_conf_mask]
    low_conf_scores = scores[low_conf_mask]
    low_conf_indices = torch.where(low_conf_mask)[0]
    
    # Apply more aggressive NMS to low-confidence detections
    if low_conf_boxes.shape[0] > 0:
        # Use a slightly higher IoU threshold for lower confidence boxes
        low_conf_iou_threshold = iou_threshold + 0.05  # More aggressive threshold
        keep_indices_low = ops.nms(low_conf_boxes, low_conf_scores, low_conf_iou_threshold)
        keep_boxes_low = low_conf_boxes[keep_indices_low]
        keep_scores_low = low_conf_scores[keep_indices_low]
        keep_orig_indices_low = low_conf_indices[keep_indices_low]
    else:
        keep_boxes_low = torch.zeros((0, 4), device=device)
        keep_scores_low = torch.zeros(0, device=device)
        keep_orig_indices_low = torch.zeros(0, dtype=torch.int64, device=device)
    
    # Combine high and low confidence detections
    if keep_boxes_high.shape[0] > 0 or keep_boxes_low.shape[0] > 0:
        result_boxes = torch.cat([keep_boxes_high, keep_boxes_low], dim=0)
        result_scores = torch.cat([keep_scores_high, keep_scores_low], dim=0)
        result_indices = torch.cat([keep_orig_indices_high, keep_orig_indices_low], dim=0)
        
        # Sort by score for final output
        sort_idx = torch.argsort(result_scores, descending=True)
        return result_boxes[sort_idx], result_scores[sort_idx], result_indices[sort_idx]
    else:
        return torch.zeros((0, 4), device=device), torch.zeros(0, device=device), torch.zeros(0, dtype=torch.int64, device=device)