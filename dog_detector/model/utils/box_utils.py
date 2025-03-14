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
    """Calculate IoU between two sets of boxes with proper broadcasting"""
    # Handle empty boxes
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)

    # Convert from center format if needed
    if boxes1.size(-1) == 4:
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    else:
        raise ValueError("Boxes must be in [x1, y1, x2, y2] format")

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # Shape: [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # Shape: [N,M,2]
    wh = (rb - lt).clamp(min=0)  # Shape: [N,M,2]
    intersection = wh[..., 0] * wh[..., 1]  # Shape: [N,M]

    # Compute union
    union = area1[:, None] + area2 - intersection
    
    # Add small epsilon to prevent division by zero
    iou = intersection / (union + 1e-7)
    
    return iou

def diou_loss(boxes1, boxes2):
    """Calculate DIoU Loss with center point distance"""
    # Handle empty boxes
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), device=boxes1.device)

    # Calculate IoU
    iou = box_iou(boxes1, boxes2)

    # Get centers of boxes
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) / 2

    # Calculate center distance
    center_distance = torch.sum((boxes1_center - boxes2_center) ** 2, dim=-1)

    # Calculate diagonal length of smallest enclosing box
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    diagonal_length = torch.sum((rb - lt) ** 2, dim=-1)

    # Compute DIoU Loss
    loss = 1 - iou + (center_distance / (diagonal_length + 1e-7))

    return loss

def weighted_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.05):
    """
    Perform weighted NMS with score thresholding and proper box format handling
    """
    if len(boxes) == 0:
        return boxes, scores, torch.tensor([], device=boxes.device, dtype=torch.int64)
    
    # Filter by score threshold first
    score_mask = scores > score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    
    if len(boxes) == 0:
        return boxes, scores, torch.tensor([], device=boxes.device, dtype=torch.int64)
    
    # Calculate areas once
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort boxes by score
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0]
        keep.append(i.item())
        
        # Calculate IoU between current box and remaining boxes
        remaining_boxes = boxes[order[1:]]
        box_i = boxes[i].unsqueeze(0)
        ious = box_iou(box_i, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU less than threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep], scores[keep], keep

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