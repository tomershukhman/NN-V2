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
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)
    
    # Handle single boxes (shape: [4])
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    
    # Ensure proper dimensions
    if boxes1.dim() != 2 or boxes2.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got boxes1: {boxes1.dim()}D and boxes2: {boxes2.dim()}D")
    
    # Ensure shapes are [N, 4] and [M, 4]
    if boxes1.size(1) != 4 or boxes2.size(1) != 4:
        raise ValueError(f"Boxes must be in [x1, y1, x2, y2] format with shape [N, 4]. Got {boxes1.shape} and {boxes2.shape}")

    # Calculate areas for both sets of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Compute union
    union = area1[:, None] + area2 - intersection
    
    # Add small epsilon to prevent division by zero
    return intersection / (union + 1e-7)

def diou_loss(boxes1, boxes2):
    """Calculate DIoU Loss with center point distance"""
    # Handle empty boxes
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(len(boxes1), device=boxes1.device)
    
    # Handle single boxes
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    # Calculate IoU
    iou = box_iou(boxes1, boxes2)
    
    # Diagonal of matrix entries will be the IoU of each box with its corresponding GT
    diag_iou = torch.diag(iou)

    # Get centers of boxes
    boxes1_center = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    boxes2_center = (boxes2[:, :2] + boxes2[:, 2:]) / 2

    # Calculate center distance for each pair of boxes
    center_distance = torch.sum((boxes1_center - boxes2_center) ** 2, dim=-1)

    # Calculate diagonal length of smallest enclosing box
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    diagonal_length = torch.sum((rb - lt) ** 2, dim=-1)

    # Compute DIoU Loss
    loss = 1 - diag_iou + (center_distance / (diagonal_length + 1e-7))

    return loss



