import torch

def box_iou(boxes1, boxes2, batch_dim=False):
    """
    Calculate IoU between all pairs of boxes between boxes1 and boxes2
    Args:
        boxes1: [..., N, 4] or [N, 4] boxes
        boxes2: [..., M, 4] or [M, 4] boxes
        batch_dim: bool, whether inputs include batch dimension
    Returns: [..., N, M] or [N, M] IoU matrix
    """
    if batch_dim:
        # Calculate intersection areas with batch dimension
        left = torch.max(boxes1[..., 0], boxes2[..., 0])
        top = torch.max(boxes1[..., 1], boxes2[..., 1])
        right = torch.min(boxes1[..., 2], boxes2[..., 2])
        bottom = torch.min(boxes1[..., 3], boxes2[..., 3])
    else:
        # Calculate intersection areas without batch dimension
        left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    width = (right - left).clamp(min=0)
    height = (bottom - top).clamp(min=0)
    intersection = width * height
    
    # Calculate union areas
    if batch_dim:
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    else:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        area1 = area1[:, None]  # Add broadcasting dimension
    
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)