import torch

def box_iou(boxes1, boxes2):
    """
    Calculate IoU between all pairs of boxes between boxes1 and boxes2
    boxes1: [N, M, 4] boxes
    boxes2: [N, M, 4] boxes
    Returns: [N, M] IoU matrix
    """
    # Calculate intersection areas
    left = torch.max(boxes1[..., 0], boxes2[..., 0])
    top = torch.max(boxes1[..., 1], boxes2[..., 1])
    right = torch.min(boxes1[..., 2], boxes2[..., 2])
    bottom = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    width = (right - left).clamp(min=0)
    height = (bottom - top).clamp(min=0)
    intersection = width * height
    
    # Calculate union areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def diou_loss(boxes1, boxes2):
    """
    Calculate DIoU loss between boxes1 and boxes2
    DIoU = 1 - IoU + ρ²(b,b^gt)/c² where ρ is the Euclidean distance between centers
    and c is the diagonal length of the smallest enclosing box
    """
    # Calculate IoU
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
    
    # Calculate DIoU
    diou = 1 - iou + center_dist / enclose_diag
    
    return diou