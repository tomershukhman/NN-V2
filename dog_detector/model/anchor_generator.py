import torch
import math
from config import ANCHOR_SCALES, ANCHOR_RATIOS

def generate_anchors(feature_map_size):
    """Generate anchor boxes for each cell in the feature map"""
    anchors = []
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) / feature_map_size
            cy = (i + 0.5) / feature_map_size
            
            for scale in ANCHOR_SCALES:
                for ratio in ANCHOR_RATIOS:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return torch.tensor(anchors, dtype=torch.float32)

def decode_boxes(box_pred, anchors):
    """
    Convert predicted box offsets back to absolute coordinates
    with improved variance scaling and clamping
    
    Args:
        box_pred: Predicted box offsets [batch_size, num_anchors, 4]
        anchors: Anchor boxes [num_anchors, 4]
        
    Returns:
        Absolute box coordinates [batch_size, num_anchors, 4]
    """
    # Convert anchors to center form
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchor_sizes = anchors[:, 2:] - anchors[:, :2]
    
    # Get the device from input tensors
    device = box_pred.device
    
    # Scale factor for offsets - optimized values based on testing
    variance = torch.tensor([0.08, 0.08, 0.15, 0.15], device=device).reshape(1, 1, 4)
    
    # Apply variance to make the regression more stable
    # For center coordinates, apply linear transformation
    pred_centers = box_pred[..., :2] * variance[..., :2] * anchor_sizes + anchor_centers
    
    # For width and height, use exponential transformation with clamping
    # to prevent extremely large boxes
    pred_sizes = torch.exp(torch.clamp(box_pred[..., 2:] * variance[..., 2:], min=-4.0, max=2.5)) * anchor_sizes
    
    # Convert back to [x1, y1, x2, y2] format
    boxes = torch.cat([
        pred_centers - pred_sizes/2,  # x1, y1
        pred_centers + pred_sizes/2   # x2, y2
    ], dim=-1)
    
    # Ensure all coordinates are within valid range [0, 1]
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    
    return boxes