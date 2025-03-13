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
    """Convert predicted box offsets back to absolute coordinates"""
    # Center form
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchor_sizes = anchors[:, 2:] - anchors[:, :2]
    
    # Decode predictions with improved scale handling
    pred_centers = box_pred[..., :2] * anchor_sizes + anchor_centers
    pred_sizes = torch.exp(torch.clamp(box_pred[..., 2:], max=4.0)) * anchor_sizes
    
    # Convert back to [x1, y1, x2, y2] format
    boxes = torch.cat([
        pred_centers - pred_sizes/2,
        pred_centers + pred_sizes/2
    ], dim=-1)
    
    return boxes