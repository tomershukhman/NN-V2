import torch
import math
from config import ANCHOR_SCALES, ANCHOR_RATIOS

class AnchorGenerator:
    def __init__(self):
        super().__init__()
        self.base_anchors = None
        
    def __call__(self, images):
        """Generate anchor boxes for detection"""
        device = images.device
        
        # Generate base anchors once and cache them
        if self.base_anchors is None:
            self.base_anchors = self._generate_base_anchors().to(device)
            
        return self.base_anchors
    
    def _generate_base_anchors(self):
        """Generate base anchor boxes with improved scale handling"""
        from config import FEATURE_MAP_SIZE, IMAGE_SIZE, ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS
        
        # Calculate stride between anchors
        stride = IMAGE_SIZE / FEATURE_MAP_SIZE
        
        # Generate grid coordinates
        shifts_x = torch.arange(0, FEATURE_MAP_SIZE, dtype=torch.float32) * stride
        shifts_y = torch.arange(0, FEATURE_MAP_SIZE, dtype=torch.float32) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        anchors = []
        for x, y in zip(shift_x, shift_y):
            cx = (x + stride/2) / IMAGE_SIZE
            cy = (y + stride/2) / IMAGE_SIZE
            
            # Generate anchors at each position with different scales and aspect ratios
            for size in ANCHOR_SIZES:
                scale = size / IMAGE_SIZE
                for ratio in ANCHOR_ASPECT_RATIOS:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)
                    
                    # Convert to XYXY format
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    # Only add valid anchors
                    if x2 > x1 and y2 > y1:
                        # Clip to image boundaries
                        x1 = max(0.0, min(1.0, x1))
                        y1 = max(0.0, min(1.0, y1))
                        x2 = max(0.0, min(1.0, x2))
                        y2 = max(0.0, min(1.0, y2))
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)

def decode_boxes(box_pred, anchors):
    """Convert predicted box offsets to absolute coordinates"""
    device = box_pred.device
    
    # Get centers and dimensions of anchors
    anchor_x1 = anchors[..., 0]
    anchor_y1 = anchors[..., 1]
    anchor_x2 = anchors[..., 2]
    anchor_y2 = anchors[..., 3]
    anchor_w = anchor_x2 - anchor_x1
    anchor_h = anchor_y2 - anchor_y1
    anchor_cx = (anchor_x1 + anchor_x2) / 2
    anchor_cy = (anchor_y1 + anchor_y2) / 2
    
    # Scale factors for better gradient flow
    scale_factors = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device)
    
    # Extract predicted offsets
    pred_cx = box_pred[..., 0] * scale_factors[0] * anchor_w + anchor_cx
    pred_cy = box_pred[..., 1] * scale_factors[1] * anchor_h + anchor_cy
    pred_w = torch.exp(torch.clamp(box_pred[..., 2] * scale_factors[2], min=-4.0, max=4.0)) * anchor_w
    pred_h = torch.exp(torch.clamp(box_pred[..., 3] * scale_factors[3], min=-4.0, max=4.0)) * anchor_h
    
    # Convert to XYXY format
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2
    
    # Clip predictions to [0, 1]
    pred_x1 = torch.clamp(pred_x1, min=0.0, max=1.0)
    pred_y1 = torch.clamp(pred_y1, min=0.0, max=1.0)
    pred_x2 = torch.clamp(pred_x2, min=0.0, max=1.0)
    pred_y2 = torch.clamp(pred_y2, min=0.0, max=1.0)
    
    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)