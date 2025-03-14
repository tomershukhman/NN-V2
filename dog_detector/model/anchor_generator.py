import torch
import math
from config import ANCHOR_SCALES, ANCHOR_RATIOS

class AnchorGenerator:
    def __init__(self):
        super().__init__()
        self.base_anchors = None
        
    def __call__(self, images):
        """Generate anchor boxes optimized for dog detection"""
        device = images.device
        
        # Generate base anchors once and cache them
        if self.base_anchors is None:
            self.base_anchors = self._generate_base_anchors().to(device)
        
        return self.base_anchors
    
    def _generate_base_anchors(self):
        """Generate base anchor boxes optimized for typical dog aspect ratios and sizes"""
        from config import (
            FEATURE_MAP_SIZE, IMAGE_SIZE, 
            ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS
        )
        
        # Calculate stride between anchors
        stride = IMAGE_SIZE / FEATURE_MAP_SIZE
        
        # Generate grid coordinates
        shifts_x = torch.arange(0, FEATURE_MAP_SIZE, dtype=torch.float32) * stride
        shifts_y = torch.arange(0, FEATURE_MAP_SIZE, dtype=torch.float32) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        # Generate anchors with sizes optimized for dogs
        anchors = []
        
        # Scale relative to image size for better generalization
        base_sizes = [s / IMAGE_SIZE for s in ANCHOR_SIZES]
        
        for size in base_sizes:
            for ratio in ANCHOR_ASPECT_RATIOS:
                # Calculate width and height based on size and aspect ratio
                w = size * math.sqrt(ratio)
                h = size / math.sqrt(ratio)
                
                # Generate anchors for each grid point
                for x, y in zip(shift_x, shift_y):
                    # Convert to normalized coordinates
                    cx = (x + stride/2) / IMAGE_SIZE  # Center x
                    cy = (y + stride/2) / IMAGE_SIZE  # Center y
                    
                    # Convert from center format to XYXY format
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    # Clip to image boundaries
                    x1 = torch.clamp(x1, min=0.0, max=1.0)
                    y1 = torch.clamp(y1, min=0.0, max=1.0)
                    x2 = torch.clamp(x2, min=0.0, max=1.0)
                    y2 = torch.clamp(y2, min=0.0, max=1.0)
                    
                    # Only add valid anchors
                    if x2 > x1 and y2 > y1:
                        anchors.append([x1, y1, x2, y2])
        
        # Create tensor WITHOUT gradients - anchors should be fixed
        return torch.tensor(anchors, dtype=torch.float32)

def decode_boxes(box_pred, anchors):
    """Convert predicted box offsets to absolute coordinates with improved scaling"""
    # Get the device from input tensors
    device = box_pred.device
    
    # Convert predictions to center form
    pred_ctr_x = box_pred[..., 0]
    pred_ctr_y = box_pred[..., 1]
    pred_w = box_pred[..., 2]
    pred_h = box_pred[..., 3]
    
    # Get anchor centers and dimensions
    anchor_ctr_x = (anchors[..., 0] + anchors[..., 2]) * 0.5
    anchor_ctr_y = (anchors[..., 1] + anchors[..., 3]) * 0.5
    anchor_w = anchors[..., 2] - anchors[..., 0]
    anchor_h = anchors[..., 3] - anchors[..., 1]
    
    # Scale factors for better gradient flow
    scale_factors = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device)
    
    # Apply transformations with scaling
    pred_ctr_x = pred_ctr_x * scale_factors[0] * anchor_w + anchor_ctr_x
    pred_ctr_y = pred_ctr_y * scale_factors[1] * anchor_h + anchor_ctr_y
    pred_w = torch.exp(torch.clamp(pred_w * scale_factors[2], min=-4.0, max=4.0)) * anchor_w
    pred_h = torch.exp(torch.clamp(pred_h * scale_factors[3], min=-4.0, max=4.0)) * anchor_h
    
    # Convert back to xyxy format
    pred_x1 = pred_ctr_x - pred_w * 0.5
    pred_y1 = pred_ctr_y - pred_h * 0.5
    pred_x2 = pred_ctr_x + pred_w * 0.5
    pred_y2 = pred_ctr_y + pred_h * 0.5
    
    # Clip predictions to [0, 1]
    pred_x1 = torch.clamp(pred_x1, min=0.0, max=1.0)
    pred_y1 = torch.clamp(pred_y1, min=0.0, max=1.0)
    pred_x2 = torch.clamp(pred_x2, min=0.0, max=1.0)
    pred_y2 = torch.clamp(pred_y2, min=0.0, max=1.0)
    
    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)