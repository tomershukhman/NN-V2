import torch
import torch.nn as nn
import torchvision.ops as ops
from .backbone import ResNetBackbone
from .detection_head import DetectionHead
from config import (
    CONFIDENCE_THRESHOLD, MAX_DETECTIONS, NMS_THRESHOLD,
    MIN_BOX_SIZE
)

class DogDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Initialize backbone with pretrained weights
        self.backbone = ResNetBackbone()
        
        # Initialize detection head
        self.detection_head = DetectionHead()
        
        # Initialize anchor generator
        from .anchor_generator import AnchorGenerator
        self.anchor_generator = AnchorGenerator()

    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        batch_size = images.size(0)
        
        # Generate anchors if not cached
        if not hasattr(self, 'cached_anchors') or self.cached_anchors is None:
            self.cached_anchors = self.anchor_generator(images)
        
        # Get raw predictions from detection head
        predictions = self.detection_head(features)
        bbox_pred = predictions['bbox_pred']
        conf_pred = predictions['conf_pred']
        
        if self.training or targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': self.cached_anchors
            }
        
        # Inference mode - decode boxes and apply NMS
        from .anchor_generator import decode_boxes
        detections = []
        
        for i in range(batch_size):
            # Get confidence scores
            scores = torch.sigmoid(conf_pred[i])
            
            # Decode box predictions
            boxes = decode_boxes(bbox_pred[i], self.cached_anchors)
            
            # Initial confidence threshold
            mask = scores > CONFIDENCE_THRESHOLD
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            
            if len(filtered_boxes) > 0:
                # Convert boxes to absolute coordinates for NMS
                boxes_abs = filtered_boxes.clone()
                
                # Apply NMS with clustering
                keep_indices = ops.nms(
                    boxes_abs,
                    filtered_scores,
                    iou_threshold=NMS_THRESHOLD
                )
                
                # Apply max detections limit
                if len(keep_indices) > MAX_DETECTIONS:
                    # Sort by score and take top detections
                    _, sorted_idx = filtered_scores[keep_indices].sort(descending=True)
                    keep_indices = keep_indices[sorted_idx[:MAX_DETECTIONS]]
                
                final_boxes = filtered_boxes[keep_indices]
                final_scores = filtered_scores[keep_indices]
                
                # Additional filtering
                valid_mask = (final_boxes[:, 2] - final_boxes[:, 0] >= MIN_BOX_SIZE) & (final_boxes[:, 3] - final_boxes[:, 1] >= MIN_BOX_SIZE)
                
                final_boxes = final_boxes[valid_mask]
                final_scores = final_scores[valid_mask]
            else:
                final_boxes = torch.empty((0, 4), device=boxes.device)
                final_scores = torch.empty(0, device=scores.device)
            
            detections.append({
                'boxes': final_boxes,
                'scores': final_scores
            })
        
        return detections

def get_model(device):
    """Create and return a DogDetector model instance moved to the specified device."""
    model = DogDetector()
    model = model.to(device)
    return model