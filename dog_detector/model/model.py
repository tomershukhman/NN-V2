import torch
import torch.nn as nn
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    ANCHOR_SCALES, ANCHOR_RATIOS, FEATURE_MAP_SIZE
)
from .backbone import ResNetBackbone
from .detection_head import DetectionHead
from .anchor_generator import generate_anchors, decode_boxes

class DogDetector(nn.Module):
    def __init__(self, feature_map_size=None):
        super(DogDetector, self).__init__()
        self.feature_map_size = feature_map_size if feature_map_size is not None else FEATURE_MAP_SIZE
        
        # Create backbone and detection head
        self.backbone = ResNetBackbone()
        num_anchors_per_cell = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.detection_head = DetectionHead(num_anchors_per_cell)
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', generate_anchors(self.feature_map_size))
    
    def forward(self, x, targets=None):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Get predictions from detection head
        bbox_pred, conf_pred = self.detection_head(features, self.feature_map_size)
        
        # Transform bbox predictions from offsets to actual coordinates
        bbox_pred = decode_boxes(bbox_pred, self.default_anchors)
        
        if targets is not None:
            # Return raw predictions for loss calculation
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': self.default_anchors
            }
        else:
            # Post-process predictions for inference
            return self._post_process(bbox_pred, conf_pred)
            
    def _post_process(self, bbox_pred, conf_pred):
        results = []
        for boxes, scores in zip(bbox_pred, conf_pred):
            # Apply confidence threshold
            mask = scores > CONFIDENCE_THRESHOLD
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) > 0:
                # Clip boxes to image boundaries
                boxes = torch.clamp(boxes, min=0, max=1)
                
                # Apply NMS
                keep_idx = nms(boxes, scores, NMS_THRESHOLD)
                
                # Limit maximum detections
                if len(keep_idx) > MAX_DETECTIONS:
                    scores_for_topk = scores[keep_idx]
                    _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                    keep_idx = keep_idx[topk_indices]
                
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
            else:
                boxes = torch.empty((0, 4), device=boxes.device)
                scores = torch.empty(0, device=scores.device)
            
            results.append({
                'boxes': boxes,
                'scores': scores,
                'anchors': self.default_anchors
            })
        
        return results

def get_model(device):
    """Create and return a DogDetector model instance moved to the specified device."""
    model = DogDetector()
    model = model.to(device)
    return model