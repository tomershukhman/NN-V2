import torch
import torch.nn as nn
import torchvision.ops as ops
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    ANCHOR_SCALES, ANCHOR_RATIOS, FEATURE_MAP_SIZE,
    NUM_ANCHORS_PER_CELL, MIN_BOX_SIZE, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO
)
from .backbone import ResNetBackbone
from .detection_head import DetectionHead
from .anchor_generator import generate_anchors, decode_boxes
from .utils.box_utils import nms_with_high_confidence_priority

class DogDetector(nn.Module):
    def __init__(self, feature_map_size=None):
        super(DogDetector, self).__init__()
        self.feature_map_size = feature_map_size if feature_map_size is not None else FEATURE_MAP_SIZE
        
        # Create backbone and detection head
        self.backbone = ResNetBackbone()
        self.detection_head = DetectionHead(NUM_ANCHORS_PER_CELL)
        
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
        batch_size = bbox_pred.shape[0]
        
        for b in range(batch_size):
            boxes = bbox_pred[b]
            scores = conf_pred[b]
            
            # Apply confidence threshold
            mask = scores > CONFIDENCE_THRESHOLD
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) > 0:
                # Clip boxes to image boundaries
                boxes = torch.clamp(boxes, min=0, max=1)
                
                # Filter out very small boxes that are likely noise
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                area_mask = (widths > MIN_BOX_SIZE) & (heights > MIN_BOX_SIZE)
                boxes = boxes[area_mask]
                scores = scores[area_mask]
                
                # Filter out extreme aspect ratio boxes (very thin/wide)
                aspect_ratios = widths / (heights + 1e-6)
                aspect_ratio_mask = (aspect_ratios > MIN_ASPECT_RATIO) & (aspect_ratios < MAX_ASPECT_RATIO)
                boxes = boxes[aspect_ratio_mask]
                scores = scores[aspect_ratio_mask]
                
                if len(boxes) > 0:
                    # Use NMS with high confidence priority
                    # For potential future improvement: Consider using torchvision.ops.batched_nms instead
                    boxes, scores, _ = nms_with_high_confidence_priority(
                        boxes, scores, 
                        iou_threshold=NMS_THRESHOLD,
                        confidence_threshold=CONFIDENCE_THRESHOLD
                    )
                    
                    # Limit maximum detections
                    if len(boxes) > MAX_DETECTIONS:
                        _, topk_indices = torch.topk(scores, k=MAX_DETECTIONS)
                        boxes = boxes[topk_indices]
                        scores = scores[topk_indices]
            
            # Create empty tensors if no boxes detected
            if len(boxes) == 0:
                boxes = torch.empty((0, 4), device=boxes.device if 'boxes' in locals() else conf_pred.device)
                scores = torch.empty(0, device=scores.device if 'scores' in locals() else conf_pred.device)
            
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