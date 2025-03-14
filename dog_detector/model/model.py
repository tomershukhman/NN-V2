import torch
import torch.nn as nn
import torchvision.ops as ops
from torchvision.models import resnet50, ResNet50_Weights
from .backbone import ResNetBackbone
from .detection_head import DetectionHead
from .utils.box_utils import coco_to_xyxy, xyxy_to_coco
from config import (
    NUM_CLASSES, FEATURE_MAP_SIZE, IMAGE_SIZE,
    CONFIDENCE_THRESHOLD, MAX_DETECTIONS, NMS_THRESHOLD
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
            # Training mode - return raw predictions for loss computation
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': self.cached_anchors
            }
        else:
            # Inference mode
            from .anchor_generator import decode_boxes
            
            # Apply sigmoid to get confidence scores
            conf_scores = torch.sigmoid(conf_pred)
            
            # Process each image in the batch
            detections = []
            for i in range(batch_size):
                # Get scores and boxes for this image
                scores = conf_scores[i]
                # Decode box predictions relative to anchors
                boxes = decode_boxes(bbox_pred[i], self.cached_anchors)
                
                # Filter by confidence threshold
                mask = scores > CONFIDENCE_THRESHOLD
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
                
                # Prepare empty tensor with proper device
                keep_indices = torch.tensor([], dtype=torch.int64, device=boxes.device)
                
                # Apply NMS if we have any detections
                if len(filtered_boxes) > 0:
                    keep_indices = ops.nms(
                        filtered_boxes,
                        filtered_scores,
                        iou_threshold=NMS_THRESHOLD
                    )
                    
                    # Limit number of detections
                    if len(keep_indices) > MAX_DETECTIONS:
                        keep_indices = keep_indices[:MAX_DETECTIONS]
                    
                    final_boxes = filtered_boxes[keep_indices]
                    final_scores = filtered_scores[keep_indices]
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