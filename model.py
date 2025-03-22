import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    ANCHOR_SCALES, ANCHOR_RATIOS,
    NUM_CLASSES, CLASS_CONFIDENCE_THRESHOLDS, CLASS_NMS_THRESHOLDS, CLASS_MAX_DETECTIONS,
    CLASS_NAMES
)
import math

class ObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use ResNet18 backbone but stop at layer3 for higher resolution features
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-3])  # Up to layer3
        
        # Simple detection head with 3x3 convs
        self.det_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Separate heads for classification and bounding box regression
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.cls_head = nn.Conv2d(256, self.num_anchors * NUM_CLASSES, kernel_size=3, padding=1)
        self.bbox_head = nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, padding=1)
        
        # Generate and register default anchors
        self.register_buffer('default_anchors', self._generate_anchors())
        
    def forward(self, x, targets=None):
        # Extract features
        features = self.backbone(x)
        features = self.det_head(features)
        
        # Get batch size and feature dimensions
        batch_size = x.shape[0]
        
        # Predict class scores and bounding box offsets
        class_pred = self.cls_head(features)
        bbox_pred = self.bbox_head(features)
        
        # Reshape predictions
        class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
        class_pred = class_pred.view(batch_size, -1, NUM_CLASSES)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, -1, 4)
        
        # Convert bbox predictions to absolute coordinates
        bbox_pred = self._decode_boxes(bbox_pred)
        
        if self.training:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': class_pred,
                'anchors': self.default_anchors
            }
        
        # Process predictions for inference
        results = []
        for boxes, scores in zip(bbox_pred, class_pred):
            result = self._process_single_image_predictions(boxes, scores)
            results.append(result)
        
        return results
    
    def _process_single_image_predictions(self, boxes, scores):
        """Process predictions for a single image"""
        final_boxes = []
        final_scores = []
        final_labels = []
        
        # Process each class separately (skip background)
        for class_idx in range(1, NUM_CLASSES):
            class_scores = scores[:, class_idx]
            
            # Filter by confidence
            mask = class_scores > CLASS_CONFIDENCE_THRESHOLDS[CLASS_NAMES[class_idx]]
            if not mask.any():
                continue
            
            class_boxes = boxes[mask]
            class_scores = class_scores[mask]
            
            # Apply NMS
            keep_idx = nms(
                class_boxes,
                class_scores,
                CLASS_NMS_THRESHOLDS[CLASS_NAMES[class_idx]]
            )
            
            # Keep only top detections
            max_dets = CLASS_MAX_DETECTIONS[CLASS_NAMES[class_idx]]
            if len(keep_idx) > max_dets:
                keep_idx = keep_idx[:max_dets]
            
            final_boxes.append(class_boxes[keep_idx])
            final_scores.append(class_scores[keep_idx])
            final_labels.append(torch.full((len(keep_idx),), class_idx, device=boxes.device))
        
        # Combine all detections
        if final_boxes:
            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)
        else:
            final_boxes = torch.zeros((0, 4), device=boxes.device)
            final_scores = torch.zeros((0,), device=boxes.device)
            final_labels = torch.zeros((0,), device=boxes.device, dtype=torch.long)
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }
    
    def _generate_anchors(self):
        """Generate anchor boxes for feature map"""
        anchors = []
        for i in range(20):  # 20x20 feature map
            for j in range(20):
                cx = (j + 0.5) / 20
                cy = (i + 0.5) / 20
                for scale in ANCHOR_SCALES:
                    for ratio in ANCHOR_RATIOS:
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        # Convert to [x1, y1, x2, y2] format
                        anchors.append([
                            cx - w/2, cy - h/2,
                            cx + w/2, cy + h/2
                        ])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def _decode_boxes(self, box_pred):
        """Convert predicted offsets to absolute coordinates"""
        anchors = self.default_anchors.to(box_pred.device)
        
        # Extract center coordinates and dimensions from anchors
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions
        pred_centers = box_pred[..., :2] * anchor_sizes + anchor_centers
        pred_sizes = torch.exp(box_pred[..., 2:]) * anchor_sizes
        
        # Convert to [x1, y1, x2, y2] format
        boxes = torch.cat([
            pred_centers - pred_sizes/2,
            pred_centers + pred_sizes/2
        ], dim=-1)
        
        # Clip to image bounds
        boxes = boxes.clamp(0, 1)
        
        return boxes

def get_model(device):
    model = ObjectDetector()
    return model.to(device)