import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    TRAIN_CONFIDENCE_THRESHOLD, TRAIN_NMS_THRESHOLD
)
import math

class DogDetector(nn.Module):
    def __init__(self, num_anchors_per_cell=9, feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights)
        
        # Remove the last two layers (avg pool and fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Store feature map size
        self.feature_map_size = feature_map_size
        
        # Freeze the first few layers of ResNet18
        for param in list(self.backbone.parameters())[:-6]:
            param.requires_grad = False
        
        # FPN-like feature pyramid with fixed pooling to ensure MPS compatibility
        self.lateral_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.smooth_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Replace adaptive pooling with fixed pooling
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Generate anchor boxes with different scales and aspect ratios
        self.anchor_scales = [0.5, 1.0, 2.0]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors_per_cell = num_anchors_per_cell
        
        # Prediction heads
        self.bbox_head = nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(256, num_anchors_per_cell, kernel_size=3, padding=1)
        
        # Count estimator - a small network that predicts the number of objects from global features
        self.count_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Predict a single value for object count
            nn.ReLU()  # Ensure positive output
        )
        
        # Initialize weights
        for m in [self.lateral_conv, self.smooth_conv, self.conv1, self.conv2, self.bbox_head, self.cls_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())

    def _generate_anchors(self):
        """Generate anchor boxes for each cell in the feature map"""
        anchors = []
        for i in range(self.feature_map_size):
            for j in range(self.feature_map_size):
                cx = (j + 0.5) / self.feature_map_size
                cy = (i + 0.5) / self.feature_map_size
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        # Convert to [x1, y1, x2, y2] format
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors, dtype=torch.float32)
        
    def forward(self, x, targets=None):
        # Extract features using backbone
        features = self.backbone(x)
        
        # FPN-like feature processing with fixed pooling
        lateral = self.lateral_conv(features)
        features = self.smooth_conv(lateral)
        
        # Apply fixed-size pooling operations until we reach desired size
        while features.shape[-1] > self.feature_map_size:
            features = self.pool(features)
            
        # If the feature map is too small, use interpolation to reach target size
        if features.shape[-1] < self.feature_map_size:
            features = F.interpolate(
                features, 
                size=(self.feature_map_size, self.feature_map_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Detection head
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(x)
        conf_pred = torch.sigmoid(self.cls_head(x))
        
        # Predict object count (only used during inference)
        estimated_count = self.count_estimator(x).squeeze(-1)
        
        # Get shapes
        batch_size = x.shape[0]
        feature_size = x.shape[2]  # Should be self.feature_map_size
        total_anchors = feature_size * feature_size * self.num_anchors_per_cell
        
        # Reshape bbox predictions to [batch, total_anchors, 4]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, total_anchors, 4)
        
        # Transform bbox predictions from offsets to actual coordinates
        default_anchors = self.default_anchors.to(bbox_pred.device)
        bbox_pred = self._decode_boxes(bbox_pred, default_anchors)
        
        # Reshape confidence predictions
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
        conf_pred = conf_pred.view(batch_size, total_anchors)
        
        if self.training and targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors
            }
        else:
            # Process each image in the batch
            results = []
            for i, (boxes, scores, count) in enumerate(zip(bbox_pred, conf_pred, estimated_count)):
                # Determine adaptive threshold based on expected count and confidence distribution
                if not self.training:
                    adaptive_threshold = self._get_adaptive_threshold(scores, count)
                    threshold = min(adaptive_threshold, CONFIDENCE_THRESHOLD)  # Don't go higher than config
                else:
                    threshold = TRAIN_CONFIDENCE_THRESHOLD
                
                # Filter by confidence threshold
                mask = scores > threshold
                boxes = boxes[mask]
                scores = scores[mask]
                
                if len(boxes) > 0:
                    # Clip boxes to image boundaries
                    boxes = torch.clamp(boxes, min=0, max=1)
                    
                    # Apply NMS with appropriate threshold
                    keep_idx = nms(boxes, scores, 
                                 TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD)
                    
                    # Limit maximum detections, but try to respect estimated count
                    if not self.training and count is not None:
                        # Aim for the estimated count, but don't exceed MAX_DETECTIONS
                        target_count = min(int(count.item() + 0.5), MAX_DETECTIONS)
                        if len(keep_idx) > target_count:
                            scores_for_topk = scores[keep_idx]
                            _, topk_indices = torch.topk(scores_for_topk, k=target_count)
                            keep_idx = keep_idx[topk_indices]
                    elif len(keep_idx) > MAX_DETECTIONS:
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
                    'anchors': default_anchors,  # Include anchors in each result
                    'estimated_count': count.item() if not self.training else None
                })
            
            return results
            
    def _get_adaptive_threshold(self, scores, count_estimate):
        """Calculate an adaptive confidence threshold to match the expected object count"""
        # Sort scores in descending order
        sorted_scores, _ = torch.sort(scores, descending=True)
        
        # Get target count from count estimator (round to nearest integer)
        target_count = max(1, min(int(count_estimate.item() + 0.5), MAX_DETECTIONS))
        
        # If we have enough scores, set threshold just below the target count's score
        if len(sorted_scores) > target_count:
            # Use the score at target_count position as threshold, but add a small buffer
            adaptive_threshold = sorted_scores[target_count - 1] * 0.95
            # Don't let the threshold go too low
            adaptive_threshold = max(adaptive_threshold, 0.15)
        else:
            # If we don't have enough detected objects, lower the threshold
            adaptive_threshold = 0.15
            
        return adaptive_threshold
        
    def _decode_boxes(self, box_pred, anchors):
        """Convert predicted box offsets back to absolute coordinates"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions
        pred_centers = box_pred[:, :, :2] * anchor_sizes + anchor_centers
        pred_sizes = torch.exp(box_pred[:, :, 2:]) * anchor_sizes
        
        # Convert back to [x1, y1, x2, y2] format
        boxes = torch.cat([
            pred_centers - pred_sizes/2,
            pred_centers + pred_sizes/2
        ], dim=-1)
        
        return boxes

def get_model(device):
    model = DogDetector()
    model = model.to(device)
    return model