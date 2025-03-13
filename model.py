import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    DROPOUT_RATE, ANCHOR_SCALES, ANCHOR_RATIOS
)
import math

class DogDetector(nn.Module):
    def __init__(self, num_anchors_per_cell=None, feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Load pretrained ResNet18 backbone - as required
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the last two layers (avg pool and fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Store feature map size
        self.feature_map_size = feature_map_size
        
        # Unfreeze only the last layers of the backbone
        for i, param in enumerate(self.backbone.parameters()):
            if i < 30:  # Freeze more layers to avoid overfitting
                param.requires_grad = False
        
        # Enhanced FPN-like feature pyramid
        self.lateral_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.smooth_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Replace adaptive pooling with fixed pooling
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with dropout for regularization
        self.conv_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE),  # Add dropout for regularization
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE)   # Add dropout for regularization
        )
        
        # Generate anchor boxes with different scales and aspect ratios
        # Use scales and ratios from config
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        
        if num_anchors_per_cell is None:
            self.num_anchors_per_cell = len(self.anchor_scales) * len(self.anchor_ratios)
        else:
            self.num_anchors_per_cell = num_anchors_per_cell
        
        # Prediction heads with specific initialization
        self.bbox_head = nn.Conv2d(256, self.num_anchors_per_cell * 4, kernel_size=3, padding=1)
        # Separate confidence head for more stable training
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(256, self.num_anchors_per_cell, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())

    def _initialize_weights(self):
        """Initialize model weights with better schemes for different parts"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for prediction heads
        nn.init.normal_(self.bbox_head.weight, std=0.01)
        if hasattr(self.bbox_head, 'bias') and self.bbox_head.bias is not None:
            nn.init.zeros_(self.bbox_head.bias)
        
        # Initialize last layer of cls_head with slight negative bias for fewer initial predictions
        last_conv = list(self.cls_head.children())[-1]
        nn.init.normal_(last_conv.weight, std=0.01)
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, -4.0)  # Start with low confidence

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
        
        # Apply the detection head
        x = self.conv_head(features)
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(x)
        conf_pred = torch.sigmoid(self.cls_head(x))
        
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
        
        if targets is not None:
            # Return raw predictions for loss calculation regardless of training mode
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors
            }
        else:
            # Apply more careful filtering for inference
            results = []
            for boxes, scores in zip(bbox_pred, conf_pred):
                # Apply confidence threshold more strictly during inference
                mask = scores > CONFIDENCE_THRESHOLD
                boxes = boxes[mask]
                scores = scores[mask]
                
                if len(boxes) > 0:
                    # Clip boxes to image boundaries
                    boxes = torch.clamp(boxes, min=0, max=1)
                    
                    # Apply NMS with stricter threshold
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
                    'anchors': default_anchors  # Include anchors in each result
                })
            
            return results

    def _decode_boxes(self, box_pred, anchors):
        """Convert predicted box offsets back to absolute coordinates with improved handling"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions with improved scale handling
        pred_centers = box_pred[:, :, :2] * anchor_sizes + anchor_centers
        pred_sizes = torch.exp(torch.clamp(box_pred[:, :, 2:], max=4.0)) * anchor_sizes  # Clamp to avoid extreme boxes
        
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