import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    TRAIN_CONFIDENCE_THRESHOLD, TRAIN_NMS_THRESHOLD,
    ANCHOR_SCALES, ANCHOR_RATIOS
)
import math

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scale_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.scale_attention(x)
        return x * attention

class DogDetector(nn.Module):
    def __init__(self, num_anchors_per_cell=len(ANCHOR_SCALES) * len(ANCHOR_RATIOS), feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Initialize anchor parameters first
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        self.num_anchors_per_cell = num_anchors_per_cell
        self.feature_map_size = feature_map_size
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())
        
        # Load pretrained ResNet18 backbone with improved feature extraction
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Freeze early layers but allow more layers to train
        for param in list(self.backbone.parameters())[:-8]:
            param.requires_grad = False
        
        # Enhanced feature pyramid with attention
        self.lateral_conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral_bn1 = nn.BatchNorm2d(256)
        self.lateral_bn2 = nn.BatchNorm2d(256)
        
        self.smooth_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_bn1 = nn.BatchNorm2d(256)
        self.smooth_bn2 = nn.BatchNorm2d(256)
        
        # Multi-scale attention modules
        self.attention1 = MultiScaleAttention(256)
        self.attention2 = MultiScaleAttention(256)
        
        # Enhanced pooling with residual connections
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Improved detection heads with better feature utilization
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Enhanced confidence prediction head
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors_per_cell, kernel_size=3, padding=1)
        )
        
        # Refined bbox prediction head
        self.bbox_head = nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        
        # Initialize weights with better scaling
        for m in [self.lateral_conv1, self.lateral_conv2, self.smooth_conv1, self.smooth_conv2,
                 self.conv1, self.conv2, self.bbox_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for confidence head
        for m in self.cls_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -math.log((1 - 0.01) / 0.01))
        
        # Add confidence calibration parameters
        self.register_parameter('conf_scaling', nn.Parameter(torch.ones(1)))
        self.register_parameter('conf_bias', nn.Parameter(torch.zeros(1)))

    def _generate_anchors(self):
        """Generate anchor boxes for each cell in the feature map with improved coverage"""
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
        """Enhanced forward pass with improved feature pyramid and attention"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Enhanced feature pyramid with attention
        lateral1 = self.lateral_conv1(features)
        lateral1 = self.lateral_bn1(lateral1)
        lateral1 = self.attention1(lateral1)
        
        features = self.smooth_conv1(lateral1)
        features = self.smooth_bn1(features)
        
        lateral2 = self.lateral_conv2(features)
        lateral2 = self.lateral_bn2(lateral2)
        lateral2 = self.attention2(lateral2)
        
        features = self.smooth_conv2(lateral2)
        features = self.smooth_bn2(features)
        
        # Adaptive feature map sizing
        while features.shape[-1] > self.feature_map_size:
            features = self.pool(features)
        
        if features.shape[-1] < self.feature_map_size:
            features = F.interpolate(
                features, 
                size=(self.feature_map_size, self.feature_map_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Enhanced detection heads
        x = self.conv1(features)
        x = self.conv2(x)
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(x)
        conf_pred = self.cls_head(x)
        
        # Reshape predictions
        batch_size = x.shape[0]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, -1, 4)
        
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
        conf_pred = conf_pred.view(batch_size, -1)
        
        # Apply confidence calibration
        conf_pred = conf_pred * self.conf_scaling + self.conf_bias
        conf_pred = torch.sigmoid(conf_pred)
        
        if self.training and targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': self.default_anchors
            }
        
        # Process each image in the batch
        results = []
        for boxes, scores in zip(bbox_pred, conf_pred):
            confidence_threshold = TRAIN_CONFIDENCE_THRESHOLD if self.training else CONFIDENCE_THRESHOLD
            
            # Filter by confidence threshold
            mask = scores > confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) > 0:
                # Ensure valid boxes
                boxes = torch.clamp(boxes, min=0, max=1)
                
                # Apply NMS with dynamic thresholding
                nms_threshold = TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD
                keep_idx = nms(boxes, scores, nms_threshold)
                
                # Handle multiple detections
                if len(keep_idx) > MAX_DETECTIONS:
                    scores_for_topk = scores[keep_idx]
                    _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                    keep_idx = keep_idx[topk_indices]
                
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
            
            # Ensure at least one prediction with better default box
            if len(boxes) == 0:
                boxes = torch.tensor([[0.3, 0.3, 0.7, 0.7]], device=bbox_pred.device)
                scores = torch.tensor([confidence_threshold], device=bbox_pred.device)
            
            results.append({
                'boxes': boxes,
                'scores': scores
            })
        
        return results

def get_model(device):
    model = DogDetector()
    model = model.to(device)
    return model