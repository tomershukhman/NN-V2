import torch
import torch.nn as nn
from config import (
    DROPOUT_RATE, DETECTION_HEAD_CHANNELS,
    LATERAL_CHANNELS, CONF_BIAS_INIT
)

class DetectionHead(nn.Module):
    def __init__(self, num_anchors_per_cell):
        super().__init__()
        self.num_anchors_per_cell = num_anchors_per_cell
        
        # Feature processing with FPN-like structure
        self.lateral_conv = nn.Conv2d(512, LATERAL_CHANNELS, kernel_size=1)
        self.smooth_conv = nn.Conv2d(LATERAL_CHANNELS, LATERAL_CHANNELS, kernel_size=3, padding=1)
        
        # Fixed pooling layers
        self.pool = nn.Sequential(
            nn.Conv2d(LATERAL_CHANNELS, LATERAL_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(LATERAL_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with dropout for regularization
        self.conv_head = nn.Sequential(
            nn.Conv2d(LATERAL_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE),
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE)
        )
        
        # Prediction heads
        self.bbox_head = nn.Conv2d(DETECTION_HEAD_CHANNELS, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(DETECTION_HEAD_CHANNELS, num_anchors_per_cell, kernel_size=3, padding=1)
        )
        
        self._initialize_weights()
        
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
        
        # Initialize last layer of cls_head with configurable bias for confidence predictions
        last_conv = list(self.cls_head.children())[-1]
        nn.init.normal_(last_conv.weight, std=0.01)
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, CONF_BIAS_INIT)
            
    def forward(self, features, target_size):
        # FPN-like feature processing
        lateral = self.lateral_conv(features)
        features = self.smooth_conv(lateral)
        
        # Apply fixed-size pooling operations until we reach desired size
        while features.shape[-1] > target_size:
            features = self.pool(features)
            
        # If the feature map is too small, use interpolation
        if features.shape[-1] < target_size:
            features = nn.functional.interpolate(
                features, 
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Apply detection head
        x = self.conv_head(features)
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(x)
        conf_pred = torch.sigmoid(self.cls_head(x))
        
        # Reshape predictions
        batch_size = x.shape[0]
        total_anchors = target_size * target_size * self.num_anchors_per_cell
        
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, total_anchors, 4)
        
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
        conf_pred = conf_pred.view(batch_size, total_anchors)
        
        return bbox_pred, conf_pred