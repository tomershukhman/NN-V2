import torch
import torch.nn as nn
from config import (
    DROPOUT_RATE, DETECTION_HEAD_CHANNELS,
    LATERAL_CHANNELS, CONF_BIAS_INIT, NUM_ANCHORS_PER_CELL
)
from .anchor_generator import AnchorGenerator


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature pyramid network layers with better initialization
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(512, LATERAL_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(LATERAL_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Deeper feature extraction with skip connections
        self.feature_block1 = nn.Sequential(
            nn.Conv2d(LATERAL_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Create anchor generator
        self.anchor_generator = AnchorGenerator()
        
        # Separate prediction heads with better initialization and deeper architecture
        self.bbox_head = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS//2, NUM_ANCHORS_PER_CELL * 4, kernel_size=3, padding=1)
        )
        
        self.conf_head = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS//2, NUM_ANCHORS_PER_CELL, kernel_size=3, padding=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with improved schemes for early training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.conf_head[-1]:
                    # Initialize final confidence layer with slight positive bias
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    # Initialize bias to make initial predictions close to 0.5
                    nn.init.constant_(m.bias, 0.0)
                elif m == self.bbox_head[-1]:
                    # Initialize final bbox layer with small weights but larger than before
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
                else:
                    # Better initialization for feature layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch norm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """Forward pass with skip connections"""
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Generate anchors
        anchors = self.anchor_generator(features)
        
        # Feature extraction with skip connections
        x = self.fpn_conv(features)
        identity = x
        
        x = self.feature_block1(x)
        x = x + identity  # Skip connection
        
        # Prediction heads
        bbox_output = self.bbox_head(x)
        conf_output = self.conf_head(x)
        
        # Reshape outputs
        batch_size = bbox_output.size(0)
        bbox_output = bbox_output.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        conf_output = conf_output.permute(0, 2, 3, 1).reshape(batch_size, -1)
        
        return {
            'bbox_pred': bbox_output,
            'conf_pred': conf_output,
            'anchors': anchors
        }
