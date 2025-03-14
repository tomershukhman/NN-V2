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
        
        # Create anchor generator
        self.anchor_generator = AnchorGenerator()
        
        # Feature processing with FPN-like structure
        self.lateral_conv = nn.Conv2d(512, LATERAL_CHANNELS, kernel_size=1)
        self.smooth_conv = nn.Conv2d(LATERAL_CHANNELS, LATERAL_CHANNELS, kernel_size=3, padding=1)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(DROPOUT_RATE)
        
        # Detection head with residual connections
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(LATERAL_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DETECTION_HEAD_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Separate prediction heads
        self.bbox_head = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS // 2, NUM_ANCHORS_PER_CELL * 4, kernel_size=3, padding=1)
        )
        
        self.conf_head = nn.Sequential(
            nn.Conv2d(DETECTION_HEAD_CHANNELS, DETECTION_HEAD_CHANNELS // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(DETECTION_HEAD_CHANNELS // 2, NUM_ANCHORS_PER_CELL, kernel_size=3, padding=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize feature processing layers with kaiming initialization
        for m in [self.lateral_conv, self.smooth_conv, self.conv_block1, self.conv_block2, self.conv_block3]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize bbox prediction head
        for m in self.bbox_head.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.bbox_head[-1]:  # Final bbox layer
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize confidence prediction head
        for m in self.conf_head.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.conf_head[-1]:  # Final confidence layer
                    # Initialize to predict low confidence initially
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    # Set bias to favor low initial confidence predictions
                    if m.bias is not None:
                        # Use a lower initial bias than before (-2.944 -> -4.595)
                        # This corresponds to initial confidence of ~0.01
                        nn.init.constant_(m.bias, -4.595)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, features):
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take last feature map if multiple levels
        
        # Generate anchors
        anchors = self.anchor_generator(features)
        
        # Process features through the network with residual connections
        x = self.lateral_conv(features)
        x = self.smooth_conv(x)
        
        # Apply residual blocks with dropout
        identity = x
        x = self.conv_block1(x)
        x = self.dropout(x)
        x = x + identity
        
        identity = x
        x = self.conv_block2(x)
        x = self.dropout(x)
        x = x + identity
        
        identity = x
        x = self.conv_block3(x)
        x = self.dropout(x)
        x = x + identity
        
        # Get predictions
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
