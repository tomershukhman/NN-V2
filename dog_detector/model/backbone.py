import torch.nn as nn
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from config import BACKBONE_FROZEN_LAYERS

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract different parts of the ResNet for multi-scale features
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Additional convolutional layers to transform features
        self.lateral_layer4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_layer3 = nn.Conv2d(256, 256, kernel_size=1)
        
        # Upsample and smooth layers
        # Using interpolate in forward instead of fixed upsampling
        self.smooth = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Freeze early layers for transfer learning
        layers_to_freeze = [
            list(self.layer0.parameters()),
            list(self.layer1.parameters()),
            list(self.layer2.parameters())[:BACKBONE_FROZEN_LAYERS]
        ]
        
        for layer_params in layers_to_freeze:
            for param in layer_params:
                param.requires_grad = False
                
    def forward(self, x):
        # Extract features from different layers
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN-like lateral connections
        p5 = self.lateral_layer4(c5)
        
        # Dynamically upsample p5 to match c4's spatial dimensions
        p5_upsampled = nn.functional.interpolate(
            p5, 
            size=c4.shape[2:],  # Use c4's exact spatial dimensions
            mode='nearest'
        )
        
        # Add upsampled features
        p4 = self.lateral_layer3(c4) + p5_upsampled
        
        # Final feature map with rich multi-scale information
        out = self.smooth(p4)
        
        return out