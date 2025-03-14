import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from config import BACKBONE_FROZEN_LAYERS

def get_backbone():
    """Factory function to create and return a configured ResNetBackbone"""
    return ResNetBackbone()

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained ResNet18 with proper initialization
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract layers for multi-scale feature extraction
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32
        
        # FPN layers with proper initialization
        self.fpn_transforms = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),  # P5
            nn.Conv2d(256, 256, kernel_size=1),  # P4
            nn.Conv2d(128, 256, kernel_size=1),  # P3
        ])
        
        # Additional convolutions for feature refinement
        self.fpn_smooths = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        ])
        
        # Final output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_fpn()
        self._freeze_early_layers()
        
    def _initialize_fpn(self):
        """Initialize FPN layers with proper weight distribution"""
        for modules in [self.fpn_transforms, self.fpn_smooths, self.output_conv]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def _freeze_early_layers(self):
        """Freeze early layers based on config"""
        layers_to_freeze = [
            self.layer0,
            self.layer1,
            list(self.layer2.children())[:BACKBONE_FROZEN_LAYERS]
        ]
        
        for layer in layers_to_freeze:
            if isinstance(layer, (list, tuple)):
                for sublayer in layer:
                    for param in sublayer.parameters():
                        param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        # Extract features from ResNet layers
        c1 = self.layer0(x)        # 1/4
        c2 = self.layer1(c1)       # 1/4
        c3 = self.layer2(c2)       # 1/8
        c4 = self.layer3(c3)       # 1/16
        c5 = self.layer4(c4)       # 1/32
        
        # FPN top-down pathway
        p5 = self.fpn_transforms[0](c5)
        p4 = self._upsample_add(p5, self.fpn_transforms[1](c4))
        p3 = self._upsample_add(p4, self.fpn_transforms[2](c3))
        
        # Smooth layers
        p5 = self.fpn_smooths[0](p5)
        p4 = self.fpn_smooths[1](p4)
        p3 = self.fpn_smooths[2](p3)
        
        # Choose appropriate feature level based on image size
        # Here we use P4 (1/16) as it's a good balance for dog detection
        out = self.output_conv(p4)
        
        return out
    
    def _upsample_add(self, x, y):
        """Upsample and add two feature maps"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

def get_backbone():
    """Factory function to create and return a configured ResNetBackbone"""
    return ResNetBackbone()