import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the last two layers (avg pool and fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Unfreeze only the last layers of the backbone
        for i, param in enumerate(self.backbone.parameters()):
            if i < 30:  # Freeze more layers to avoid overfitting
                param.requires_grad = False
                
    def forward(self, x):
        return self.backbone(x)