import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    TRAIN_CONFIDENCE_THRESHOLD, TRAIN_NMS_THRESHOLD,
    ANCHOR_SCALES, ANCHOR_RATIOS, FEATURE_MAP_SIZE
)
import math

class DogDetector(nn.Module):
    def __init__(self, num_anchors_per_cell=None, feature_map_size=None):
        super(DogDetector, self).__init__()
        
        # Use values from config if not provided
        self.feature_map_size = feature_map_size or FEATURE_MAP_SIZE
        
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Split backbone for multi-scale features
        self.backbone_layers = nn.ModuleList([
            nn.Sequential(*list(backbone.children())[:5]),    # C2 output
            nn.Sequential(*list(backbone.children())[5:6]),   # C3 output
            nn.Sequential(*list(backbone.children())[6:7]),   # C4 output
            nn.Sequential(*list(backbone.children())[7:8])    # C5 output
        ])
        
        # Freeze the first layers of ResNet18 to avoid overfitting
        for param in self.backbone_layers[0].parameters():
            param.requires_grad = False
        
        # Multi-scale feature fusion (FPN-like approach)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),  # C5 lateral
            nn.Conv2d(256, 256, kernel_size=1),  # C4 lateral
            nn.Conv2d(128, 256, kernel_size=1)   # C3 lateral
        ])
        
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P5 smooth
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # P4 smooth
            nn.Conv2d(256, 256, kernel_size=3, padding=1)   # P3 smooth
        ])
        
        # Use fixed-size pooling for consistent output shape
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with skip connections for better gradient flow
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Generate anchor boxes with different scales and aspect ratios
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        self.num_anchors_per_cell = num_anchors_per_cell or len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Prediction heads
        self.bbox_head = nn.Conv2d(256, self.num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(256, self.num_anchors_per_cell, kernel_size=3, padding=1)
        
        # Initialize weights using Kaiming initialization for better training dynamics
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        # Extract multi-scale features using backbone
        c2 = self.backbone_layers[0](x)
        c3 = self.backbone_layers[1](c2)
        c4 = self.backbone_layers[2](c3)
        c5 = self.backbone_layers[3](c4)
        
        # FPN-like feature fusion for multi-scale awareness
        p5 = self.lateral_convs[0](c5)
        p5_upsampled = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.lateral_convs[1](c4) + p5_upsampled
        p4 = self.smooth_convs[1](p4)
        p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[2](c3) + p4_upsampled
        p3 = self.smooth_convs[2](p3)
        
        # Use the most appropriate feature level based on target size
        features = p3
        
        # Resize to target feature map size if needed
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
        
        # Detection head with residual connection for better gradient flow
        identity = features
        x = F.relu(self.bn1(self.conv1(features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity  # Residual connection
        
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
        
        if self.training and targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors
            }
        else:
            # Process each image in the batch
            results = []
            for boxes, scores in zip(bbox_pred, conf_pred):
                # Use appropriate confidence threshold based on mode
                confidence_threshold = TRAIN_CONFIDENCE_THRESHOLD if self.training else CONFIDENCE_THRESHOLD
                
                # Filter by confidence threshold
                mask = scores > confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                
                if len(boxes) > 0:
                    # Clip boxes to image boundaries
                    boxes = torch.clamp(boxes, min=0, max=1)
                    
                    # Apply NMS with appropriate threshold
                    nms_threshold = TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD
                    keep_idx = nms(boxes, scores, nms_threshold)
                    
                    # Limit maximum detections
                    if len(keep_idx) > MAX_DETECTIONS:
                        scores_for_topk = scores[keep_idx]
                        _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                        keep_idx = keep_idx[topk_indices]
                    
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                
                # Always ensure we have at least one prediction for stability
                if len(boxes) == 0:
                    # Default box covers most of the image
                    boxes = torch.tensor([[0.2, 0.2, 0.8, 0.8]], device=bbox_pred.device)
                    scores = torch.tensor([confidence_threshold], device=bbox_pred.device)
                
                results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'anchors': default_anchors  # Include anchors in each result
                })
            
            return results
        
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