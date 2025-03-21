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
    def __init__(self, num_anchors_per_cell=12, feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Store initial convolution layers
        self.backbone_conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        # Add adaptive pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Store intermediate layers for FPN
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # Store feature map size and input size
        self.feature_map_size = feature_map_size
        self.input_size = (224, 224)  # Standard input size
        
        # Freeze only the first few layers for better feature extraction
        for param in list(self.backbone_conv1.parameters()) + list(self.bn1.parameters()) + list(self.layer1.parameters()):
            param.requires_grad = False
        
        # Enhanced FPN with multi-scale features
        self.fpn_convs = nn.ModuleDict({
            'p5': nn.Conv2d(512, 256, kernel_size=1),
            'p4': nn.Conv2d(256, 256, kernel_size=1),
            'p3': nn.Conv2d(128, 256, kernel_size=1),
            'p2': nn.Conv2d(64, 256, kernel_size=1)  # Add p2 for finer details
        })
        
        self.fpn_bns = nn.ModuleDict({
            'p5': nn.BatchNorm2d(256),
            'p4': nn.BatchNorm2d(256),
            'p3': nn.BatchNorm2d(256),
            'p2': nn.BatchNorm2d(256)  # Add corresponding BN
        })
        
        # Smooth convs for feature refinement
        self.smooth_convs = nn.ModuleDict({
            'p5': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'p4': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'p3': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'p2': nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Add p2 smooth conv
        })
        
        self.smooth_bns = nn.ModuleDict({
            'p5': nn.BatchNorm2d(256),
            'p4': nn.BatchNorm2d(256),
            'p3': nn.BatchNorm2d(256),
            'p2': nn.BatchNorm2d(256)  # Add corresponding BN
        })

        # Improved detection head with better attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced CBAM-inspired attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Detection head with improved regularization
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)  # Increased dropout
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)  # Increased dropout
        )
        
        # Expanded anchor configuration for better coverage of different dog sizes and poses
        self.anchor_scales = [0.25, 0.4, 0.6, 0.8, 1.2]  # Better scaled steps for diverse sizes
        self.anchor_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]  # Enhanced ratios for different poses
        self.num_anchors_per_cell = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Enhanced prediction heads with better initialization
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_anchors_per_cell, kernel_size=3, padding=1),
        )
        
        self.bbox_head = nn.Conv2d(256, self.num_anchors_per_cell * 4, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())
        
        # Add confidence calibration parameters with better initialization
        self.register_parameter('conf_scaling', nn.Parameter(torch.ones(1) * 1.7))  # Higher initial scaling for confidence
        self.register_parameter('conf_bias', nn.Parameter(torch.zeros(1) - 0.3))  # Less aggressive bias to avoid missing dogs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        # Initial convolution layers
        x = self.backbone_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features using backbone layers
        c2 = self.layer1(x)      # 64 channels
        c3 = self.layer2(c2)     # 128 channels
        c4 = self.layer3(c3)     # 256 channels
        c5 = self.layer4(c4)     # 512 channels
        
        # FPN-like feature processing with improved lateral connections
        p5 = self.fpn_convs['p5'](c5)
        p5 = self.fpn_bns['p5'](p5)
        p5 = self.relu(p5)
        p5 = self.smooth_convs['p5'](p5)
        p5 = self.smooth_bns['p5'](p5)
        p5 = self.relu(p5)

        # Upsample p5 to match p4's size with improved interpolation
        p5_upsampled = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=True)
        
        p4 = self.fpn_convs['p4'](c4)
        p4 = self.fpn_bns['p4'](p4)
        p4 = self.relu(p4)
        p4 = p4 + p5_upsampled
        p4 = self.smooth_convs['p4'](p4)
        p4 = self.smooth_bns['p4'](p4)
        p4 = self.relu(p4)

        # Upsample combined p4 to match p3's size with improved interpolation
        p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=True)
        
        p3 = self.fpn_convs['p3'](c3)
        p3 = self.fpn_bns['p3'](p3)
        p3 = self.relu(p3)
        p3 = p3 + p4_upsampled
        p3 = self.smooth_convs['p3'](p3)
        p3 = self.smooth_bns['p3'](p3)
        p3 = self.relu(p3)
        
        # Add p2 pathway for finer details
        p3_upsampled = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=True)
        
        p2 = self.fpn_convs['p2'](c2)
        p2 = self.fpn_bns['p2'](p2)
        p2 = self.relu(p2)
        p2 = p2 + p3_upsampled
        p2 = self.smooth_convs['p2'](p2)
        p2 = self.smooth_bns['p2'](p2)
        p2 = self.relu(p2)

        # Multi-scale feature fusion
        p3_resized = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=True)
        p4_resized = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear', align_corners=True)
        p5_resized = F.interpolate(p5, size=p2.shape[-2:], mode='bilinear', align_corners=True)
        
        # Weighted feature fusion
        features = p2 * 0.5 + p3_resized * 0.25 + p4_resized * 0.15 + p5_resized * 0.1
        
        # Apply improved attention mechanism
        spatial_attention = self.attention(features)
        channel_attention = self.channel_attention(features)
        features = features * spatial_attention * channel_attention
        
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
        
        # Detection head processing
        features = self.det_conv1(features)
        features = self.det_conv2(features)
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(features)
        conf_pred = self.cls_head(features)
        conf_pred = conf_pred * self.conf_scaling + self.conf_bias
        conf_pred = torch.sigmoid(conf_pred)
        
        # Get shapes
        batch_size = features.shape[0]
        total_anchors = self.feature_map_size * self.feature_map_size * self.num_anchors_per_cell
        
        # Reshape predictions
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, total_anchors, 4)
        
        # Transform bbox predictions from offsets to actual coordinates
        default_anchors = self.default_anchors.to(bbox_pred.device)
        bbox_pred = self._decode_boxes(bbox_pred, default_anchors)
        
        # Normalize box coordinates to be within valid range, but maintain order of coordinates
        # This uses a safe clamping approach that preserves the box structure
        x1 = torch.clamp(bbox_pred[..., 0], min=0.0, max=1.0)
        y1 = torch.clamp(bbox_pred[..., 1], min=0.0, max=1.0)
        x2 = torch.clamp(bbox_pred[..., 2], min=0.0, max=1.0)
        y2 = torch.clamp(bbox_pred[..., 3], min=0.0, max=1.0)
        
        # Ensure x2 > x1 and y2 > y1 with a small epsilon to prevent zero-area boxes
        eps = 1e-5
        x2 = torch.max(x2, x1 + eps)
        y2 = torch.max(y2, y1 + eps)
        
        # Reassemble the boxes
        bbox_pred = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Reshape confidence predictions
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
        conf_pred = conf_pred.view(batch_size, total_anchors)
        
        # Return predictions based on training/inference mode
        if self.training and targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': default_anchors
            }
        
        # Process each image in the batch for inference
        results = []
        for boxes, scores in zip(bbox_pred, conf_pred):
            # Filter by confidence
            mask = scores > (TRAIN_CONFIDENCE_THRESHOLD if self.training else CONFIDENCE_THRESHOLD)
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) > 0:
                # Apply NMS
                keep_idx = nms(
                    boxes, 
                    scores, 
                    TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD
                )
                
                # Limit detections
                if len(keep_idx) > MAX_DETECTIONS:
                    scores_for_topk = scores[keep_idx]
                    _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                    keep_idx = keep_idx[topk_indices]
                
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
            
            # Ensure at least one prediction
            if len(boxes) == 0:
                boxes = torch.tensor([[0.3, 0.3, 0.7, 0.7]], device=bbox_pred.device)
                scores = torch.tensor([CONFIDENCE_THRESHOLD], device=bbox_pred.device)
            
            results.append({
                'boxes': boxes,
                'scores': scores,
                'anchors': default_anchors
            })
        
        return results
        
    def _decode_boxes(self, box_pred, anchors):
        """Convert predicted box offsets back to absolute coordinates with improved precision"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions with better scaling for more accurate box dimensions
        pred_centers = box_pred[:, :, :2] * anchor_sizes + anchor_centers
        pred_sizes = torch.exp(torch.clamp(box_pred[:, :, 2:], -4.0, 4.0)) * anchor_sizes
        
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