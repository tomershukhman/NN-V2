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
    def __init__(self, num_anchors_per_cell=9, feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Load pretrained ResNet18 backbone
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the last two layers (avg pool and fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Store feature map size
        self.feature_map_size = feature_map_size
        
        # Freeze more layers of ResNet18 to prevent overfitting
        for param in list(self.backbone.parameters())[:-3]:  # Freeze all but last ResNet block
            param.requires_grad = False
        
        # FPN-like feature pyramid with fixed pooling to ensure MPS compatibility
        self.lateral_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.smooth_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Add batch norm and dropout to improve regularization
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with added regularization
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # Use more appropriate scales and ratios for dog detection
        self.anchor_scales = [0.4, 0.8, 1.2]  # Smaller range of scales
        self.anchor_ratios = [0.7, 1.0, 1.3]  # Less extreme ratios for dogs
        self.num_anchors_per_cell = num_anchors_per_cell
        
        # Prediction heads with proper initialization
        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors_per_cell, kernel_size=3, padding=1)
        )
        
        # Initialize weights with Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
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
        
        # Detection head
        x = self.conv1(features)
        x = self.conv2(x)
        
        # Predict bounding boxes and confidence scores
        bbox_pred = self.bbox_head(x)
        conf_pred = self.cls_head(x)
        
        # Get shapes
        batch_size = x.shape[0]
        feature_size = x.shape[2]  # Should be self.feature_map_size
        total_anchors = feature_size * feature_size * self.num_anchors_per_cell
        
        # Reshape predictions
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, total_anchors, 4)
        
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
        conf_pred = conf_pred.view(batch_size, total_anchors)
        conf_pred = torch.sigmoid(conf_pred)  # Apply sigmoid after reshaping
        
        # Transform bbox predictions from offsets to actual coordinates with normalization
        default_anchors = self.default_anchors.to(bbox_pred.device)
        bbox_pred = self._decode_boxes(bbox_pred, default_anchors)
        
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
                # Use appropriate threshold based on actual training state
                threshold = TRAIN_CONFIDENCE_THRESHOLD if targets is not None else CONFIDENCE_THRESHOLD
                mask = scores > threshold
                boxes = boxes[mask]
                scores = scores[mask]
                
                if len(boxes) > 0:
                    # Clip boxes to image boundaries
                    boxes = torch.clamp(boxes, min=0, max=1)
                    
                    # Apply NMS
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
                    'anchors': default_anchors
                })
            
            return results

    def _decode_boxes(self, box_pred, anchors):
        """Convert predicted box offsets back to absolute coordinates with normalization"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Scale factors for more stable gradients
        scale_factor = torch.tensor([0.1, 0.1, 0.2, 0.2], device=box_pred.device)
        box_pred = box_pred * scale_factor
        
        # Decode predictions with gradient-friendly computations
        pred_centers = box_pred[:, :, :2].tanh() * 0.5 + anchor_centers
        pred_sizes = torch.exp(box_pred[:, :, 2:].clamp(-4, 4)) * anchor_sizes
        
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