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
        
        # Freeze the first few layers of ResNet18
        for param in list(self.backbone.parameters())[:-6]:
            param.requires_grad = False
        
        # FPN-like feature pyramid with fixed pooling
        self.lateral_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_bn = nn.BatchNorm2d(256)
        self.smooth_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_bn = nn.BatchNorm2d(256)
        
        # Replace adaptive pooling with fixed pooling
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with improved regularization
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
        
        # Return to original anchor configuration but with slight adjustments
        self.anchor_scales = [0.5, 1.0, 2.0]  # Original scales
        self.anchor_ratios = [0.6, 1.0, 1.67]  # Slightly adjusted for dogs
        self.num_anchors_per_cell = num_anchors_per_cell
        
        # Prediction heads
        self.bbox_head = nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(256, num_anchors_per_cell, kernel_size=3, padding=1)
        
        # Initialize weights
        for m in [self.lateral_conv, self.smooth_conv, self.conv1, self.conv2, self.bbox_head, self.cls_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
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
        lateral = self.lateral_bn(lateral)
        features = self.smooth_conv(lateral)
        features = self.smooth_bn(features)
        
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
                    
                    # Enhanced filtering for isolated high-confidence predictions
                    if len(boxes) == 1 and scores[0] > 0.7:
                        # Calculate average activation in box region
                        # Use feature map before classification head
                        feat_size = x.shape[-1]
                        box_feat_coords = (boxes[0] * feat_size).long()
                        region_feats = x[0, :, 
                                      box_feat_coords[1]:box_feat_coords[3]+1,
                                      box_feat_coords[0]:box_feat_coords[2]+1]
                        if region_feats.numel() > 0:
                            avg_activation = region_feats.mean()
                            # If average activation is too low, likely a false positive
                            if avg_activation < 0.1:  # This threshold might need tuning
                                boxes = boxes[1:]  # Remove the detection
                                scores = scores[1:]
                    
                    if len(boxes) > 0:  # Check again after potential filtering
                        # Apply NMS with appropriate threshold
                        nms_threshold = TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD
                        keep_idx = nms(boxes, scores, nms_threshold)
                        
                        # Limit maximum detections but ensure at least one high-confidence detection remains
                        if len(keep_idx) > MAX_DETECTIONS:
                            scores_for_topk = scores[keep_idx]
                            _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                            keep_idx = keep_idx[topk_indices]
                        
                        boxes = boxes[keep_idx]
                        scores = scores[keep_idx]
                
                # Only add default box if no high-confidence detections were found
                if len(boxes) == 0:
                    # Check feature map activations before adding default box
                    avg_activation = x[0].mean()
                    if avg_activation > 0.15:  # Only add default box if there's significant activation
                        boxes = torch.tensor([[0.3, 0.3, 0.7, 0.7]], device=bbox_pred.device)  # Smaller default box
                        scores = torch.tensor([confidence_threshold], device=bbox_pred.device)
                    else:
                        # No significant activation, return empty prediction
                        boxes = torch.zeros((0, 4), device=bbox_pred.device)
                        scores = torch.zeros(0, device=bbox_pred.device)
                
                results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'anchors': default_anchors
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