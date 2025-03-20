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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention mask
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention_mask = self.sigmoid(attention)
        
        # Apply attention mask
        return x * attention_mask

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.smooth_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.attention = SpatialAttention()
        
        # BatchNorm layers to stabilize training
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        lateral = self.lateral_conv(x)
        features = F.relu(self.bn1(self.smooth_conv1(lateral)))
        features = self.attention(features)  # Apply spatial attention
        features = F.relu(self.bn2(self.smooth_conv2(features)))
        return features

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
        for param in list(self.backbone.parameters())[:-8]:  # Freeze fewer layers for better feature extraction
            param.requires_grad = False
        
        # Enhanced Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(512, 256)
        
        # Replace adaptive pooling with fixed pooling
        self.pool = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Detection head with improved capacity
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Generate anchor boxes with more varied scales and aspect ratios
        # Adding more varied scales improves detection of different sized dogs
        self.anchor_scales = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # More varied scales
        self.anchor_ratios = [0.5, 0.75, 1.0, 1.33, 2.0]  # More varied aspect ratios
        self.num_anchors_per_cell = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Prediction heads
        self.bbox_head = nn.Conv2d(256, self.num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(256, self.num_anchors_per_cell, kernel_size=3, padding=1)
        
        # Initialize weights with improved method
        for m in [self.fpn.lateral_conv, self.fpn.smooth_conv1, self.fpn.smooth_conv2, 
                 self.conv1, self.conv2, self.bbox_head, self.cls_head]:
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for better convergence
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())

    def _generate_anchors(self):
        """Generate anchor boxes for each cell in the feature map with improved coverage"""
        anchors = []
        for i in range(self.feature_map_size):
            for j in range(self.feature_map_size):
                cx = (j + 0.5) / self.feature_map_size
                cy = (i + 0.5) / self.feature_map_size
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        # Clip widths and heights to prevent extreme values
                        w = min(w, 0.95)
                        h = min(h, 0.95)
                        # Convert to [x1, y1, x2, y2] format
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        # Ensure all anchors are valid
                        x1 = max(0, min(x1, 1.0 - 0.01))
                        y1 = max(0, min(y1, 1.0 - 0.01))
                        x2 = max(x1 + 0.01, min(x2, 1.0))
                        y2 = max(y1 + 0.01, min(y2, 1.0))
                        anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors, dtype=torch.float32)
        
    def forward(self, x, targets=None):
        # Extract features using backbone
        features = self.backbone(x)
        
        # FPN-like feature processing with fixed pooling
        features = self.fpn(features)
        
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
        
        # Detection head with improved feature processing
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        
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
                    
                    # Apply Soft-NMS instead of standard NMS for better overlapping dog detection
                    keep_indices = self._soft_nms(boxes, scores, 
                                                 nms_threshold=TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD,
                                                 method='gaussian')
                    
                    # Limit maximum detections
                    if len(keep_indices) > MAX_DETECTIONS:
                        scores_for_topk = scores[keep_indices]
                        _, topk_indices = torch.topk(scores_for_topk, k=MAX_DETECTIONS)
                        keep_indices = keep_indices[topk_indices]
                    
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]
                
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
    
    def _soft_nms(self, boxes, scores, nms_threshold=0.3, sigma=0.5, method='gaussian'):
        """
        Soft-NMS implementation to better handle overlapping dogs
        Returns indices of kept boxes according to input method.
        """
        # Convert to CPU for processing
        device = boxes.device
        boxes_cpu = boxes.detach().cpu()
        scores_cpu = scores.detach().cpu()
        
        N = len(boxes_cpu)
        
        if N == 0:
            return torch.tensor([], dtype=torch.long, device=device)
        
        indices = torch.arange(N)
        
        # Sort scores in descending order
        _, order = torch.sort(scores_cpu, descending=True)
        
        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            idx = order[0].item()
            keep.append(idx)
            
            # If only 1 box left, break
            if order.numel() == 1:
                break
            
            # Get IoU between the selected box and the rest
            order = order[1:]
            iou = self._compute_iou(boxes_cpu[idx], boxes_cpu[order])
            
            # Apply soft-NMS weight decay
            if method == 'gaussian':
                weight = torch.exp(-(iou * iou) / sigma)
            else:  # Linear penalty
                weight = 1 - iou
                weight = torch.clamp(weight, min=0.0)
            
            # Update scores
            scores_cpu[order] *= weight
            
            # Remove boxes below threshold
            remaining = torch.where(scores_cpu[order] >= nms_threshold)[0]
            order = order[remaining]
        
        return torch.tensor(keep, dtype=torch.long, device=device)
    
    def _compute_iou(self, box, boxes):
        """Compute IoU between a box and a set of boxes"""
        # Calculate intersection areas
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])
        
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        intersection = w * h
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        return intersection / torch.clamp(union, min=1e-6)
        
    def _decode_boxes(self, box_pred, anchors):
        """Convert predicted box offsets back to absolute coordinates with improved decoding"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions with scaled offsets for better stability
        # Use scale factors to control the influence of predictions
        center_scale = 0.1
        size_scale = 0.2
        
        # Adjust centers with scaled offsets
        pred_centers = box_pred[:, :, :2] * center_scale * anchor_sizes + anchor_centers
        
        # Decode sizes with exponential to ensure positive values
        # Use tanh to limit extreme size changes
        pred_sizes = torch.exp(torch.clamp(box_pred[:, :, 2:] * size_scale, min=-4, max=4)) * anchor_sizes
        
        # Convert back to [x1, y1, x2, y2] format
        boxes = torch.cat([
            pred_centers - pred_sizes/2,
            pred_centers + pred_sizes/2
        ], dim=-1)
        
        # Ensure valid boxes (no negative widths/heights)
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]
        
        # Force correct ordering and minimum size
        min_size = 0.001
        x1, x2 = torch.min(x1, x2 - min_size), torch.max(x2, x1 + min_size)
        y1, y2 = torch.min(y1, y2 - min_size), torch.max(y2, y1 + min_size)
        
        valid_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return valid_boxes

def get_model(device):
    model = DogDetector()
    model = model.to(device)
    return model