import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    TRAIN_CONFIDENCE_THRESHOLD, TRAIN_NMS_THRESHOLD, ANCHOR_SCALES, ANCHOR_RATIOS,
    NUM_CLASSES, CLASS_CONFIDENCE_THRESHOLDS, CLASS_NMS_THRESHOLDS, CLASS_MAX_DETECTIONS,
    CLASS_NAMES
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
        for param in list(self.backbone_conv1.parameters()) + list(self.bn1.parameters()) + list(self.layer1.parameters()) + list(self.layer2.parameters())[:-4]:
            param.requires_grad = False
        
        # Enhanced FPN with multi-scale features
        self.fpn_convs = nn.ModuleDict({
            'p5': nn.Conv2d(512, 256, kernel_size=1),
            'p4': nn.Conv2d(256, 256, kernel_size=1),
            'p3': nn.Conv2d(128, 256, kernel_size=1)
        })
        
        self.fpn_bns = nn.ModuleDict({
            'p5': nn.BatchNorm2d(256),
            'p4': nn.BatchNorm2d(256),
            'p3': nn.BatchNorm2d(256)
        })
        
        # Smooth convs for feature refinement
        self.smooth_convs = nn.ModuleDict({
            'p5': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'p4': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'p3': nn.Conv2d(256, 256, kernel_size=3, padding=1)
        })
        
        self.smooth_bns = nn.ModuleDict({
            'p5': nn.BatchNorm2d(256),
            'p4': nn.BatchNorm2d(256),
            'p3': nn.BatchNorm2d(256)
        })

        # Improved detection head with attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
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
        
        # Expanded anchor configuration from config.py
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        self.num_anchors_per_cell = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Enhanced prediction heads with better initialization for multi-class
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_anchors_per_cell * NUM_CLASSES, kernel_size=3, padding=1),
        )
        
        self.bbox_head = nn.Conv2d(256, self.num_anchors_per_cell * 4, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Generate and register anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())
        
        # Improved confidence calibration parameters for multi-object detection
        self.register_parameter('conf_scaling', nn.Parameter(torch.ones(1) * 1.3))  # Slightly reduced from 1.5
        self.register_parameter('conf_bias', nn.Parameter(torch.zeros(1) - 0.3))    # Reduced negative bias
        
        # Add separate calibration for multi-object scenarios
        self.register_parameter('multi_obj_conf_boost', nn.Parameter(torch.ones(1) * 0.2))  # Boost for multiple objects

        # Add separate calibration for each class
        self.register_parameter('class_conf_scaling', 
            nn.Parameter(torch.ones(NUM_CLASSES) * 1.3))
        self.register_parameter('class_conf_bias',
            nn.Parameter(torch.zeros(NUM_CLASSES) - 0.3))
        self.register_parameter('class_multi_obj_conf_boost',
            nn.Parameter(torch.ones(NUM_CLASSES) * 0.2))

    def _initialize_weights(self):
        for m in [self.fpn_convs, self.smooth_convs, self.det_conv1, self.det_conv2, self.bbox_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
        
        for m in self.cls_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
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
        x = self.layer1(x)
        c3 = self.layer2(x)      # 128 channels
        c4 = self.layer3(c3)     # 256 channels
        c5 = self.layer4(c4)     # 512 channels
        
        # FPN-like feature processing
        p5 = self.fpn_convs['p5'](c5)
        p5 = self.fpn_bns['p5'](p5)
        p5 = self.smooth_convs['p5'](p5)
        p5 = self.smooth_bns['p5'](p5)

        # Upsample p5 to match p4's size
        p5_upsampled = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        
        p4 = self.fpn_convs['p4'](c4)
        p4 = self.fpn_bns['p4'](p4)
        p4 = self.smooth_convs['p4'](p4)
        p4 = self.smooth_bns['p4'](p4)
        p4 = p4 + p5_upsampled

        # Upsample combined p4 to match p3's size
        p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        
        p3 = self.fpn_convs['p3'](c3)
        p3 = self.fpn_bns['p3'](p3)
        p3 = self.smooth_convs['p3'](p3)
        p3 = self.smooth_bns['p3'](p3)
        p3 = p3 + p4_upsampled

        # Use p3 as our final features
        features = p3
        
        # Apply attention mechanism
        attention = self.attention(features)
        features = features * attention
        
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
        
        # Get batch size from features tensor
        batch_size = features.shape[0]
        
        # Detection head processing
        features = self.det_conv1(features)
        features = self.det_conv2(features)
        
        # Predict bounding boxes and class scores
        bbox_pred = self.bbox_head(features)
        conf_pred = self.cls_head(features)
        
        # Reshape predictions
        bbox_pred = bbox_pred.view(batch_size, -1, 4)
        conf_pred = conf_pred.view(batch_size, -1, NUM_CLASSES)
        
        # Transform bbox predictions from offsets to actual coordinates
        default_anchors = self.default_anchors.to(bbox_pred.device)
        bbox_pred = self._decode_boxes(bbox_pred, default_anchors)
        
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
            all_boxes = []
            all_scores = []
            all_labels = []
            
            # Process each class separately
            for class_idx in range(1, NUM_CLASSES):  # Skip background class (0)
                class_name = CLASS_NAMES[class_idx]
                class_scores = scores[:, class_idx]
                
                # Filter by confidence threshold
                confidence_threshold = CLASS_CONFIDENCE_THRESHOLDS[class_name]
                mask = class_scores > confidence_threshold
                if mask.any():
                    class_boxes = boxes[mask]
                    class_scores = class_scores[mask]
                    
                    # Apply NMS per class
                    nms_threshold = CLASS_NMS_THRESHOLDS[class_name]
                    keep_idx = nms(class_boxes, class_scores, nms_threshold)
                    
                    # Keep only top-k detections for this class
                    max_dets = CLASS_MAX_DETECTIONS[class_name]
                    if len(keep_idx) > max_dets:
                        _, top_k = torch.topk(class_scores[keep_idx], k=max_dets)
                        keep_idx = keep_idx[top_k]
                    
                    # Add kept detections to results
                    all_boxes.append(class_boxes[keep_idx])
                    all_scores.append(class_scores[keep_idx])
                    all_labels.append(torch.full((len(keep_idx),), class_idx, 
                                              device=boxes.device))
            
            # Combine all class predictions
            if all_boxes:
                final_boxes = torch.cat(all_boxes)
                final_scores = torch.cat(all_scores)
                final_labels = torch.cat(all_labels)
                
                # Sort all detections by confidence
                sorted_idx = torch.argsort(final_scores, descending=True)
                final_boxes = final_boxes[sorted_idx]
                final_scores = final_scores[sorted_idx]
                final_labels = final_labels[sorted_idx]
            else:
                # Return empty tensors if no detections
                final_boxes = torch.zeros((0, 4), device=boxes.device)
                final_scores = torch.zeros((0,), device=boxes.device)
                final_labels = torch.zeros((0,), device=boxes.device, dtype=torch.long)
            
            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels,
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