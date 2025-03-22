import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms
from config import (
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    TRAIN_CONFIDENCE_THRESHOLD, TRAIN_NMS_THRESHOLD, ANCHOR_SCALES, ANCHOR_RATIOS
)
import math
import torch.jit

@torch.jit.script
def efficient_nms(boxes, scores, iou_threshold: float):
    """JIT-compiled NMS for faster inference"""
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return keep

class DogDetector(nn.Module):
    def __init__(self, num_anchors_per_cell=12, feature_map_size=7):
        super(DogDetector, self).__init__()
        
        # Load pretrained ResNet18 backbone with torch.jit
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.eval()  # Set to eval mode for JIT
        self.backbone = torch.jit.trace(backbone, torch.randn(1, 3, 224, 224))
        
        # Cache static tensors
        self.register_buffer('anchor_scales', torch.tensor(ANCHOR_SCALES))
        self.register_buffer('anchor_ratios', torch.tensor(ANCHOR_RATIOS))
        
        # Use nn.Sequential for better performance
        self.fpn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Simplified detection head
        self.det_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Prediction heads with better initialization
        self.cls_head = nn.Conv2d(256, num_anchors_per_cell, kernel_size=3, padding=1)
        self.bbox_head = nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Generate and cache anchor boxes
        self.register_buffer('default_anchors', self._generate_anchors())
        
        # JIT-compile decode boxes function
        self.decode_boxes = torch.jit.script(self._decode_boxes)

    @torch.jit.ignore
    def _initialize_weights(self):
        for m in [self.fpn, self.det_conv, self.cls_head, self.bbox_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _generate_anchors(self):
        """Generate anchor boxes efficiently using vectorized operations"""
        feature_map = torch.arange(self.feature_map_size, device=self.device)
        cx, cy = torch.meshgrid(feature_map, feature_map, indexing='ij')
        centers = torch.stack(
            [(cx + 0.5) / self.feature_map_size,
             (cy + 0.5) / self.feature_map_size], dim=-1
        ).view(-1, 2)
        
        scales = self.anchor_scales.view(-1, 1)
        ratios = torch.sqrt(self.anchor_ratios).view(1, -1)
        
        ws = (scales * ratios).view(-1, 1)
        hs = (scales / ratios).view(-1, 1)
        
        # Generate all anchor boxes at once
        deltas = torch.cat([-ws, -hs, ws, hs], dim=1) / 2
        anchors = (centers.view(-1, 1, 2) + deltas).view(-1, 4)
        
        return anchors

    def forward(self, x, targets=None):
        if self.training and targets is None:
            self.eval()
            with torch.no_grad():
                result = self._forward_eval(x)
            self.train()
            return result
        elif not self.training and targets is None:
            return self._forward_eval(x)
        else:
            return self._forward_train(x, targets)

    @torch.jit.ignore
    def _forward_train(self, x, targets):
        """Training forward pass with full features"""
        features = self.backbone(x)
        features = self.fpn(features)
        features = self.det_conv(features)
        
        # Predict boxes and scores
        bbox_pred = self.bbox_head(features)
        conf_pred = self.cls_head(features)
        conf_pred = torch.sigmoid(conf_pred)
        
        # Reshape predictions
        batch_size = x.shape[0]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        conf_pred = conf_pred.permute(0, 2, 3, 1).reshape(batch_size, -1)
        
        # Transform bbox predictions
        bbox_pred = self.decode_boxes(bbox_pred, self.default_anchors)
        
        return {
            'bbox_pred': bbox_pred,
            'conf_pred': conf_pred,
            'anchors': self.default_anchors
        }

    def _forward_eval(self, x):
        """Optimized inference forward pass"""
        with torch.cuda.amp.autocast(enabled=True):
            features = self.backbone(x)
            features = self.fpn(features)
            features = self.det_conv(features)
            
            # Predict boxes and scores
            bbox_pred = self.bbox_head(features)
            conf_pred = torch.sigmoid(self.cls_head(features))
            
            # Reshape predictions efficiently
            batch_size = x.shape[0]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            conf_pred = conf_pred.permute(0, 2, 3, 1).reshape(batch_size, -1)
            
            # Transform bbox predictions
            bbox_pred = self.decode_boxes(bbox_pred, self.default_anchors)
        
        # Process each image in batch
        results = []
        for boxes, scores in zip(bbox_pred, conf_pred):
            # Filter by confidence
            mask = scores > (TRAIN_CONFIDENCE_THRESHOLD if self.training else CONFIDENCE_THRESHOLD)
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) > 0:
                # Apply efficient NMS
                nms_threshold = TRAIN_NMS_THRESHOLD if self.training else NMS_THRESHOLD
                keep_idx = efficient_nms(boxes, scores, nms_threshold)
                
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
                
                # Limit detections efficiently
                if len(keep_idx) > MAX_DETECTIONS:
                    _, topk_indices = torch.topk(scores, k=MAX_DETECTIONS)
                    boxes = boxes[topk_indices]
                    scores = scores[topk_indices]
            
            # Ensure at least one prediction
            if len(boxes) == 0:
                boxes = torch.tensor([[0.3, 0.3, 0.7, 0.7]], device=bbox_pred.device)
                scores = torch.tensor([CONFIDENCE_THRESHOLD], device=bbox_pred.device)
            
            results.append({
                'boxes': boxes,
                'scores': scores
            })
        
        return results

    @staticmethod
    def _decode_boxes(box_pred, anchors):
        """Optimized box decoding"""
        # Center form
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        # Decode predictions efficiently
        pred_centers = box_pred[..., :2] * anchor_sizes + anchor_centers
        pred_sizes = torch.exp(box_pred[..., 2:]) * anchor_sizes
        
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