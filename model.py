import torch
import torch.nn as nn
from torchvision import models
import math


class DogDetector(nn.Module):
    def __init__(self, pretrained=True, num_anchors=9, num_classes=1, feature_map_size=7):
        super(DogDetector, self).__init__()

        # Backbone using ResNet18 (or any other model you prefer)
        resnet = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # The ResNet18 backbone outputs 512 channels
        backbone_output_channels = 512

        # Store the expected feature map size for anchor generation
        self.expected_feature_map_size = feature_map_size

        # Bounding box head with adaptive normalization
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 256),  # Replace LayerNorm with GroupNorm
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(16, 128),  # Replace LayerNorm with GroupNorm
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(16, 128),  # Replace LayerNorm with GroupNorm
            # 4 values per anchor for each position
            nn.Conv2d(128, 4 * num_anchors, kernel_size=1, stride=1)
        )

        # Classification head using convolutional layers with adaptive normalization
        self.classification_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 256),  # Replace LayerNorm with GroupNorm
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(16, 128),  # Replace LayerNorm with GroupNorm
            nn.Conv2d(128, num_classes * num_anchors, kernel_size=1,
                      stride=1)  # One value per anchor for each position
        )

        # Initialize weights
        for m in [self.bbox_head, self.classification_head]:
            for layer in m.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Anchor box generation
        self.default_anchors = self._generate_anchors(feature_map_size)
        self.default_anchors = self.default_anchors.to(
            next(self.parameters()).device)
        self.num_anchors = num_anchors
        self.feature_map_size = feature_map_size

    def _generate_anchors(self, feature_map_size):
        """Generate anchor boxes across the feature map"""
        anchors = []
        for i in range(feature_map_size):
            for j in range(feature_map_size):
                cx = (j + 0.5) / feature_map_size
                cy = (i + 0.5) / feature_map_size
                for scale in [0.5, 1.0, 2.0]:  # Different scales for anchor boxes
                    for ratio in [0.5, 1.0, 2.0]:  # Different aspect ratios
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, x, targets=None):
        # Feature extraction from backbone
        features = self.backbone(x)  # [batch_size, 512, H, W]
        batch_size = x.size(0)

        # Get actual feature map size from backbone output
        _, _, feature_h, feature_w = features.shape

        # If feature map size doesn't match expected size, regenerate anchors
        if feature_h != self.feature_map_size or feature_w != self.feature_map_size:
            self.feature_map_size = feature_h  # Assuming square feature maps
            self.default_anchors = self._generate_anchors(
                self.feature_map_size)
            self.default_anchors = self.default_anchors.to(features.device)

        # Classification head (fully convolutional)
        conf_pred = self.classification_head(
            features)  # [batch_size, num_anchors, 7, 7]
        conf_pred = conf_pred.permute(0, 2, 3, 1).reshape(
            batch_size, -1)  # [batch_size, num_total_anchors]
        conf_pred = torch.sigmoid(conf_pred)

        # Bounding box head
        # [batch_size, 4*num_anchors, 7, 7]
        bbox_pred = self.bbox_head(features)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            batch_size, -1, 4)  # [batch_size, num_total_anchors, 4]

        # During training, return the raw predictions
        if self.training and targets is not None:
            return {
                'bbox_pred': bbox_pred,
                'conf_pred': conf_pred,
                'anchors': self.default_anchors
            }

        # During inference, handle multiple detections
        results = []
        for boxes, scores in zip(bbox_pred, conf_pred):
            mask = scores > 0.5  # Confidence threshold
            boxes = boxes[mask]
            scores = scores[mask]

            if len(boxes) > 0:
                # Apply NMS
                boxes = torch.clamp(boxes, 0, 1)
                keep_idx = self._nms(boxes, scores, 0.3)

                if len(keep_idx) > 50:
                    scores_for_topk = scores[keep_idx]
                    _, topk_indices = torch.topk(scores_for_topk, k=50)
                    # Ensure topk_indices is on the same device as keep_idx
                    topk_indices = topk_indices.to(keep_idx.device)
                    keep_idx = keep_idx[topk_indices]

                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
            else:
                boxes = torch.empty((0, 4), device=boxes.device)
                scores = torch.empty(0, device=scores.device)

            results.append({
                'boxes': boxes,
                'scores': scores,
                'anchors': self.default_anchors
            })

        return results

    def _nms(self, boxes, scores, iou_threshold):
        """Apply Non-Maximum Suppression (NMS)"""
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        boxes = boxes[sorted_idx]

        keep_idx = []
        while len(boxes) > 0:
            keep_idx.append(sorted_idx[0])
            if len(boxes) == 1:
                break
            iou = self._compute_iou(boxes[0], boxes[1:])
            non_overlap = iou < iou_threshold
            # Make sure we're only selecting from the remaining boxes (indices 1 and onwards)
            # This ensures sorted_idx and boxes have the same length after filtering
            boxes = boxes[1:][non_overlap]
            sorted_idx = sorted_idx[1:][non_overlap]

        return torch.tensor(keep_idx)

    def _compute_iou(self, box1, boxes):
        """Calculate Intersection over Union (IoU)"""
        x1 = torch.max(box1[0], boxes[:, 0])
        y1 = torch.max(box1[1], boxes[:, 1])
        x2 = torch.min(box1[2], boxes[:, 2])
        y2 = torch.min(box1[3], boxes[:, 3])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - inter_area

        iou = inter_area / union_area
        return iou


def get_model(device):
    model = DogDetector()
    model = model.to(device)
    return model
