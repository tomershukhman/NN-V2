import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config import (
    NUM_CLASSES,CONFIDENCE_THRESHOLD,NMS_THRESHOLD,MAX_DETECTIONS, ANCHOR_SCALES,ANCHOR_RATIOS,IMAGE_SIZE
)
from torchvision.ops import nms


class DogDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(DogDetector, self).__init__()
        # Load pretrained ResNet18 backbone (excluding final classification layers)
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Detection head layers
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # Output heads:
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        # Classification: (num_classes + 1) channels per anchor (background + classes)
        self.cls_head = nn.Conv2d(256, (num_classes + 1) * self.num_anchors, kernel_size=3, padding=1)
        # Regression: 4 values per anchor
        self.reg_head = nn.Conv2d(256, 4 * self.num_anchors, kernel_size=3, padding=1)
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        # Store the input image size for proper coordinate mapping
        if isinstance(IMAGE_SIZE, (list, tuple)):
            self.input_size = (IMAGE_SIZE[0], IMAGE_SIZE[1])  # width, height
        else:
            self.input_size = (IMAGE_SIZE, IMAGE_SIZE)  # square image
        self._anchors = None
        self._feature_map_size = None

    def forward(self, x):
        features = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Get current feature map size
        batch_size, _, feat_h, feat_w = x.shape
        
        # Generate anchors if feature map size changed
        if self._feature_map_size != (feat_h, feat_w):
            self._feature_map_size = (feat_h, feat_w)
            self._anchors = self.generate_anchors((feat_h, feat_w), x.device)

        # Classification head: [B, (num_classes+1)*num_anchors, H, W]
        cls_output = self.cls_head(x)
        # Reshape to [B, num_classes+1, num_anchors, H, W]
        cls_output = cls_output.reshape(batch_size, self.num_anchors, -1, feat_h, feat_w)
        cls_output = cls_output.permute(0, 2, 1, 3, 4).contiguous()

        # Regression head: [B, 4*num_anchors, H, W]
        reg_output = self.reg_head(x)
        # Reshape to [B, 4, num_anchors, H, W]
        reg_output = reg_output.reshape(batch_size, self.num_anchors, 4, feat_h, feat_w)
        reg_output = reg_output.permute(0, 2, 1, 3, 4).contiguous()

        return cls_output, reg_output, self._anchors

    def generate_anchors(self, feature_map_size, device):
        """
        Generate anchor boxes for the given feature map size.
        Maps feature map coordinates to the original image space.
        """
        fm_height, fm_width = feature_map_size
        
        # Calculate stride based on input image size and feature map size
        stride_x = self.input_size[0] / fm_width
        stride_y = self.input_size[1] / fm_height
        
        # Generate grid centers properly scaled to image coordinates
        shifts_x = torch.arange(0, fm_width, device=device, dtype=torch.float32) * stride_x + stride_x / 2
        shifts_y = torch.arange(0, fm_height, device=device, dtype=torch.float32) * stride_y + stride_y / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        centers = torch.stack((shift_x, shift_y), dim=1)
        
        # Generate anchors with absolute pixel values 
        anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                # Calculate width and height in pixels
                h = scale  # Base height is the scale
                w = scale * ratio  # Width is scale * ratio
                
                # Create anchors for this scale and ratio
                anchors.append(torch.tensor([
                    -w/2, -h/2, w/2, h/2
                ], device=device, dtype=torch.float32))
        
        # Stack all base anchors
        base_anchors = torch.stack(anchors, dim=0)
        
        # Add anchors at each grid cell by broadcasting
        num_anchors = base_anchors.size(0)
        num_centers = centers.size(0)
        
        # Reshape for broadcasting
        # centers: [num_centers, 2] -> [num_centers, 1, 2] -> broadcast to [num_centers, num_anchors, 2]
        # base_anchors: [num_anchors, 4] -> [1, num_anchors, 4] -> broadcast to [num_centers, num_anchors, 4]
        expanded_centers = centers.unsqueeze(1).expand(num_centers, num_anchors, 2)
        expanded_anchors = base_anchors.unsqueeze(0).expand(num_centers, num_anchors, 4)
        
        # Apply centers to anchors
        all_anchors = torch.zeros((num_centers, num_anchors, 4), device=device)
        all_anchors[..., 0] = expanded_centers[..., 0] + expanded_anchors[..., 0]  # x1 = center_x + (-w/2)
        all_anchors[..., 1] = expanded_centers[..., 1] + expanded_anchors[..., 1]  # y1 = center_y + (-h/2)
        all_anchors[..., 2] = expanded_centers[..., 0] + expanded_anchors[..., 2]  # x2 = center_x + (w/2)
        all_anchors[..., 3] = expanded_centers[..., 1] + expanded_anchors[..., 3]  # y2 = center_y + (h/2)
        
        # Reshape to [num_centers * num_anchors, 4]
        all_anchors = all_anchors.view(-1, 4)

        # Clip anchors to stay within image bounds
        all_anchors[:, 0].clamp_(min=0, max=self.input_size[0])
        all_anchors[:, 1].clamp_(min=0, max=self.input_size[1])
        all_anchors[:, 2].clamp_(min=0, max=self.input_size[0])
        all_anchors[:, 3].clamp_(min=0, max=self.input_size[1])

        return all_anchors

    @property
    def stride(self):
        return 32

    def post_process(self, cls_output, reg_output, anchors, conf_threshold=None, nms_threshold=None):
        """Post-process outputs to get final detections"""
        conf_threshold = conf_threshold or CONFIDENCE_THRESHOLD
        nms_threshold = nms_threshold or NMS_THRESHOLD

        batch_size = cls_output.size(0)
        num_classes = cls_output.size(1)

        # Reshape classification output to [batch_size, H*W*num_anchors, num_classes]
        # Original shape: [B, num_classes, num_anchors, H, W]
        cls_output = cls_output.permute(
            0, 2, 3, 4, 1).reshape(batch_size, -1, num_classes)

        # Reshape regression output to [batch_size, H*W*num_anchors, 4]
        # Original shape: [B, 4, num_anchors, H, W]
        reg_output = reg_output.permute(
            0, 2, 3, 4, 1).reshape(batch_size, -1, 4)

        # Apply softmax to get proper probabilities
        cls_probs = F.softmax(cls_output, dim=-1)
        scores = cls_probs[..., 1]  # Dog class probability

        processed_boxes = []
        processed_scores = []

        for i in range(batch_size):
            # Decode box coordinates
            boxes = self._decode_boxes(reg_output[i], anchors)

            # Filter by confidence threshold
            mask = scores[i] > conf_threshold
            filtered_boxes = boxes[mask]
            filtered_scores = scores[i][mask]

            if filtered_boxes.size(0) > 0:
                # Apply standard NMS
                keep_indices = nms(
                    filtered_boxes, filtered_scores, nms_threshold)

                # Limit maximum detections according to config
                if len(keep_indices) > MAX_DETECTIONS:
                    keep_indices = keep_indices[:MAX_DETECTIONS]

                processed_boxes.append(filtered_boxes[keep_indices])
                processed_scores.append(filtered_scores[keep_indices])
            else:
                processed_boxes.append(
                    torch.zeros((0, 4), device=boxes.device))
                processed_scores.append(torch.zeros(0, device=scores.device))

        return processed_boxes, processed_scores

    def _decode_boxes(self, reg_output, anchors):
        """
        Decode regression output into box coordinates using anchor boxes.
        reg_output format: (tx, ty, tw, th) where:
        - tx, ty: center offset relative to anchor
        - tw, th: width and height scaling factors
        """
        # Extract anchor coordinates
        anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchors.unbind(1)
        anchor_w = anchor_x2 - anchor_x1
        anchor_h = anchor_y2 - anchor_y1
        anchor_cx = (anchor_x1 + anchor_x2) / 2
        anchor_cy = (anchor_y1 + anchor_y2) / 2

        # Extract regression values and apply sigmoid to constrain offsets
        tx = torch.sigmoid(reg_output[:, 0]) * 2 - 1  # [-1, 1]
        ty = torch.sigmoid(reg_output[:, 1]) * 2 - 1  # [-1, 1]
        tw = reg_output[:, 2]
        th = reg_output[:, 3]

        # Apply transformations with scale factor for better stability
        scale = 4.0  # Scale factor for offsets
        cx = anchor_cx + tx * anchor_w / scale
        cy = anchor_cy + ty * anchor_h / scale
        w = torch.exp(torch.clamp(tw, -4, 4)) * anchor_w  # Clamp to prevent extreme scaling
        h = torch.exp(torch.clamp(th, -4, 4)) * anchor_h

        # Convert back to x1,y1,x2,y2 format
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        # Stack and ensure valid boxes
        boxes = torch.stack([
            x1.clamp(min=0, max=self.input_size[0]),
            y1.clamp(min=0, max=self.input_size[1]),
            x2.clamp(min=0, max=self.input_size[0]),
            y2.clamp(min=0, max=self.input_size[1])
        ], dim=1)
        
        # Ensure x2 > x1 and y2 > y1
        boxes = torch.stack([
            boxes[:, 0],
            boxes[:, 1],
            torch.max(boxes[:, 2], boxes[:, 0] + 1),  # Ensure x2 > x1
            torch.max(boxes[:, 3], boxes[:, 1] + 1)   # Ensure y2 > y1
        ], dim=1)

        return boxes


def get_model(device):
    """Initialize and return the DogDetector model"""
    model = DogDetector()
    model = model.to(device)
    return model
