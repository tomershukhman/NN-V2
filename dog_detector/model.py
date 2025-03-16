import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dog_detector.config import config
from torchvision.ops import nms

def compute_iou(boxes1, boxes2):
    """
    Compute Intersection over Union between two sets of bounding boxes.
    Args:
        boxes1: (N, 4) tensor of first set of boxes (x1, y1, x2, y2)
        boxes2: (M, 4) tensor of second set of boxes (x1, y1, x2, y2)
    Returns:
        iou: (N, M) tensor of IoU values for each box combination
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unsqueeze(1).chunk(4, dim=2)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unsqueeze(0).chunk(4, dim=2)

    # Get the coordinates of the intersection rectangle
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    return iou.squeeze(-1)

class DogDetector(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED):
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
        self.num_anchors = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)
        # Classification: (num_classes + 1) channels per anchor (background + classes)
        self.cls_head = nn.Conv2d(256, (num_classes + 1) * self.num_anchors, kernel_size=3, padding=1)
        # Regression: 4 values per anchor
        self.reg_head = nn.Conv2d(256, 4 * self.num_anchors, kernel_size=3, padding=1)
        self.anchor_scales = config.ANCHOR_SCALES
        self.anchor_ratios = config.ANCHOR_RATIOS
        # Store the input image size for proper coordinate mapping
        self.input_size = config.IMAGE_SIZE  # (width, height)

    def forward(self, x):
        features = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        cls_output = self.cls_head(x)   # [B, 2*num_anchors, H, W]
        reg_output = self.reg_head(x)   # [B, 4*num_anchors, H, W]
        return cls_output, reg_output

    def generate_anchors(self, feature_map_size, device):
        """
        Generate anchor boxes for the given feature map size.
        Maps feature map coordinates to the original image space.
        """
        fm_height, fm_width = feature_map_size
        
        # Calculate stride based on input image size and feature map size
        stride_x = self.input_size[0] / fm_width
        stride_y = self.input_size[1] / fm_height
        
        # Generate grid centers in image coordinates
        shifts_x = torch.arange(0, fm_width, device=device, dtype=torch.float32) * stride_x + stride_x / 2
        shifts_y = torch.arange(0, fm_height, device=device, dtype=torch.float32) * stride_y + stride_y / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # Generate different scale and aspect ratio anchors
        anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                w = float(scale) * np.sqrt(1.0 / ratio)  # Ensure float32
                h = float(scale) * np.sqrt(ratio)        # Ensure float32
                anchor = torch.tensor([-w/2, -h/2, w/2, h/2], device=device, dtype=torch.float32)
                anchors.append(anchor)
        
        anchors = torch.stack(anchors, dim=0)
        # Add anchors at each grid cell
        anchors = anchors[None, :, :] + centers[:, None, :]
        anchors = anchors.reshape(-1, 4)
        
        # Clip anchors to stay within image bounds
        anchors[:, 0].clamp_(min=0, max=self.input_size[0])
        anchors[:, 1].clamp_(min=0, max=self.input_size[1])
        anchors[:, 2].clamp_(min=0, max=self.input_size[0])
        anchors[:, 3].clamp_(min=0, max=self.input_size[1])
        
        return anchors

    @property
    def stride(self):
        return 32

    def post_process(self, cls_output, reg_output, anchors, conf_threshold=None, nms_threshold=None):
        conf_threshold = conf_threshold or config.CONF_THRESHOLD
        nms_threshold = nms_threshold or config.NMS_THRESHOLD
        
        batch_size = cls_output.size(0)
        
        # Reshape outputs properly based on feature map dimensions
        # Handle both 3D and 4D tensors
        if len(cls_output.shape) == 4:
            # Standard 4D tensor: [batch_size, channels, height, width]
            _, _, height, width = cls_output.shape
        else:
            # Handle 3D tensor: likely [batch_size, channels, flattened_spatial_dim]
            # Need to infer height and width
            _, channels, flattened_dim = cls_output.shape
            # Compute a reasonable feature map size based on the input image size and model stride
            feature_map_size = [dim // self.stride for dim in self.input_size]
            width, height = feature_map_size
            # Reshape to 4D for consistent processing
            cls_output = cls_output.reshape(batch_size, channels, height, width)
            reg_output = reg_output.reshape(batch_size, -1, height, width)
        
        cls_output = cls_output.reshape(batch_size, 2, self.num_anchors, height, width)
        cls_output = cls_output.permute(0, 3, 4, 2, 1).contiguous().reshape(batch_size, -1, 2)
        
        reg_output = reg_output.reshape(batch_size, 4, self.num_anchors, height, width)
        reg_output = reg_output.permute(0, 3, 4, 2, 1).contiguous().reshape(batch_size, -1, 4)
        
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
                keep_indices = nms(filtered_boxes, filtered_scores, nms_threshold)
                
                # Limit maximum detections according to config
                if len(keep_indices) > config.MAX_DETECTIONS_PER_IMAGE:
                    keep_indices = keep_indices[:config.MAX_DETECTIONS_PER_IMAGE]
                
                processed_boxes.append(filtered_boxes[keep_indices])
                processed_scores.append(filtered_scores[keep_indices])
            else:
                processed_boxes.append(torch.zeros((0, 4), device=boxes.device))
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

        # Extract regression values
        tx, ty, tw, th = reg_output.unbind(1)

        # Apply transformations
        cx = tx * anchor_w + anchor_cx
        cy = ty * anchor_h + anchor_cy
        w = torch.exp(tw) * anchor_w
        h = torch.exp(th) * anchor_h

        # Convert back to x1,y1,x2,y2 format
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        return torch.stack([x1, y1, x2, y2], dim=1)
