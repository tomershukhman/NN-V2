# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision.ops import nms
# from dog_detector.utils import compute_iou
# from config import (
#     PRETRAINED  , NUM_CLASSES,ANCHOR_SCALES,ANCHOR_RATIOS, IMAGE_SIZE,CONFIDENCE_THRESHOLD, NMS_THRESHOLD,MAX_DETECTIONS
# )

# class DogDetector(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
#         super(DogDetector, self).__init__()
#         # Load pretrained ResNet18 backbone (excluding final classification layers)
#         resnet = torchvision.models.resnet18(pretrained=pretrained)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         # Detection head layers
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         # Output heads:
#         self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
#         # Classification: (num_classes + 1) channels per anchor (background + classes)
#         self.cls_head = nn.Conv2d(256, (num_classes + 1) * self.num_anchors, kernel_size=3, padding=1)
#         # Regression: 4 values per anchor
#         self.reg_head = nn.Conv2d(256, 4 * self.num_anchors, kernel_size=3, padding=1)
#         self.anchor_scales = ANCHOR_SCALES
#         self.anchor_ratios = ANCHOR_RATIOS
#         # Store the input image size as width,height tuple
#         if isinstance(IMAGE_SIZE, (list, tuple)):
#             self.input_size = (IMAGE_SIZE[0], IMAGE_SIZE[1])  # width, height
#         else:
#             self.input_size = (IMAGE_SIZE, IMAGE_SIZE)  # square image
#         self._anchors = None
#         self._feature_map_size = None

#     def forward(self, x):
#         features = self.backbone(x)
#         x = F.relu(self.bn1(self.conv1(features)))
#         x = F.relu(self.bn2(self.conv2(x)))
        
#         # Get current feature map size
#         batch_size, _, feat_h, feat_w = x.shape
        
#         # Generate anchors if feature map size changed
#         if self._feature_map_size != (feat_h, feat_w):
#             self._feature_map_size = (feat_h, feat_w)
#             self._anchors = self.generate_anchors((feat_h, feat_w), x.device)
        
#         # Classification head: [B, (num_classes+1)*num_anchors, H, W]
#         cls_output = self.cls_head(x)
#         # Reshape to [B, num_classes+1, num_anchors, H, W]
#         cls_output = cls_output.reshape(batch_size, self.num_anchors, -1, feat_h, feat_w)
#         cls_output = cls_output.permute(0, 2, 1, 3, 4).contiguous()
        
#         # Regression head: [B, 4*num_anchors, H, W]
#         reg_output = self.reg_head(x)
#         # Reshape to [B, 4, num_anchors, H, W]
#         reg_output = reg_output.reshape(batch_size, self.num_anchors, 4, feat_h, feat_w)
#         reg_output = reg_output.permute(0, 2, 1, 3, 4).contiguous()
        
#         return cls_output, reg_output, self._anchors

#     def generate_anchors(self, feature_map_size, device):
#         """
#         Generate anchor boxes for the given feature map size.
#         Maps feature map coordinates to the original image space.
#         """
#         fm_height, fm_width = feature_map_size
        
#         # Calculate stride based on input image size and feature map size
#         stride_x = self.input_size[0] / fm_width
#         stride_y = self.input_size[1] / fm_height
#         print(f"\nAnchor Generation Debug:")
#         print(f"Feature map size: {fm_width}x{fm_height}")
#         print(f"Input image size: {self.input_size}")
#         print(f"Stride: ({stride_x:.1f}, {stride_y:.1f})")
        
#         # Generate grid centers in image coordinates
#         shifts_x = torch.arange(0, fm_width, device=device, dtype=torch.float32) * stride_x + stride_x / 2
#         shifts_y = torch.arange(0, fm_height, device=device, dtype=torch.float32) * stride_y + stride_y / 2
#         shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
#         print(f"Grid centers x range: {shifts_x.min().item():.1f} to {shifts_x.max().item():.1f}")
#         print(f"Grid centers y range: {shifts_y.min().item():.1f} to {shifts_y.max().item():.1f}")
        
#         # Generate base anchors around (0, 0)
#         base_anchors = []
#         for scale in self.anchor_scales:
#             for ratio in self.anchor_ratios:
#                 h = scale
#                 w = scale * ratio
#                 base_anchors.append([-w/2, -h/2, w/2, h/2])
                
#         base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
#         print(f"\nBase anchor sizes:")
#         for i, anchor in enumerate(base_anchors):
#             w = anchor[2] - anchor[0]
#             h = anchor[3] - anchor[1]
#             print(f"Anchor {i}: {w.item():.1f}x{h.item():.1f}")
        
#         # For each grid cell center, add all base anchors
#         all_anchors = []
#         for cy, cx in zip(shift_y.flatten(), shift_x.flatten()):
#             # Move base anchors to grid cell center
#             cell_anchors = base_anchors.clone()
#             cell_anchors[:, [0, 2]] += cx  # Add cx to x coordinates
#             cell_anchors[:, [1, 3]] += cy  # Add cy to y coordinates
#             all_anchors.append(cell_anchors)
        
#         # Stack all anchors into single tensor
#         all_anchors = torch.cat(all_anchors, dim=0)
        
#         # Clip to image bounds
#         all_anchors = torch.stack([
#             all_anchors[:, 0].clamp(min=0, max=self.input_size[0]),
#             all_anchors[:, 1].clamp(min=0, max=self.input_size[1]),
#             all_anchors[:, 2].clamp(min=0, max=self.input_size[0]),
#             all_anchors[:, 3].clamp(min=0, max=self.input_size[1])
#         ], dim=1)
        
#         # Print sample of final anchors
#         print(f"\nGenerated {len(all_anchors)} total anchors")
#         print("Sample of final anchors:")
#         for i in range(min(5, len(all_anchors))):
#             box = all_anchors[i]
#             w = box[2] - box[0]
#             h = box[3] - box[1]
#             cx = (box[0] + box[2]) / 2
#             cy = (box[1] + box[3]) / 2
#             print(f"Anchor {i}: center=({cx:.1f}, {cy:.1f}), size=({w:.1f}, {h:.1f})")
            
#         # Verify no invalid boxes
#         invalid = (all_anchors[:, 2] <= all_anchors[:, 0]) | (all_anchors[:, 3] <= all_anchors[:, 1])
#         if invalid.any():
#             print(f"\nWARNING: Found {invalid.sum().item()} invalid boxes!")
#             print("Sample invalid boxes:")
#             invalid_idx = torch.where(invalid)[0]
#             for i in range(min(3, len(invalid_idx))):
#                 idx = invalid_idx[i]
#                 print(f"Invalid box {idx}: {all_anchors[idx].tolist()}")

#         return all_anchors

#     @property
#     def stride(self):
#         return 32

#     def post_process(self, cls_output, reg_output, anchors, conf_threshold=None, nms_threshold=None):
#         """Post-process outputs to get final detections"""
#         conf_threshold = conf_threshold or CONFIDENCE_THRESHOLD  # Changed from CONF_THRESHOLD
#         nms_threshold = nms_threshold or NMS_THRESHOLD
        
#         batch_size = cls_output.size(0)
#         num_classes = cls_output.size(1)
        
#         # Reshape classification output to [batch_size, H*W*num_anchors, num_classes]
#         # Original shape: [B, num_classes, num_anchors, H, W]
#         cls_output = cls_output.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, num_classes)
        
#         # Reshape regression output to [batch_size, H*W*num_anchors, 4]
#         # Original shape: [B, 4, num_anchors, H, W]
#         reg_output = reg_output.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, 4)
        
#         # Apply softmax to get proper probabilities
#         cls_probs = F.softmax(cls_output, dim=-1)
#         scores = cls_probs[..., 1]  # Dog class probability
        
#         processed_boxes = []
#         processed_scores = []
        
#         for i in range(batch_size):
#             # Decode box coordinates
#             boxes = self._decode_boxes(reg_output[i], anchors)
            
#             # Filter by confidence threshold
#             mask = scores[i] > conf_threshold
#             filtered_boxes = boxes[mask]
#             filtered_scores = scores[i][mask]
            
#             if filtered_boxes.size(0) > 0:
#                 # Apply standard NMS
#                 keep_indices = nms(filtered_boxes, filtered_scores, nms_threshold)
                
#                 # Limit maximum detections according to config
#                 if len(keep_indices) > MAX_DETECTIONS:
#                     keep_indices = keep_indices[:MAX_DETECTIONS]
                
#                 processed_boxes.append(filtered_boxes[keep_indices])
#                 processed_scores.append(filtered_scores[keep_indices])
#             else:
#                 processed_boxes.append(torch.zeros((0, 4), device=boxes.device))
#                 processed_scores.append(torch.zeros(0, device=scores.device))
        
#         return processed_boxes, processed_scores

#     def _decode_boxes(self, reg_output, anchors):
#         """
#         Decode regression output into box coordinates using anchor boxes.
#         reg_output format: (tx, ty, tw, th) where:
#         - tx, ty: center offset relative to anchor
#         - tw, th: width and height scaling factors
#         """
#         # Extract anchor coordinates
#         anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchors.unbind(1)
#         anchor_w = anchor_x2 - anchor_x1
#         anchor_h = anchor_y2 - anchor_y1
#         anchor_cx = (anchor_x1 + anchor_x2) / 2
#         anchor_cy = (anchor_y1 + anchor_y2) / 2

#         # Extract regression values and apply sigmoid to constrain offsets
#         tx = torch.sigmoid(reg_output[:, 0]) * 2 - 1  # [-1, 1]
#         ty = torch.sigmoid(reg_output[:, 1]) * 2 - 1  # [-1, 1]
#         tw = reg_output[:, 2]
#         th = reg_output[:, 3]

#         # Apply transformations with scale factor for better stability
#         scale = 4.0  # Scale factor for offsets
#         cx = anchor_cx + tx * anchor_w / scale
#         cy = anchor_cy + ty * anchor_h / scale
#         w = torch.exp(torch.clamp(tw, -4, 4)) * anchor_w  # Clamp to prevent extreme scaling
#         h = torch.exp(torch.clamp(th, -4, 4)) * anchor_h

#         # Convert back to x1,y1,x2,y2 format
#         x1 = cx - w/2
#         y1 = cy - h/2
#         x2 = cx + w/2
#         y2 = cy + h/2

#         # Stack and return
#         boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
#         # Ensure valid boxes (no negative dimensions)
#         boxes = torch.stack([
#             boxes[:, 0],
#             boxes[:, 1],
#             torch.max(boxes[:, 2], boxes[:, 0] + 1),  # Ensure x2 > x1
#             torch.max(boxes[:, 3], boxes[:, 1] + 1)   # Ensure y2 > y1
#         ], dim=1)

#         return boxes
