"""
Model definitions and loss functions for dog detection.
"""

from .model import DogDetector, get_model
from .losses import FocalLoss, DetectionLoss
from .backbone import ResNetBackbone, get_backbone
from .detection_head import DetectionHead
from .anchor_generator import AnchorGenerator, decode_boxes

__all__ = [
    'DogDetector',
    'get_model',
    'get_backbone',
    'FocalLoss',
    'DetectionLoss',
    'ResNetBackbone',
    'DetectionHead',
    'AnchorGenerator',
    'decode_boxes'
]