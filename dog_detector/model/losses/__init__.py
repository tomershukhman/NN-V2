"""
Loss functions for object detection models.
"""

from .focal_loss import FocalLoss
from .detection_loss import DetectionLoss

__all__ = ['FocalLoss', 'DetectionLoss']