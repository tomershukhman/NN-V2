"""
Dataset package for multi-class object detection.

This package provides the dataset, transforms, and loaders for multi-class object detection
using the Open Images dataset. It includes tools for data loading, augmentation, and batching.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger('dog_detector')
logger.setLevel(logging.INFO)

# Import main components for easy access using relative imports
from .dataset import ObjectDetectionDataset, LABEL_MAP, download_and_prepare_dataset, create_balanced_samples
from .transforms import get_augmentations
from .collate import collate_fn
from .loaders import create_dataloaders

# Export these symbols when using "from dataset_package import *"
__all__ = [
    "ObjectDetectionDataset",
    "LABEL_MAP",
    "download_and_prepare_dataset",
    "create_balanced_samples",
    "get_augmentations",
    "collate_fn",
    "create_dataloaders",
]
