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

# Import main components for easy access
from dataset_package.dataset import DogDetectionDataset, CLASS_NAMES
from dataset_package.transforms import (
    get_train_transform, 
    get_val_transform, 
    get_multi_dog_transform,
    TransformedSubset
)
from dataset_package.collate import collate_fn
from dataset_package.loaders import get_data_loaders, get_total_samples

# Export these symbols when using "from dataset_package import *"
__all__ = [
    'DogDetectionDataset',
    'CLASS_NAMES',
    'get_train_transform',
    'get_val_transform',
    'get_multi_dog_transform',
    'TransformedSubset',
    'collate_fn',
    'get_data_loaders',
    'get_total_samples',
]