"""
Image and bounding box transformations for object detection.

This module defines the transformations applied to images and bounding
boxes during training and validation of the dog detector model.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE


def get_train_transform():
    """
    Returns an Albumentations transform pipeline for training data augmentation.
    
    Includes random crops, flips, color adjustments and other augmentations
    to improve model robustness.
    """
    return A.Compose([
        # More aggressive random crop to help with large dog detection
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE), 
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        # Color augmentations for better normalization and robustness
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=1.0
            ),
        ], p=0.5),
        # Add perspective transforms to simulate different viewpoints
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        # Scale-specific augmentations
        A.OneOf([
            A.RandomScale(scale_limit=(-0.3, 0.1), p=1.0),
            A.RandomScale(scale_limit=(0.1, 0.3), p=1.0),
        ], p=0.3),
        # Normalize and convert to tensor
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3  # Ensure boxes remain mostly visible
    ))


def get_val_transform():
    """
    Returns an Albumentations transform pipeline for validation data.
    
    Simple resizing and normalization without data augmentation for consistent evaluation.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['labels']
    ))


class TransformedSubset:
    """
    A dataset wrapper that applies transforms to a subset of another dataset.
    
    This is useful for applying different transformations to subsets of a dataset,
    such as when splitting a dataset into training and validation sets.
    """
    def __init__(self, subset, transform):
        """
        Initialize a TransformedSubset.
        
        Args:
            subset: The dataset subset to transform
            transform: The Albumentations transform to apply
        """
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        """
        Get a transformed item from the underlying dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            tuple: (transformed_image, transformed_target)
        """
        import numpy as np
        import torch
        from PIL import Image as PILImage
        
        img, target = self.subset[idx]
        
        # If img is a PIL image, convert to numpy array before transforming
        if isinstance(img, PILImage.Image):
            img = np.array(img)
            
        # Convert target boxes from tensor to list of lists
        bboxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.long)
        
        return img, target
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.subset)