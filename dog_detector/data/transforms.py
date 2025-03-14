"""
Image and bounding box transformations for object detection.

This module defines the transformations applied to images and bounding
boxes during training and validation of the dog detector model.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE


def get_train_transform():
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),  # Less aggressive cropping
                ratio=(0.85, 1.15),  # Less extreme aspect ratios
                p=1.0
            ),
            A.Resize(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.0
            ),
        ], p=1.0),
        
        # Color augmentations - reduced intensity
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Reduced from 0.2
                contrast_limit=0.1,    # Reduced from 0.2
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.1,        # Reduced from 0.2
                contrast=0.1,          # Reduced from 0.2
                saturation=0.1,        # Reduced from 0.2
                hue=0.05,             # Reduced from 0.1
                p=1.0
            ),
        ], p=0.3),  # Reduced from 0.5
        
        # Noise and blur - significantly reduced
        A.OneOf([
            A.GaussNoise(
                std_range=(0.03, 0.1),  # Significantly reduced noise
                mean_range=(0.0, 0.0),
                per_channel=False,      # Changed to false to reduce noise
                noise_scale_factor=0.5,  # Reduced from 1.0
                p=1.0
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),  # Fixed blur size
        ], p=0.2),  # Reduced from 0.3
        
        # Geometric transforms - reduced intensity
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.03, 0.03),  # Reduced translation
            scale=(0.95, 1.05),              # Reduced scale variation
            rotate=(-10, 10),                # Reduced rotation
            border_mode=0,
            p=0.2                           # Reduced from 0.3
        ),
        
        # Final normalization
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.5,  # Increased from 0.3 to ensure better box visibility
        label_fields=['labels']
    ))


def get_val_transform():
    """
    Returns an Albumentations transform pipeline for validation data.
    
    Simple resizing and normalization without data augmentation for consistent evaluation.
    """
    return A.Compose([
        A.Resize(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            p=1.0
        ),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.1,
        label_fields=['labels']
    ))

class TransformedSubset:
    """
    A dataset wrapper that applies transforms to a subset of another dataset.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def _validate_and_clip_boxes(self, boxes):
        """
        Validate and clip bounding boxes in Pascal VOC format ([x1, y1, x2, y2])
        """
        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Ensure coordinates are in range [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            # Validate box dimensions
            w = x2 - x1
            h = y2 - y1
            if w > 0.01 and h > 0.01:  # Minimum size threshold
                valid_boxes.append([x1, y1, x2, y2])
        return valid_boxes

    def __getitem__(self, idx):
        import numpy as np
        import torch
        from PIL import Image as PILImage
        
        img, target = self.subset[idx]
        
        # Convert PIL image to numpy array if needed
        if isinstance(img, PILImage.Image):
            img = np.array(img)
            
        # Convert target boxes from tensor to list and validate them
        bboxes = target['boxes'].tolist()
        bboxes = self._validate_and_clip_boxes(bboxes)
        
        # Adjust labels to match the number of valid boxes
        if bboxes:
            labels = target['labels'].tolist()[:len(bboxes)]
        else:
            labels = []
        
        # Instead of returning None when no valid boxes are found,
        # return the transformed image with an empty target.
        if len(bboxes) == 0:
            transformed = self.transform(image=img, bboxes=[], labels=[])
            img = transformed['image']
            new_target = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.long),
                'image_id': target.get('image_id', torch.tensor([]))
            }
            return img, new_target
        
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.long)
        
        return img, target

    def __len__(self):
        return len(self.subset)