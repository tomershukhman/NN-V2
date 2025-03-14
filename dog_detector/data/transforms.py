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
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=1.0
            ),
            A.Resize(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.0
            ),
        ], p=1.0),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(
                std_range=(0.2, 0.44),     # noise standard deviation as a fraction of max value
                mean_range=(0.0, 0.0),       # noise mean as a fraction of max value
                per_channel=True,            # sample noise for each channel independently
                noise_scale_factor=1.0,      # sample noise per pixel independently
                p=1.0
            ),            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.0625, 0.0625),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=0,
            p=0.3
        ),
        
        # Final normalization
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.3,
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