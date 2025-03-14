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
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE), 
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
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
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomScale(scale_limit=(-0.3, 0.1), p=1.0),
            A.RandomScale(scale_limit=(0.1, 0.3), p=1.0),
        ], p=0.3),
        # Ensure all images are resized to IMAGE_SIZE
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels'],
        min_visibility=0.3
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
        format='coco',  # Changed from pascal_voc to coco format
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
        Validate and clip bounding boxes in COCO format ([x, y, w, h]).
        Ensures that x, y are in [0,1] and that the box does not extend past 1.
        """
        valid_boxes = []
        for box in boxes:
            x, y, w, h = box
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            # Ensure the box does not go past the image boundary
            w = max(0.0, min(w, 1.0 - x))
            h = max(0.0, min(h, 1.0 - y))
            if w > 0 and h > 0:
                valid_boxes.append([x, y, w, h])
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