"""
Transform functions for data augmentation and preprocessing.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """
    Create training transforms for data augmentation.
    
    Returns:
        A.Compose: Albumentations transform pipeline for training data
    """
    return A.Compose([
        # Improved random crop to better handle single objects
        A.RandomResizedCrop(
            size=(224, 224), 
            scale=(0.7, 1.0),
            ratio=(0.8, 1.25),
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        # Improved color augmentations
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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.4  # Increased from 0.3 for better quality detections
    ))

def get_val_transform():
    """
    Create validation transforms for consistent evaluation.
    
    Returns:
        A.Compose: Albumentations transform pipeline for validation data
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_multi_dog_transform():
    """
    Special transform for multi-dog images that's more careful about preserving all dogs.
    Uses less aggressive cropping and ensures all dogs remain visible.
    
    Returns:
        A.Compose: Albumentations transform pipeline optimized for multi-dog images
    """
    return A.Compose([
        # Less aggressive crop for multi-dog images to preserve more context
        A.RandomResizedCrop(
            size=(224, 224), 
            scale=(0.8, 1.0),  # Less aggressive scaling
            ratio=(0.85, 1.15),  # Less variation in aspect ratio
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),  # Less rotation to avoid cutting off dogs
        # Color augmentations for better normalization and robustness
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=15, 
                val_shift_limit=10, 
                p=1.0
            ),
        ], p=0.4),
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.5  # Higher visibility threshold for multi-dog cases
    ))

class TransformedSubset:
    """
    A class that applies transforms to a dataset subset.
    """
    def __init__(self, subset, transform):
        """
        Initialize the transformed subset.
        
        Args:
            subset: The original dataset subset
            transform: Albumentations transform to apply
        """
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        """
        Get a transformed item from the subset.
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (transformed_image, transformed_target)
        """
        from PIL import Image as PILImage
        import numpy as np
        import torch
        
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
        """
        Get the length of the subset.
        
        Returns:
            int: Length of the subset
        """
        return len(self.subset)