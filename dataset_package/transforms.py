"""
Transform functions for data augmentation and preprocessing.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transform(has_dog=False):
    """Transform pipeline optimized based on dataset statistics"""
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(
                size=(320, 320),
                scale=(0.6 if has_dog else 0.7, 1.0),  # More aggressive scaling for dogs
                ratio=(0.7 if has_dog else 0.8, 1.3 if has_dog else 1.2)  # Wider ratio range for dogs
            ),
            A.Resize(320, 320)
        ], p=1.0),
        
        # Color augmentations - more aggressive for dogs
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3 if has_dog else 0.2,
                contrast_limit=0.3 if has_dog else 0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20 if has_dog else 10,
                sat_shift_limit=30 if has_dog else 20,
                val_shift_limit=30 if has_dog else 20,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=20 if has_dog else 10,
                g_shift_limit=20 if has_dog else 10,
                b_shift_limit=20 if has_dog else 10,
                p=1.0
            )
        ], p=0.7 if has_dog else 0.5),  # Higher probability for dogs
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.15 if has_dog else 0.1,
                scale_limit=0.25 if has_dog else 0.2,
                rotate_limit=20 if has_dog else 15,
                p=1.0
            ),
            A.Affine(
                scale=(0.7, 1.3) if has_dog else (0.8, 1.2),
                translate_percent={"x": (-0.15, 0.15) if has_dog else (-0.1, 0.1),
                                 "y": (-0.15, 0.15) if has_dog else (-0.1, 0.1)},
                rotate=(-20, 20) if has_dog else (-15, 15),
                p=1.0
            )
        ], p=0.4 if has_dog else 0.3),
        
        # Additional augmentations for dog images
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50) if has_dog else (5, 30), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7) if has_dog else (3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7) if has_dog else (3, 5), p=1.0)
        ], p=0.3 if has_dog else 0.2),
        
        # Dog-specific augmentations
        *([A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomToneCurve(p=1.0),
            A.RandomBrightnessContrast(p=1.0)
        ], p=0.3)] if has_dog else []),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3 if has_dog else 0.4  # More permissive for dogs
    ))

def get_val_transform():
    """Simple validation transform pipeline"""
    return A.Compose([
        A.Resize(320, 320),  # Simplified resize parameters
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

def get_multi_dog_transform():
    """Special transform for multi-dog images with careful augmentation"""
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(
                size=(320, 320),
                scale=(0.8, 1.0),  # Less aggressive scaling for multiple objects
                ratio=(0.9, 1.1)    # Less aspect ratio distortion
            ),
            A.Resize(320, 320)
        ], p=1.0),
        
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Reduced shift for multiple objects
            scale_limit=0.15,  # Reduced scale changes
            rotate_limit=10,   # Reduced rotation
            p=0.3
        ),
        
        # Subtle color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=1.0
            ),
        ], p=0.4),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.4  # Higher visibility threshold for multi-object cases
    ))

def get_augmentations(train=True):
    """
    Wrapper to get augmentations.
    
    Args:
        train (bool): If True, returns training transforms; otherwise returns validation transforms.
    
    Returns:
        A.Compose: Albumentations transform pipeline.
    """
    return get_train_transform() if train else get_val_transform()

class TransformedSubset(torch.utils.data.Subset):
    """A custom Subset class that can handle transforms"""
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, boxes, labels = self.dataset[self.indices[idx]]
        
        # Check if the sample contains a dog (label 2)
        has_dog = 2 in labels if isinstance(labels, torch.Tensor) else 2 in labels
        
        # Get appropriate transform based on content
        if self.transform is None:
            transform = get_train_transform(has_dog=has_dog)
        else:
            transform = self.transform
        
        # Ensure boxes and labels are in the correct format
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy().tolist()
        
        # Apply transforms
        transformed = transform(
            image=image,
            bboxes=boxes,
            labels=labels
        )
        
        return (
            transformed['image'],
            torch.tensor(transformed['bboxes'], dtype=torch.float32),
            torch.tensor(transformed['labels'], dtype=torch.int64)
        )
    
    def set_transform(self, transform):
        """Set a new transform for the subset"""
        self.transform = transform
