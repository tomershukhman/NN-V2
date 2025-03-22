"""
Transform functions for data augmentation and preprocessing.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transform():
    """Basic transform pipeline for training that preserves detection quality"""
    return A.Compose([
        A.Resize(
            height=640,
            width=640,
            always_apply=True
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))

def get_val_transform():
    """Simple validation transform pipeline"""
    return A.Compose([
        A.Resize(
            height=640,
            width=640,
            always_apply=True
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

def get_multi_dog_transform():
    """
    Special transform for multi-dog images that's more careful about preserving all dogs.
    Uses less aggressive cropping and ensures all dogs remain visible.
    
    Returns:
        A.Compose: Albumentations transform pipeline optimized for multi-dog images
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(640,640),
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
        # Explicitly assign the attributes to ensure they're available in DataLoader workers
        self.dataset = dataset
        self.indices = indices
        self.transform = transform if transform is not None else get_val_transform()
    
    def __getitem__(self, idx):
        image, boxes, labels = self.dataset[self.indices[idx]]
        
        # Ensure boxes and labels are in the correct format for albumentations
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy().tolist()
        
        # Apply transforms
        sample = {
            'image': image,
            'bboxes': boxes,
            'labels': labels
        }
        transformed = self.transform(**sample)
        
        return (
            transformed['image'],  # Already a tensor due to ToTensorV2
            torch.tensor(transformed['bboxes'], dtype=torch.float32),
            torch.tensor(transformed['labels'], dtype=torch.int64)
        )
    
    def set_transform(self, transform):
        """Set a new transform for the subset"""
        self.transform = transform
