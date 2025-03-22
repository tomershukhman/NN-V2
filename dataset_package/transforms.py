"""
Transform functions for data augmentation and preprocessing.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transform():
    """Transform pipeline optimized based on dataset statistics"""
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(
                size=(320, 320),  # Using size parameter instead of height/width
                scale=(0.7, 1.0),  # Wider scale range for size variation
                ratio=(0.8, 1.2)
            ),
            A.Resize(320, 320)  # Simplified resize parameters
        ], p=1.0),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(p=1.0)
        ], p=0.5),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=1.0
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                p=1.0
            )
        ], p=0.3),
        
        # Noise and blur for robustness
        A.OneOf([
            A.GaussNoise(var_limit=(5, 30), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0)
        ], p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3  # Reduced from default to handle partial occlusions
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
