import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """Returns the training transform pipeline using Albumentations"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(224, 224), 
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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))

def get_val_transform():
    """Returns the validation transform pipeline using Albumentations"""
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))