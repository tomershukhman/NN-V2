import os
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    DATA_ROOT
)
from .dataset import DogDetectionDataset
from .transforms import get_train_transform, get_val_transform
from .utils.utils import collate_fn

def get_data_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE, download=True):
    """Create data loaders for Open Images dog detection dataset"""
    os.makedirs(root, exist_ok=True)
    print(f"Using data root directory: {os.path.abspath(root)}")
    
    print("Creating training dataset...")
    try:
        train_dataset = DogDetectionDataset(
            root=root,
            split='train',
            transform=get_train_transform(),
            download=download
        )
    except Exception as e:
        print(f"Error creating training dataset: {e}")
        raise RuntimeError(f"Failed to create training dataset: {e}")
    
    print("Creating validation dataset...")
    try:
        val_dataset = DogDetectionDataset(
            root=root,
            split='validation',
            transform=get_val_transform(),
            download=download
        )
    except Exception as e:
        print(f"Error creating validation dataset: {e}")
        raise RuntimeError(f"Failed to create validation dataset: {e}")
    
    # Update the reporting to show both total images and images with dogs
    print(f"Train set: {len(train_dataset)} total images")
    print(f"Val set: {len(val_dataset)} total images")
    
    num_workers = min(8, NUM_WORKERS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers // 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def get_total_samples():
    """Get the total number of samples in the dataset"""
    cache_file = os.path.join(DATA_ROOT, 'dog_detection_combined_cache.pt')
    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        return len(cache_data['samples'])
    return 0

def create_datasets():
    """Create training and validation datasets with the specified split ratio"""
    print("Creating training dataset...")
    train_dataset = DogDetectionDataset(
        DATA_ROOT,
        split='train'
    )
    
    print("Creating validation dataset...")
    val_dataset = DogDetectionDataset(
        DATA_ROOT,
        split='validation'
    )
    
    return train_dataset, val_dataset