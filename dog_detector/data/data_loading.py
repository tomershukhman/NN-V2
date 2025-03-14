"""
Data loading utilities for the dog detection model.
"""

from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
from .dataset import DogDetectionDataset
from .transforms import get_train_transform, get_val_transform, TransformedSubset
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, DATA_SET_TO_USE, TRAIN_VAL_SPLIT

def collate_fn(batch):
    """Custom collate function with proper error handling and filtering"""
    # Remove None items
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Ensure all items have valid structure
    filtered_batch = []
    for item in batch:
        if isinstance(item, tuple) and len(item) == 2:
            img, target = item
            if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
                filtered_batch.append(item)
    
    if len(filtered_batch) == 0:
        return None
        
    return tuple(zip(*filtered_batch))

def create_datasets(root=DATA_ROOT, download=False):
    """Create training and validation datasets with memory-efficient data handling"""
    # Create base dataset
    full_dataset = DogDetectionDataset(
        root=root, 
        split='train', 
        download=download,
        load_all_splits=True
    )
    
    # Calculate dataset split sizes
    total_images = len(full_dataset)
    num_images_to_use = int(total_images * DATA_SET_TO_USE)
    num_train = int(num_images_to_use * TRAIN_VAL_SPLIT)
    
    # Create deterministic indices for reproducibility
    all_indices = torch.arange(total_images)
    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(total_images, generator=generator)
    selected_indices = shuffled_indices[:num_images_to_use]
    
    train_indices = selected_indices[:num_train]
    val_indices = selected_indices[num_train:]
    
    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Add transforms
    train_dataset = TransformedSubset(train_subset, get_train_transform())
    val_dataset = TransformedSubset(val_subset, get_val_transform())
    
    print(f"\nDataset split summary:")
    print(f"Total available images: {total_images}")
    print(f"Using {num_images_to_use} images ({DATA_SET_TO_USE * 100:.1f}% of total)")
    print(f"- Training set: {len(train_indices)} images")
    print(f"- Validation set: {len(val_indices)} images")
    
    return train_dataset, val_dataset

def get_data_loaders(root=DATA_ROOT, download=False, batch_size=None):
    """Create data loaders with proper memory management and error handling"""
    actual_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    
    train_dataset, val_dataset = create_datasets(root, download)
    
    # Create data loaders with proper worker initialization
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    
    return train_loader, val_loader

def get_total_samples():
    """Get the total number of samples available in the dataset"""
    try:
        dataset = DogDetectionDataset(
            root=DATA_ROOT, 
            split='train',
            download=False,
            load_all_splits=True
        )
        return len(dataset)
    except Exception as e:
        print(f"Error getting total samples: {e}")
        return 0