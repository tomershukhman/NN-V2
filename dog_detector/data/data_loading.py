"""
Data loading utilities for the dog detection model.
"""

from torch.utils.data import DataLoader, Subset
import numpy as np
from .dataset import DogDetectionDataset
from .transforms import get_train_transform, get_val_transform, TransformedSubset
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, DATA_SET_TO_USE, TRAIN_VAL_SPLIT

def collate_fn(batch):
    """Custom collate function that filters out None items in the batch."""
    # Remove any items that are None
    batch = [item for item in batch if item is not None]
    # Optionally, check if batch is empty
    if len(batch) == 0:
        return None  # or raise an exception if needed
    return tuple(zip(*batch))

def create_datasets(root=DATA_ROOT, download=False):
    """
    Create training and validation datasets.
    
    Args:
        root: Root directory for dataset storage
        download: Whether to download the dataset if not found
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create base dataset with all splits
    full_dataset = DogDetectionDataset(root=root, split='train', download=download, load_all_splits=True)
    
    # Calculate how many images to use based on DATA_SET_TO_USE
    total_images = len(full_dataset)
    num_images_to_use = int(total_images * DATA_SET_TO_USE)
    
    # Create random indices for the subset we want to use
    all_indices = np.arange(total_images)
    np.random.shuffle(all_indices)
    selected_indices = all_indices[:num_images_to_use]
    
    # Split the selected indices into train and validation sets
    num_train = int(num_images_to_use * TRAIN_VAL_SPLIT)
    train_indices = selected_indices[:num_train]
    val_indices = selected_indices[num_train:]
    
    # Create train and validation subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Add transforms
    train_dataset = TransformedSubset(train_subset, get_train_transform())
    val_dataset = TransformedSubset(val_subset, get_val_transform())
    
    print("Dataset split summary:")
    print(f"Total available images: {total_images}")
    print(f"Using {num_images_to_use} images ({DATA_SET_TO_USE * 100:.1f}% of total)")
    print(f"- Training set: {len(train_indices)} images")
    print(f"- Validation set: {len(val_indices)} images")
    
    return train_dataset, val_dataset

def get_data_loaders(root=DATA_ROOT, download=False):
    """
    Create data loaders for training and validation.
    
    Args:
        root: Root directory for dataset storage
        download: Whether to download the dataset if not found
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset, val_dataset = create_datasets(root, download)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_total_samples():
    """Get the total number of samples in all splits of the dataset."""
    dataset = DogDetectionDataset(root=DATA_ROOT, split='train', download=False, load_all_splits=True)
    return len(dataset)