"""
DataLoader creation and configuration for the dataset.
"""
import os
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import BATCH_SIZE, NUM_WORKERS, DATA_ROOT, TRAIN_VAL_SPLIT
from dataset_package.dataset import DogDetectionDataset
from dataset_package.transforms import get_train_transform, get_val_transform
from dataset_package.collate import collate_fn

logger = logging.getLogger('dog_detector')

def get_data_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE, download=True, max_samples=25000):
    """
    Create data loaders for Open Images multi-class detection dataset.
    
    Args:
        root (str): Root directory for the dataset
        batch_size (int): Batch size for dataloaders
        download (bool): If True, downloads the dataset if not available
        max_samples (int): Maximum number of samples to load
        
    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoader instances
    """
    os.makedirs(root, exist_ok=True)
    logger.info(f"Using data root directory: {os.path.abspath(root)}")
    
    # Get transforms for training and validation
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create the training and validation datasets
    logger.info("Creating training dataset...")
    try:
        train_dataset = DogDetectionDataset(
            root=root,
            split='train',
            transform=train_transform,
            download=download,
            max_samples=int(max_samples * TRAIN_VAL_SPLIT)  # Adjust for train split
        )
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise RuntimeError(f"Failed to create training dataset: {e}")
    
    logger.info("Creating validation dataset...")
    try:
        val_dataset = DogDetectionDataset(
            root=root,
            split='validation',
            transform=val_transform,
            download=download,
            max_samples=int(max_samples * (1 - TRAIN_VAL_SPLIT))  # Adjust for validation split
        )
    except Exception as e:
        logger.error(f"Error creating validation dataset: {e}")
        raise RuntimeError(f"Failed to create validation dataset: {e}")
    
    logger.info(f"Train set: {len(train_dataset)} images with objects")
    logger.info(f"Val set: {len(val_dataset)} images with objects")
    
    # Create weighted sampler for training to balance single/multi-object examples
    sample_weights = train_dataset.get_sample_weights()
    train_sampler = None
    
    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        logger.info("Using weighted sampler to balance single-object and multi-object examples")
    
    num_workers = min(8, NUM_WORKERS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,  # Ensure pinned memory
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
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
    """
    Get the total number of samples in the dataset
    
    Returns:
        int: Total number of samples in the dataset cache
    """
    cache_file = os.path.join(DATA_ROOT, 'multiclass_detection_cache.pt')
    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        return len(cache_data['samples'])
    return 0