"""
Data loading utilities for the dog detection model.

This module provides functions for loading and batching dog detection data.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT, BATCH_SIZE, NUM_WORKERS
)
from .transforms import get_train_transform, get_val_transform
from .dataset import DogDetectionDataset


def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image.
    
    Args:
        batch: List of (image, target) tuples from the dataset
        
    Returns:
        tuple: (images, num_dogs, all_bboxes)
    """
    images = []
    num_dogs = []
    all_bboxes = []
    
    debug_batch = False  # Set to True for debugging batch content
    
    for img, target in batch:
        images.append(img)
        boxes = target['boxes']
        num_dogs.append(len(boxes))
        
        # Check if boxes are normalized and swap coordinates if needed
        if len(boxes) > 0:
            is_normalized = True
            for i in range(len(boxes)):
                for j in range(4):
                    if boxes[i, j] < 0.0 or boxes[i, j] > 1.0:
                        is_normalized = False
                        break
                if not is_normalized:
                    break
            if not is_normalized:
                if debug_batch:
                    print(f"WARNING: Found unnormalized boxes in collate_fn: {boxes}")
                boxes = torch.clamp(boxes, min=0.0, max=1.0)
            
            # Fix incorrect box coordinates (where x1 > x2 or y1 > y2)
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
        
        all_bboxes.append(boxes)
    
    # Ensure all images are the same size
    if len(images) > 0:
        expected_shape = images[0].shape
        resized_images = []
        
        for i, img in enumerate(images):
            if img.shape != expected_shape:
                # If image has different size, resize it to match the first image
                if debug_batch:
                    print(f"WARNING: Image {i} has shape {img.shape}, expected {expected_shape}")
                
                img = F.interpolate(img.unsqueeze(0), size=(expected_shape[1], expected_shape[2]), 
                                   mode='bilinear', align_corners=False).squeeze(0)
                
            resized_images.append(img)
        
        images = resized_images
    
    # Convert to tensors
    images = torch.stack(images)
    num_dogs = torch.tensor(num_dogs)
    
    # Optional debug info
    if debug_batch:
        print(f"Batch stats: {len(images)} images, avg dogs per image: {num_dogs.float().mean().item():.2f}")
        means = images.view(images.size(0), images.size(1), -1).mean(dim=2).mean(dim=0)
        stds = images.view(images.size(0), images.size(1), -1).std(dim=2).mean(dim=0)
        print(f"Image channel means: {means}, stds: {stds}")
    
    return images, num_dogs, all_bboxes


def get_data_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE, download=True):
    """
    Create data loaders for training and validation.
    
    Args:
        root: Root directory for the dataset
        batch_size: Batch size for data loading
        download: Whether to download the dataset if not available
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    os.makedirs(root, exist_ok=True)
    print(f"Using data root directory: {os.path.abspath(root)}")
    
    # Get transforms for training and validation
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create the training and validation datasets
    print("Creating training dataset...")
    try:
        train_dataset = DogDetectionDataset(
            root=root,
            split='train',
            transform=train_transform,
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
            transform=val_transform,
            download=download
        )
    except Exception as e:
        print(f"Error creating validation dataset: {e}")
        raise RuntimeError(f"Failed to create validation dataset: {e}")
    
    print(f"Train set: {len(train_dataset)} images with dogs")
    print(f"Val set: {len(val_dataset)} images with dogs")
    
    # Set up DataLoaders with optimized parameters
    num_workers = min(8, NUM_WORKERS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Ensure pinned memory for faster transfer to GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers // 2,  # Use fewer workers for validation
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def create_datasets(root=DATA_ROOT):
    """
    Create training and validation datasets without transforms.
    
    Args:
        root: Root directory for the dataset
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print("Creating training dataset...")
    train_dataset = DogDetectionDataset(
        root=root,
        split='train'
    )
    
    print("Creating validation dataset...")
    val_dataset = DogDetectionDataset(
        root=root,
        split='validation'
    )
    
    return train_dataset, val_dataset


def get_total_samples(root=DATA_ROOT):
    """
    Get the total number of samples in the dataset.
    
    Args:
        root: Root directory for the dataset
        
    Returns:
        int: Number of samples
    """
    cache_file = os.path.join(root, 'dog_detection_combined_cache.pt')
    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        return len(cache_data['samples'])
    return 0