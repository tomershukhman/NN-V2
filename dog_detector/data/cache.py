"""
Dataset caching utilities.

This module provides functions for caching and retrieving dataset samples
to avoid repeatedly loading from the original source.
"""

import os
import torch
import zipfile
from config import DATA_SET_TO_USE, TRAIN_VAL_SPLIT


def get_cache_path(root_dir, filename='dog_detection_combined_cache.pt'):
    """
    Get the path to the cache file.
    
    Args:
        root_dir: Root directory for storing cache
        filename: Name of the cache file
        
    Returns:
        str: Path to the cache file
    """
    return os.path.join(root_dir, filename)


def load_from_cache(root_dir, split='train'):
    """
    Load dataset samples from cache.
    
    Args:
        root_dir: Root directory where cache is stored
        split: Dataset split to load ('train' or 'validation')
        
    Returns:
        tuple: (samples, dogs_per_image) or (None, None) if cache not found
    """
    cache_file = get_cache_path(root_dir)
    
    if not os.path.exists(cache_file):
        return None, None
    
    print(f"Loading combined dataset from cache: {cache_file}")
    cache_data = torch.load(cache_file)
    all_samples = cache_data['samples']
    all_dogs_per_image = cache_data['dogs_per_image']
    
    # Apply DATA_SET_TO_USE to reduce total dataset size
    total_samples = len(all_samples)
    num_samples_to_use = int(total_samples * DATA_SET_TO_USE)
    all_samples = all_samples[:num_samples_to_use]
    all_dogs_per_image = all_dogs_per_image[:num_samples_to_use]
    
    # Split into train/val using TRAIN_VAL_SPLIT
    train_size = int(len(all_samples) * TRAIN_VAL_SPLIT)
    
    if split == 'train':
        samples = all_samples[:train_size]
        dogs_per_image = all_dogs_per_image[:train_size]
    else:  # validation split
        samples = all_samples[train_size:]
        dogs_per_image = all_dogs_per_image[train_size:]
    
    print(f"Successfully loaded {len(samples)} samples for {split} split")
    print(f"Using {DATA_SET_TO_USE*100:.1f}% of total data with {TRAIN_VAL_SPLIT*100:.1f}% train split")
    
    return samples, dogs_per_image


def save_to_cache(root_dir, samples, dogs_per_image):
    """
    Save dataset samples to cache.
    
    Args:
        root_dir: Root directory where cache will be stored
        samples: List of (image_path, boxes) tuples
        dogs_per_image: List of number of dogs per image
        
    Returns:
        str: Path to the created cache file
    """
    cache_file = get_cache_path(root_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    print(f"Saving combined dataset to cache: {cache_file}")
    torch.save({
        'samples': samples,
        'dogs_per_image': dogs_per_image
    }, cache_file)
    
    return cache_file


def extract_zip(zip_path, extract_path):
    """Extract a zip file to the specified path"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)  # Clean up zip file after extraction

