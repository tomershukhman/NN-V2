import os

from config import (
    DATA_ROOT, BATCH_SIZE, NUM_WORKERS
)
from .open_images_manager import OpenImagesV7Manager
from torch.utils.data import Dataset, DataLoader


def create_dataloaders():
    """
    Create PyTorch DataLoaders for dog/non-dog classification.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        data_set_to_use (float, optional): Override DATASET_TO_USE constant
        train_val_split (float, optional): Override TRAIN_VAL_SPLIT constant
    """
    # Use module constants if not overridden
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(DATA_ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get dataset manager and datasets
    print("\n" + "="*50)
    print("Dataset Statistics Summary")
    print("="*50)
    
    manager = OpenImagesV7Manager()
    
    # Get total counts first
    # total_counts = manager.get_downloaded_images_totals()
    # total_available = total_counts['total_dogs'] + total_counts['total_non_dogs']
    # print("\nTotal Downloaded Images in Dataset:")
    # print(f"Total Downloaded Dogs: {total_counts['total_dogs']:,}")
    # print(f"Total Downloaded Non-Dogs: {total_counts['total_non_dogs']:,}")
    # print(f"Total Downloaded Images: {total_available:,}")
        
    # Get datasets with splits info
    train_dataset, val_dataset = manager.get_datasets()
    
    # Count dogs and non-dogs in train dataset
    train_dogs = sum(1 for sample in train_dataset.samples if sample['label'] == 1)
    train_non_dogs = sum(1 for sample in train_dataset.samples if sample['label'] == 0)
    train_total = len(train_dataset.samples)
    
    # Count dogs and non-dogs in val dataset
    val_dogs = sum(1 for sample in val_dataset.samples if sample['label'] == 1)
    val_non_dogs = sum(1 for sample in val_dataset.samples if sample['label'] == 0)
    val_total = len(val_dataset.samples)
    
    print("\nCurrent Dataset Split Statistics:")
    print("Training Set:")
    print(f"  - Dogs: {train_dogs:,} ({train_dogs/train_total*100:.1f}% of training set)")
    print(f"  - Non-Dogs: {train_non_dogs:,} ({train_non_dogs/train_total*100:.1f}% of training set)")
    print(f"  - Total: {train_total:,}")
    
    print("\nValidation Set:")
    print(f"  - Dogs: {val_dogs:,} ({val_dogs/val_total*100:.1f}% of validation set)")
    print(f"  - Non-Dogs: {val_non_dogs:,} ({val_non_dogs/val_total*100:.1f}% of validation set)")
    print(f"  - Total: {val_total:,}")
    
    total_dogs_used = train_dogs + val_dogs
    total_non_dogs_used = train_non_dogs + val_non_dogs
    total_used = total_dogs_used + total_non_dogs_used
    
    # print("\nTotal Used in Current Split:")
    # print(f"  - Dogs Being Used: {total_dogs_used:,} ({total_dogs_used/total_counts['total_dogs']*100:.1f}% of available dogs)")
    # print(f"  - Non-Dogs Being Used: {total_non_dogs_used:,} ({total_non_dogs_used/total_counts['total_non_dogs']*100:.1f}% of available non-dogs)")
    # print(f"  - Total Images Being Used: {total_used:,} ({total_used/total_available*100:.1f}% of available images)")
    print("="*50 + "\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader

