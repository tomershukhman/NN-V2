# loaders.py
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import download_and_prepare_dataset, create_balanced_samples, ObjectDetectionDataset, LABEL_MAP
from .transforms import get_augmentations
from .collate import collate_fn

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def create_dataloaders(batch_size=8, num_workers=4, max_samples=500, train_split=0.8):
    """
    Creates training and validation DataLoaders.
    1. Downloads dataset using FiftyOne.
    2. Balances the dataset using sample weights.
    3. Splits samples into training and validation sets.
    4. Constructs dataset objects with proper augmentations.
    5. Returns DataLoaders, using a WeightedRandomSampler for training.
    """
    foset = download_and_prepare_dataset(max_samples=max_samples)
    samples, sample_weights = create_balanced_samples(foset)
    
    num_train = int(len(samples) * train_split)
    train_samples = samples[:num_train]
    val_samples = samples[num_train:]
    train_weights = sample_weights[:num_train]
    
    train_transforms = get_augmentations(train=True)
    val_transforms = get_augmentations(train=False)
    
    train_dataset = ObjectDetectionDataset(train_samples, train_transforms, LABEL_MAP)
    val_dataset = ObjectDetectionDataset(val_samples, val_transforms, LABEL_MAP)
    
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(batch_size=8, max_samples=500)
    
    # Example: iterate over one batch from train_loader
    for images, targets in train_loader:
        print("Batch of images shape:", images.shape)
        print("Example target:", targets[0])
        break
