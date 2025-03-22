# collate.py
import torch

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    Stacks images and aggregates targets.
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets
