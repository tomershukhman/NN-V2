# collate.py
import torch

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    Stacks images and separates box and label targets to match trainer expectations.
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    
    # Extract boxes and labels from targets dict
    boxes = [target["boxes"] for target in targets]
    labels = [target["labels"] for target in targets]
    
    # Return format expected by trainer: (images, boxes, labels)
    return images, boxes, labels
