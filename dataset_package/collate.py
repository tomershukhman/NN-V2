# collate.py
import torch

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    """
    images = []
    boxes = []
    labels = []
    
    for img, target in batch:
        images.append(img)
        boxes.append(target['boxes'])
        labels.append(target['labels'])
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    
    return images, boxes, labels
