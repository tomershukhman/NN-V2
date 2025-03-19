import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import numpy as np

def collate_fn(batch):
    """Custom collate function to handle padded bounding boxes per image."""
    images = []
    num_dogs = []
    boxes = []
    
    for img, n_dogs, box in batch:
        images.append(img)
        num_dogs.append(n_dogs)
        boxes.append(box)
    
    # Stack all tensors
    images = torch.stack(images)
    num_dogs = torch.tensor(num_dogs)
    boxes = torch.stack(boxes)  # Now safe to stack since all tensors are same size
    
    return images, num_dogs, boxes

class TransformedSubset(torch.utils.data.Dataset):
    """Dataset wrapper that applies transforms to a subset of data."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, target = self.subset[idx]
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        bboxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.long)
        return img, target
    
    def __len__(self):
        return len(self.subset)