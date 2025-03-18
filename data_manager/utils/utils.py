import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import numpy as np

def collate_fn(batch):
    """Custom collate function to handle variable number of bounding boxes per image."""
    images = []
    num_dogs = []
    all_bboxes = []
    
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
                boxes = torch.clamp(boxes, min=0.0, max=1.0)
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
        all_bboxes.append(boxes)
    
    # Ensure consistent image sizes
    if len(images) > 0:
        expected_shape = images[0].shape
        resized_images = []
        
        for img in images:
            if img.shape != expected_shape:
                img = F.interpolate(img.unsqueeze(0), size=(expected_shape[1], expected_shape[2]), 
                                mode='bilinear', align_corners=False).squeeze(0)
            resized_images.append(img)
        
        images = resized_images
    
    images = torch.stack(images)
    num_dogs = torch.tensor(num_dogs)
    
    return images, num_dogs, all_bboxes

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